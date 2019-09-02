from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import numpy as np
from scipy.fftpack import next_fast_len
from scipy.signal import detrend
import mne

from tqdm.auto import tqdm

__all__ = [
    'maxima_of_transform_mne_source_estimate',
    'make_epochs',
    'PValueFilterFn',
    'hertz_from_radians_per_ms',
    'radians_per_ms_from_hertz',
    'gaussian_mixtures_per_epoch',
    'common_label_count']


class PValueFilterFn:

    def __init__(self, p_value_func, p_value_threshold):
        self.p_value_func = p_value_func
        self.p_value_threshold = p_value_threshold

    def __call__(
            self,
            ea_morse,
            scale_frequencies,
            wavelet_coefficients,
            indices_maxima,
            maxima_coefficients,
            interp_fs):
        # estimate sigma for noise by assuming the highest frequency is only noise
        sigma = np.sqrt(np.mean(np.square(np.abs(wavelet_coefficients[np.argmax(scale_frequencies)]))))
        normalized_maxima = maxima_coefficients / (sigma * np.sqrt(interp_fs / np.max(scale_frequencies)))
        p_values = self.p_value_func(normalized_maxima)
        return p_values <= self.p_value_threshold


def _maxima_of_transform_mne_source_estimate_worker(
        ea_morse,
        scale_frequencies,
        psi_f,
        source,
        time_offset,
        index_time_start,
        index_time_end,
        filter_fn,
        isolated_only):
    from analytic_wavelet import analytic_wavelet_transform, maxima_of_transform

    w = analytic_wavelet_transform(source, psi_f, False)
    indices_maxima, maxima_coefficients, interp_fs = maxima_of_transform(w, scale_frequencies)

    # adjust the indices for the time offset
    indices_maxima = indices_maxima[:-1] + (indices_maxima[-1] + time_offset,)

    # remove maxima that are in the padding
    indicator_event = np.logical_and(
        indices_maxima[-1] >= index_time_start, indices_maxima[-1] < index_time_end)

    if filter_fn is not None:
        indicator_event = np.logical_and(
            indicator_event,
            filter_fn(ea_morse, scale_frequencies, w, indices_maxima, maxima_coefficients, interp_fs))

    indices_maxima = tuple(ind[indicator_event] for ind in indices_maxima)
    maxima_coefficients = maxima_coefficients[indicator_event]
    interp_fs = interp_fs[indicator_event]

    indicator_isolated, influence_regions = ea_morse.isolated_maxima(
        indices_maxima, maxima_coefficients, interp_fs)

    if isolated_only:
        indices_maxima = tuple(ind[indicator_isolated] for ind in indices_maxima)
        maxima_coefficients = maxima_coefficients[indicator_isolated]
        interp_fs = interp_fs[indicator_isolated]
        return indices_maxima, maxima_coefficients, interp_fs
    return indices_maxima, maxima_coefficients, interp_fs, indicator_isolated


def maxima_of_transform_mne_source_estimate(
        ea_morse,
        scale_frequencies,
        mne_raw,
        inv,
        epoch_start_times=None,
        epoch_duration=None,
        source_estimate_label=None,
        filter_fn=None,
        isolated_only=True,
        num_timepoints_per_batch=10000,
        **apply_inverse_raw_kwargs):
    """
    This is essentially equivalent to calling:

        indices_maxima, maxima_coefficients, interp_scale_freq = maxima_of_transform(
            analytic_wavelet_transform(mne.minimum_norm.apply_inverse_raw(...), ...), ...)
        indicator_keep = filter_fn(indices_maxima, maxima_coefficients, interp_scale_freq)
        indices_maxima, maxima_coefficients, interp_scale_freq = (
            tuple(ind[indicator_keep] for ind in indices_maxima),
            maxima_coefficients[indicator_keep], interp_scale_freq[indicator_keep])
        indicator_keep, _ = ea_morse.isolated_maxima(indices_maxima, maxima_coefficients, interp_fs)
        indices_maxima, maxima_coefficients, interp_scale_freq = (
            tuple(ind[indicator_keep] for ind in indices_maxima),
            maxima_coefficients[indicator_keep], interp_scale_freq[indicator_keep])

    But run in batches to reduce memory usage and run over multiple processors to speed things up. Source estimates
    are computed for batches of timepoints. Then dipoles within each batch are run across processors to give something
    like:

            for start_time, end_time in time_batches:
                mne_batch = mne_raw.copy().crop(start_time, end_time)
                stc = mne.minimum_norm.apply_inverse_raw(mne_batch, ...)
                # computes maxima for each dipole individually
                batch_result = process_pool_executor.map(_maxima_of_transform_worker, ..., dipoles, ...)

    Args:
        ea_morse: An instance of ElementAnalysisMorse used to run analytic_wavelet_transform and isolated_maxima
        scale_frequencies: The frequencies (in radians per time step, usually radians per ms) at which to sample
            the wavelet transform
        mne_raw: An instance of mne.io.Raw. This is the data which will be analyzed
        inv: An instance of mne.minimum_norm.InverseOperator, used to estimate the sources
        epoch_start_times: If provided, the returned values will also be "epoched". An epoch coordinate will be
            added to the result, and maxima not falling within any epoch will be filtered out. Start times should
            be in the same units as mne_raw.time. Providing epoch_start_times can also save a little computation.
            If provided, data beyond the relevant boundaries will not be processed.
        epoch_duration: How long the epochs should be in number of timepoints. Required if epoch_start_times is
            provided, ignored otherwise.
        source_estimate_label: If not None, then the analysis will be restricted to just the sources within this label
        filter_fn: Used to filter the results. Typically, this is used to place a threshold on the maxima, for example
            applying a minimum to the absolute value of the wavelet coefficients, or creating a p-value estimate
            for the wavelet coefficients and filtering using this p-value. See
            GeneralizedMorseWavelet.distribution_of_maxima_of_transformed_noise, MaximaPValueInterp1d, and
            PValueFilterFn for creating a function to estimate the p-values compared to white or red noise. filter_fn
            can also be used to filter in other ways, such as restricting the range of frequencies for the maxima.
            Note that this function must be pickle-able
        isolated_only: If True, maxima are filtered to only those which are isolated
            (see ElementAnalysisMorse.isolated_maxima). If False, an indicator will be returned describing which
            maxima are isolated.
        num_timepoints_per_batch: How large the batch should be. This controls how much memory is used. A good value
            seems to be around 10000 (the default)
        **apply_inverse_raw_kwargs: Arguments to give the mne.minimum_norm.apply_inverse_raw.
            According to the mne people, SNR of 1 (i.e. setting lambda2=1) is what you want for single trial data
            https://github.com/mne-tools/mne-python/issues/4131

    Returns:
        indices_maxima: The coordinates of each maximum. If epoch_start_times is provided, this is a tuple of
                (epochs, vertices, indices_scale_frequency, indices_time)
            otherwise, this is a tuple of
                (vertices, indices_scale_frequency, indices_time)
            Note that vertices is not necessarily the index within the stc, but is the actual vertex number. Thus,
            maxima that are discovered from a run restricted to a specific label will have the same coordinates as
            maxima that are discovered from an unrestricted run.
        maxima_coefficients: The wavelet coefficients at the maxima
        interp_scale_freq: The interpolated scale-frequencies at the maxima. These can make some of the downstream
            estimates better than using the discrete scale-frequency indices.
        indicator_isolated: A boolean array which is True for isolated events and False otherwise. Only returned if
            isolated_only is False
    """

    # we'll use this as 'padding'
    max_footprint = ea_morse.analyzing_morse.footprint(np.max(scale_frequencies))

    start_index = 0
    end_index = len(mne_raw.times)
    if epoch_start_times is not None:
        if epoch_duration is None:
            raise ValueError('epoch_duration must be provided if epoch_start_times is provided')

        start_indices = np.searchsorted(mne_raw.times, np.array(epoch_start_times), 'left')
        start_index = np.min(start_indices)
        end_index = np.max(start_indices) + 500

    if source_estimate_label is not None:
        num_sources = 0
        if source_estimate_label.hemi == 'lh' or source_estimate_label.hemi == 'both':
            num_sources += np.count_nonzero(np.in1d(inv['src'][0]['vertno'], source_estimate_label.vertices))
        if source_estimate_label.hemi == 'rh' or source_estimate_label.hemi == 'both':
            num_sources += np.count_nonzero(np.in1d(inv['src'][0]['vertno'], source_estimate_label.vertices))
    else:
        num_sources = inv['nsource']

    p_bar = tqdm(total=int(np.ceil((end_index - start_index) / num_timepoints_per_batch)) * num_sources)

    result_indices_maxima = None
    result_maxima_coefficients = None
    result_interp_fs = None
    result_indicator_isolated = None
    vertices = None

    try:
        for index_time_start in range(start_index, end_index, num_timepoints_per_batch):
            batch_start = max(index_time_start - max_footprint, 0)
            batch_length = min(num_timepoints_per_batch, end_index - index_time_start)
            padded_batch_length = next_fast_len(index_time_start + batch_length + max_footprint - batch_start)
            batch_end = min(batch_start + padded_batch_length, len(mne_raw.times))

            with mne.utils.use_log_level(False):
                mne_raw_batch = mne_raw.copy().crop(mne_raw.times[batch_start], mne_raw.times[batch_end - 1])
                # this is roughly (9000, 700000)
                source_estimate = mne.minimum_norm.apply_inverse_raw(mne_raw_batch, inv, **apply_inverse_raw_kwargs)
                if source_estimate_label is not None:
                    source_estimate = source_estimate.in_label(source_estimate_label)

            sources = source_estimate.data
            vertices = np.concatenate(source_estimate.vertices)
            del mne_raw_batch  # save some memory
            del source_estimate  # save some memory

            assert(num_sources == sources.shape[0])

            sources = detrend(sources)

            _, psi_f = ea_morse.analyzing_morse.make_wavelet(sources.shape[-1], scale_frequencies)

            with ProcessPoolExecutor() as ex:
                for index_source, source_result in enumerate(ex.map(
                        _maxima_of_transform_mne_source_estimate_worker,
                        repeat(ea_morse, sources.shape[0]),
                        repeat(scale_frequencies, sources.shape[0]),
                        repeat(psi_f, sources.shape[0]),
                        sources,
                        repeat(batch_start, sources.shape[0]),
                        repeat(index_time_start, sources.shape[0]),
                        repeat(index_time_start + batch_length, sources.shape[0]),
                        repeat(filter_fn, sources.shape[0]),
                        repeat(isolated_only, sources.shape[0]))):

                    if isolated_only:
                        indices_maxima, maxima_coefficients, interp_fs = source_result
                    else:
                        indices_maxima, maxima_coefficients, interp_fs, indicator_isolated = source_result

                    if len(maxima_coefficients) > 0:
                        if result_indices_maxima is None:
                            result_indices_maxima = ([index_source] * len(indices_maxima[0]),) \
                                                    + tuple(ind.tolist() for ind in indices_maxima)
                            result_maxima_coefficients = maxima_coefficients.tolist()
                            result_interp_fs = interp_fs.tolist()
                            if not isolated_only:
                                result_indicator_isolated = indicator_isolated.tolist()
                        else:
                            result_indices_maxima[0].extend([index_source] * len(indices_maxima[0]))
                            for i in range(1, len(result_indices_maxima)):
                                result_indices_maxima[i].extend(indices_maxima[i - 1])
                            result_maxima_coefficients.extend(maxima_coefficients)
                            result_interp_fs.extend(interp_fs)
                            if not isolated_only:
                                result_indicator_isolated.extend(indicator_isolated)
                    p_bar.update()
        p_bar.refresh()
    finally:
        p_bar.close()

    result_indices_maxima = tuple(np.array(ind) for ind in result_indices_maxima)
    result_maxima_coefficients = np.array(result_maxima_coefficients)
    result_interp_fs = np.array(result_interp_fs)
    if not isolated_only:
        result_indicator_isolated = np.array(result_indicator_isolated)

    # convert source indices to vertices
    result_indices_maxima = (vertices[result_indices_maxima[0]],) + result_indices_maxima[1:]

    # put these back in row-major order
    assert(len(result_indices_maxima) == 3)
    indices_sort = np.lexsort(result_indices_maxima)
    result_indices_maxima = tuple(ind[indices_sort] for ind in result_indices_maxima)
    result_maxima_coefficients = result_maxima_coefficients[indices_sort]
    result_interp_fs = result_interp_fs[indices_sort]
    if not isolated_only:
        result_indicator_isolated = result_indicator_isolated[indices_sort]

    if epoch_start_times is not None:
        result_indices_maxima = make_epochs(result_indices_maxima, mne_raw.times, epoch_start_times, epoch_duration)
        # a few events may not be within epoch bounds
        indicator_epoch = result_indices_maxima[0] >= 0
        result_indices_maxima = tuple(ind[indicator_epoch] for ind in result_indices_maxima)
        result_maxima_coefficients = result_maxima_coefficients[indicator_epoch]
        result_interp_fs = result_interp_fs[indicator_epoch]
        if not isolated_only:
            result_indicator_isolated = result_indicator_isolated[indicator_epoch]

    if isolated_only:
        return result_indices_maxima, result_maxima_coefficients, result_interp_fs
    return result_indices_maxima, result_maxima_coefficients, result_interp_fs, result_indicator_isolated


def make_epochs(
        coordinates,
        time,
        start_times,
        duration,
        make_time_relative=True,
        time_axis=-1):
    """
    Modifies the coordinates of maxima to account for epochs in the data
    Args:
        coordinates: A tuple of maxima coordinates with the index in time as one of the coordinates
        time: The times corresponding to each time index, usually mne_raw.times
        start_times: The start times of each epoch in the same units as time
        duration: The duration of each epoch in number of timepoints
        make_time_relative: If True, then the index in time in the returned coordinates will be relative to the
            start of the epoch
        time_axis: Which axis contains the time indices
    Returns:
        New coordinates. The first coordinate will be the epoch, the other coordinates are in the same order as
            input. items which do not belong to any epoch will have epoch == -1 and the time coordinate unmodified.
    """
    start_indices = np.searchsorted(time, start_times, 'left')
    indices_epochs = -1 * np.ones_like(coordinates[0])
    if make_time_relative:
        # copy the time axis so this operation doesn't overwrite
        coordinates = (
                coordinates[:time_axis] + np.copy(coordinates[time_axis]) + coordinates[time_axis + 1:])
    for index_stimulus, start_index in enumerate(start_indices):
        indicator_stimulus = np.logical_and(
            coordinates[time_axis] >= start_index,
            coordinates[time_axis] < start_index + duration)
        indices_epochs[indicator_stimulus] = index_stimulus
        if make_time_relative:
            coordinates[time_axis][indicator_stimulus] = coordinates[time_axis][indicator_stimulus] - start_index

    return (indices_epochs,) + coordinates


def radians_per_ms_from_hertz(x):
    return x * 2 * np.pi / 1000


def hertz_from_radians_per_ms(x):
    return x * 1000 / (2 * np.pi)


def gaussian_mixtures_per_epoch(indices_epoch, indices_time, c_hat, f_hat, n_components=1, **mixture_kwargs):
    """
    Fits a sklearn.GaussianMixture model independently for each epoch using indices_time, c_hat, and f_hat as the
    coordinates of the samples. Useful for identifying maxima across dipoles that are likely to be associated with
    the same event
    Args:
        indices_epoch: The epoch coordinates of each point
        indices_time: The time coordinates of each point
        c_hat: The estimated c parameter of each point (output from ElementAnalysisMorse.event_parameters)
        f_hat: The estimated frequency of each point (output from ElementAnalysisMorse.event_parameters)
        n_components: The number of Gaussians in each Gaussian mixture
        **mixture_kwargs: Other arguments to the GaussianMixture model

    Returns:
        labels: The cluster labels for each point. The labels are different for each epoch, so the total number of
            unique labels is len(np.unique(indices_epoch)) * n_components
        prob: The probability that a point belongs to the given label
    """
    from sklearn.mixture import GaussianMixture
    from scipy.stats import zscore

    labels = -1 * np.ones_like(indices_epoch)
    prob = np.zeros_like(f_hat)
    for epoch in np.unique(indices_epoch):

        indicator_epoch = indices_epoch == epoch

        points = np.concatenate([
            np.expand_dims(zscore(indices_time[indicator_epoch]), 1),
            np.expand_dims(zscore(f_hat[indicator_epoch]), 1),
            np.expand_dims(zscore(np.abs(c_hat[indicator_epoch])), 1)], axis=1)

        gm = GaussianMixture(n_components, **mixture_kwargs)
        epoch_labels = gm.fit_predict(points)
        labels[indicator_epoch] = epoch * n_components + epoch_labels
        prob[indicator_epoch] = np.max(gm.predict_proba(points), axis=1)

    return labels, prob


def common_label_count(vertices, labels):
    """
    For every pair of vertices v1 and v2, computes: sum_label(min(count v1 in label, count v2 in label))
    Args:
        vertices: The vertices for every sample (1d)
        labels: The labels for every sample (1d)

    Returns:
        A 2d array of shape (len(np.unique(vertices)), len(np.unique(vertices))) giving the number of common points
    """
    unique_vertices = np.unique(vertices)
    counts = list()
    for i in range(len(unique_vertices)):
        counts.append(dict(zip(*np.unique(labels[vertices == unique_vertices[i]], return_counts=True))))

    result = np.zeros((len(unique_vertices), len(unique_vertices)), np.int64)
    for i in range(len(unique_vertices)):
        for j in range(i, len(unique_vertices)):
            for label in counts[j]:
                if label in counts[i]:
                    common_count = min(counts[i][label], counts[j][label])
                    result[i, j] += common_count
                    if i != j:
                        result[j, i] += common_count
    return result
