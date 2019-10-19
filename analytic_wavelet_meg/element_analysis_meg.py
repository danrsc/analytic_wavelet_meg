from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from datetime import datetime
import numpy as np
from numba import njit, prange
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
    'common_label_count',
    'make_grid',
    'assign_grid_labels',
    'cumulative_indices_from_unique_inverse',
    'segment_combine_events',
    'segment_median',
    'kendall_tau']


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
        indices_maxima: The flat index of each maximum, which can be converted to coordinates using the returned shape
        maxima_coefficients: The wavelet coefficients at the maxima
        interp_scale_freq: The interpolated scale-frequencies at the maxima. These can make some of the downstream
            estimates better than using the discrete scale-frequency indices.
        shape: If epoch_start_times is provided, this is a tuple of
                (num_epochs, num_sources, num_scale_frequencies, num_timepoints_per_epoch)
            otherwise, this is a tuple of
                (num_sources, num_scale_frequencies, num_timepoints)
        vertices: A tuple of 2 1d arrays (left hemisphere, right hemisphere) giving the vertex numbers associated
            with the source estimate.
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
        if source_estimate_label.hemi == 'both':
            num_sources = np.count_nonzero(np.in1d(inv['src'][0]['vertno'], source_estimate_label.lh.vertices))
            num_sources += np.count_nonzero(np.in1d(inv['src'][1]['vertno'], source_estimate_label.rh.vertices))
        elif source_estimate_label.hemi == 'lh':
            num_sources = np.count_nonzero(np.in1d(inv['src'][0]['vertno'], source_estimate_label.vertices))
        else:
            assert(source_estimate_label.hemi == 'rh')
            num_sources = np.count_nonzero(np.in1d(inv['src'][1]['vertno'], source_estimate_label.vertices))
    else:
        num_sources = inv['nsource']

    total_batches = int(np.ceil((end_index - start_index) / num_timepoints_per_batch))
    p_bar = tqdm(total=total_batches * num_sources)

    result_indices_maxima = None
    result_maxima_coefficients = None
    result_interp_fs = None
    result_indicator_isolated = None
    vertices = None
    num_sources = None

    try:
        # tqdm doesn't have great support for writing to a file write now, so do this manually
        start_time = datetime.now()
        for index_of_time_batch, index_time_start in enumerate(range(start_index, end_index, num_timepoints_per_batch)):
            batch_start = max(index_time_start - max_footprint, 0)
            batch_length = min(num_timepoints_per_batch, end_index - index_time_start)
            padded_batch_length = next_fast_len(index_time_start + batch_length + max_footprint - batch_start)
            batch_end = min(batch_start + padded_batch_length, len(mne_raw.times))

            with mne.utils.use_log_level(False):
                mne_raw_batch = mne_raw.copy().crop(mne_raw.times[batch_start], mne_raw.times[batch_end - 1])
                # this is roughly (9000, 700000) if we don't crop
                source_estimate = mne.minimum_norm.apply_inverse_raw(mne_raw_batch, inv, **apply_inverse_raw_kwargs)
                if source_estimate_label is not None:
                    source_estimate = source_estimate.in_label(source_estimate_label)

            sources = source_estimate.data
            vertices = source_estimate.vertices
            num_sources = sources.shape[0]
            assert(len(vertices[0]) + len(vertices[1]) == num_sources)
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

            # manually write to output so we get some indication in the log file of where
            # we are. There are ways to do this with tqdm itself, but it's a bit complicated
            # and there are issues so we just do this way
            elapsed_time = datetime.now() - start_time
            avg_time = elapsed_time / (index_of_time_batch + 1)
            est_time = (total_batches - index_of_time_batch - 1) * avg_time
            tqdm.write('{percent:.3f}% | {count} / {total} [{elapsed}<{est}, {hours_per_it:.3f} h/it'.format(
                percent=(index_of_time_batch + 1) / total_batches * 100,
                count=index_of_time_batch + 1,
                total=total_batches,
                elapsed=tqdm.format_interval(int(elapsed_time.total_seconds())),
                est=tqdm.format_interval(int(est_time.total_seconds())),
                hours_per_it=avg_time.total_seconds() / 3600))

        p_bar.refresh()
    finally:
        p_bar.close()

    result_indices_maxima = tuple(np.array(ind) for ind in result_indices_maxima)
    result_maxima_coefficients = np.array(result_maxima_coefficients)
    result_interp_fs = np.array(result_interp_fs)
    if not isolated_only:
        result_indicator_isolated = np.array(result_indicator_isolated)

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

        shape = len(epoch_start_times), num_sources, len(scale_frequencies), epoch_duration
    else:
        shape = num_sources, len(scale_frequencies), end_index - start_index

    result_indices_maxima = np.ravel_multi_index(result_indices_maxima, shape)

    to_return = result_indices_maxima, result_maxima_coefficients, result_interp_fs, shape, vertices
    if not isolated_only:
        to_return = to_return + (result_indicator_isolated,)

    return to_return


def make_epochs(
        coordinates,
        time,
        start_times,
        duration,
        time_axis=-1):
    """
    Modifies the coordinates of maxima to account for epochs in the data
    Args:
        coordinates: A tuple of maxima coordinates with the index in time as one of the coordinates
        time: The times corresponding to each time index, usually mne_raw.times
        start_times: The start times of each epoch in the same units as time
        duration: The duration of each epoch in number of timepoints
        time_axis: Which axis contains the time indices
    Returns:
        New coordinates. The first coordinate will be the epoch, the other coordinates are in the same order as
            input. items which do not belong to any epoch will have epoch == -1 and the time coordinate unmodified.
    """
    start_indices = np.searchsorted(time, start_times, 'left')
    indices_epochs = -1 * np.ones_like(coordinates[0])
    # copy the time axis so this operation doesn't overwrite
    if time_axis < 0:
        time_axis = time_axis + len(coordinates)
        assert(0 <= time_axis < len(coordinates))
    coordinates = (
        coordinates[:time_axis] + (np.copy(coordinates[time_axis]),) + coordinates[time_axis + 1:])
    for index_stimulus, start_index in enumerate(start_indices):
        indicator_stimulus = np.logical_and(
            coordinates[time_axis] >= start_index,
            coordinates[time_axis] < start_index + duration)
        indices_epochs[indicator_stimulus] = index_stimulus
        coordinates[time_axis][indicator_stimulus] = coordinates[time_axis][indicator_stimulus] - start_index

    return (indices_epochs,) + coordinates


def radians_per_ms_from_hertz(x):
    return x * 2 * np.pi / 1000


def hertz_from_radians_per_ms(x):
    return x * 1000 / (2 * np.pi)


def make_grid(ea_morse, num_timepoints, scale_frequencies):
    """
    Divides the time-frequency plane of each epoch into a grid, so that each point described by the coordinates
    (indices_time, f_hat) can later be assigned a grid-element label indicating which grid-element the point belongs
    to. The given scale_frequencies are used as the lower and upper frequency bounds in the grid, so they should be
    sub-sampled before being passed into this function if fewer frequency bins are desired. The time bins are
    created by using the footprint of the generalized morse wavelet at each scale as the width of the bin. The
    footprint is computed by using the higher frequency bound of each frequency bin.
    Args:
        ea_morse: An instance of ElementAnalysisMorse, used to compute the bins.
        num_timepoints: The number of timepoints in the shape of the time-scale plane
        scale_frequencies: The scale_frequencies that define the bin edges along the scale axis

    Returns:
        grid: A (num_grid_elements, 4) array giving the bounds of each bin as
            (lower_time, upper_time, lower_freq, upper_freq)
    """

    # make sure these are in descending order
    scale_frequencies = scale_frequencies[np.argsort(-scale_frequencies)]
    footprint = ea_morse.analyzing_morse.footprint(scale_frequencies)
    grid = list()
    for i in range(1, len(scale_frequencies)):
        foot = footprint[i - 1]
        low_freq = scale_frequencies[i]
        high_freq = scale_frequencies[i - 1]
        for j in range(foot, num_timepoints + foot, foot):
            grid.append((j - foot, j, low_freq, high_freq))
    return np.array(grid)


def assign_grid_labels(grid, indices_time, f_hat):
    """
    Assigns to each point described by the coordinates (indices_time, f_hat) a grid-element label indicating which
    grid-element the point belongs to, where grid is computed by make_grid. Points which have a higher frequency than
    the upper bound on frequency in the bin with the maximum upper bound on frequency are assigned to the highest
    frequency bin (at the same time interval). Similarly with points having a frequency below the lower bound of the
    lowest frequency bin.
    Args:
        grid: A (num_grid_elements, 4) array giving the bounds of each bin as
            (lower_time, upper_time, lower_freq, upper_freq). Typically computed by make_grid
        indices_time: The time coordinates of each point.
        f_hat: The frequency coordinates of each point (in radians).

    Returns:
        grid_assignments: A 1d array of grid labels between 0 and num_grid_elements which is the same length
            as indices_time
    """

    grid_assignments = -1 * np.ones_like(indices_time)
    max_upper_bound = np.max(grid[:, 3])
    min_lower_bound = np.min(grid[:, 2])
    for index_grid, grid_element in enumerate(grid):
        indicator_element = np.logical_and(indices_time >= grid_element[0], indices_time < grid_element[1])
        if grid_element[3] != max_upper_bound:  # no upper bound on frequency for the highest frequency
            indicator_element = np.logical_and(indicator_element, f_hat < grid_element[3])
        if grid_element[2] != min_lower_bound:  # no lower bound on frequency for the lowest frequency
            indicator_element = np.logical_and(indicator_element, f_hat >= grid_element[2])
        grid_assignments[indicator_element] = index_grid

    assert (np.all(grid_assignments >= 0))

    return grid_assignments


def cumulative_indices_from_unique_inverse(inverse, counts):
    """
    Given the inverse and counts output from np.unique, computes the index of each occurrence of each unique item
    within inverse. For example if inverse is
        [0, 1, 0, 3, 0, 2, 2, 1, 3]
    then this function will return
        [0, 0, 1, 0, 2, 0, 1, 1, 1]
    This can be used to create a dense representation like:
        cumulative_indices = cumulative_indices_from_unique_inverse(inverse, counts)
        dense = np.full((np.max(inverse) + 1, np.max(cumulative_indices) + 1), np.nan)
        dense[(inverse, cumulative_indices)] = 1d_data
    Args:
        inverse: The inverse output from np.unique called with return_inverse=True
        counts: The count output from np.unique called with return_counts=True

    Returns:
        The index of each occurrence of each unique item within inverse. Same size as inverse.
    """
    indices = np.cumsum(counts)
    result = np.ones(indices[-1], dtype=counts.dtype)
    result[0] = 0
    result[indices[:-1]] = -counts[:-1] + 1
    result = np.cumsum(result)
    indices_sort = np.argsort(np.argsort(inverse))
    return result[indices_sort]


@njit(parallel=True)
def _numba_sum_min(x):

    count = (x.shape[1] * (x.shape[1] + 1)) // 2

    result = np.zeros(count, x.dtype)

    for lbl in prange(x.shape[0]):
        label_result = np.zeros(count, x.dtype)
        k = 0
        for i in range(x.shape[1]):
            for j in range(i, x.shape[1]):
                if x[lbl, i] < x[lbl, j]:
                    label_result[k] = x[lbl, i]
                else:
                    label_result[k] = x[lbl, j]
                k += 1
        result += label_result

    final_result = np.zeros((x.shape[1], x.shape[1]), result.dtype)
    k = 0
    for i in range(x.shape[1]):
        for j in range(i, x.shape[1]):
            final_result[i, j] = result[k]
            final_result[j, i] = result[k]
            k += 1

    return final_result


@njit(parallel=True)
def _parallel_quantile(v, q, bounds):
    m = np.full(len(bounds), np.nan, dtype=np.float64)
    for i in prange(len(bounds)):
        if q is None:
            m[i] = np.median(v[bounds[i][0]:bounds[i][1]])
        else:
            m[i] = np.quantile(v[bounds[i][0]:bounds[i][1]])
    return m


def segment_median(segment_labels, *values, min_count=None):

    segment_label_encoder_shape = None
    if isinstance(segment_labels, (tuple, list)):
        segment_label_encoder_shape = tuple(np.max(lbl) + 1 for lbl in segment_labels)
        segment_labels = np.ravel_multi_index(segment_labels, segment_label_encoder_shape)

    segment_labels, inverse, indices_sort = _numba_unique(segment_labels)
    counts = _numba_bincount(inverse)

    if segment_label_encoder_shape is not None:
        segment_labels = np.unravel_index(segment_labels, segment_label_encoder_shape)

    values = [(v[0][indices_sort], v[1])
              if (isinstance(v, (tuple, list)) and len(v) == 2) else v[indices_sort] for v in values]
    bounds = np.concatenate([np.array([0]), _numba_cumsum(counts)])
    bounds = np.concatenate([np.expand_dims(bounds[:-1], 1), np.expand_dims(bounds[1:], 1)], axis=1)
    if min_count is not None:
        bounds = bounds[counts >= min_count]
    result = [segment_labels, counts]
    for v in values:
        if isinstance(v, (tuple, list)):
            v, q = v
            result.append(_parallel_quantile(v, q, bounds))
        else:
            result.append(_parallel_quantile(v, None, bounds))

    return tuple(result)


@njit
def _numba_unique(x):
    indices_sort = np.argsort(x)
    x = x[indices_sort]
    inverse = np.zeros_like(indices_sort)
    indicator_keep = np.full(len(indices_sort), False)
    indicator_keep[0] = True
    for i in range(1, len(inverse)):
        if x[i] != x[i - 1]:
            inverse[indices_sort[i]] = inverse[indices_sort[i - 1]] + 1
            indicator_keep[i] = True
        else:
            inverse[indices_sort[i]] = inverse[indices_sort[i - 1]]
    return x[indicator_keep], inverse, indices_sort


def _unsorted_segment_sum(segment_labels, *x):
    segment_labels, inverse, _ = _numba_unique(segment_labels)
    return (segment_labels,) + tuple(_numba_bincount(inverse, x_) for x_ in x)


def segment_combine_events(
        segment_labels, c_hat, min_count=None, power_weighted_dict=None, unweighted_dict=None, use_amplitude=False):
    """
    Combines events together according to segment_labels. The result will have 1 event per unique segment label
    Args:
        segment_labels: A 1d array or tuple giving the "segments" for each event. This would typically be something like
            (epoch, combined-source, grid-label) for each event. The number of events is along axis=0
        c_hat: The estimated complex coefficient for each event. This will be used to compute the power for
            power-weighted combinations, and will also be used to compute the magnitude of the combined event
        min_count: If not None, then unique segment_labels having fewer than this many instances will be dropped from
            the data
        use_amplitude: If True, items in power_weighted dict are weighted by abs(c_hat) rather than abs(c_hat)**2 and
            modulus is mean(abs(c_hat)) rather than L2
        power_weighted_dict: Each item in the dictionary should be a 1d array which will be combined according to
            segment_labels by using a weighted average within label, weighted by the power at each point
            (the power is computed from c_hat)
        unweighted_dict: Each item in the dictionary should be a 1d array which will be combined according to
            segment_labels using a simple average within label.

    Returns:
        unique_segment_labels: An array of the unique segment labels
        counts: The number of events in each unique segment
        modulus: The L2 norm of of the modulus of c_hat within label (or sum of the modulus if use_amplitude=True)
        modulus_rms: The rms of the modulus of c_hat within label (or mean of the modulus if use_amplitude=True)
        power_weighted_result: A dictionary having the same keys as power_weighted_dict, containing the
            weighted averages of each item from power_weighted_dict within label. Only returned if power_weighted_dict
            is not None
        unweighted_result: A dictionary having the same keys as unweighted_dict, containing the averages of each item
            from unweighted_dict within label. Only returned if unweighted_dict is not None
    """

    segment_label_encoder_shape = None
    if isinstance(segment_labels, (tuple, list)):
        segment_label_encoder_shape = tuple(np.max(lbl) + 1 for lbl in segment_labels)
        segment_labels = np.ravel_multi_index(segment_labels, segment_label_encoder_shape)

    if len(segment_labels) != len(c_hat):
        raise ValueError('The number of segment_labels must be the same as the number of items in c_hat')

    if c_hat.ndim != 1:
        raise ValueError('c_hat must be 1d')

    if use_amplitude:
        w = np.abs(c_hat)
    else:
        w = np.square(np.abs(c_hat))

    x = [w, np.ones_like(w)]
    if power_weighted_dict is not None:
        for k in power_weighted_dict:
            if not np.array_equal(power_weighted_dict[k].shape, c_hat.shape):
                raise ValueError('power_weighted_dict[{}] has a different shape than c_hat'.format(k))
            x.append(w * power_weighted_dict[k])
    if unweighted_dict is not None:
        for k in unweighted_dict:
            if not np.array_equal(unweighted_dict[k].shape, c_hat.shape):
                raise ValueError('unweighted_dict[{}] has a different shape than c_hat'.format(k))
            x.append(unweighted_dict[k])

    x = _unsorted_segment_sum(segment_labels, *x)
    segment_labels = x[0]
    x = list(x[1:])

    counts = x[1]
    if min_count is not None:
        indicator_enough = counts >= min_count
        for i in range(len(x)):
            x[i] = x[i][indicator_enough]
        counts = counts[indicator_enough]
        segment_labels = segment_labels[indicator_enough]

    if segment_label_encoder_shape is not None:
        segment_labels = np.unravel_index(segment_labels, segment_label_encoder_shape)

    w = x[0]

    offset = 2

    power_weighted_result = None
    if power_weighted_dict is not None:
        power_weighted_result = type(power_weighted_dict)()
        for i, k in enumerate(power_weighted_dict):
            power_weighted_result[k] = x[i + offset] / w
        offset += len(power_weighted_dict)

    unweighted_result = None
    if unweighted_dict is not None:
        unweighted_result = type(unweighted_dict)()
        for i, k in enumerate(unweighted_dict):
            unweighted_result[k] = x[i + offset]
        offset += len(unweighted_dict)

    if use_amplitude:
        total = w
        avg = w / counts
    else:
        total = np.sqrt(w)
        avg = np.sqrt(w / counts)

    if power_weighted_dict is not None and unweighted_dict is not None:
        return segment_labels, counts, total, avg, power_weighted_result, unweighted_result
    if power_weighted_dict is not None:
        return segment_labels, counts, total, avg, power_weighted_result
    if unweighted_dict is not None:
        return segment_labels, counts, total, avg, unweighted_result
    return segment_labels, counts, total, avg


@njit
def _numba_bincount(bins, weights=None):
    return np.bincount(bins, weights)


@njit
def _numba_cumsum(x):
    return np.cumsum(x)


def common_label_count(indices_source, indices_label, weights=None):
    """
    For every pair of vertices v1 and v2, computes: sum_label(min(count v1 in label, count v2 in label))
    Args:
        indices_source: The vertices for every sample (1d)
        indices_label: The labels for every sample (1d)
        weights: If not None, the counts are weighted by these values
    Returns:
        A 2d array of shape (len(np.unique(vertices)), len(np.unique(vertices))) giving the number of common points
    """
    num_labels = np.max(indices_label) + 1
    num_sources = np.max(indices_source) + 1
    counts = _numba_bincount(
        np.ravel_multi_index((indices_label, indices_source), (num_labels, num_sources)), weights=weights)
    if len(counts) < num_labels * num_sources:
        counts_ = np.zeros(num_labels * num_sources, dtype=counts.dtype)
        counts_[:len(counts)] = counts
        counts = counts_
    counts = np.reshape(counts, (num_labels, num_sources))

    # sum out the labels to give a count per-source. This can be useful
    # if we decide to use something like pointwise-mutual information
    # downstream
    independent_counts = np.sum(counts, axis=0, dtype=np.int64)

    # note: this step takes 1 - 4 hours for a full block on the GPU servers
    # You need to have a lot of memory to run this, it uses about 80GB
    joint_counts = _numba_sum_min(counts)

    return joint_counts, independent_counts


def kendall_tau(x):
    """
    Computes tau-b (i.e. corrects for ties) pairwise for each pair of rows in a, omitting pairs of values where either
    value is nan (same as scipy.stats.kendalltau with nan_policy='omit').
    This is a very fast implementation of Kendall's tau which uses Knight's algorithm and runs in parallel across
    each pair of rows in x. The core algorithm is JIT-compiled using numba. This runs hundreds of times faster
    than scipy's implementation.
    Args:
        x: A 2d array of shape (num_variables, num_samples_per_variable) containing ranked data.
            nan is handled by omitting any pair (x[i, k], x[j, k]) from the computation of tau_b(x[i], x[j])
            when either x[i, k] or x[j, k] is nan

    Returns:
        tau: A 2d array of shape (num_variables, num_variables) containing where tau[i, j] = tau_b(x[i], x[j])
        num_common: A 2d array of shape (num_variables, num_variables) where num_common[i, j] gives how many samples
            were actually used in the computation (i.e. x.shape[1] - num_omitted(i, j))
    """
    tau, num_common = _kendall_tau(x)
    clipped = np.clip(tau, a_min=-1, a_max=1)
    assert(np.allclose(clipped, tau, equal_nan=True))
    return clipped, num_common


@njit(parallel=True)
def _kendall_tau(a):
    # primarily ported from the C++ code at https://github.com/fruttasecca/kendall/blob/master/include/kendall.h
    # see also https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient which describes the algorithm

    result = np.full((a.shape[0], a.shape[0]), np.nan, dtype=a.dtype)
    num_common = np.zeros(result.shape, dtype=np.int64)

    # set up the upper triangle indexes
    indices_x = np.full((a.shape[0] * (a.shape[0] + 1)) // 2, -1, dtype=np.int64)
    indices_y = np.full(len(indices_x), -1, dtype=np.int64)
    m = 0
    for j in range(a.shape[0]):
        for k in range(j, a.shape[0]):
            indices_x[m] = j
            indices_y[m] = k
            m += 1

    for i in prange(len(indices_x)):
        discordant = 0

        # take the values where neither x or y is nan
        x_ = list()
        y_ = list()
        for j in range(a.shape[1]):
            if not np.isnan(a[indices_x[i], j]) and not np.isnan(a[indices_y[i], j]):
                x_.append(a[indices_x[i], j])
                y_.append(a[indices_y[i], j])
        x = np.array(x_)
        y = np.array(y_)

        # sort by y
        indices_sort = np.argsort(y)
        x = x[indices_sort]
        y = y[indices_sort]

        # stable-sort by x, now the sort order is keyed by x then y
        indices_sort = np.argsort(x, kind='mergesort')
        x = x[indices_sort]
        y = y[indices_sort]

        # n_1: sum_i((t_i * (t_i - 1)) // 2) where t_i is the number of tied values in the ith group of ties for x
        same_x = 0
        # n_3: computed like n_1, but where t_i is the number of tied values jointly in the ith group of ties for x, y
        same_xy = 0
        consecutive_same_x = 1  # t_i in n_1: current streak of ties in x
        consecutive_same_xy = 1  # t_i in n_3: current streak of pairs with same x, y

        for j in range(1, len(x)):
            if x[j] == x[j - 1]:
                consecutive_same_x += 1
                if y[j] == y[j - 1]:
                    consecutive_same_xy += 1
                else:
                    same_xy += (consecutive_same_xy * (consecutive_same_xy - 1)) // 2
                    consecutive_same_xy = 1
            else:
                same_x += (consecutive_same_x * (consecutive_same_x - 1)) // 2
                consecutive_same_x = 1

                same_xy += (consecutive_same_xy * (consecutive_same_xy - 1)) // 2
                consecutive_same_xy = 1
        # needed if the values are all equal
        same_x += (consecutive_same_x * (consecutive_same_x - 1)) // 2
        same_xy += (consecutive_same_xy * (consecutive_same_xy - 1)) // 2

        holder = np.full(len(y), np.nan, dtype=y.dtype)

        # non recursive merge sort of y
        # start from chunks of size 1 to n, merge (and count swaps)
        chunk = 1
        while chunk < len(x):
            # take 2 sorted chunks and make them one sorted chunk
            for start_chunk in range(0, len(x), 2 * chunk):
                # start and end of the left half
                start_left = start_chunk
                end_left = min(start_left + chunk, len(x))

                # start and end of the right half
                start_right = end_left
                end_right = min(start_right + chunk, len(x))

                # merge the 2 halves
                # index is used to point to the right place in the holder array
                index = start_left
                while start_left < end_left and start_right < end_right:
                    # if the pairs (ordered by x) discord when checked by y
                    # increment the number of discordant pairs by 1 for each
                    # remaining pair on the left half, because if the pair on the right
                    # half discords with the pair on the left half it surely discords
                    # with all the remaining pairs on the left half, since they all
                    # have a y greater than the y of the left half pair currently
                    # being checked
                    if y[start_left] > y[start_right]:
                        holder[index] = y[start_right]
                        start_right += 1
                        discordant += end_left - start_left
                    else:
                        holder[index] = y[start_left]
                        start_left += 1
                    index += 1

                # if the left half is over there are no more discordant pairs in this
                # chunk, the remaining pairs in the right half can be copied
                while start_right < end_right:
                    holder[index] = y[start_right]
                    start_right += 1
                    index += 1

                # if the right half is over (and the left one is not) all the
                # discordant pairs have been accounted for already
                while start_left < end_left:
                    holder[index] = y[start_left]
                    start_left += 1
                    index += 1

            # swap y and holder
            y, holder = holder, y
            chunk *= 2

        # n_2: sum_i((t_i * (t_i - 1)) // 2) where t_i is the number of tied values in the ith group of ties for y
        same_y = 0
        consecutive_same_y = 1  # t_i in n_2
        for j in range(1, len(y)):
            if y[j] == y[j - 1]:
                consecutive_same_y += 1
            else:
                same_y += (consecutive_same_y * (consecutive_same_y - 1)) // 2
                consecutive_same_y = 1
        same_y += (consecutive_same_y * (consecutive_same_y - 1)) // 2
        total_pairs = (len(x) * (len(x) - 1)) // 2
        numerator = total_pairs - same_x - same_y + same_xy - 2 * discordant
        denominator = np.sqrt((total_pairs - same_x) * (total_pairs - same_y))
        if len(x) < 2:
            result[indices_x[i], indices_y[i]] = np.nan
        else:
            if denominator == 0:
                if same_x == same_y:
                    result[indices_x[i], indices_y[i]] = 1.0
                else:
                    result[indices_x[i], indices_y[i]] = 0.0
            else:
                result[indices_x[i], indices_y[i]] = numerator / denominator
        num_common[indices_x[i], indices_y[i]] = len(x)
        result[indices_y[i], indices_x[i]] = result[indices_x[i], indices_y[i]]
        num_common[indices_y[i], indices_x[i]] = num_common[indices_x[i], indices_y[i]]

    return result, num_common
