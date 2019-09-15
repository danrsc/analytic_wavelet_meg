from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from datetime import datetime
import numpy as np
from numba import njit, prange
from scipy.fftpack import next_fast_len
from scipy.signal import detrend
from scipy.spatial.distance import cdist
import mne

from tqdm.auto import tqdm

__all__ = [
    'maxima_of_transform_mne_source_estimate',
    'make_epochs',
    'PValueFilterFn',
    'hertz_from_radians_per_ms',
    'radians_per_ms_from_hertz',
    'common_label_count',
    'source_cluster',
    'assign_grid_labels',
    'cumulative_indices_from_unique_inverse',
    'segment_combine_events']


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


def assign_grid_labels(ea_morse, scale_frequencies, indices_time, f_hat, batch_size=None):
    """
    Divides the time-scale plane of each epoch into a grid, and assigns to each point described by the coordinates
    (indices_time, f_hat) a grid-element label indicating which grid-element the point belongs to. The given
    scale_frequencies are used as the lower and upper frequency bounds in the grid, so they should be sub-sampled before
    being passed into this function if fewer frequency bins are desired. The time bins are created by using the
    footprint of the generalized morse wavelet at each scale as the width of the bin. The footprint is computed by
    using the higher frequency bound of each frequency bin. The grid-element centers are computed from these bounds,
    and the points are assigned the the grid-element centers that are nearest to them.
    Args:
        ea_morse: An instance of ElementAnalysisMorse, used to compute the bins.
        scale_frequencies: The scale_frequencies that define the bin edges along the scale axis
        indices_time: The time coordinates of each point.
        gf_hat: The frequency coordinates of each point (in radians).
        batch_size: The number of points to assign simultaneously. This keeps the memory down.

    Returns:
        grid: A (num_grid_elements, 4) array giving the bounds of each bin as
            (lower_time, upper_time, lower_freq, upper_freq)
        grid_assignments: A 1d array of grid labels between 0 and num_grid_elements which is the same length
            as indices_time
    """
    # make sure these are in descending order
    scale_frequencies = scale_frequencies[np.argsort(-scale_frequencies)]
    footprint = ea_morse.analyzing_morse.footprint(scale_frequencies)
    grid = list()
    for i in range(1, len(scale_frequencies)):
        foot = footprint[i - 1]
        low_freq = scale_frequencies[i]
        high_freq = scale_frequencies[i - 1]
        for j in range(foot, np.max(indices_time), foot):
            grid.append((j - foot, j, low_freq, high_freq))
    grid = np.array(grid)
    grid_centers = np.concatenate([
        np.mean(grid[:, :2], axis=1, keepdims=True),
        np.mean(grid[:, 2:], axis=1, keepdims=True)], axis=1)
    if batch_size is None:
        distances = cdist(
            np.concatenate([np.expand_dims(indices_time, 1), np.expand_dims(f_hat, 1)], axis=1), grid_centers)
        grid_assignments = np.argmin(distances, axis=1)
    else:
        grid_assignments = -1 * np.ones_like(indices_time)
        for batch in range(0, len(indices_time), batch_size):
            distances = cdist(
                np.concatenate([np.expand_dims(indices_time[batch:(batch + batch_size)], 1),
                                np.expand_dims(f_hat[batch:(batch + batch_size)], 1)], axis=1), grid_centers)
            grid_assignments[batch:(batch + batch_size)] = np.argmin(distances, axis=1)
    return grid, grid_assignments


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


def _vertex_label_count(vertex_label):
    vertex_label, counts = np.unique(vertex_label, axis=0, return_counts=True)
    unique_labels, label_id = np.unique(vertex_label[:, 1:], return_inverse=True, axis=0)
    unique_vertices, vertex_id = np.unique(vertex_label[:, 0], return_inverse=True)

    # put counts in a dense representation that is addressable by vertex and label
    counts_ = np.zeros((len(unique_labels), len(unique_vertices)), counts.dtype)
    counts_[(label_id, vertex_id)] = counts
    return unique_labels, unique_vertices, counts_


def _vertex_label_count_from_flat_indices(flat_indices, shape, vertex_axis, label_axes):
    indices = np.unravel_index(flat_indices, shape)
    if not np.isscalar(label_axes):
        vertex_label = [indices[vertex_axis]] + list(np.expand_dims(indices[ax], 1) for ax in label_axes)
    else:
        vertex_label = [indices[vertex_axis], indices[label_axes]]

    vertex_label = np.concatenate(list(np.expand_dims(ind, 1) for ind in vertex_label), axis=1)

    vertex_label, counts = np.unique(vertex_label, axis=0, return_counts=True)
    unique_labels, label_id = np.unique(vertex_label[:, 1:], return_inverse=True, axis=0)
    unique_vertices, vertex_id = np.unique(vertex_label[:, 0], return_inverse=True)

    # put counts in a dense representation that is addressable by vertex and label
    counts_ = np.zeros((len(unique_labels), len(unique_vertices)), counts.dtype)
    counts_[(label_id, vertex_id)] = counts
    return unique_labels, unique_vertices, counts_


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


def segment_combine_events(segment_labels, c_hat, power_weighted_dict=None, unweighted_dict=None):
    """
    Combines events together according to segment_labels. The result will have 1 event per unique segment label
    Args:
        segment_labels: A 1d or 2d array giving the "segments" for each event. This would typically be something like
            (epoch, combined-source, grid-label) for each event. The number of events is along axis=0
        c_hat: The estimated complex coefficient for each event. This will be used to compute the power for
            power-weighted combinations, and will also be used to compute the magnitude of the combined event
        power_weighted_dict: Each item in the dictionary should be a 1d array which will be combined according to
            segment_labels by using a weighted average within label, weighted by the power at each point
            (the power is computed from c_hat)
        unweighted_dict: Each item in the dictionary should be a 1d array which will be combined according to
            segment_labels using a simple average within label.

    Returns:
        unique_segment_labels: An array of the unique segment labels
        modulus: The L2 norm of of the modulus of c_hat within label
        power_weighted_result: A dictionary having the same keys as power_weighted_dict, containing the
            weighted averages of each item from power_weighted_dict within label. Only returned if power_weighted_dict
            is not None
        unweighted_result: A dictionary having the same keys as unweighted_dict, containing the averages of each item
            from unweighted_dict within label. Only returned if unweighted_dict is not None
    """
    unique_segment_labels, segment_labels = np.unique(segment_labels, axis=0, return_inverse=True)
    indices_sort = np.argsort(segment_labels)
    segment_labels = segment_labels[indices_sort]

    w = np.square(np.abs(c_hat[indices_sort]))
    total_w = np.bincount(segment_labels, weights=w)
    assert (len(total_w) == len(unique_segment_labels))

    power_weighted_result = None
    if power_weighted_dict is not None:
        power_weighted_result = type(power_weighted_dict)()
        for k in power_weighted_dict:
            power_weighted_result[k] = (
                np.bincount(segment_labels, weights=power_weighted_dict[k][indices_sort] * w) / total_w)

    total_segment = np.bincount(segment_labels)
    w = np.sqrt(total_w / total_segment)

    unweighted_result = None
    if unweighted_dict is not None:
        unweighted_result = type(unweighted_dict)()
        for k in unweighted_dict:
            unweighted_result[k] = np.bincount(segment_labels, weights=unweighted_dict[k][indices_sort])

    if power_weighted_dict is not None and unweighted_dict is not None:
        return unique_segment_labels, w, power_weighted_result, unweighted_result
    if power_weighted_dict is not None:
        return unique_segment_labels, w, power_weighted_result
    if unweighted_dict is not None:
        return unique_segment_labels, w, unweighted_result
    return unique_segment_labels, w


def common_label_count(vertices, labels, batch_size=100000):
    """
    For every pair of vertices v1 and v2, computes: sum_label(min(count v1 in label, count v2 in label))
    Args:
        vertices: The vertices for every sample (1d)
        labels: The labels for every sample (vertices.shape[0], ...)
        batch_size: If given, counting is done in parallel over batches
    Returns:
        A 2d array of shape (len(np.unique(vertices)), len(np.unique(vertices))) giving the number of common points
    """

    if labels.ndim == 1:
        labels = np.expand_dims(labels, 1)

    # unique is pretty slow if we run it on the number of points we expect, so we multi-thread it
    vertex_label = np.concatenate([np.expand_dims(vertices, 1), labels], axis=1)
    if batch_size is None:
        unique_labels, unique_vertices, counts = _vertex_label_count(vertex_label)
    else:
        unique_vertices = list()
        unique_labels = list()
        counts = list()
        with ProcessPoolExecutor() as ex:
            num_splits = int(np.ceil(len(vertex_label) / batch_size))
            for current_labels, current_vertices, current_counts in ex.map(
                    _vertex_label_count, np.array_split(vertex_label, num_splits)):
                unique_vertices.append(current_vertices)
                unique_labels.append(current_labels)
                counts.append(current_counts)

        splits_vertices = np.cumsum(list(len(v) for v in unique_vertices))[:-1]
        splits_labels = np.cumsum(list(len(l_) for l_ in unique_labels))[:-1]
        unique_vertices = np.concatenate(unique_vertices)
        unique_labels = np.concatenate(unique_labels)

        unique_vertices, destination_vertices = np.unique(unique_vertices, return_inverse=True)
        unique_labels, destination_labels = np.unique(unique_labels, return_inverse=True, axis=0)

        destination_vertices = np.split(destination_vertices, splits_vertices)
        destination_labels = np.split(destination_labels, splits_labels)

        batch_counts = counts
        counts = np.zeros((len(unique_labels), len(unique_vertices)), batch_counts[0].dtype)

        for map_labels, map_vertices, source_counts in zip(destination_labels, destination_vertices, batch_counts):
            map_labels, map_vertices = np.meshgrid(map_labels, map_vertices, indexing='ij')
            counts[(np.reshape(map_labels, -1), np.reshape(map_vertices, -1))] += np.reshape(source_counts, -1)

    # sum out the labels to give a count per-source. This can be useful
    # if we decide to use something like pointwise-mutual information
    # downstream
    independent_counts = np.sum(counts, axis=0)

    # note: this step takes 1 - 4 hours for a full block on the GPU servers
    # You need to have a lot of memory to run this, it uses about 80GB
    joint_counts = _numba_sum_min(counts)

    return unique_vertices, unique_labels, joint_counts, independent_counts


def source_cluster(source_affinity_matrix, n_clusters, sources, **spectral_kwargs):
    from sklearn.cluster import SpectralClustering
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', **spectral_kwargs)
    source_clusters = spectral_clustering.fit_predict(source_affinity_matrix)
    _, indices_vertices = np.unique(sources, return_inverse=True)
    return source_clusters, source_clusters[indices_vertices]
