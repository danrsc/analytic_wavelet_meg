import numpy as np
from numba import njit, prange


__all__ = ['segment_median', 'segment_combine_events']


@njit(parallel=True)
def _parallel_quantile(v, q, bounds):
    m = np.full(len(bounds), np.nan, dtype=np.float64)
    for i in prange(len(bounds)):
        if q is None:
            m[i] = np.median(v[bounds[i][0]:bounds[i][1]])
        else:
            m[i] = np.quantile(v[bounds[i][0]:bounds[i][1]], q)
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
