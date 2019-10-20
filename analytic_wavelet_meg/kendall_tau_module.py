import numpy as np
from numba import njit, prange


__all__ = ['kendall_tau']


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
