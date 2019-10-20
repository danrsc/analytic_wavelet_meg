import os
import gc
from datetime import datetime
import numpy as np
from sklearn.cluster import SpectralClustering
from bottleneck import nanrankdata, nanmedian, nansum, nanmean
import mne

from tqdm.auto import tqdm

from analytic_wavelet import ElementAnalysisMorse
from analytic_wavelet_meg import radians_per_ms_from_hertz, PValueFilterFn, maxima_of_transform_mne_source_estimate, \
    make_grid, assign_grid_labels, common_label_count, kendall_tau, segment_median, segment_combine_events

all_blocks = 1, 2, 3, 4
all_subjects = 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'


def load_harry_potter(subject, block):
    from paradigms import Loader

    structural_map = {
        ('harryPotter', 'A'): 'struct4',
        ('harryPotter', 'B'): 'struct5',
        ('harryPotter', 'C'): 'struct6',
        ('harryPotter', 'D'): 'krns5D',
        ('harryPotter', 'E'): 'struct2',
        ('harryPotter', 'F'): 'krns5A',
        ('harryPotter', 'G'): 'struct1',
        ('harryPotter', 'H'): 'struct3',
        ('harryPotter', 'I'): 'krns5C'
    }

    loader = Loader(
        session_stimuli_path_format='/share/volume0/newmeg/{experiment}/meta/{subject}/sentenceBlock.mat',
        data_root='/share/volume0/newmeg/',
        recording_tuple_regex=Loader.make_standard_recording_tuple_regex(
            'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas'),
        inverse_operator_path_format=(
            '/share/volume0/newmeg/{experiment}/data/inv/{subject}/'
            '{subject}_{experiment}_trans-D_nsb-5_cb-0_raw-{structural}-7-0.2-0.8-limitTrue-rankNone-inv.fif'),
        structural_directory='/share/volume0/drschwar/structural',
        experiment_subject_to_structural_subject=structural_map)

    with mne.utils.use_log_level(False):
        inv, labels = loader.load_structural('harryPotter', subject)
        mne_raw, stimuli, _ = loader.load_block('harryPotter', subject, block)

    epoch_start_times = np.array(list(s['time_stamp'] for s in stimuli))
    duration = 500
    stimuli = list(s.text for s in stimuli)

    return mne_raw, inv, labels, stimuli, epoch_start_times, duration


def make_element_analysis_block(output_path, load_fn, subject, block, label_name=None):
    from analytic_wavelet import ElementAnalysisMorse, MaximaPValueInterp1d

    mne_raw, inv, labels, stimuli, epoch_start_times, epoch_duration = load_fn(subject, block)

    label = None
    if label_name is not None:
        label = [lbl for lbl in labels if lbl.name == label_name]
        if len(label) == 0:
            raise ValueError('No matching label for label_name: {}'.format(label_name))
        label = label[0]

    ea_morse = ElementAnalysisMorse(gamma=4, analyzing_beta=32, element_beta=32)

    fs = ea_morse.analyzing_morse.log_spaced_frequencies(
        nyquist_overlap=0.05,
        high=radians_per_ms_from_hertz(150),
        endpoint_overlap=3,
        num_timepoints=20 * 500,
        low=radians_per_ms_from_hertz(0.5))

    hist, bin_edges = ea_morse.analyzing_morse.distribution_of_maxima_of_transformed_noise(
        spectral_slope=0, scale_ratio=fs[0] / fs[1], num_monte_carlo_realizations=int(1e7))
    p_value_func = PValueFilterFn(MaximaPValueInterp1d.from_histogram(hist, bin_edges), p_value_threshold=.1)

    indices_flat, maxima_coefficients, interp_fs, shape, vertices = maxima_of_transform_mne_source_estimate(
        ea_morse, fs, mne_raw, inv, epoch_start_times, epoch_duration, source_estimate_label=label,
        filter_fn=p_value_func, lambda2=1)

    output_dir = os.path.split(output_path)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    assert(shape[0] == len(stimuli))
    assert(shape[1] == len(vertices[0]) + len(vertices[1]))
    assert(shape[2] == len(fs))

    np.savez(
        output_path,
        stimuli=np.array(stimuli),
        gamma=ea_morse.gamma,
        analyzing_beta=ea_morse.analyzing_beta,
        element_beta=ea_morse.element_beta,
        scale_frequencies=fs,
        indices_flat=indices_flat,
        maxima_coefficients=maxima_coefficients,
        interpolated_scale_frequencies=interp_fs,
        shape=shape,
        shape_order=np.array(('num_epochs', 'num_sources', 'num_scale_frequencies', 'num_timepoints_per_epoch')),
        num_timepoints_per_epoch=shape[-1],
        left_vertices=vertices[0],
        right_vertices=vertices[1])


def _local_thread_compute_shared_grid_element_counts(result_queue, ea_path, time_slice_ms, weight_kind, index_slice):
    with np.load(ea_path) as ea_block:
        ea_morse = ElementAnalysisMorse(ea_block['gamma'], ea_block['analyzing_beta'], ea_block['element_beta'])
        c_hat, _, f_hat = ea_morse.event_parameters(
            ea_block['maxima_coefficients'], ea_block['interpolated_scale_frequencies'])

        scale_frequencies = ea_block['scale_frequencies']

        indices_stimuli, indices_source, _, indices_time = np.unravel_index(
            ea_block['indices_flat'], ea_block['shape'])

        # noinspection PyTypeChecker
        grid = make_grid(ea_morse, time_slice_ms, scale_frequencies[np.arange(0, len(scale_frequencies), 10)])

        indices_time_slice = indices_time - index_slice * time_slice_ms
        indicator_slice = np.logical_and(indices_time_slice >= 0, indices_time_slice < time_slice_ms)
        grid_labels = assign_grid_labels(grid, indices_time_slice[indicator_slice], f_hat[indicator_slice])

        # flatten this index to save some memory
        combine_source_labels = np.ravel_multi_index(
            (indices_stimuli[indicator_slice], grid_labels), (ea_block['shape'][0], len(grid)))

        weights = None
        if weight_kind == 'power':
            weights = np.square(np.abs(c_hat[indicator_slice]))
        elif weight_kind == 'amplitude':
            weights = np.abs(c_hat[indicator_slice])
        elif weight_kind is not None:
            raise ValueError('Unknown weight_kind: {}'.format(weight_kind))

        joint_count, independent_count = common_label_count(
            indices_source[indicator_slice], combine_source_labels, weights=weights)

    result_queue.put((joint_count, independent_count, grid, grid_labels))


def _compute_shared_grid_element_counts(ea_path, time_slice_ms=None, weight_kind=None):
    # we invoke the main counting code in a separate process to workaround a memory leak.
    # that way, the memory gets cleaned up after each time slice
    from multiprocessing import get_context

    with np.load(ea_path) as ea_block:
        shape = ea_block['shape']

    has_slices = True
    if time_slice_ms is None:
        has_slices = False
        time_slice_ms = shape[-1]

    if shape[-1] // time_slice_ms * time_slice_ms != shape[-1]:
        raise ValueError('time_slice_ms must divide epochs exactly. Epoch has {} ms, time_slice_ms is {}'.format(
            shape[-1], time_slice_ms))

    num_slices = shape[-1] // time_slice_ms

    ctx = get_context()
    q = ctx.Queue()

    joint_count = list()
    independent_count = list()
    grid_labels = list()
    grid = None

    for index_slice in range(num_slices):
        p = ctx.Process(
            target=_local_thread_compute_shared_grid_element_counts,
            args=(q, ea_path, time_slice_ms, weight_kind, index_slice))
        p.start()
        slice_joint_count, slice_independent_count, slice_grid, slice_grid_labels = q.get()
        if grid is None:
            grid = slice_grid
        else:
            assert(np.array_equal(grid, slice_grid))
        joint_count.append(slice_joint_count)
        independent_count.append(slice_independent_count)
        grid_labels.append(slice_grid_labels)

    if has_slices:
        joint_count = np.array(joint_count)
        independent_count = np.array(independent_count)
        grid_labels = np.concatenate(grid_labels)
    else:
        joint_count = joint_count[0]
        independent_count = independent_count[0]
        grid_labels = grid_labels[0]

    return joint_count, independent_count, grid, grid_labels


def compute_shared_grid_element_counts(element_analysis_dir, subjects=None, time_slice_ms=None, weight_kind=None):
    """
    Computes how often two dipoles have events in the same 'grid element'. A grid element divides
    the time-scale plane into boxes that are log-scaled on the scale axis and scaled according to the
    wavelet footprint at the appropriate frequency on the time axis. Thus, boxes are larger in the scale-axis
    and smaller in the time axis as the frequency increases.

    Notes:
        For some reason, this function appears to "leak" memory -- it doesn't really free up all of the resources
        for one subject when it moves to the next. Possibly due to the use of numba jit code? In any case, it
        may be best to run this on 2-3 subjects at a time, then reset the kernel (if running from Jupyter) or
        restart the process and do the next 2-3.

    Args:
        element_analysis_dir: The directory where element analysis files can be found. For example:
            '/share/volume0/<your user name>/data_sets/harry_potter/element_analysis'
        subjects: Which subjects to run the counts on. If None, runs on all subjects
        time_slice_ms: If provided, counts are computed within slices of the time spanning this many ms
        weight_kind: 
            If 'power' counts are weighted by the square of the modulus of the wavelet coefficient
            If 'amplitude' counts are weighted by the modulus of the wavelet coefficient
            If None (default), counts are not weighted
    Returns:
        None. Saves an output file for each subject, containing, for each block, the keys:
            sources_<block>: The unique sources for the block, in the order of the grid element counts. As long
                as the element analysis has been run in a consistent way, these should be the same across all blocks.
            source_shared_grid_element_count_<block>: The (num_vertices, num_vertices) matrix of counts giving how
                often two vertices have events in the same grid-cell. The events must occur in the grid element
                within the same epoch in order to be counted as being shared. The count for a grid element within an
                epoch is the minimum of the count of events from vertex 1 and vertex 2 within that element in that
                epoch
            grid_<block>: A (num_grid_elements, 4) array giving the bounding boxes for each grid element as:
                (time_lower, time_upper, freq_lower, freq_upper). Note that grid-cell assignments are done by
                nearest-neighbor to the center point of each grid element. As long as element analysis has been run
                in a consistent way, these should be the same across all blocks.
            grid_elements_<block>: A 1d array of the labels of which grid element each event is assigned to.
        The output also contains the key "blocks", which gives the list of blocks in the output.
    """

    if subjects is None:
        subjects = all_subjects
    result = {'blocks': all_blocks}
    subject_joint_shape = None
    subject_grid = None

    for subject in subjects:
        print('Starting subject {} at {}'.format(subject, datetime.now()))
        for block in tqdm(all_blocks):
            ea_path = os.path.join(
                element_analysis_dir, 'harry_potter_element_analysis_{}_{}.npz'.format(subject, block))
            joint_counts, independent_counts, grid, grid_labels = _compute_shared_grid_element_counts(
                ea_path, time_slice_ms, weight_kind=weight_kind)

            # try to release some memory
            gc.collect()

            if subject_joint_shape is None:
                subject_joint_shape = joint_counts.shape
                subject_grid = grid
                result['grid'] = grid
            else:
                assert(np.array_equal(subject_joint_shape, joint_counts.shape))
                assert(np.array_equal(subject_grid, grid))

            result['joint_grid_element_count_{}'.format(block)] = joint_counts
            result['independent_grid_element_count_{}'.format(block)] = independent_counts
            result['grid_elements_{}'.format(block)] = grid_labels

        if time_slice_ms is not None:
            output_file_name = 'harry_potter_shared_grid_element_counts' \
                               '_{subject}_time_slice_ms_{time_slice_ms}{weight}.npz'
        else:
            output_file_name = 'harry_potter_shared_grid_element_counts_{subject}{weight}.npz'
        np.savez(
            os.path.join(element_analysis_dir, output_file_name.format(
                subject=subject,
                time_slice_ms=time_slice_ms,
                weight='_{}'.format(weight_kind) if weight_kind is not None else None)),
            **result)


def _renumber_clusters(clusters_in_dipole_order, previous_clusters=None):
    if previous_clusters is not None:
        # use overlap to keep consistency across slices
        from scipy.optimize import linear_sum_assignment
        if not np.array_equal(clusters_in_dipole_order.shape, previous_clusters.shape):
            raise ValueError('previous_clusters must have the same shape as clusters_in_dipole_order')
        overlap = np.zeros((np.max(clusters_in_dipole_order) + 1, np.max(previous_clusters) + 1), np.intp)
        if overlap.shape[0] != overlap.shape[1]:
            raise ValueError('clusters_in_dipole_order and previous_clusters must have the same range')
        pair_indices, overlap_ = np.unique(
            np.ravel_multi_index((
                np.reshape(clusters_in_dipole_order, -1), np.reshape(previous_clusters, -1)), overlap.shape),
            return_counts=True)
        overlap[np.unravel_index(pair_indices, overlap.shape)] = overlap_
        _, new_numbers = linear_sum_assignment(np.max(overlap) - overlap)
        return new_numbers[clusters_in_dipole_order]
    else:
        # use dipole order to try to keep consistency across runs
        indices_sort = np.argsort(clusters_in_dipole_order)
        cluster_cum_count = np.cumsum(np.bincount(clusters_in_dipole_order))
        median_ranks = list()
        for i in range(len(cluster_cum_count)):
            start = 0 if i == 0 else cluster_cum_count[i - 1]
            median_ranks.append(np.median(indices_sort[start:cluster_cum_count[i]]))
        new_numbers = np.argsort(np.array(median_ranks))
        return new_numbers[clusters_in_dipole_order]


def co_occurrence_source_cluster(
        element_analysis_dir,
        subject,
        n_clusters=100,
        use_pointwise_mutual_information=True,
        aggregation='power',
        min_events=10,
        time_slice_ms=None,
        ignore_grid_on_aggregate=False,
        knn=None,
        **spectral_kwargs):
    """
    Uses spectral clustering on an affinity matrix computed between each dipole and each other dipole, where the
    affinity matrix is based on co-occurrence of events from element-analysis.
    Args:
        element_analysis_dir: The directory where element analysis files can be found. For example:
            '/share/volume0/<your user name>/data_sets/harry_potter/element_analysis'
        subject: Which subject to run the source clustering on
        n_clusters: The number of clusters to produce
        use_pointwise_mutual_information: If True normalizes joint counts by the product of the independent counts,
            otherwise no normalization is done
        aggregation: One of 'power' (default), 'amplitude', 'median'
            Events in the same cluster, epoch, and grid-element are aggregated together by this method.
            If 'power', the aggregated event magnitude is the L2 norm of the wavelet coefficients of each
                event being aggregated. The other properties of the aggregated event are the power-weighted average
                of each property of the events being aggregated. The root-mean-squared wavelet coefficient is also
                computed and stored
            If 'amplitude', the aggregated event magnitude is the sum of the absolute wavelet coefficients of each
                event being aggregated. The other properties of the aggregated event are the absolute-coefficient
                weighted average of each property of the events being aggregated. The mean absolute wavelet coefficient
                is also computed and stored
            If 'median', the aggregated event properties are the medians over the events being aggregated
                of each property, and the aggregated event magnitude is the median absolute wavelet coefficient
        min_events: If not None, then (cluster, epoch, grid-element) tuples containing less than this many events
            will be dropped
        time_slice_ms: If provided, clusters are computed within slices of the time spanning this many ms. Must have
            run compute_shared_grid_element_counts with this value of time_slice_ms before running this step.
        ignore_grid_on_aggregate: If True, then aggregation occurs within (cluster, epoch), completely ignoring
            grid_element after clustering has occurred. Typically this would be used in combination with time_slice_ms
            so that the aggregation is within a time-slice instead of within a grid-element.
        knn: If set, this number of nearest-neighbors is used to build the affinity matrix rather than using the
            fully connected graph
        **spectral_kwargs: Additional arguments to sklearn.cluster.SpectralClustering

    Returns:
        Saves a new file into element_analysis_dir for each held-out block with name
        harry_potter_co_occurrence_clustered_{aggregation}_{subject}_hold_out_{hold_out}.npz
        Note that each hold-out file has data for all of the blocks, with the held out block mapped according to the
        clustering done on the other blocks.

        source_clusters: The cluster assignment for each source
        magnitude:
            If aggregation == 'power', the L2 norm of the modulus of the events occurring within the same
                (block, stimulus, cluster, grid_element) tuple
            If aggregation == 'amplitude', the L1 norm of the modulus of the events occurring within the same
                (block, stimulus, cluster, grid_element) tuple
            If aggregation == 'median', the median of the modulus of the events occurring within the same
                (block, stimulus, cluster, grid_element) tuple
            If aggregation == 'percentile_75', the 75th percentile of the modulus of the events occurring within
                the same (block, stimulus, cluster, grid_element) tuple
        magnitude_mean:
            If aggregation == 'power', the root-mean-squared value of the modulus of the events occurring within
                the same (block, stimulus, cluster, grid_element) tuple
            If aggregation == 'amplitude', the mean absolute value of the modulus of the events occurring within the
                same (block, stimulus, cluster, grid_element) tuple
            If aggregation == 'median', not present
        f_hat: The power-weighted average of the estimated frequency events occurring within the same
            (block, stimulus, cluster, grid_element) tuple
        time: The power-weighted average of the time of the events occurring within the same
            (block, stimulus, cluster, grid_element) tuple
        interpolated_scale_frequencies: The power-weighted average of the interpolated scale-frequencies of the events
            occurring within the same (block, stimulus, cluster, grid_element) tuple
        stimuli: The index of the stimulus (within block) corresponding to each event average
        blocks: The block corresponding to each event average
        sample_source_clusters: The cluster corresponding to each event average
        grid_elements: The grid-element corresponding to each event average
    """
    indices_stimuli = list()
    indices_block = list()
    indices_source = list()
    indices_time = list()
    interp_fs = list()
    c_hat = list()
    f_hat = list()
    for block in all_blocks:
        ea_path = os.path.join(
            element_analysis_dir, 'harry_potter_element_analysis_{}_{}.npz'.format(subject, block))
        with np.load(ea_path) as ea_block:
            indices_stimuli_block, indices_source_block, _, indices_time_block = np.unravel_index(
                ea_block['indices_flat'], ea_block['shape'])
            indices_stimuli.append(indices_stimuli_block)
            indices_block.append(np.array([block] * len(indices_stimuli_block), dtype=indices_stimuli_block.dtype))
            indices_source.append(indices_source_block)
            indices_time.append(indices_time_block)

            interp_fs.append(ea_block['interpolated_scale_frequencies'])
            ea_morse = ElementAnalysisMorse(ea_block['gamma'], ea_block['analyzing_beta'], ea_block['element_beta'])
            c_hat_block, _, f_hat_block = ea_morse.event_parameters(
                ea_block['maxima_coefficients'], ea_block['interpolated_scale_frequencies'])
            c_hat.append(c_hat_block)
            f_hat.append(f_hat_block)

    indices_stimuli = np.concatenate(indices_stimuli)
    indices_block = np.concatenate(indices_block)
    indices_sources = np.concatenate(indices_source)
    indices_time = np.concatenate(indices_time)
    interp_fs = np.concatenate(interp_fs)
    c_hat = np.concatenate(c_hat)
    f_hat = np.concatenate(f_hat)

    if time_slice_ms is not None:
        count_path = os.path.join(
            element_analysis_dir,
            'harry_potter_shared_grid_element_counts_{subject}_time_slice_ms_{time_slice_ms}.npz'.format(
                subject=subject, time_slice_ms=time_slice_ms))
    else:
        count_path = os.path.join(
            element_analysis_dir, 'harry_potter_shared_grid_element_counts_{subject}.npz'.format(subject=subject))
    with np.load(count_path) as counts:

        grid_elements = np.concatenate(list(counts['grid_elements_{}'.format(b)] for b in all_blocks))

        for hold_out_block in tqdm(all_blocks):
            train_blocks = [b for b in all_blocks if b != hold_out_block]
            joint_counts = None
            independent_counts = None
            for block in train_blocks:
                if joint_counts is None:
                    joint_counts = counts['joint_grid_element_count_{}'.format(block)]
                    if use_pointwise_mutual_information:
                        independent_counts = counts['independent_grid_element_count_{}'.format(block)]
                else:
                    joint_counts += counts['joint_grid_element_count_{}'.format(block)]
                    if use_pointwise_mutual_information:
                        independent_counts += counts['independent_grid_element_count_{}'.format(block)]

            if use_pointwise_mutual_information:
                if time_slice_ms is not None:
                    independent_counts = np.expand_dims(independent_counts, 2) * np.expand_dims(independent_counts, 1)
                    joint_counts = joint_counts / independent_counts
                else:
                    independent_counts = np.expand_dims(independent_counts, 1) * np.expand_dims(independent_counts, 0)
                    joint_counts = [joint_counts / independent_counts]

            sample_clusters = np.full_like(indices_sources, -1)
            source_clusters = np.full((joint_counts[0].shape[0], len(joint_counts)), -1, dtype=sample_clusters.dtype)
            for index_slice, jc in enumerate(joint_counts):

                if knn is not None:
                    indices_sort = np.argsort(-jc, axis=1)
                    indices_sort = indices_sort[:, :knn]
                    indices_sort_b = np.tile(
                        np.expand_dims(np.arange(indices_sort.shape[0]), 1), (1, indices_sort.shape[1]))
                    indices_sort = np.reshape(indices_sort, -1)
                    indices_sort_b = np.reshape(indices_sort_b, -1)
                    indicator_nearest = np.full_like(jc, False)
                    indicator_nearest[(indices_sort, indices_sort_b)] = True
                    indicator_nearest[(indices_sort_b, indices_sort)] = True
                    jc = np.where(indicator_nearest, jc, 0)

                spectral_clustering = SpectralClustering(
                    n_clusters=n_clusters, affinity='precomputed', **spectral_kwargs)
                # renumber the clusters here according to dipole order to
                # try to keep the same cluster numbers across runs
                source_clusters_ = _renumber_clusters(
                    spectral_clustering.fit_predict(jc),
                    source_clusters[:, index_slice - 1] - (index_slice - 1) * n_clusters if index_slice > 0 else None)
                source_clusters_ = source_clusters_ + index_slice * n_clusters
                source_clusters[:, index_slice] = source_clusters_
                if time_slice_ms is not None:
                    indices_time_slice = indices_time - index_slice * time_slice_ms
                    indicator_slice = np.logical_and(indices_time_slice >= 0, indices_time_slice < time_slice_ms)
                    sample_clusters[indicator_slice] = source_clusters_[indices_sources[indicator_slice]]
                else:
                    sample_clusters[:] = source_clusters_[indices_sources]

            del joint_counts
            del independent_counts

            if time_slice_ms is None:
                source_clusters = np.squeeze(source_clusters, axis=1)

            if ignore_grid_on_aggregate:
                segments = (indices_stimuli, indices_block, sample_clusters)
            else:
                segments = (indices_stimuli, indices_block, sample_clusters, grid_elements)

            mean_w_segment = None
            if aggregation == 'median' or aggregation == 'percentile_75':
                c_hat_val = np.abs(c_hat)
                if aggregation == 'percentile_75':
                    c_hat_val = (c_hat_val, 0.75)

                (segments,
                 segment_counts,
                 w_segment,
                 f_hat_segment,
                 time_segment,
                 interpolated_scale_frequencies_segment) = segment_median(
                    segments, c_hat_val, f_hat, indices_time, interp_fs, min_count=min_events)
            elif aggregation == 'power' or aggregation == 'amplitude':
                segments, segment_counts, w_segment, mean_w_segment, power_weighted = segment_combine_events(
                    segments, c_hat, min_count=min_events,
                    power_weighted_dict={'f_hat': f_hat, 'time': indices_time, 'interp_fs': interp_fs},
                    use_amplitude=aggregation == 'amplitude')

                f_hat_segment, time_segment, interpolated_scale_frequencies_segment = (
                    power_weighted['f_hat'], power_weighted['time'], power_weighted['interp_fs'])
            else:
                raise ValueError('Unrecognized aggregation: {}'.format(aggregation))

            result_dict = dict(
                source_clusters=source_clusters,
                counts=segment_counts,
                magnitude=w_segment,
                f_hat=f_hat_segment,
                time=time_segment,
                interpolated_scale_frequencies=interpolated_scale_frequencies_segment,
                stimuli=segments[0],
                blocks=segments[1],
                sample_source_clusters=segments[2],
                grid=counts['grid'])

            if not ignore_grid_on_aggregate:
                result_dict['grid_elements'] = segments[3]

            if mean_w_segment is not None:
                result_dict['magnitude_mean'] = mean_w_segment

            output_path = os.path.join(
                element_analysis_dir,
                'harry_potter_co_occurrence_clustered'
                '_{aggregation}_{subject}{time_slice}_hold_out_{hold_out}.npz'.format(
                    aggregation=aggregation,
                    subject=subject,
                    time_slice='_time_slice_ms_{}'.format(time_slice_ms) if time_slice_ms is not None else '',
                    hold_out=hold_out_block))

            np.savez(output_path, **result_dict)


def _insert_missing(block_stimuli, data, index):
    indicator_block = block_stimuli[:, 0] == block_stimuli[index - 1, 0]
    indices_block = np.flatnonzero(indicator_block)
    start, end = indices_block[0], indices_block[-1] + 1
    assert(np.all(indicator_block[start:end]))
    assert(np.array_equal(block_stimuli[indicator_block, 1], np.arange(np.count_nonzero(indicator_block))))

    block_stimuli = np.concatenate([
        block_stimuli[:index],
        np.array([[block_stimuli[index - 1, 0], -1]], dtype=block_stimuli.dtype),
        block_stimuli[index:]], axis=0)

    block_stimuli[start:end + 1, 1] = np.arange(end + 1 - start)

    data = np.concatenate([
        data[..., :index],
        np.full(data.shape[:-1] + (1,), np.nan, dtype=data.dtype),
        data[..., index:]], axis=-1)

    return block_stimuli, data


def _aggregate_dense(dense_array, cluster_assignments, aggregation, **unique_kwargs):
    unique_result = np.unique(cluster_assignments, **unique_kwargs)
    if not isinstance(unique_result, tuple):
        unique_result = (unique_result,)

    unique_cluster_assignments = unique_result[0]
    unique_cluster_assignments = unique_cluster_assignments[unique_cluster_assignments >= 0]

    assert (np.array_equal(unique_cluster_assignments, np.arange(len(unique_cluster_assignments))))

    result_data = np.full((dense_array.shape[1], len(unique_cluster_assignments)), np.nan, dtype=dense_array.dtype)

    for c in unique_cluster_assignments:
        indicator_cluster = c == cluster_assignments
        if aggregation == 'sum' or aggregation == 'counts':
            result_data[:, c] = nansum(dense_array[indicator_cluster], axis=0)
        elif aggregation == 'mean':
            result_data[:, c] = nanmean(dense_array[indicator_cluster], axis=0)
        elif aggregation == 'median':
            result_data[:, c] = nanmedian(dense_array[indicator_cluster], axis=0)
        elif aggregation == 'percentile_75':
            result_data[:, c] = np.nanquantile(dense_array[indicator_cluster], q=0.75, axis=0)
        elif aggregation == 'L2':
            result_data[:, c] = np.sqrt(nansum(np.square(dense_array[indicator_cluster]), axis=0))
        elif aggregation == 'rms':
            result_data[:, c] = np.sqrt(nanmean(np.square(dense_array[indicator_cluster]), axis=0))
        else:
            raise ValueError('Unknown aggregation_mode: {}'.format(aggregation))

    return (result_data,) + unique_result


def _rank_cluster_helper(
        dense_array,
        blocks,
        held_out_block,
        maximum_proportion_missing,
        aggregation,
        n_clusters,
        n_clusters_2,
        spectral_kwargs,
        spectral_radius):

    indicator_held_out = blocks == held_out_block

    co_occurrence_rank_clusters = np.full(dense_array.shape[:-1], -1, dtype=np.int32)
    co_occurrence_rank_clusters_2 = None
    if n_clusters_2 is not None:
        co_occurrence_rank_clusters_2 = np.copy(co_occurrence_rank_clusters)
    dense_array = np.reshape(dense_array, (int(np.prod(dense_array.shape[:-1])), dense_array.shape[-1]))

    train_array = dense_array[:, np.logical_not(indicator_held_out)]

    indicator_enough_data = None
    if maximum_proportion_missing is not None:
        indicator_enough_data = (
                np.count_nonzero(np.isnan(train_array), axis=1) / train_array.shape[1] <= maximum_proportion_missing)
        train_array = train_array[indicator_enough_data]

    # the user can pass n_clusters >= dense_array.shape[0] to bypass rank-clustering and just cause dense
    # aggregation on the input, so we skip a bunch of work if that is the case
    if n_clusters < dense_array.shape[0] or (n_clusters_2 is not None and n_clusters_2 < dense_array.shape[0]):
        train_array = nanrankdata(train_array, axis=1)
        tau, _ = kendall_tau(train_array)
        # can be nan if num_common is < 2
        affinity = np.where(np.isnan(tau), 0, tau + 1)
        if spectral_radius is not None:
            affinity = np.where(affinity >= spectral_radius, affinity - spectral_radius, 0)
    else:
        tau = None
        affinity = None
        del train_array

    result = list()
    for n, co_occurrence_clusters in [
            (n_clusters, co_occurrence_rank_clusters), (n_clusters_2, co_occurrence_rank_clusters_2)]:
        if n is None:
            break

        if n < dense_array.shape[0]:
            spectral = SpectralClustering(n_clusters=n, affinity='precomputed', **spectral_kwargs)
            indicator_connected = np.count_nonzero(affinity > 0, axis=0) > 1
            rank_clusters_ = spectral.fit_predict(affinity[indicator_connected][:, indicator_connected])
            rank_clusters = np.full(indicator_connected.shape, -1, dtype=rank_clusters_.dtype)
            rank_clusters[indicator_connected] = rank_clusters_
            del rank_clusters_
        else:
            rank_clusters = np.arange(dense_array.shape[0])

        if indicator_enough_data is not None:
            dense_array = dense_array[indicator_enough_data]

        result_data, unique_rank_clusters, inverse = _aggregate_dense(
            dense_array, rank_clusters, aggregation=aggregation, return_inverse=True)

        if indicator_enough_data is not None:
            indices_co_occurrence_rank_clusters = np.unravel_index(
                np.arange(co_occurrence_clusters.size)[indicator_enough_data], co_occurrence_clusters.shape)
        else:
            indices_co_occurrence_rank_clusters = np.unravel_index(
                np.arange(co_occurrence_clusters.size), co_occurrence_clusters.shape)

        co_occurrence_clusters[indices_co_occurrence_rank_clusters] = unique_rank_clusters[inverse]
        result.append(result_data)
        result.append(co_occurrence_clusters)

    result.append(tau)
    result.append(affinity)

    return tuple(result)


def rank_cluster_only_combine_block(element_analysis_dir, subject, block, grid, source_rank_clusters, aggregation):
    """
    Computes aggregate events for each (source, rank-cluster) tuple bypassing co-occurrence cluster aggregation. This
    can be useful for comparing a predicted result to a ground-truth value at each dipole.
    Args:
        element_analysis_dir: The directory where element analysis files can be found. For example:
            '/share/volume0/<your user name>/data_sets/harry_potter/element_analysis'
        subject: Which subject to compute the events for.
        block: Which block to compute the events for.
        grid: The time-frequency grid, available in the rank-cluster file as 'grid'
        source_rank_clusters: A 2d array of shape (num_dipoles, num_grid_elements) giving the rank-cluster assignment
            for each (source, grid-element) tuple. Available in the rank-cluster file as
            'source_rank_clusters_<subject>_hold_out_<held-out-block>'
        aggregation:
            If 'median', uses the median absolute wavelet coefficient of the events in all dipoles belonging to a
                rank cluster as the value for that cluster for an epoch.
            If 'counts', uses the total number of events in all dipoles belonging to a rank cluster as the value for
                that cluster for an epoch.
            If 'rms', uses the root-mean-squared absolute wavelet coefficient of all events in all dipoles belonging to
                a rank cluster as the value for that cluster for an epoch.
            If 'L2' (default), uses the L2-norm of the absolute wavelet coefficient of all events in all dipoles
                belonging to a rank cluster as the value for that cluster for an epoch.
            If 'mean', uses the mean of the absolute wavelet coefficient of all events in all dipoles
                belonging to a rank cluster as the value for that cluster for an epoch.
            If 'sum', uses the total of the absolute wavelet coefficient of all events in all dipoles
                belonging to a rank cluster as the value for that cluster for an epoch.
    Returns:
        combined_events: A 3d array with shape (num_epochs, num_sources, num_rank_clusters) containing the aggregated
            events
    """
    ea_path = os.path.join(
        element_analysis_dir, 'harry_potter_element_analysis_{}_{}.npz'.format(subject, block))
    with np.load(ea_path) as ea_block:
        indices_stimuli, indices_source, _, indices_time = np.unravel_index(
            ea_block['indices_flat'], ea_block['shape'])
        ea_morse = ElementAnalysisMorse(ea_block['gamma'], ea_block['analyzing_beta'], ea_block['element_beta'])
        c_hat, _, f_hat = ea_morse.event_parameters(
            ea_block['maxima_coefficients'], ea_block['interpolated_scale_frequencies'])
    indices_grid = assign_grid_labels(grid, indices_time, f_hat)
    del indices_time
    del f_hat
    indices_cluster = source_rank_clusters[(indices_source, indices_grid)]
    del indices_grid

    segments = np.concatenate([
        np.expand_dims(indices_stimuli, 1),
        np.expand_dims(indices_source, 1),
        np.expand_dims(indices_cluster, 1)], axis=1)

    dense_array = np.full(
        (np.max(indices_stimuli) + 1, np.max(indices_source) + 1, np.max(indices_cluster) + 1), np.nan)

    if aggregation == 'counts':
        unique_segments, counts = np.unique(segments, axis=0, return_counts=True)
        dense_array[(
            unique_segments[:, 0], unique_segments[:, 1], unique_segments[:, 2])] = counts
    elif aggregation == 'median':
        unique_segments, _, w = segment_median(segments, np.abs(c_hat))
        dense_array[(
            unique_segments[:, 0], unique_segments[:, 1], unique_segments[:, 2])] = w
    elif aggregation == 'L2' or aggregation == 'rms':
        unique_segments, _, w, w_rms = segment_combine_events(segments, c_hat)
        if aggregation == 'rms':
            w = w_rms
        dense_array[(
            unique_segments[:, 0], unique_segments[:, 1], unique_segments[:, 2])] = w
    elif aggregation == 'sum' or aggregation == 'mean':
        unique_segments, _, w, w_mean = segment_combine_events(segments, c_hat, use_amplitude=True)
        if aggregation == 'mean':
            w = w_mean
        dense_array[(
            unique_segments[:, 0], unique_segments[:, 1], unique_segments[:, 2])] = w
    else:
        raise ValueError('Unrecognized mode: {}'.format(aggregation))

    return dense_array


def rank_cluster(
        element_analysis_dir,
        aggregation='L2',
        maximum_proportion_missing=None,
        single_subject=None,
        spectral_radius=1.1,
        n_clusters=300,
        n_multi_subject_clusters=None,
        n_multi_subject_input_clusters=None,
        time_slice_ms=None,
        **spectral_kwargs):
    """
    Uses spectral clustering, computing an affinity matrix between each (source-cluster, grid-element) tuple and each
    other (source-cluster, grid-element) tuple. The affinity matrix is computed ranking the epochs according to the
    values for each epoch within a (source-cluster, grid-element) tuple and then computing Kendall's tau-b between each
    pair of (source-cluster, grid-element) tuples on this ranked data.
    Args:
        element_analysis_dir: The directory where element analysis files can be found. For example:
            '/share/volume0/<your user name>/data_sets/harry_potter/element_analysis'
        aggregation:
            If 'median', uses harry_potter_co_occurrence_clustered_median_{subject}_hold_out_{hold_out}.npz as
                input, taking the median absolute wavelet coefficient of the co-occurrence clusters to rank
                epochs when rank clustering. Also combines values from the co-occurrence clusters using the median.
            If 'counts', uses harry_potter_co_occurrence_clustered_median_{subject}_hold_out_{hold_out}.npz as
                input, taking the number of events in each co-occurrence cluster instead of the wavelet coefficients
                to rank epochs when rank clustering. Combines values (counts) from the co-occurrence clusters by
                summing.
            If 'rms', uses harry_potter_co_occurrence_clustered_{subject}_hold_out_{hold_out}.npz as input, taking
                the root-mean-squared aggregates from that file to rank epochs when rank clustering. Combines values
                from co-occurrence clusters using the root-mean-squared value.
            If 'L2', uses harry_potter_co_occurrence_clustered_{subject}_hold_out_{hold_out}.npz as input,
                taking the L2-norm of the absolute value of the wavelet coefficients of the co-occurrence
                clusters to rank epochs when rank clustering. Combines values from co-occurrence clusters using the
                L2-norm.
            If 'mean', uses harry_potter_co_occurrence_clustered_amplitude_{subject}_hold_out_{hold_out}.npz as input,
                taking the mean of the absolute value of the wavelet coefficients of the co-occurrence clusters to rank
                epochs when rank clustering. Combines values from co-occurrence clusters using the mean.
            If 'sum', uses harry_potter_co_occurrence_clustered_amplitude_{subject}_hold_out_{hold_out}.npz as input,
                taking the sum of the absolute value of the wavelet coefficients of the co-occurrence clusters to rank
                epochs when rank clustering. Combines values from co-occurrence clusters using the sum.
        maximum_proportion_missing: (source-cluster, grid-element) tuples having more than maximum_proportion_missing
            missing epochs will be dropped from the data instead of clustered. If None (default), no tuples will be
            dropped.
        n_clusters: The number of single-subject clusters to produce. If None, then rank-clustering is bypassed for
            single subjects, and this function only converts the input data to a dense array. Multi-subject
            rank-clustering can still be used in that case.
        n_multi_subject_clusters:
            If None, then the number of multi-subject clusters will be the same as the number of single-subject
            clusters. Set to 0 to turn off multi-subject clustering.
        n_multi_subject_input_clusters:
            If None or not 0, multi-subject clustering is done by first clustering each subject independently and then
            by clustering the clusters. If set to 0, multi-subject clustering is done directly on the input data by
            concatenating the subject data together. Direct clustering is impractical when the aggregated co-occurrence
            data is too large, so use n_multi_subject_input_clusters=0 with caution. When not 0, this determines the
            number of clusters used in the first stage of multi-subject clustering (i.e. the stage in which each
            subject's data is clustered independently. If this value is the same as n_clusters, then the input to the
            multi-subject clustering is identical to the single-subject clusters, but for other values two different
            subject level clusterings are computed. Setting this value to None (the default) is equivalent to using
            2 * n_multi_subject_clusters (or 2 * n_clusters when n_multi_subject_clusters is None).
        single_subject: If specified, clustering is performed only on the specified subject. No multi-subject
            clustering is performed, and the output file is suffixed with the subject name. This is useful for
            trialing changes to the clustering without waiting for all subjects to be processed.
        spectral_radius: If specified, affinity below this value will be set to 0 to create a knn-graph for
            spectral clustering instead of using a fully-connected graph. This value is applied after
            the Kendall's tau is transformed to affinity. affinity < 1 is a negative correlation, affinity = 2 is
            perfect correlation, so 1.1 is a reasonable value here.
        time_slice_ms: If provided, clusters are computed within slices of the time spanning this many ms. Must have
            run co_occurrence_source_cluster with this value of time_slice_ms before running this step.
        **spectral_kwargs: Additional arguments to sklearn.cluster.SpectralClustering

    Returns:
        Saves a new file into element_analysis_dir with name harry_potter_meg_rank_clustered_<aggregation>.npz
        Note that the file has data for each subject and hold out.

        description: Comments about what this data file contains.
        grid: The boundaries of the grid elements. A 2d array with shape (num_grid_cells, 4) giving
            (low-time, high-time, low-freq, high-freq) coordinates for each grid-cell. The high-time
            and high-freq boundaries are exclusive, the low-time and low-freq boundaries are inclusive.
            Grid-cells at the lowest-frequency have an open-lower bound, meaning events which fall below the
            lowest-frequency are put into the lowest frequency bins at the same time interval. Similarly, events
            which occur at a higher frequency than the highest frequency bins are put into the highest frequency
            bins at the same time interval.
        stimuli: The text of the stimulus corresponding to each row of data_<subject>_hold_out_<held-out-block>
        blocks: The block corresponding to each row of data_<subject>_hold_out_<held-out-block>
        co_occurrence_clusters_<subject>_hold_out_<held-out-block>:
            The co-occurrence cluster assignment for each source. Spatial clustering only (copied from input data).
            A 1d array with shape (num_sources,). This is the result of co_occurrence_source_cluster.
        source_rank_clusters_<subject>_hold_out_<held-out-block>: The rank-cluster assignment for each
            (source, grid-cell) tuple. A 2d array with shape (num_sources, num_grid_cells)
        co_occurrence_rank_clusters_<subject>_hold_out_<held-out-block>: The rank-cluster assignment for each
            (co_occurrence_cluster, grid-cell) tuple.
            A 2d array with shape (num_co_occurrence_clusters, num_grid_cells)
        data_<subject>_hold_out_<held-out-block>:
            The aggregate values for each cluster, computed according the specified aggregation method
            (see aggregation). A 2d array with shape (num_stimuli, n_clusters). Note that this contains data for all
            blocks, but the clustering is fit without the held out block.
        source_rank_clusters_multi_subject_<subject>_hold_out_<held-out-block>:
            Similar to source_rank_clusters_<subject>_hold_out_<held-out-block>, but contains the cluster
            assignments when all of the subjects are clustered jointly. These are still separated out by subject
            even in the multi-subject clustering case
        co_occurrence_rank_clusters_multi_subject_<subject>_hold_out_<held-out-block>:
            Similar to spectral_rank_clusters_<subject>_hold_out_<held-out-block>, but contains the cluster
            assignments when all of the subjects are clustered jointly. These are still separated out by subject
            even in the multi-subject clustering case
        data_multi_subject_hold_out_<held-out-block>:
            The aggregate values for each cluster when the subjects are clustered jointly.
            A 2d array with shape (num_stimuli, n_clusters). Note that this contains data for all blocks,
            but the clustering is fit without the held out block.
    """
    subjects = all_subjects
    if single_subject is not None:
        subjects = (single_subject,)

    # subject A has 2 '+' stimuli missing
    a_insert_indices = 1302, 2653

    if aggregation == 'median' or aggregation == 'counts':
        co_occurrence_aggregation = 'median'
    elif aggregation == 'percentile_75':
        co_occurrence_aggregation = 'percentile_75'
    elif aggregation == 'L2' or aggregation == 'rms':
        co_occurrence_aggregation = 'power'
    elif aggregation == 'mean' or aggregation == 'sum':
        co_occurrence_aggregation = 'amplitude'
    else:
        raise ValueError('Unrecognized aggregation: {}'.format(aggregation))

    result = dict()

    is_n_clusters_explicit = n_clusters is not None

    for held_out_block in tqdm(all_blocks, desc='hold_out_block'):

        multi_subject_arrays = list()
        multi_subject_input_aggregates = list()
        multi_subject_source_clusters = list()
        multi_subject_co_occurrence_clusters = list()

        for subject in tqdm(subjects, desc='subject', leave=False):

            co_occurrence_clustered_path = os.path.join(
                element_analysis_dir,
                'harry_potter_co_occurrence_clustered'
                '_{co_occurrence_aggregation}_{subject}{time_slice}_hold_out_{block}.npz'.format(
                    co_occurrence_aggregation=co_occurrence_aggregation,
                    subject=subject,
                    time_slice='_time_slice_ms_{}'.format(time_slice_ms) if time_slice_ms is not None else '',
                    block=held_out_block))

            with np.load(co_occurrence_clustered_path) as co_occurrence_clustered:
                grid = co_occurrence_clustered['grid']
                if aggregation == 'rms' or aggregation == 'mean':
                    w = co_occurrence_clustered['magnitude_mean']
                else:
                    w = co_occurrence_clustered['magnitude']
                counts = co_occurrence_clustered['counts']
                blocks = co_occurrence_clustered['blocks']
                stimuli = co_occurrence_clustered['stimuli']
                elements = co_occurrence_clustered['grid_elements'] \
                    if 'grid_elements' in co_occurrence_clustered else None
                sample_co_occurrence_clusters = co_occurrence_clustered['sample_source_clusters']
                source_clusters = co_occurrence_clustered['source_clusters']

            block_stimuli, virtual_stim = np.unique(
                np.concatenate([np.expand_dims(blocks, 1), np.expand_dims(stimuli, 1)], axis=1),
                axis=0, return_inverse=True)

            if elements is None:
                dense_array = np.full(
                    (np.max(sample_co_occurrence_clusters) + 1, np.max(virtual_stim) + 1), np.nan)

                if aggregation == 'counts':
                    dense_array[(sample_co_occurrence_clusters, virtual_stim)] = counts
                else:
                    dense_array[(sample_co_occurrence_clusters, virtual_stim)] = w
            else:
                dense_array = np.full(
                    (np.max(sample_co_occurrence_clusters) + 1, np.max(elements) + 1, np.max(virtual_stim) + 1), np.nan)

                if aggregation == 'counts':
                    dense_array[(sample_co_occurrence_clusters, elements, virtual_stim)] = counts
                else:
                    dense_array[(sample_co_occurrence_clusters, elements, virtual_stim)] = w

            if n_clusters is None:
                n_clusters = int(np.prod(dense_array.shape[:-1]))
            elif not is_n_clusters_explicit:
                # we require that all of the subjects have the same number of clusters in this mode
                if n_clusters != int(np.prod(dense_array.shape[:-1])):
                    raise ValueError('The number of input clusters must be consistent across all subjects '
                                     'if n_clusters is set to None')

            if n_multi_subject_clusters is None:
                n_multi_subject_clusters = n_clusters

            if n_multi_subject_input_clusters is None:
                n_multi_subject_input_clusters = 2 * n_multi_subject_clusters

            if subject == 'A':
                for idx in a_insert_indices:
                    block_stimuli, dense_array = _insert_missing(block_stimuli, dense_array, idx)

            blocks = block_stimuli[:, 0]
            stimuli = block_stimuli[:, 1]

            result_data_ms_input = None
            co_occurrence_rank_clusters_ms_input = None
            if n_multi_subject_clusters != 0 \
                    and single_subject is None \
                    and n_multi_subject_input_clusters != n_clusters \
                    and n_multi_subject_input_clusters != 0:
                (result_data,
                 co_occurrence_rank_clusters,
                 result_data_ms_input,
                 co_occurrence_rank_clusters_ms_input,
                 tau,
                 affinity) = \
                    _rank_cluster_helper(
                        dense_array, blocks, held_out_block,
                        maximum_proportion_missing,
                        aggregation=aggregation,
                        n_clusters=n_clusters,
                        n_clusters_2=n_multi_subject_input_clusters,
                        spectral_kwargs=spectral_kwargs,
                        spectral_radius=spectral_radius)
            else:
                result_data, co_occurrence_rank_clusters, tau, affinity = \
                    _rank_cluster_helper(
                        dense_array, blocks, held_out_block,
                        maximum_proportion_missing,
                        aggregation=aggregation,
                        n_clusters=n_clusters,
                        n_clusters_2=None,
                        spectral_kwargs=spectral_kwargs,
                        spectral_radius=spectral_radius)
                if n_multi_subject_clusters != 0 and single_subject is None:
                    result_data_ms_input, co_occurrence_rank_clusters_ms_input = \
                        result_data, co_occurrence_rank_clusters

            # these could be None if n_clusters >= num_co_occurrence_clusters
            if tau is not None or affinity is not None:
                affinity_path = os.path.join(
                    element_analysis_dir,
                    'harry_potter_rank_affinity'
                    '_{co_occurrence_aggregation}_{subject}{time_slice}_hold_out_{block}.npz'.format(
                        co_occurrence_aggregation=co_occurrence_aggregation,
                        subject=subject,
                        time_slice='_time_slice_ms_{}'.format(time_slice_ms) if time_slice_ms is not None else '',
                        block=held_out_block))

                np.savez(affinity_path, tau=tau, affinity=affinity)
                del tau
                del affinity

            source_rank_clusters = np.reshape(
                co_occurrence_rank_clusters[np.reshape(source_clusters, -1)],
                source_clusters.shape + co_occurrence_rank_clusters.shape[1:])

            if 'grid' in result:
                assert np.array_equal(result['grid'], grid)
                assert np.array_equal(result['stimuli'], stimuli)
                assert np.array_equal(result['blocks'], blocks)
            else:
                result['grid'] = grid
                result['stimuli'] = stimuli
                result['blocks'] = blocks

            result['co_occurrence_clusters_{}_hold_out_{}'.format(subject, held_out_block)] = source_clusters
            result['source_rank_clusters_{}_hold_out_{}'.format(subject, held_out_block)] = source_rank_clusters
            result['co_occurrence_rank_clusters_{}_hold_out_{}'.format(subject, held_out_block)] = \
                co_occurrence_rank_clusters
            result['data_{}_hold_out_{}'.format(subject, held_out_block)] = result_data

            if result_data_ms_input is not None:
                if n_multi_subject_input_clusters != 0:  # not direct clustering
                    source_rank_clusters_ms_input = np.reshape(
                        co_occurrence_rank_clusters_ms_input[np.reshape(source_clusters, -1)],
                        source_clusters.shape + co_occurrence_rank_clusters_ms_input.shape[1:])
                    multi_subject_source_clusters.append(source_rank_clusters_ms_input)
                    multi_subject_co_occurrence_clusters.append(co_occurrence_rank_clusters_ms_input)
                    multi_subject_input_aggregates.append(result_data_ms_input)
                else:  # direct clustering
                    multi_subject_source_clusters.append(source_clusters)

                multi_subject_arrays.append(dense_array)

        if n_multi_subject_clusters != 0 and single_subject is None:

            if n_multi_subject_input_clusters == 0:
                result_data, ms_rank_clusters, _, _ = _rank_cluster_helper(
                    np.concatenate(multi_subject_arrays),
                    result['blocks'], held_out_block, maximum_proportion_missing,
                    aggregation=aggregation,
                    n_clusters=n_multi_subject_clusters,
                    n_clusters_2=None,
                    spectral_radius=spectral_radius,
                    spectral_kwargs=spectral_kwargs)
                ms_rank_clusters = np.split(ms_rank_clusters, len(multi_subject_arrays), axis=0)
            else:
                ms_input = np.concatenate(multi_subject_input_aggregates, axis=1)
                ms_train = ms_input[np.logical_not(result['blocks'] == held_out_block)]
                ms_train = nanrankdata(ms_train, axis=0)
                ms_tau, _ = kendall_tau(ms_train.T)
                # can be nan if num_common is < 2
                ms_affinity = np.where(np.isnan(ms_tau), 0, ms_tau + 1)
                if spectral_radius is not None:
                    ms_affinity = np.where(ms_affinity >= spectral_radius, ms_affinity, 0)
                ms_spectral = SpectralClustering(
                    n_clusters=n_multi_subject_clusters, affinity='precomputed', **spectral_kwargs)
                # remove nodes that are not connected to anything
                indicator_connected = np.count_nonzero(ms_affinity > 0, axis=0) > 1
                ms_rank_clusters_ = ms_spectral.fit_predict(ms_affinity[indicator_connected][:, indicator_connected])
                ms_rank_clusters = np.full(indicator_connected.shape, -1, dtype=ms_rank_clusters_.dtype)
                ms_rank_clusters[indicator_connected] = ms_rank_clusters_
                del ms_rank_clusters_

                ms_rank_clusters = np.split(ms_rank_clusters, len(multi_subject_input_aggregates), axis=0)
                result_data = None

            for idx_subject, (subject, ms_rank_clusters_subject) in enumerate(zip(subjects, ms_rank_clusters)):
                # ms_rank_clusters maps from multi-subject-input-cluster to multi-subject-cluster, so just index into
                # it with our multi-subject-input-cluster information to replace multi-subject-input-cluster
                # information with multi-subject-cluster information
                multi_subject_source_clusters[idx_subject] = np.reshape(
                    ms_rank_clusters_subject[np.reshape(multi_subject_source_clusters[idx_subject], -1)],
                    multi_subject_source_clusters[idx_subject].shape)
                result['source_rank_clusters_multi_subject_{}_hold_out_{}'.format(subject, held_out_block)] = \
                    multi_subject_source_clusters[idx_subject]

                if n_multi_subject_input_clusters == 0:  # direct clustering
                    result['co_occurrence_rank_clusters_multi_subject_{}_hold_out_{}'.format(
                        subject, held_out_block)] = ms_rank_clusters_subject
                else:
                    multi_subject_co_occurrence_clusters[idx_subject] = np.reshape(
                        ms_rank_clusters_subject[np.reshape(multi_subject_co_occurrence_clusters[idx_subject], -1)],
                        multi_subject_co_occurrence_clusters[idx_subject].shape)
                    result['co_occurrence_rank_clusters_multi_subject_{}_hold_out_{}'.format(
                        subject, held_out_block)] = multi_subject_co_occurrence_clusters[idx_subject]

            if result_data is None:
                multi_subject_arrays = np.concatenate(multi_subject_arrays, axis=0)
                result_data, _ = _aggregate_dense(
                    np.reshape(
                        multi_subject_arrays,
                        (multi_subject_arrays.shape[0] * multi_subject_arrays.shape[1], multi_subject_arrays.shape[2])),
                    np.reshape(
                        np.concatenate(multi_subject_co_occurrence_clusters, axis=0),
                        (multi_subject_arrays.shape[0] * multi_subject_arrays.shape[1])),
                    aggregation)

            result['data_multi_subject_hold_out_{}'.format(held_out_block)] = result_data

    # convert stimuli from indices into text for convenience
    stimuli = list()
    # anyone other than A should be fine for this. Choose the first in all_subjects that is not A, or if
    # that is empty, set to B
    stimulus_subject = [s for s in subjects if s != 'A']
    if len(stimulus_subject) > 0:
        stimulus_subject = stimulus_subject[0]
    else:
        stimulus_subject = 'B'
    for block in all_blocks:
        ea_path = os.path.join(
            element_analysis_dir, 'harry_potter_element_analysis_{}_{}.npz'.format(stimulus_subject, block))
        with np.load(ea_path) as ea_block:
            ea_stimuli = ea_block['stimuli']
            stimuli.append(ea_stimuli[result['stimuli'][result['blocks'] == block]])
    result['stimuli'] = np.concatenate(stimuli)

    # check that data we inserted is at '+'
    for idx in a_insert_indices:
        assert(result['stimuli'][idx]) == '+'

    result['subjects'] = subjects

    if aggregation == 'median' or aggregation == 'counts':
        description_aggregation_co_occurrence = \
            """
            Taking the median value of the properties of each event as the aggregate-event properties. The total 
            number of events mapping to a cluster is also stored.
            """
        if aggregation == 'median':
            description_aggregation_rank = \
                """
            Taking the median of the values of all (dipole-cluster, grid-cell) tuples which map to the same 
            rank cluster.   
                """
        elif aggregation == 'counts':
            description_aggregation_rank = \
                """
            Summing the counts of events from all (dipole-cluster, grid-cell) tuples which map to the same rank cluster.
                """
        else:
            raise ValueError('Unknown aggregation: {}'.format(aggregation))
    elif aggregation == 'percentile_75':
        description_aggregation_co_occurrence = \
            """
            Taking the median value of the properties of each event as the aggregate-event properties. For the 
            magnitude of the event, the 75th percentile is used rather than the median.
            """
        description_aggregation_rank = \
            """
            Taking the 75th percentile of the values of all (dipole-cluster, grid-cell) tuples which map to the same 
            rank cluster.   
            """
    elif aggregation == 'L2' or aggregation == 'rms':
        description_aggregation_co_occurrence = \
            """
            Taking a power-weighted average of the properties of each event as the aggregate-event properties. 
            The L2-norm and root-mean-squared value of the absolute wavelet coefficients are also computed,
            and the total number of events mapping to a cluster is also stored.
            """
        if aggregation == 'L2':
            description_aggregation_rank = \
                """
            Taking the L2-norm of the values of all (dipole-cluster, grid-cell) tuples which map to the same
            rank cluster.
                """
        elif aggregation == 'rms':
            description_aggregation_rank = \
                """
            Taking the root of the mean of the square of the values of the aggregate-events from all 
            (dipole-cluster, grid-cell) tuples which map to the same rank cluster.
                """
        else:
            raise ValueError('Unknown aggregation: {}'.format(aggregation))
    elif aggregation == 'mean' or aggregation == 'sum':
        description_aggregation_co_occurrence = \
            """
            Taking the amplitude-weighted average of the properties of each event as the aggregate-event properties. 
            The total absolute value and mean absolute value of the wavelet coefficients are also computed, and the 
            total number of events mapping to a cluster is also stored.
            """
        if aggregation == 'sum':
            description_aggregation_rank = \
                """
            Taking the sum of the values of all (dipole-cluster, grid-cell) tuples which map to the same
            rank cluster.
                """
        elif aggregation == 'mean':
            description_aggregation_rank = \
                """
            Taking the mean of the values of all (dipole-cluster, grid-cell) tuples which map to the same
            rank cluster.
                """
        else:
            raise ValueError('Unknown aggregation: {}'.format(aggregation))
    else:
        raise ValueError('Unknown aggregation: {}'.format(aggregation))

    result['description'] = \
        """
        This data file contains the result of clustering MEG data using a 4-step process:
        1. Data is source-localized using minimum norm estimation
        2. An analytic wavelet transform is run on each dipole, and element analysis (Lilly, 2017) is applied. Roughly
            speaking, this extracts local maxima from the wavelet transformed data.
        3. Each epoch (i.e. 500 ms of time corresponding to a word) is subdivided into a grid based on log-spaced
            frequencies and time-windows which vary with the footprint of the analyzing wavelet. Pointwise-mutual 
            information is computed between each pair of dipoles. The number of maxima within a grid cell for a given
            dipole and epoch is used as a count for each individual dipole, and the joint count is the minimum of the
            individual count for each pair of dipoles. Summing over all epochs and grid-elements gives the final joint
            and individual counts from which PMI is computed. Spectral clustering is applied to the
            PMI matrix to cluster the dipoles. This can roughly be thought of as clustering the dipoles according to
            phase-locking value, i.e. this is an approximation to clustering by functional connectivity. The events 
            (maxima) within a dipole-cluster are then combined together within each (grid-cell, epoch) tuple. Events
            are aggregated within a (dipole-cluster, grid_cell, epoch) according to the specified aggregation type. In
            the case of {aggregation} aggregation, which has been used for the current results, events are combined by:
        """.format(aggregation=aggregation) \
        + description_aggregation_co_occurrence + \
        """
        4. A matrix is formed which has on one axis each (dipole-cluster, grid-cell) tuple, and on the other axis each
            epoch. The epochs are ranked within a (dipole-cluster, grid-cell) tuple, and Kendall's tau-b is computed 
            between each (dipole-cluster, grid-cell) tuple and each other (dipole-cluster, grid-cell) tuple, to produce
            a matrix of rank-correlations. The rank-correlations are then converted to an affinity matrix, and spectral
            clustering is on these rank-correlations to produce a new clustering, which groups together 
            similar (dipole-cluster, grid-cell) tuples where similarity is defined as their ranking of the epochs.
            We call these rank-clusters. Aggregate-events from the co-occurrence clustering step which occurring within 
            the same rank cluster are then further aggregated together according to the specified aggregation type. In
            the case of {aggregation} aggregation, which has been used for the current results, the final representation
            of an epoch (word) for a cluster is given by:
        """.format(aggregation=aggregation) \
        + description_aggregation_rank + \
        """
            This gives us num_rank_cluster values as the final data. Joint clustering over all subjects is also applied
            at this step, and proceeds in a similar manner. However, under typical parameters of the computation, the
            total number of (dipole-cluster, grid-cell) tuples across all subjects becomes too large for direct spectral
            clustering. Instead, a hierarchical clustering is employed. First, within-subject clusters are computed
            as described above (but note that the number of single-subject clusters for hierarchical clustering may
            be more than the number of single-subject clusters used for independent clustering), and the data resulting
            from the single-subject clustering is then clustered across subjects, again by using an affinity matrix
            computed using Kendall's tau and applying spectral clustering to this matrix. The final data from the joint 
            clustering is aggregated directly from the (dipole-cluster, grid-cell) tuple aggregates resulting 
            from step 3 (i.e. the intermediate result of the hierarchical clustering is discarded).
        
        The code which produces this data can be found at 
        https://github.com/danrsc/analytic_wavelet_meg and https://github.com/danrsc/analytic_wavelet
        
        Note that all fields in this file of the form *_<subject>_hold_out_<held-out-block> contain information
        for one subject where the values were computed with the appropriate block held out of all training steps 
        (the PMI is computed without this block for the spectral clustering, and the Kendall's tau is computed without 
        this block for the rank clustering).
        
        Fields:
            description: Comments about what this data file contains.
            grid: The boundaries of the grid elements. A 2d array with shape (num_grid_cells, 4) giving 
                (low-time, high-time, low-freq, high-freq) coordinates for each grid-cell. The high-time 
                and high-freq boundaries are exclusive, the low-time and low-freq boundaries are inclusive. 
                Grid-cells at the lowest-frequency have an open-lower bound, meaning events which fall below the
                lowest-frequency are put into the lowest frequency bins at the same time interval. Similarly, events
                which occur at a higher frequency than the highest frequency bins are put into the highest frequency
                bins at the same time interval.
            stimuli: The text of the stimulus corresponding to each row of data_<subject>_hold_out_<held-out-block>
            blocks: The block corresponding to each row of data_<subject>_hold_out_<held-out-block>
            co_occurrence_clusters_<subject>_hold_out_<held-out-block>:
                The co-occurrence cluster assignment for each source. Spatial clustering only (copied from input data).
                A 1d array with shape (num_sources,). This is the result of step 3 above.
            source_rank_clusters_<subject>_hold_out_<held-out-block>: The rank-cluster assignment for each
                (source, grid-cell) tuple. A 2d array with shape (num_sources, num_grid_cells)
            co_occurrence_rank_clusters_<subject>_hold_out_<held-out-block>: The rank-cluster assignment for each
                (co_occurrence_cluster, grid-cell) tuple. 
                A 2d array with shape (num_co_occurrence_clusters, num_grid_cells)
            data_<subject>_hold_out_<held-out-block>:
                The aggregate values for each cluster, computed as described in steps 3 and 4 above. 
                A 2d array with shape (num_stimuli, n_clusters). Note that this contains data for all 
                blocks, but the clustering is fit without the held out block.
            source_rank_clusters_multi_subject_<subject>_hold_out_<held-out-block>:
                Similar to source_rank_clusters_<subject>_hold_out_<held-out-block>, but contains the cluster 
                assignments when all of the subjects are clustered jointly. These are still separated out by subject
                even in the multi-subject clustering case
            co_occurrence_rank_clusters_multi_subject_<subject>_hold_out_<held-out-block>: 
                Similar to spectral_rank_clusters_<subject>_hold_out_<held-out-block>, but contains the cluster
                assignments when all of the subjects are clustered jointly. These are still separated out by subject
                even in the multi-subject clustering case
            data_multi_subject_hold_out_<held-out-block>: 
                The aggregate values for each cluster when the subjects are clustered jointly. 
                A 2d array with shape (num_stimuli, n_clusters). Note that this contains data for all blocks, 
                but the clustering is fit without the held out block.
        """

    output_name = 'harry_potter_meg_rank_clustered_{aggregation}{time_slice}{single_subject}.npz'.format(
        aggregation=aggregation,
        time_slice='' if time_slice_ms is None else '_time_slice_ms_{}'.format(time_slice_ms),
        single_subject='' if single_subject is None else '_' + single_subject)

    np.savez(os.path.join(element_analysis_dir, output_name), **result)
