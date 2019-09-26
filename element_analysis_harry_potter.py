import os
import gc
from datetime import datetime
import numpy as np
from bottleneck import nanrankdata, nanmedian, nansum, nanmean
import mne

from tqdm.auto import tqdm

from analytic_wavelet import ElementAnalysisMorse
from analytic_wavelet_meg import radians_per_ms_from_hertz, PValueFilterFn, maxima_of_transform_mne_source_estimate, \
    make_grid, assign_grid_labels, common_label_count, segment_combine_events, segment_median, k_pod


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


def _compute_shared_grid_element_counts(ea_path):
    ea_block = np.load(ea_path)

    ea_morse = ElementAnalysisMorse(ea_block['gamma'], ea_block['analyzing_beta'], ea_block['element_beta'])
    c_hat, _, f_hat = ea_morse.event_parameters(
        ea_block['maxima_coefficients'], ea_block['interpolated_scale_frequencies'])

    scale_frequencies = ea_block['scale_frequencies']

    indices_stimuli, indices_source, _, indices_time = np.unravel_index(
        ea_block['indices_flat'], ea_block['shape'])

    # noinspection PyTypeChecker
    grid = make_grid(ea_morse, ea_block['shape'][-1], scale_frequencies[np.arange(0, len(scale_frequencies), 10)])
    grid_labels = assign_grid_labels(grid, indices_time, f_hat)

    # flatten this index to save some memory
    combine_source_labels = np.ravel_multi_index(
        (indices_stimuli, grid_labels), (ea_block['shape'][0], len(grid)))

    del indices_stimuli

    unique_sources, _, joint_count, independent_count = common_label_count(indices_source, combine_source_labels)
    return unique_sources, joint_count, independent_count, grid, grid_labels


def compute_shared_grid_element_counts(element_analysis_dir, subjects=None, label_name=None):
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
        subjects: Which subjects to run the counts on. Defaults to all subjects.
        label_name: If specified, analysis is restricted to a single FreeSurfer label

    Returns:
        None. Saves an output file for each subject, containing, for each block, the keys:
            sources_<block>: The unique sources for the block, in the order of the grid element counts. As long
                as the element analysis has been run in a consistent way, these should be the same across all blocks.
            source_shared_grid_element_count_<block>: The (num_vertices, num_vertices) matrix of counts giving how
                often two vertices have events in the same grid-element. The events must occur in the grid element
                within the same epoch in order to be counted as being shared. The count for a grid element within an
                epoch is the minimum of the count of events from vertex 1 and vertex 2 within that element in that
                epoch
            grid_<block>: A (num_grid_elements, 4) array giving the bounding boxes for each grid element as:
                (time_lower, time_upper, freq_lower, freq_upper). Note that grid-element assignments are done by
                nearest-neighbor to the center point of each grid element. As long as element analysis has been run
                in a consistent way, these should be the same across all blocks.
            grid_elements_<block>: A 1d array of the labels of which grid element each event is assigned to.
        The output also contains the key "blocks", which gives the list of blocks in the output.
    """

    all_subjects = 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'
    all_blocks = '1', '2', '3', '4'

    if subjects is None:
        subjects = all_subjects

    for s in subjects:
        if s not in all_subjects:
            raise ValueError('Unknown subject: {}'.format(s))

    # this takes about 1 hour per block
    for subject in tqdm(subjects):

        tqdm.write('Beginning analysis for subject {} at {}'.format(subject, datetime.now()))

        result = {'blocks': all_blocks}
        subject_sources = None
        subject_grid = None
        for block in tqdm(all_blocks, leave=False):
            if label_name is not None:
                ea_path = os.path.join(
                    element_analysis_dir,
                    'harry_potter_element_analysis_{}_{}_{}.npz'.format(subject, block, label_name))
            else:
                ea_path = os.path.join(element_analysis_dir,
                                       'harry_potter_element_analysis_{}_{}.npz'.format(subject, block))
            unique_sources, joint_counts, independent_counts, grid, grid_labels = \
                _compute_shared_grid_element_counts(ea_path)

            # try to release some memory
            gc.collect()

            if subject_sources is None:
                subject_sources = unique_sources
                subject_grid = grid
                result['sources'] = unique_sources
                result['grid'] = grid
            else:
                assert(np.array_equal(subject_sources, unique_sources))
                assert(np.array_equal(subject_grid, grid))

            result['joint_grid_element_count_{}'.format(block)] = joint_counts
            result['independent_grid_element_count_{}'.format(block)] = independent_counts
            result['grid_elements_{}'.format(block)] = grid_labels

            # get rid of these references so we can free up memory after we write result (i.e. on the last block)
            del unique_sources
            del joint_counts
            del independent_counts
            del grid
            del grid_labels

        output_file_name = 'harry_potter_shared_grid_element_counts_{subject}.npz'
        if label_name is not None:
            output_file_name = 'harry_potter_shared_grid_element_counts_{subject}_{label}.npz'
        np.savez(
            os.path.join(element_analysis_dir, output_file_name.format(subject=subject, label=label_name)), **result)


def spectral_source_cluster(
        element_analysis_dir,
        subject,
        n_clusters=100,
        use_pointwise_mutual_information=True,
        use_median=False,
        **spectral_kwargs):
    """
    Uses spectral clustering on an affinity matrix computed between each dipole and each other dipole.
    Args:
        element_analysis_dir: The directory where element analysis files can be found. For example:
            '/share/volume0/<your user name>/data_sets/harry_potter/element_analysis'
        subject: Which subject to run the source clustering on
        n_clusters: The number of clusters to produce
        use_pointwise_mutual_information: If True normalizes joint counts by the product of the independent counts,
            otherwise no normalization is done
        use_median: Events in the same cluster, epoch, and grid-element are aggregated together by this method.
            If use_median=True, the aggregated event properties are the medians over the events being aggregated
                of each property
            If use_median=False, the aggregated event magnitude is the L2 norm of the wavelet coefficients of each
                event being aggregated. The other properties of the aggregated event are the power-weighted average
                of each property of the events being aggregated.
        **spectral_kwargs: Additional arguments to sklearn.cluster.SpectralClustering

    Returns:
        Saves a new file into element_analysis_dir for each held-out block with name
        harry_potter_spectral_clustered_{subject}_hold_out_{hold_out}.npz
        Note that each hold-out file has data for all of the blocks, with the held out block mapped according to the
        clustering done on the other blocks.

        source_clusters: The cluster assignment for each source
        magnitude: The L2 norm of the modulus of the events occurring within the same
            (block, stimulus, cluster, grid_element) tuple
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
    from sklearn.cluster import SpectralClustering

    all_blocks = 1, 2, 3, 4

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
        ea_block = np.load(ea_path)

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
        del ea_block

    indices_stimuli = np.concatenate(indices_stimuli)
    indices_block = np.concatenate(indices_block)
    indices_sources = np.concatenate(indices_source)
    indices_time = np.concatenate(indices_time)
    interp_fs = np.concatenate(interp_fs)
    c_hat = np.concatenate(c_hat)
    f_hat = np.concatenate(f_hat)

    count_path = os.path.join(
        element_analysis_dir, 'harry_potter_shared_grid_element_counts_{subject}.npz'.format(subject=subject))
    counts = np.load(count_path)

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
            independent_counts = np.expand_dims(independent_counts, 1) * np.expand_dims(independent_counts, 0)
            joint_counts = joint_counts / independent_counts

        spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', **spectral_kwargs)
        source_clusters = spectral_clustering.fit_predict(joint_counts)
        sample_clusters = source_clusters[indices_sources]

        segments = np.concatenate([
            np.expand_dims(indices_stimuli, 1),
            np.expand_dims(indices_block, 1),
            np.expand_dims(sample_clusters, 1),
            np.expand_dims(grid_elements, 1)], axis=1)

        if use_median:
            (segments,
             segment_counts,
             w_segment,
             f_hat_segment,
             time_segment,
             interpolated_scale_frequencies_segment) = segment_median(
                segments, np.abs(c_hat), f_hat, indices_time, interp_fs)
            output_path = os.path.join(
                element_analysis_dir,
                'harry_potter_spectral_clustered_median_{subject}_hold_out_{hold_out}.npz'.format(
                    subject=subject, hold_out=hold_out_block))
        else:
            segments, segment_counts, w_segment, power_weighted = segment_combine_events(
                segments, c_hat, power_weighted_dict={'f_hat': f_hat, 'time': indices_time, 'interp_fs': interp_fs})
            f_hat_segment, time_segment, interpolated_scale_frequencies_segment = (
                power_weighted['f_hat'], power_weighted['time'], power_weighted['interp_fs'])
            output_path = os.path.join(
                element_analysis_dir, 'harry_potter_spectral_clustered_{subject}_hold_out_{hold_out}.npz'.format(
                    subject=subject, hold_out=hold_out_block))

        np.savez(
            output_path,
            source_clusters=source_clusters,
            counts=segment_counts,
            magnitude=w_segment,
            f_hat=f_hat_segment,
            time=time_segment,
            interpolated_scale_frequencies=interpolated_scale_frequencies_segment,
            stimuli=segments[:, 0],
            blocks=segments[:, 1],
            sample_source_clusters=segments[:, 2],
            grid_elements=segments[:, 3],
            grid=counts['grid'])


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


def _rank_cluster_helper(
        dense_array,
        block_stimuli,
        source_clusters,
        grid,
        held_out_block,
        maximum_proportion_missing,
        aggregation_mode,
        use_mini_batch,
        k_pod_kwargs):

    indicator_held_out = block_stimuli[:, 0] == held_out_block

    spectral_rank_clusters = np.full(dense_array.shape[:2], -1, dtype=np.int32)
    indices_source_cluster, indices_element = np.unravel_index(
        np.arange(dense_array.shape[0] * dense_array.shape[1]), dense_array.shape[:2])
    dense_array = np.reshape(dense_array, (dense_array.shape[0] * dense_array.shape[1], dense_array.shape[2]))

    train_array = dense_array[:, np.logical_not(indicator_held_out)]

    indicator_enough_data = (np.count_nonzero(np.isnan(train_array), axis=1) / train_array.shape[1]
                             <= maximum_proportion_missing)

    train_array = train_array[indicator_enough_data]

    train_array = nanrankdata(train_array, axis=1)

    rank_clusters, centroids, _ = k_pod(train_array, use_mini_batch=use_mini_batch, **k_pod_kwargs)
    unique_rank_clusters, inverse = np.unique(rank_clusters, return_inverse=True)
    assert (np.array_equal(unique_rank_clusters, np.arange(len(unique_rank_clusters))))

    indices_spectral_rank_clusters = np.unravel_index(
        np.arange(spectral_rank_clusters.size)[indicator_enough_data], spectral_rank_clusters.shape)
    spectral_rank_clusters[indices_spectral_rank_clusters] = inverse

    dense_array = dense_array[indicator_enough_data]
    indices_source_cluster = indices_source_cluster[indicator_enough_data]
    indices_element = indices_element[indicator_enough_data]
    result_data = np.full((dense_array.shape[1], len(unique_rank_clusters)), np.nan, dtype=dense_array.dtype)
    source_rank_clusters = np.full((len(source_clusters), len(grid)), -1, dtype=np.int32)
    for c in unique_rank_clusters:
        indicator_cluster = c == rank_clusters
        if aggregation_mode == 'sum' or aggregation_mode == 'counts':
            result_data[:, c] = nansum(dense_array[indicator_cluster], axis=0)
        elif aggregation_mode == 'mean':
            result_data[:, c] = nanmean(dense_array[indicator_cluster], axis=0)
        elif aggregation_mode == 'median':
            result_data[:, c] = nanmedian(dense_array[indicator_cluster], axis=0)
        elif aggregation_mode == 'L2':
            result_data[:, c] = np.sqrt(nansum(np.square(dense_array[indicator_cluster]), axis=0))
        else:
            raise ValueError('Unknown aggregation_mode: {}'.format(aggregation_mode))
        indices_source_cluster_match = indices_source_cluster[indicator_cluster]
        indices_element_match = indices_element[indicator_cluster]
        for s, e in zip(indices_source_cluster_match, indices_element_match):
            indices_source = np.flatnonzero(source_clusters == s)
            source_rank_clusters[indices_source, e] = c

    return (
        result_data,
        block_stimuli[:, 1],
        block_stimuli[:, 0],
        source_rank_clusters,
        spectral_rank_clusters,
        centroids)


def _rank_cluster_only_combine_block(
        element_analysis_dir, subject, block, grid, source_rank_clusters, mode):
    """
    Computes aggregate events for each (source, rank-cluster) tuple bypassing spectral cluster aggregation. This
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
        mode:
            If 'median', uses harry_potter_spectral_clustered_median_{subject}_hold_out_{hold_out}.npz as
                input, thereby using the median absolute wavelet coefficient of the spectral clusters to rank
                epochs when rank clustering. Also combines values from the spectral clusters using the median.
            If 'counts', uses harry_potter_spectral_clustered_median_{subject}_hold_out_{hold_out}.npz as
                input, but uses the number of events in each spectral cluster instead of the wavelet coefficients
                to rank epochs when rank clustering. Combines values (counts) from the spectral clusters by summing.
            If 'L2' (default), uses harry_potter_spectral_clustered_{subject}_hold_out_{hold_out}.npz as input, thereby
                using the L2-norm of the absolute value of the wavelet coefficients of the spectral clusters to rank
                epochs when rank clustering. Combines values from spectral clusters using the L2-norm.

    Returns:
        combined_events: A 3d array with shape (num_epochs, num_sources, num_rank_clusters) containing the aggregated
            events
    """
    ea_path = os.path.join(
        element_analysis_dir, 'harry_potter_element_analysis_{}_{}.npz'.format(subject, block))
    ea_block = np.load(ea_path)

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

    dense_array = np.full((np.max(indices_stimuli) + 1,) + source_rank_clusters.shape, np.nan)

    if mode == 'counts':
        unique_segments, counts = np.unique(segments, axis=0, return_counts=True)
        dense_array[unique_segments] = counts
    elif mode == 'median':
        unique_segments, _, w = segment_median(segments, np.abs(c_hat))
        dense_array[unique_segments] = w
    elif mode == 'L2':
        unique_segments, _, w = segment_combine_events(segments, c_hat)
        dense_array[unique_segments] = w
    else:
        raise ValueError('Unrecognized mode: {}'.format(mode))

    return dense_array


def rank_cluster(
        element_analysis_dir,
        maximum_proportion_missing=0.1,
        mode='L2',
        n_multi_subject_clusters=None,
        **k_pod_kwargs):
    """
    Uses k-means clustering treating each (source-cluster, grid-element) tuple as a sample and the rank-order
    of the epochs as the features. (source-cluster, grid-element) tuples having more than maximum_proportion_missing
    missing epochs will be dropped from the data instead of clustered. The k-pod variant of k-means is used to
    handle missing epochs in the remaining data.
    Args:
        element_analysis_dir: The directory where element analysis files can be found. For example:
            '/share/volume0/<your user name>/data_sets/harry_potter/element_analysis'
        maximum_proportion_missing: (source-cluster, grid-element) tuples having more than maximum_proportion_missing
            missing epochs will be dropped from the data instead of clustered.
        mode:
            If 'median', uses harry_potter_spectral_clustered_median_{subject}_hold_out_{hold_out}.npz as
                input, thereby using the median absolute wavelet coefficient of the spectral clusters to rank
                epochs when rank clustering. Also combines values from the spectral clusters using the median.
            If 'counts', uses harry_potter_spectral_clustered_median_{subject}_hold_out_{hold_out}.npz as
                input, but uses the number of events in each spectral cluster instead of the wavelet coefficients
                to rank epochs when rank clustering. Combines values (counts) from the spectral clusters by summing.
            If 'L2' (default), uses harry_potter_spectral_clustered_{subject}_hold_out_{hold_out}.npz as input, thereby
                using the L2-norm of the absolute value of the wavelet coefficients of the spectral clusters to rank
                epochs when rank clustering. Combines values from spectral clusters using the L2-norm.
        n_multi_subject_clusters:
            If None, then the number of multi-subject clusters will be the same as the number of single-subject
            clusters. Set to 0 to turn off multi-subject clustering.
        **k_pod_kwargs: Additional arguments to sklearn.cluster.SpectralClustering

    Returns:
        Saves a new file into element_analysis_dir with name harry_potter_rank_clustered.npz
        Note that the file has data for each subject and hold out.

        description: Comments about what this data file contains.
        grid: The boundaries of the grid elements (copied from input data). A 2d array with shape
            (num_grid_elements, 4) giving (low-time, high-time, low-freq, high-freq) coordinates for each grid-element.
        stimuli: The index of the stimulus (within block) corresponding to each row of data_<held-out-block>
        blocks: The block corresponding to each row of data_<held-out-block>
        spectral_source_clusters_<subject>_hold_out_<held-out-block>:
            The spectral cluster assignment for each source (copied from input data).
            A 1d array with shape (num_sources,)
        source_rank_clusters_<subject>_hold_out_<held-out-block>: The rank-cluster assignment for each
            (source, grid-element) tuple. A 2d array with shape (num_sources, num_grid_elements)
        spectral_rank_clusters_<subject>_hold_out_<held-out-block>: The rank-cluster assignment for each
            (spectral_cluster, grid-element) tuple. A 2d array with shape (num_spectral_clusters, num_grid_elements)
        rank_cluster_centroids_<subject>_hold_out_<held-out-block>: The centroids for each rank-cluster.
            A 2d array of shape (num_rank_clusters, num_train_epochs)
        data_<subject>_hold_out_<held-out-block>:
            The L2 norm of the input magnitudes for each cluster. A 2d array with shape
            (num_stimuli, n_clusters)
    """
    all_blocks = 1, 2, 3, 4
    all_subjects = 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'

    # subject A has 2 '+' stimuli missing
    a_insert_indices = 1302, 2653

    result = dict()
    n_clusters = None

    for held_out_block in tqdm(all_blocks, desc='hold_out_block'):

        multi_subject_arrays = list()
        multi_subject_block_stimuli = None
        multi_subject_source_clusters = list()
        multi_subject_source_cluster_offset = 0

        for subject in tqdm(all_subjects, desc='subject', leave=False):
            if mode == 'median' or mode == 'counts':
                spectral_clustered_path = os.path.join(
                    element_analysis_dir,
                    'harry_potter_spectral_clustered_median_{subject}_hold_out_{block}.npz'.format(
                        subject=subject, block=held_out_block))
            elif mode == 'L2':
                spectral_clustered_path = os.path.join(
                    element_analysis_dir,
                    'harry_potter_spectral_clustered_{subject}_hold_out_{block}.npz'.format(
                        subject=subject, block=held_out_block))
            else:
                raise ValueError('Unrecognized mode: {}'.format(mode))

            spectral_clustered = np.load(spectral_clustered_path)

            grid = spectral_clustered['grid']
            w = spectral_clustered['magnitude']
            counts = spectral_clustered['counts']
            blocks = spectral_clustered['blocks']
            stimuli = spectral_clustered['stimuli']
            elements = spectral_clustered['grid_elements']
            sample_source_clusters = spectral_clustered['sample_source_clusters']
            source_clusters = spectral_clustered['source_clusters']

            block_stimuli, virtual_stim = np.unique(
                np.concatenate([np.expand_dims(blocks, 1), np.expand_dims(stimuli, 1)], axis=1),
                axis=0, return_inverse=True)

            dense_array = np.full(
                (np.max(sample_source_clusters) + 1, np.max(elements) + 1, np.max(virtual_stim) + 1), np.nan)

            if mode == 'counts':
                dense_array[(sample_source_clusters, elements, virtual_stim)] = counts
            else:
                dense_array[(sample_source_clusters, elements, virtual_stim)] = w

            if subject == 'A':
                for idx in a_insert_indices:
                    block_stimuli, dense_array = _insert_missing(block_stimuli, dense_array, idx)

            if n_multi_subject_clusters != 0:
                multi_subject_arrays.append(dense_array)
                if multi_subject_block_stimuli is None:
                    multi_subject_block_stimuli = block_stimuli
                else:
                    assert(np.array_equal(multi_subject_block_stimuli, block_stimuli))
                multi_subject_source_clusters.append(source_clusters + multi_subject_source_cluster_offset)
                multi_subject_source_cluster_offset += np.max(source_clusters) + 1

            result_data, stimuli, blocks, source_rank_clusters, spectral_rank_clusters, centroids = \
                _rank_cluster_helper(
                    dense_array, block_stimuli, source_clusters, grid, held_out_block, maximum_proportion_missing,
                    aggregation_mode=mode, use_mini_batch=False, k_pod_kwargs=k_pod_kwargs)
            n_clusters = result_data.shape[1]

            if 'grid' in result:
                assert np.array_equal(result['grid'], grid)
                assert np.array_equal(result['stimuli'], stimuli)
                assert np.array_equal(result['blocks'], blocks)
            else:
                result['grid'] = grid
                result['stimuli'] = stimuli
                result['blocks'] = blocks

            result['spectral_source_clusters_{}_hold_out_{}'.format(subject, held_out_block)] = source_clusters
            result['source_rank_clusters_{}_hold_out_{}'.format(subject, held_out_block)] = source_rank_clusters
            result['spectral_rank_clusters_{}_hold_out_{}'.format(subject, held_out_block)] = spectral_rank_clusters
            result['rank_cluster_centroids_{}_hold_out_{}'.format(subject, held_out_block)] = centroids
            result['data_{}_hold_out_{}'.format(subject, held_out_block)] = result_data

        if n_multi_subject_clusters != 0:
            k_pod_kwargs_multi = dict(k_pod_kwargs)
            k_pod_kwargs_multi['n_clusters'] = n_multi_subject_clusters \
                if n_multi_subject_clusters is not None else n_clusters

            result_data, stimuli, blocks, source_rank_clusters, spectral_rank_clusters, centroids = \
                _rank_cluster_helper(
                    np.concatenate(multi_subject_arrays),
                    multi_subject_block_stimuli,
                    np.concatenate(multi_subject_source_clusters),
                    result['grid'],
                    held_out_block, maximum_proportion_missing,
                    aggregation_mode=mode,
                    use_mini_batch=False,
                    k_pod_kwargs=k_pod_kwargs_multi)

            assert np.array_equal(result['stimuli'], stimuli)
            assert np.array_equal(result['blocks'], blocks)

            source_rank_clusters = np.split(
                source_rank_clusters, np.cumsum(list(len(sc) for sc in multi_subject_source_clusters))[:-1])
            spectral_rank_clusters = np.split(
                spectral_rank_clusters, np.cumsum(list(a.shape[0] for a in multi_subject_arrays))[:-1])

            result['rank_cluster_centroids_multi_subject_hold_out_{}'.format(held_out_block)] = centroids
            result['data_multi_subject_hold_out_{}'.format(held_out_block)] = result_data

            for idx_subject, subject in enumerate(all_subjects):
                result['source_rank_clusters_multi_subject_{}_hold_out_{}'.format(subject, held_out_block)] = \
                    source_rank_clusters[idx_subject]
                result['spectral_rank_clusters_multi_subject_{}_hold_out_{}'.format(subject, held_out_block)] = \
                    spectral_rank_clusters[idx_subject]

    # convert stimuli from indices into text for convenience
    stimuli = list()
    # anyone other than A should be fine for this. Choose the first in all_subjects that is not A, or if
    # that is empty, set to B
    stimulus_subject = [s for s in all_subjects if s != 'A']
    if len(stimulus_subject) > 0:
        stimulus_subject = stimulus_subject[0]
    else:
        stimulus_subject = 'B'
    for block in all_blocks:
        ea_path = os.path.join(
            element_analysis_dir, 'harry_potter_element_analysis_{}_{}.npz'.format(stimulus_subject, block))
        ea_block = np.load(ea_path)
        ea_stimuli = ea_block['stimuli']
        stimuli.append(ea_stimuli[result['stimuli'][result['blocks'] == block]])
    result['stimuli'] = np.concatenate(stimuli)

    # check that data we inserted is at '+'
    for idx in a_insert_indices:
        assert(result['stimuli'][idx]) == '+'

    result['subjects'] = all_subjects

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
            (maxima) within a dipole cluster are then combined together within each grid-cell, epoch tuple. When 
            multiple events occur within the same grid-cell, they are aggregated according to one of three rules.
                In harry_potter_rank_clustered.npz the magnitude is computed as the L2-norm of the events,
                    and other properties of the events are combined using the power-weighted average.
                In harry_potter_rank_clustered_median.npz the properties of the aggregate events are the medians
                    of each property
                In harry_potter_rank_clustered_counts.npz the number of events comprising each cluster is used
                    to represent each cluster
        4. A matrix is formed which has on one axis each dipole-cluster, grid-cell tuple, and on the other axis each
            epoch. The epochs are ranked within a dipole-cluster, and k-means is applied using as the samples the 
            (dipole-cluster, grid-cell) axis and as feature the epoch ranking (actually a
            variant of k-means which can handle missing data). This gives a new clustering, which groups together 
            similar (dipole-cluster, grid-cell) tuples where similarity is defined as their ranking of the epochs.
            We call these rank-clusters. Events occurring within the same rank cluster are again aggregated together.
            For each epoch (word) using the same aggregation as described in step 3, this gives us num_rank_cluster 
            values as the final data.
            
        The code which produces this data can be found at 
        https://github.com/danrsc/analytic_wavelet_meg and https://github.com/danrsc/analytic_wavelet
        
        Note that all fields in this file of the form *_<subject>_hold_out_<held-out-block> contain information
        for one subject where the values were computed with the appropriate block held out of all training steps 
        (the PMI is computed without this block for the spectral clustering, and the k-means is computed without this
        block for the rank clustering).
        
        Fields:
            description: Comments about what this data file contains.
            grid: The boundaries of the grid elements. A 2d array with shape (num_grid_elements, 4) giving 
                (low-time, high-time, low-freq, high-freq) coordinates for each grid-element. The high-time 
                and high-freq boundaries are exclusive, the low-time and low-freq boundaries are inclusive. 
                Grid-elements at the lowest-frequency have an open-lower bound, meaning events which fall below the
                lowest-frequency are put into the lowest frequency bins at the same time interval. Similarly, events
                which occur at a higher frequency than the highest frequency bins are put into the highest frequency
                bins at the same time interval.
            stimuli: The text of the stimulus corresponding to each row of data_<subject>_hold_out_<held-out-block>
            blocks: The block corresponding to each row of data_<subject>_hold_out_<held-out-block>
            spectral_source_clusters_<subject>_hold_out_<held-out-block>:
                The spectral cluster assignment for each source (dipole).
                A 1d array with shape (num_sources,)
            source_rank_clusters_<subject>_hold_out_<held-out-block>: The rank-cluster assignment for each
                (source, grid-element) tuple. A 2d array with shape (num_sources, num_grid_elements)
            spectral_rank_clusters_<subject>_hold_out_<held-out-block>: The rank-cluster assignment for each
                (spectral_cluster, grid-element) tuple. A 2d array with shape (num_spectral_clusters, num_grid_elements)
            rank_cluster_centroids_<subject>_hold_out_<held-out-block>: The centroids for each rank-cluster.
                A 2d array of shape (num_rank_clusters, num_train_epochs)
            data_<subject>_hold_out_<held-out-block>:
                The L2 norm (or median, when mode='median', or counts when mode='counts') of the input magnitudes for 
                each cluster. A 2d array  with shape (num_stimuli, n_clusters). Note that this contains data for all 
                blocks, but the clustering is fit without the held out block.
            source_rank_clusters_multi_subject_<subject>_hold_out_<held-out-block>:
                Similar to source_rank_clusters_<subject>_hold_out_<held-out-block>, but contains the cluster 
                assignments when all of the subjects are clustered jointly. These are still separated out by subject
                even in the multi-subject clustering case
            spectral_rank_clusters_multi_subject_<subject>_hold_out_<held-out-block>: 
                Similar to spectral_rank_clusters_<subject>_hold_out_<held-out-block>, but contains the cluster
                assignments when all of the subjects are clustered jointly. These are still separated out by subject
                even in the multi-subject clustering case
            rank_cluster_centroids_multi_subject_hold_out_<held-out-block>: The centroids for each rank-cluster when
                all of the subjects are clustered jointly.
            data_multi_subject_hold_out_<held-out-block>: 
                The L2 norm (or median, when mode='median', or counts when mode='counts') of the input magnitudes for 
                each cluster when the subjects are clustered jointly. A 2d array with shape (num_stimuli, n_clusters). 
                Note that this contains data for all blocks, but the clustering is fit without the held out block.
        """
    if mode == 'counts':
        np.savez(
            os.path.join(element_analysis_dir, 'harry_potter_rank_clustered_counts.npz'),
            **result)
    elif mode == 'median':
        np.savez(
            os.path.join(element_analysis_dir, 'harry_potter_rank_clustered_median.npz'),
            **result)
    elif mode == 'L2':
        np.savez(
            os.path.join(element_analysis_dir, 'harry_potter_rank_clustered.npz'),
            **result)
    else:
        raise ValueError('Unrecognized mode: {}'.format(mode))


def source_rank_clustered_from_rank_clustered(element_analysis_dir, mode='L2'):
    """
    Computes aggregate events for each (source, rank-cluster) tuple bypassing spectral cluster aggregation, and makes
    a new file which includes both this data and all of the data from the output of rank_cluster, but separated
    into multiple files for each subject and held-out-block and since these files are big.
    Useful for comparing a predicted result to a ground-truth value at each dipole.
    Args:
        element_analysis_dir: The directory where element analysis files can be found. For example:
            '/share/volume0/<your user name>/data_sets/harry_potter/element_analysis'
        mode:
            If 'median', uses harry_potter_rank_clustered_median.npz as input, and aggregates events having the same
                (epoch, source, rank-cluster) using the median of the absolute value of the wavelet coefficient for
                those events
            If 'counts', uses harry_potter_rank_clustered_median.npz as input, but uses the number of events in
                each (epoch, source, rank-cluster) tuple as the representation for that tuple
            If 'L2' (default), uses harry_potter_rank_clustered.npz as input, and aggregates events having the same
                (epoch, source, rank-cluster) using the L2-norm of the absolute value of the wavelet coefficients of
                those events.
    Returns:
        Saves a new file into element_analysis_dir with name similar to (depending on mode)
            harry_potter_source_rank_clustered_<subject>_hold_out_<held-out-block>.npz

        Fields:
            description: Comments about what this data file contains.
            grid: The boundaries of the grid elements. A 2d array with shape (num_grid_elements, 4) giving
                (low-time, high-time, low-freq, high-freq) coordinates for each grid-element. The high-time
                and high-freq boundaries are exclusive, the low-time and low-freq boundaries are inclusive.
                Grid-elements at the lowest-frequency have an open-lower bound, meaning events which fall below
                the lowest-frequency are put into the lowest frequency bins at the same time interval.
                Similarly, events which occur at a higher frequency than the highest frequency bins are put
                into the highest frequency bins at the same time interval.
            stimuli: The text of the stimulus corresponding to each row of rank_clustered_data
            blocks: The block corresponding to each row of rank_clustered_data
            spectral_source_clusters: The spectral cluster assignment for each source (dipole).
                A 1d array with shape (num_sources,)
            source_rank_clusters: The rank-cluster assignment for each (source, grid-element) tuple.
                A 2d array with shape (num_sources, num_grid_elements)
            spectral_rank_clusters: The rank-cluster assignment for each (spectral_cluster, grid-element)
                tuple. A 2d array with shape (num_spectral_clusters, num_grid_elements)
            rank_cluster_centroids: The centroids for each rank-cluster.
                A 2d array of shape (num_rank_clusters, num_train_epochs)
            rank_clustered_data:
                The L2 norm (or median when mode='median', or counts when mode='counts') of the input
                magnitudes for each cluster. A 2d array with shape (num_stimuli, n_clusters). Note that
                this contains data for all blocks, but the clustering is fit without the held out block.
            source_rank_clustered_data:
                The L2 norm (or median when mode='median', or counts when mode='counts') of the event
                magnitudes for all events with (dipole, grid_element) coordinates that map to the same
                (dipole, rank-cluster) coordinates. A 3d array with
                shape (num_stimuli, num_dipoles, n_clusters). Note that this contains data for all blocks, but
                the clustering is fit without the held out block.
            source_rank_clusters_multi_subject: Similar to source_rank_clusters, but contains the cluster
                assignments when all of the subjects are clustered jointly.
            spectral_rank_clusters_multi_subject:
                Similar to spectral_rank_clusters, but contains the cluster assignments when all of the
                subjects are clustered jointly.
            rank_cluster_centroids_multi_subject: The centroids for each rank-cluster when
                all of the subjects are clustered jointly.
            rank_clustered_data_multi_subject:
                The L2 norm (or median, when mode='median', or counts when mode='counts') of the input
                magnitudes for each cluster when the subjects are clustered jointly. A 2d array with shape
                (num_stimuli, n_clusters). Note that this contains data for all blocks, but the clustering is
                fit without the held out block.
            source_rank_clustered_data_multi_subject:
                The L2 norm (or median when mode='median', or counts when mode='counts') of the event
                magnitudes for all events with (dipole, grid_element) coordinates that map to the same
                (dipole, rank-cluster) coordinates when the subjects are clustered jointly. A 3d array with
                shape (num_stimuli, num_dipoles, n_clusters). Note that this contains data for all blocks, but
                the clustering is fit without the held out block.
    """
    if mode == 'counts':
        rank_clustered = np.load(os.path.join(element_analysis_dir, 'harry_potter_rank_clustered_counts.npz'))
    elif mode == 'median':
        rank_clustered = np.load(os.path.join(element_analysis_dir, 'harry_potter_rank_clustered_median.npz'))
    elif mode == 'L2':
        rank_clustered = np.load(os.path.join(element_analysis_dir, 'harry_potter_rank_clustered.npz'))
    else:
        raise ValueError('Unrecognized mode: {}'.format(mode))

    blocks = np.unique(rank_clustered['blocks'])
    for held_out_block in tqdm(blocks, desc='hold_out_block'):
        for subject in tqdm(rank_clustered['subjects'], desc='subject', leave=False):
            result = {
                'grid': rank_clustered['grid'],
                'stimuli': rank_clustered['stimuli'],
                'blocks': rank_clustered['blocks'],
                'spectral_source_clusters': rank_clustered[
                    'spectral_source_clusters_{}_hold_out_{}'.format(subject, held_out_block)],
                'source_rank_clusters': rank_clustered[
                    'source_rank_clusters_{}_hold_out_{}'.format(subject, held_out_block)],
                'spectral_rank_clusters': rank_clustered[
                    'spectral_rank_clusters_{}_hold_out_{}'.format(subject, held_out_block)],
                'rank_cluster_centroids': rank_clustered[
                    'rank_cluster_centroids_{}_hold_out_{}'.format(subject, held_out_block)],
                'rank_clustered_data': rank_clustered[
                    'data_{}_hold_out_{}'.format(subject, held_out_block)],
                'source_rank_clusters_multi_subject': rank_clustered[
                    'source_rank_clusters_multi_subject_{}_hold_out_{}'.format(subject, held_out_block)],
                'spectral_rank_clusters_multi_subject': rank_clustered[
                    'spectral_rank_clusters_multi_subject_{}_hold_out_{}'.format(subject, held_out_block)],
                'rank_clustered_data_multi_subject': rank_clustered[
                    'data_multi_subject_{}_hold_out_{}'.format(subject, held_out_block)]
            }
            for is_multi in (False, True):
                if is_multi:
                    source_rank_clusters = rank_clustered[
                        'source_rank_clusters_multi_subject_{}_hold_out_{}'.format(subject, held_out_block)]
                else:
                    source_rank_clusters = rank_clustered[
                        'source_rank_clusters_{}_hold_out_{}'.format(subject, held_out_block)]

                combined_events = list()
                for block in blocks:
                    combined_events.append(_rank_cluster_only_combine_block(
                        element_analysis_dir, subject, block, rank_clustered['grid'], source_rank_clusters, mode))
                combined_events = np.concatenate(combined_events)

                # subject A is missing a couple of fixation crosses at the end of blocks 1 and 2
                if subject == 'A':
                    for idx in 1302, 2653:
                        assert (result['stimuli'][idx]) == '+'
                        combined_events = np.concatenate([
                            combined_events[:idx],
                            np.full((1,) + combined_events.shape[1:], np.nan, dtype=combined_events.dtype),
                            combined_events[idx:]], axis=0)

                assert(len(combined_events) == len(result['stimuli']))

                if is_multi:
                    result['source_rank_clustered_data_multi_subject'] = combined_events
                else:
                    result['source_rank_clustered_data'] = combined_events

            result['description'] = \
                """
                This data file contains the result of clustering MEG data using a 4-step process, with an additional
                5th step applied after.:
                1. Data is source-localized using minimum norm estimation
                2. An analytic wavelet transform is run on each dipole, and element analysis (Lilly, 2017) is applied. 
                    Roughly speaking, this extracts local maxima from the wavelet transformed data.
                3. Each epoch (i.e. 500 ms of time corresponding to a word) is subdivided into a grid based on 
                    log-spaced frequencies and time-windows which vary with the footprint of the analyzing wavelet. 
                    Pointwise-mutual information is computed between each pair of dipoles. The number of maxima within 
                    a grid cell for a given dipole and epoch is used as a count for each individual dipole, and the 
                    joint count is the minimum of the individual count for each pair of dipoles. Summing over all 
                    epochs and grid-elements gives the final joint and individual counts from which PMI is computed. 
                    Spectral clustering is applied to the PMI matrix to cluster the dipoles. This can roughly be 
                    thought of as clustering the dipoles according to phase-locking value, i.e. this is an 
                    approximation to clustering by functional connectivity. The events (maxima) within a dipole 
                    cluster are then combined together within each grid-cell, epoch tuple. When multiple events occur 
                    within the same grid-cell, they are aggregated according to one of three rules.
                        In harry_potter_rank_clustered.npz the magnitude is computed as the L2-norm of the events,
                            and other properties of the events are combined using the power-weighted average.
                        In harry_potter_rank_clustered_median.npz the properties of the aggregate events are the medians
                            of each property
                        In harry_potter_rank_clustered_counts.npz the number of events comprising each cluster is used
                            to represent each cluster
                4. A matrix is formed which has on one axis each dipole-cluster, grid-cell tuple, and on the other 
                    axis each epoch. The epochs are ranked within a dipole-cluster, and k-means is applied using as 
                    the samples the (dipole-cluster, grid-cell) axis and as feature the epoch ranking (actually a
                    variant of k-means which can handle missing data). This gives a new clustering, which groups 
                    together similar (dipole-cluster, grid-cell) tuples where similarity is defined as their ranking 
                    of the epochs. We call these rank-clusters. Events occurring within the same rank cluster are 
                    again aggregated together. For each epoch (word) using the same aggregation as described in step 3, 
                    this gives us num_rank_cluster values as the final data.
                5. After this clustering process has completed, the raw events are aggregated according to the mapping
                    from a (dipole, grid_element) tuple to a rank-cluster. That is, we take the dipole, grid_element
                    coordinates of each event, compute the rank-cluster from this, and then output aggregate events for
                    each (dipole, rank-cluster) tuple. That gives a kind of ground-truth number for each rank-cluster
                    at each dipole that can be compared to predictions for each rank-cluster.

                The code which produces this data can be found at 
                https://github.com/danrsc/analytic_wavelet_meg and https://github.com/danrsc/analytic_wavelet

                Note that all fields in this file contain information for one subject where the values were computed 
                with the appropriate block held out of all training steps (the PMI is computed without this block 
                for the spectral clustering, and the k-means is computed without this block for the rank clustering).

                Fields:
                    description: Comments about what this data file contains.
                    grid: The boundaries of the grid elements. A 2d array with shape (num_grid_elements, 4) giving 
                        (low-time, high-time, low-freq, high-freq) coordinates for each grid-element. The high-time 
                        and high-freq boundaries are exclusive, the low-time and low-freq boundaries are inclusive. 
                        Grid-elements at the lowest-frequency have an open-lower bound, meaning events which fall below 
                        the lowest-frequency are put into the lowest frequency bins at the same time interval. 
                        Similarly, events which occur at a higher frequency than the highest frequency bins are put 
                        into the highest frequency bins at the same time interval.
                    stimuli: The text of the stimulus corresponding to each row of rank_clustered_data
                    blocks: The block corresponding to each row of rank_clustered_data
                    spectral_source_clusters: The spectral cluster assignment for each source (dipole).
                        A 1d array with shape (num_sources,)
                    source_rank_clusters: The rank-cluster assignment for each (source, grid-element) tuple. 
                        A 2d array with shape (num_sources, num_grid_elements)
                    spectral_rank_clusters: The rank-cluster assignment for each (spectral_cluster, grid-element) 
                        tuple. A 2d array with shape (num_spectral_clusters, num_grid_elements)
                    rank_cluster_centroids: The centroids for each rank-cluster.
                        A 2d array of shape (num_rank_clusters, num_train_epochs)
                    rank_clustered_data:
                        The L2 norm (or median when mode='median', or counts when mode='counts') of the input 
                        magnitudes for each cluster. A 2d array with shape (num_stimuli, n_clusters). Note that 
                        this contains data for all blocks, but the clustering is fit without the held out block.
                    source_rank_clustered_data:
                        The L2 norm (or median when mode='median', or counts when mode='counts') of the event
                        magnitudes for all events with (dipole, grid_element) coordinates that map to the same
                        (dipole, rank-cluster) coordinates. A 3d array with 
                        shape (num_stimuli, num_dipoles, n_clusters). Note that this contains data for all blocks, but
                        the clustering is fit without the held out block.
                    source_rank_clusters_multi_subject: Similar to source_rank_clusters, but contains the cluster 
                        assignments when all of the subjects are clustered jointly.
                    spectral_rank_clusters_multi_subject: 
                        Similar to spectral_rank_clusters, but contains the cluster assignments when all of the 
                        subjects are clustered jointly.
                    rank_cluster_centroids_multi_subject: The centroids for each rank-cluster when
                        all of the subjects are clustered jointly.
                    rank_clustered_data_multi_subject: 
                        The L2 norm (or median, when mode='median', or counts when mode='counts') of the input 
                        magnitudes for each cluster when the subjects are clustered jointly. A 2d array with shape 
                        (num_stimuli, n_clusters). Note that this contains data for all blocks, but the clustering is 
                        fit without the held out block.
                    source_rank_clustered_data_multi_subject:
                        The L2 norm (or median when mode='median', or counts when mode='counts') of the event
                        magnitudes for all events with (dipole, grid_element) coordinates that map to the same
                        (dipole, rank-cluster) coordinates when the subjects are clustered jointly. A 3d array with 
                        shape (num_stimuli, num_dipoles, n_clusters). Note that this contains data for all blocks, but
                        the clustering is fit without the held out block.
                """
            if mode == 'counts':
                np.savez(os.path.join(
                        element_analysis_dir,
                        'harry_potter_source_rank_clustered_counts_{}_hold_out_{}.npz'.format(subject, held_out_block)),
                    **result)
            elif mode == 'median':
                np.savez(os.path.join(
                        element_analysis_dir,
                        'harry_potter_source_rank_clustered_median_{}_hold_out_{}.npz'.format(subject, held_out_block)),
                    **result)
            elif mode == 'L2':
                np.savez(os.path.join(
                        element_analysis_dir,
                        'harry_potter_source_rank_clustered_{}_hold_out_{}.npz'.format(subject, held_out_block)),
                    **result)
            else:
                raise ValueError('Unrecognized mode: {}'.format(mode))
