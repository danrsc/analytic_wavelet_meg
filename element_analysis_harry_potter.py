import os
import gc
from datetime import datetime
import numpy as np
import mne

from tqdm.auto import tqdm

from analytic_wavelet import ElementAnalysisMorse
from analytic_wavelet_meg import radians_per_ms_from_hertz, PValueFilterFn, maxima_of_transform_mne_source_estimate, \
    assign_grid_labels, common_label_count


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

    grid, grid_labels = assign_grid_labels(
        ea_morse, scale_frequencies[np.arange(0, len(scale_frequencies), 10)],
        indices_time, f_hat, batch_size=100000)

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

    all_subjects = 'A', 'B', 'C', 'D', 'E', 'F', 'H', 'I'
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

        output_file_name = 'shared_grid_element_counts_{subject}.npz'
        if label_name is not None:
            output_file_name = 'shared_grid_element_counts_{subject}_{label}.npz'
        np.savez(
            os.path.join(element_analysis_dir, output_file_name.format(subject=subject, label=label_name)), **result)
