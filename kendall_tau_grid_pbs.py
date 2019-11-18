import os
import sys
import argparse

import numpy as np


all_subjects = 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'
all_blocks = 1, 2, 3, 4


def grid_from_grid_ms(element_analysis_dir, subject, block, grid_ms):
    from analytic_wavelet_meg import make_grid
    from analytic_wavelet import ElementAnalysisMorse

    ea_path = os.path.join(
        element_analysis_dir, 'harry_potter_element_analysis_{}_{}.npz'.format(subject, block))

    with np.load(ea_path) as ea_block:
        shape = ea_block['shape']
        scale_frequencies = ea_block['scale_frequencies']
        ea_morse = ElementAnalysisMorse(
            ea_block['gamma'], ea_block['analyzing_beta'], ea_block['element_beta'])

    if grid_ms == 'time_freq':
        # noinspection PyTypeChecker
        return make_grid(ea_morse, shape[-1], scale_frequencies[np.arange(0, len(scale_frequencies), 10)])
    else:
        grid = list()
        grid_ms = int(grid_ms)
        for i in range(0, shape[-1], grid_ms):
            grid.append(
                (i, min(i + grid_ms, shape[-1]),
                 np.min(scale_frequencies), np.max(scale_frequencies)))
        return np.array(grid)


def _get_slice(element_analysis_dir, subject, block, grid_element, value_kind=None):
    from analytic_wavelet import ElementAnalysisMorse
    from analytic_wavelet_meg import assign_grid_labels, segment_median
    # noinspection PyProtectedMember
    from analytic_wavelet_meg.aggregation import _numba_bincount

    ea_path = os.path.join(
        element_analysis_dir, 'harry_potter_element_analysis_{}_{}.npz'.format(subject, block))
    with np.load(ea_path) as ea_block:
        shape = ea_block['shape']
        indices_stimuli, indices_source, _, indices_time = np.unravel_index(
            ea_block['indices_flat'], shape)
        multi = np.ravel_multi_index((indices_stimuli, indices_source), shape[:2])
        del indices_stimuli
        del indices_source

        ea_morse = ElementAnalysisMorse(ea_block['gamma'], ea_block['analyzing_beta'], ea_block['element_beta'])
        c_hat, _, f_hat = ea_morse.event_parameters(
            ea_block['maxima_coefficients'], ea_block['interpolated_scale_frequencies'])
        scale_frequencies = ea_block['scale_frequencies']
        indices_grid = assign_grid_labels(
            np.expand_dims(grid_element, 0), indices_time, f_hat,
            enforce_lower_frequency_bound=grid_element[2] != np.min(scale_frequencies),
            enforce_upper_frequency_bound=grid_element[3] != np.max(scale_frequencies),
            allow_unassigned=True)
        del f_hat
        del indices_time
        assert(np.max(indices_grid) == 0)
        indicator_grid = indices_grid == 0
        del indices_grid
        multi = multi[indicator_grid]

        if value_kind == 'percentile_75' or value_kind == 'median':
            c_hat = np.abs(c_hat[indicator_grid])
            if value_kind == 'percentile_75':
                multi, _, values = segment_median(multi, (c_hat, 0.75))
            elif value_kind == 'median':
                multi, _, values = segment_median(multi, c_hat)
            else:
                raise ValueError('Unknown value_kind: {}'.format(value_kind))
            indices_stimuli, indices_source = np.unravel_index(multi, shape[:2])
            values_ = np.zeros(shape[:2], values.dtype)
            values_[indices_stimuli, indices_source] = values
            values = values_
        else:
            weights = None
            if value_kind == 'amplitude':
                weights = np.abs(c_hat[indicator_grid])
            elif value_kind == 'power':
                weights = np.square(np.abs(c_hat[indicator_grid]))
            elif value_kind != 'count':
                raise ValueError('Unknown value_kind: {}'.format(value_kind))
            del c_hat
            values = _numba_bincount(multi, weights=weights)
            if len(values) < shape[0] * shape[1]:
                values = np.pad(values, (0, shape[0] * shape[1] - len(values)), mode='constant')
            values = np.reshape(values, shape[:2])
        return values


def kendall_tau_slice(element_analysis_dir, subject, hold_out_block, index_grid, grid_element, value_kind):
    from analytic_wavelet_meg import kendall_tau
    train_data = list()
    train_x = list()
    offset = 0
    for block in all_blocks:
        if block == hold_out_block:
            ea_path = os.path.join(
                element_analysis_dir, 'harry_potter_element_analysis_{}_{}.npz'.format(subject, block))
            with np.load(ea_path) as ea_block:
                offset += ea_block['shape'][0]
        else:
            train_data.append(_get_slice(element_analysis_dir, subject, block, grid_element, value_kind))
            train_x.append(np.arange(train_data[-1].shape[0]) + offset)
            offset += train_data[-1].shape[0]

    train_data = np.concatenate(train_data)
    train_x = np.concatenate(train_x)

    # remove any linear trend
    p = np.polyfit(train_x, train_data, deg=1)

    #      (1, num_columns)            (num_rows, 1)
    lines = np.reshape(p[0], (1, -1)) * np.reshape(train_x, (-1, 1)) + np.reshape(p[1], (1, -1))
    train_data = train_data - lines
    tau, num_common = kendall_tau(train_data.T)
    output_path = os.path.join(
        element_analysis_dir, 'kendall_tau', 'harry_potter_kendall_tau_grid_{}_{}_{}_{}.npz'.format(
            subject, hold_out_block, index_grid, value_kind))

    if not os.path.exists(os.path.split(output_path)[0]):
        os.makedirs(os.path.split(output_path)[0])

    np.savez(output_path, tau=tau, num_common=num_common)


def create_pbs_jobs(job_directory, element_analysis_dir, pool, subjects, blocks, grid_ms, grid_element_counts, kind):
    """
    Creates and queues pbs jobs for the selected subjects and blocks
    Args:
        job_directory: Directory where scripts and logs will be written
        element_analysis_dir: Working directory for element analysis results
        pool: Which pool to run the jobs on
        subjects: Which subjects to run the analysis on. Defaults to all subjects
        blocks: Which blocks to run the analysis on. Defaults to all blocks
        grid_ms: Either the string 'time_freq', or an int giving the number of milliseconds per time slice. If set
            to 'time_freq', an irregular grid will be computed which bundles the scale-frequencies into groups of
            10, and for each bundle computes a time-slice that is appropriate for the wavelet footprint at that scale
        grid_element_counts: How many grid elements there are for each subject
        kind: Determines how the kendall tau is computed. If 'amplitude', the sum of the absolute
            wavelet coefficients occurring during a slice is used. If 'power', the square of the
            modulus of the wavelet coefficient is used. If 'count', the wavelet coefficient is
            ignored, and only the count of events is used.
    """
    from element_analysis_harry_potter_pbs import create_python_exec_bash, queue_job

    if subjects is None:
        subjects = all_subjects

    for s in subjects:
        if s not in all_subjects:
            raise ValueError('Unknown subject: {}'.format(s))

    for b in blocks:
        if b not in all_blocks:
            raise ValueError('Unknown block: {}'.format(b))

    if not os.path.exists(os.path.join(job_directory, 'kendall_tau')):
        os.makedirs(os.path.join(job_directory, 'kendall_tau'))

    for subject, grid_count in zip(subjects, grid_element_counts):
        for block in blocks:
            for index_grid in range(grid_count):
                job_name = 'harry_potter_kendall_tau_{}_{}_{}'.format(subject, block, index_grid)
                bash_path = os.path.join(job_directory, 'kendall_tau', job_name + '.sh')

                arg_str = '--element_analysis_dir {} ' \
                          '--subject {} --block {} --grid_ms {} --grid_element {} --kind {}'.format(
                                element_analysis_dir, subject, block, grid_ms, index_grid, kind)

                create_python_exec_bash(
                    os.path.expanduser('~/src/analytic_wavelet_meg/'),
                    'kendall_tau_grid_pbs.py ' + arg_str,
                    bash_path,
                    os.path.join(job_directory, job_name + '.log'))

                queue_job(bash_path, None, pool)


if __name__ == '__main__':
    sys.path.append('/home/drschwar/src/analytic_wavelet')
    sys.path.append('/home/drschwar/src/analytic_wavelet_meg')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job_dir',
        action='store',
        help='A working directory where job script files and logs will be written. '
             'On the cortex cluster, a reasonable place is a subdirectory of your home directory: \n'
             '/home/<your user name>/element_analysis_jobs/',
        default='')
    parser.add_argument(
        '--element_analysis_dir',
        action='store',
        help='The element analysis working directory. '
             'On the cortex cluster, you would typically set this to go to something like: \n'
             '/share/volume0/<your user name>/data_sets/harry_potter/element_analysis',
        required=True)
    parser.add_argument('--subject', action='store', default='',
                        help='Which subjects to run the analysis on. Defaults to all')
    parser.add_argument('--block', action='store', default='',
                        help='Which blocks to run the analysis on. Defaults to all')
    parser.add_argument('--grid_ms', action='store', default='25',
                        help='Either the string \'time_freq\', or an int giving the number of milliseconds per '
                             'time slice. If set to \'time_freq\', an irregular grid will be computed which bundles'
                             ' the scale-frequencies into groups of 10, and for each bundle computes a time-slice that '
                             'is appropriate for the wavelet footprint at that scale')
    parser.add_argument('--grid_element', action='store', type=int, default=-1,
                        help='Which grid element to run on. Typically this argument would only be used by '
                             'create_pbs_jobs')
    parser.add_argument('--kind', action='store', default='amplitude',
                        help='Determines how the kendall tau is computed. If \'amplitude\', the sum of the absolute '
                             'wavelet coefficients occurring during a slice is used. If \'power\', the square of the '
                             'modulus of the wavelet coefficient is used. If \'count\', the wavelet coefficient is '
                             'ignored, and only the count of events is used. If \'percentile_75\', the 75th percentile '
                             'of the absolute wavelet coefficients is used. If \'median\' the median of the absolute '
                             'wavelet coefficients is used.')
    parser.add_argument('--queue', action='store', default='',
                        help='If specified, the analysis will be queued as jobs in the specified pool. If not '
                             'specified, the analysis will be run directly')

    args = parser.parse_args()
    arg_subjects = args.subject
    if arg_subjects == '':
        arg_subjects = all_subjects
    else:
        arg_subjects = [s.strip() for s in arg_subjects.split(',')]

    arg_blocks = args.block
    if arg_blocks == '':
        arg_blocks = all_blocks
    else:
        arg_blocks = [int(b.strip()) for b in arg_blocks.split(',')]

    arg_job_dir = None if args.job_dir == '' else args.job_dir

    arg_queue = None if args.queue == '' else args.queue

    subject_grids = list()
    for subject_ in arg_subjects:
        subject_grid = None
        for block_ in arg_blocks:
            grid_ = grid_from_grid_ms(args.element_analysis_dir, subject_, block_, args.grid_ms)
            if subject_grid is None:
                subject_grid = grid_
            else:
                if not np.array_equal(subject_grid, grid_):
                    raise ValueError('Incompatible grids across blocks')
        subject_grids.append(subject_grid)

    if arg_queue is not None:
        if arg_job_dir is None:
            parser.error('job_dir must be specified if queue is specified')
        create_pbs_jobs(
            arg_job_dir,
            args.element_analysis_dir,
            arg_queue,
            arg_subjects,
            arg_blocks,
            args.grid_ms,
            [len(g) for g in subject_grids],
            args.kind)
    else:
        for subject_, subject_grid in zip(arg_subjects, subject_grids):
            indices_grid_elements = range(len(subject_grid)) if args.grid_element < 0 else [args.grid_element]
            for block_ in arg_blocks:
                for index_grid_ in indices_grid_elements:
                    kendall_tau_slice(
                        args.element_analysis_dir, subject_, block_, index_grid_, subject_grid[index_grid_], args.kind)
