import os
import sys
import argparse

import numpy as np
from numba import njit, prange


@njit(parallel=True)
def _knn(x, num_neighbors, descending=True):
    indicator = np.full(x.shape, False)
    for i in prange(len(x)):
        if descending:
            indices_sort = np.argsort(-x[i])
        else:
            indices_sort = np.argsort(x[i])
        for j in range(num_neighbors):
            indicator[i, indices_sort[j]] = True
    indicator = np.logical_or(indicator, indicator.T)
    return np.where(indicator, x, 0)


def rank_cluster_slice(
        element_analysis_dir,
        subject, hold_out_block,
        index_grid,
        kind,
        knn,
        spectral_radius,
        n_clusters,
        **spectral_kwargs):

    from sklearn.cluster import SpectralClustering

    tau_path = os.path.join(
        element_analysis_dir,
        'kendall_tau',
        'harry_potter_kendall_tau_grid_{}_{}_{}_{}.npz'.format(subject, hold_out_block, index_grid, kind))
    if not os.path.exists(tau_path) and kind == 'amplitude':  # fall back to old format without kind
        tau_path = os.path.join(
            element_analysis_dir,
            'kendall_tau',
            'harry_potter_kendall_tau_grid_{}_{}_{}.npz'.format(subject, hold_out_block, index_grid))

    with np.load(tau_path) as tau_data:
        affinity = tau_data['tau'] + 1  # add 1 so this is all >= 0

    if knn is not None:
        affinity = _knn(affinity, knn)
    if spectral_radius is not None:
        affinity = np.where(affinity >= spectral_radius, affinity, 0)

    spectral_clustering = SpectralClustering(
        n_clusters=n_clusters, affinity='precomputed', **spectral_kwargs)

    source_clusters_ = spectral_clustering.fit_predict(affinity)

    output_path = os.path.join(
        element_analysis_dir,
        'kendall_tau',
        'harry_potter_direct_rank_cluster_{}_{}_{}_{}.npz'.format(subject, hold_out_block, index_grid, kind))

    np.savez(output_path, clusters=source_clusters_)


def create_pbs_jobs(job_directory, element_analysis_dir, pool, subjects, blocks, grid_element_counts,
                    kind, knn, spectral_radius, n_clusters, n_init):
    """
    Creates and queues pbs jobs for the selected subjects and blocks
    Args:
        job_directory: Directory where scripts and logs will be written
        element_analysis_dir: Working directory for element analysis results
        pool: Which pool to run the jobs on
        subjects: Which subjects to run the analysis on. Defaults to all subjects
        blocks: Which blocks to run the analysis on. Defaults to all blocks
        grid_element_counts: How many grid elements there are for each subject
        knn: How many neighbors to keep in the affinity matrix for spectral clustering
        kind: Determines how the kendall tau is computed. If 'amplitude', the sum of the absolute
            wavelet coefficients occurring during a slice is used. If 'power', the square of the
            modulus of the wavelet coefficient is used. If 'count', the wavelet coefficient is
            ignored, and only the count of events is used.
        spectral_radius: Values in the affinity matrix below this value will be set to 0
        n_clusters: The number of clusters per grid-element
        n_init: The number of initializations of the spectral clustering algorithm

    """
    from element_analysis_harry_potter_pbs import create_python_exec_bash, queue_job
    from kendall_tau_grid_pbs import all_subjects, all_blocks

    if subjects is None:
        subjects = all_subjects

    for s in subjects:
        if s not in all_subjects:
            raise ValueError('Unknown subject: {}'.format(s))

    for b in blocks:
        if b not in all_blocks:
            raise ValueError('Unknown block: {}'.format(b))

    for subject, grid_count in zip(subjects, grid_element_counts):
        for block in blocks:
            for index_grid in range(grid_count):
                job_name = 'harry_potter_direct_rank_cluster_{}_{}_{}'.format(subject, block, index_grid)
                bash_path = os.path.join(job_directory, 'kendall_tau', job_name + '.sh')

                arg_str = '--element_analysis_dir {} ' \
                          '--subject {} --block {} --grid_element {} --kind {} ' \
                          '--knn {} --spectral_radius {} --n_clusters {} --n_init {}'.format(
                                element_analysis_dir, subject, block, index_grid, kind,
                                knn, spectral_radius, n_clusters, n_init)

                create_python_exec_bash(
                    os.path.expanduser('~/src/analytic_wavelet_meg/'),
                    'direct_rank_cluster_pbs.py ' + arg_str,
                    bash_path,
                    os.path.join(job_directory, job_name + '.log'))

                queue_job(bash_path, None, pool)


if __name__ == '__main__':
    sys.path.append('/home/drschwar/src/analytic_wavelet')
    sys.path.append('/home/drschwar/src/analytic_wavelet_meg')

    from kendall_tau_grid_pbs import grid_from_grid_ms, all_subjects, all_blocks

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
    parser.add_argument('--knn', action='store', default='',
                        help='How many neighbors to keep in the affinity matrix for spectral clustering. '
                             'Defaults to all')
    parser.add_argument('--spectral_radius', action='store', default='',
                        help='Values in the affinity matrix below this value will be set to 0.')
    parser.add_argument('--n_clusters', action='store', default=60, type=int,
                        help='How many clusters to create')
    parser.add_argument('--n_init', action='store', default=1000, type=int,
                        help='How many initializations to use for spectral clustering')
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

    knn_ = None if args.knn == '' or args.knn.lower() == 'none' else int(args.knn)
    spectral_radius_ = None if args.spectral_radius == '' or args.spectral_radius.lower() == 'none' \
        else float(args.spectral_radius)

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
            [len(g) for g in subject_grids],
            args.kind,
            knn_,
            spectral_radius_,
            args.n_clusters,
            args.n_init)
    else:
        for subject_, subject_grid in zip(arg_subjects, subject_grids):
            indices_grid_elements = range(len(subject_grid)) if args.grid_element < 0 else [args.grid_element]
            for block_ in arg_blocks:
                for index_grid_ in indices_grid_elements:
                    rank_cluster_slice(
                        args.element_analysis_dir, subject_, block_, index_grid_, args.kind, knn_, spectral_radius_,
                        args.n_clusters, n_init=args.n_init)
