import os
import argparse
import subprocess


def create_python_exec_bash(working_directory, python_string, bash_script_path, log_path):
    """
    Creates a bash script to start Python and call the Python script indicated by python_filename.
    :param working_directory: The working directory for Python.
    :param python_string: The python command to run
    :param bash_script_path: The path at which to write the bash_script.
    :param log_path: The log path for Python.
    """

    with open(bash_script_path, 'w') as output_bash:
        output_bash.write('#!/bin/bash\n')
        output_bash.write('cd {0}\n'.format(working_directory))
        output_bash.write(
            'printf "HOSTNAME: %s\\n\\n" "$HOSTNAME" > {0}\n'.format(log_path))
        output_bash.write('python {0} | tee -a {1}'.format(python_string, log_path))
        output_bash.write('\n')


def queue_job(job_filename, resource_limits_string, pool):
    """
    Queues a single job and returns the job name
    Args:
        job_filename: The path to the bash script for the job
        resource_limits_string: The resource limits for pbs (passed as the -l argument to qsub)
        pool: Which pool to run the job on

    Returns:
        The job name for the job
    """
    if resource_limits_string is None:
        ppn = {
            'default': 16,
            'pool2': 8,
            'gpu': 16
        }

        if pool not in ppn:
            raise ValueError('Unrecognized pool. Please provide your own resource string')

        resource_limits_string = (
            'nodes=1:ppn={}'.format(ppn[pool]),
            'walltime=600:00:00')

    if not isinstance(resource_limits_string, (tuple, list)):
        resource_limits_string = (resource_limits_string,)

    resource_limits = list()
    for r in resource_limits_string:
        resource_limits.append('-l')
        resource_limits.append(r)

    job_dir = os.path.split(job_filename)[0]

    subprocess.check_output(['chmod', 'g+x', job_filename])

    subprocess_args = ['qsub'] + resource_limits + ['-q', pool, job_filename]
    # print(subprocess_args)
    job = subprocess.check_output(subprocess_args, cwd=job_dir)
    return job.strip()


def create_pbs_jobs(job_directory, output_directory, pool, subjects=None, blocks=None, label=None):
    """
    Creates and queues pbs jobs for the selected subjects and blocks
    Args:
        job_directory: Directory where scripts and logs will be written
        output_directory: Directory where results will be written
        pool: Which pool to run the jobs on
        subjects: Which subjects to run the analysis on. Defaults to all subjects
        blocks: Which blocks to run the analysis on. Defaults to all blocks.
        label: If specified, analysis is restricted to this free surfer label
    """

    from itertools import product

    all_subjects = 'A', 'B', 'C', 'D', 'E', 'F', 'H', 'I'
    all_blocks = '1', '2', '3', '4'

    if subjects is None:
        subjects = all_subjects

    if blocks is None:
        blocks = all_blocks

    for s in subjects:
        if s not in all_subjects:
            raise ValueError('Unknown subject: {}'.format(s))

    for b in blocks:
        if b not in all_blocks:
            raise ValueError('Unknown block: {}'.format(b))

    if not os.path.exists(job_directory):
        os.makedirs(job_directory)

    for s, b in product(subjects, blocks):

        if label is not None:
            job_name = 'harry_potter_element_analysis_{}_{}_{}'.format(s, b, label)
            arg_str = '--subject {subject} --block {block} --label {label} --output_dir {output}'
        else:
            job_name = 'harry_potter_element_analysis_{}_{}'.format(s, b, label)
            arg_str = '--subject {subject} --block {block} --output_dir {output}'

        output_path = os.path.join(output_directory, job_name + '.npz')
        arg_str = arg_str.format(subject=s, block=b, label=label, output=output_path)

        bash_path = os.path.join(job_directory, job_name + '.sh')

        create_python_exec_bash(
            os.path.expanduser('~/src/analytic_wavelet_meg/'),
            'element_analysis_harry_potter_pbs.py ' + arg_str,
            bash_path,
            os.path.join(job_directory, job_name + '.log'))

        queue_job(bash_path, None, pool)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job_dir',
        action='store',
        help='A working directory where job script files and logs will be written. '
             'On the cortex cluster, a reasonable place is a subdirectory of your home directory: \n'
             '/home/<your user name>/element_analysis_jobs/',
        default='')
    parser.add_argument(
        '--output_dir',
        action='store',
        help='The directory where results will be written. '
             'On the cortex cluster, you would typically set this to go to something like: \n'
             '/share/volume0/<your user name>/data_sets/harry_potter/element_analysis',
        required=True)
    parser.add_argument('--subject', action='store', default='',
                        help='Which subjects to run the analysis on. Defaults to all')
    parser.add_argument('--block', action='store', default='',
                        help='Which blocks to run the analysis on. Defaults to all')
    parser.add_argument('--label', action='store', default='',
                        help='If specified, analysis is restricted to this FreeSurfer label')
    parser.add_argument('--queue', action='store', default='',
                        help='If specified, the analysis will be queued as jobs in the specified pool. If not '
                             'specified, the analysis will be run directly')

    args = parser.parse_args()
    arg_subjects = args.subject
    if arg_subjects == '':
        arg_subjects = None
    else:
        arg_subjects = [s.strip() for s in arg_subjects.split(',')]

    arg_blocks = args.block
    if arg_blocks == '':
        arg_blocks = None
    else:
        arg_blocks = [b.strip() for b in arg_blocks.split(',')]

    arg_label = None if args.label == '' else args.label
    arg_output_dir = None if args.output_dir == '' else args.output_dir
    arg_job_dir = None if args.job_dir == '' else args.job_dir

    arg_queue = None if args.queue == '' else args.queue

    if arg_queue is not None:
        if arg_job_dir is None:
            parser.error('job_dir must be specified if queue is specified')
        create_pbs_jobs(arg_job_dir, arg_output_dir, arg_queue, arg_subjects, arg_blocks, arg_label)
    else:
        import sys
        sys.path.append('/home/drschwar/src/analytic_wavelet')
        sys.path.append('/home/drschwar/src/paradigms')

        from element_analysis_harry_potter import make_element_analysis_block, load_harry_potter

        from itertools import product
        for s, b in product(arg_subjects, arg_blocks):
            make_element_analysis_block(arg_output_dir, load_harry_potter, s, b, arg_label)
