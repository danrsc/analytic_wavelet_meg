import os
import argparse
import numpy as np
import mne

from analytic_wavelet_meg import radians_per_ms_from_hertz, PValueFilterFn, maxima_of_transform_mne_source_estimate


def load_harry_potter(subject, block):
    from paradigms import Loader

    data_root = '/share/volume0/newmeg/'
    inv_path = '/share/volume0/newmeg/{experiment}/data/inv/{subject}/' \
               '{subject}_{experiment}_trans-D_nsb-5_cb-0_raw-{structural}-7-0.2-0.8-limitTrue-rankNone-inv.fif'
    struct_dir = '/share/volume0/drschwar/structural'
    session_stimuli_path = '/share/volume0/newmeg/{experiment}/meta/{subject}/sentenceBlock.mat'

    recording_tuple_regex = Loader.make_standard_recording_tuple_regex(
        'trans-D_nsb-5_cb-0_empty-4-10-2-2_band-1-150_notch-60-120_beats-head-meas_blinks-head-meas')
    loader = Loader(session_stimuli_path, data_root, recording_tuple_regex, inv_path, struct_dir)

    structurals = {
        'A': 'struct4',
        'B': 'struct5',
        'C': 'struct6',
        'D': 'krns5D',
        'E': 'struct2',
        'F': 'krns5A',
        'G': 'struct1',
        'H': 'struct3',
        'I': 'krns5C'
    }

    with mne.utils.use_log_level(False):
        inv, labels = loader.load_structural('harryPotter', subject, structurals[subject])
        mne_raw, stimuli, _ = loader.load_block('harryPotter', subject, block)

    epoch_start_times = np.array(list(s['time_stamp'] for s in stimuli))
    duration = 500
    stimuli = list(s.text for s in stimuli)

    return mne_raw, inv, labels, stimuli, epoch_start_times, duration


def make_harry_potter_element_analysis_block(output_path, subject, block, label_name=None):
    from analytic_wavelet import ElementAnalysisMorse, MaximaPValueInterp1d

    # this part is application specific - swap out for your own loading
    mne_raw, inv, labels, stimuli, epoch_start_times, duration = load_harry_potter(subject, block)

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

    (indices_stimuli, indices_source, indices_scale, indices_time), maxima_coefficients, interp_fs = (
        maxima_of_transform_mne_source_estimate(
            ea_morse, fs, mne_raw, inv, np.array(list(s['time_stamp'] for s in stimuli)), source_estimate_label=label,
            filter_fn=p_value_func, lambda2=1))

    output_dir = os.path.split(output_path)[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savez(
        output_path,
        stimuli=np.array(list(s.text for s in stimuli)),
        gamma=ea_morse.gamma,
        analyzing_beta=ea_morse.analyzing_beta,
        element_beta=ea_morse.element_beta,
        scale_frequencies=fs,
        indices_stimuli=indices_stimuli,
        indices_source=indices_source,
        indices_scale=indices_scale,
        indices_time=indices_time,
        maxima_coefficients=maxima_coefficients,
        interpolated_scale_frequencies=interp_fs)


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

    import subprocess

    """
    queues a single job and returns the id for that job
    :param job_filename: The path to the bash script for the job.
    :param resource_limits_string: The resource limits for the job, this string is passed to qsub's -l parameter.
    :param pool_env: The pool environment to use for this job. Used to provide default resource limits.
    :type pool_env: pool_environment.PoolEnv
    :param pool: The pool on which to queue this job.
    :type pool: string
    """
    if resource_limits_string is None:
        ppn = {
            'default': 16,
            'pool2': 8,
            'gpu': 16
        }

        resource_limits_string = 'nodes=1'.format(ppn[pool])

    job_dir = os.path.split(job_filename)[0]

    subprocess.check_output(['chmod', 'g+x', job_filename])
    job = subprocess.check_output([
        'qsub',
        '-l', 'nodes=1:ppn=16',
        '-l', 'walltime=600:00:00',
        '-q', pool, job_filename], cwd=job_dir)
    return job.strip()


def create_qsub_jobs(pool, subjects=None, blocks=None, label=None):

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

    job_directory = '/home/drschwar/element_analysis_jobs'

    if not os.path.exists(job_directory):
        os.makedirs(job_directory)

    output_directory = '/share/volume0/drschwar/data_sets/harry_potter/element_analysis'

    for s, b in product(subjects, blocks):

        if label is not None:
            job_name = 'harry_potter_element_analysis_{}_{}_{}'.format(s, b, label)
            arg_str = '--subject {subject} --block {block} --label {label} --output_path {output}'
        else:
            job_name = 'harry_potter_element_analysis_{}_{}'.format(s, b, label)
            arg_str = '--subject {subject} --block {block} --output_path {output}'

        output_path = os.path.join(output_directory, job_name + '.npz')
        arg_str = arg_str.format(subject=s, block=b, label=label, output=output_path)

        bash_path = os.path.join(job_directory, job_name + '.sh')

        create_python_exec_bash(
            os.path.expanduser('~/src/bert_erp/'),
            'element_analysis_harry_potter.py ' + arg_str,
            bash_path,
            os.path.join(job_directory, job_name + '.log'))

        queue_job(bash_path, 'mem=10gb,walltime=216000', pool)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', action='store', default='')
    parser.add_argument('--block', action='store', default='')
    parser.add_argument('--label', action='store', default='')
    parser.add_argument('--output_path', action='store', default='')
    parser.add_argument('--queue', action='store', default='')

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
    arg_output_path = None if args.output_path == '' else args.output_path

    args_queue = None if args.queue == '' else args.queue

    if args_queue is not None:
        create_qsub_jobs(args_queue, arg_subjects, arg_blocks, arg_label)
    else:

        if arg_output_path is None:
            raise ValueError('Output path is required')

        import sys
        sys.path.append('/home/drschwar/src/analytic_wavelet')
        sys.path.append('/home/drschwar/src/paradigms')

        from itertools import product
        for s, b in product(arg_subjects, arg_blocks):
            make_harry_potter_element_analysis_block(arg_output_path, s, b, arg_label)
