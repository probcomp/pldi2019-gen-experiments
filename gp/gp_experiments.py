#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import json
import os
import sys
import time

from datetime import datetime

import numpy as np

import venture.shortcuts as vs

from parsable import parsable

PATH_PLUGINS = [
    'gp_synth_plugins.py',
]

PATH_GP_MODEL = './resources/gp_model_0.vnts'
PATH_RESULTS = './resources/results'

def timestamp():
    """Return current timestamp, up to the second."""
    return datetime.now().strftime('%Y%m%d-%H%M%S')

def float_list_to_str(items):
    """Return string representations of list of float items."""
    items_str = ', '.join('%1.5f' % (i,) for i in items)
    return '[%s]' % (items_str)

def rescale_linear(xs, yl, yh):
    """Rescale values linearly between [yl, yh]."""
    xl = min(xs)
    xh = max(xs)
    slope = float(yh - yl) / (xh - xl)
    intercept = yh - xh * slope
    return slope* xs + intercept

def get_results_filename(shortname, n_test, n_iters, n_epochs, schedule, seed):
    """Return filename to store results of given pipeline invocation."""
    parts = [
        ['stamp',       '%s' % (timestamp(),)],
        ['shortname',   '%s' % (shortname,)],
        ['ntest',       '%d' % (n_test,)],
        ['iters',       '%d' % (n_iters,)],
        ['epochs',      '%d' % (n_epochs,)],
        ['schedule',    '%s' % (schedule,)],
        ['seed',        '%d' % (seed,)],
    ]
    return '_'.join('@'.join(part) for part in parts)

def compute_particle_log_prior(log_joint, log_likelihood):
    """Compute the log_prior from the log_joint and log_likelihood values."""
    return np.subtract(log_joint, log_likelihood).tolist()

def compute_prediction_mae(predictions, actual):
    """Compute the mean absolute prediction errors against actual."""
    predictions_errors_abs = np.abs(np.asarray(predictions) - actual)
    predictions_errors_abs_mean = np.mean(predictions_errors_abs, axis=1)
    return predictions_errors_abs_mean.tolist()

def load_plugins(ripl, paths=None):
    """Load plugins into the given RIPL."""
    plugins = paths or PATH_PLUGINS
    for plugin in plugins:
        ripl.load_plugin(plugin)
    return ripl

def make_new_ripl(seed):
    """Make a new RIPL with given seed."""
    ripl = vs.make_lite_ripl(seed=seed)
    ripl = load_plugins(ripl)
    return ripl

def _preprocess_dataset(dataset):
    """Subtract min from x, and mean center on y."""
    assert np.ndim(dataset) == 2
    dataset = np.asarray(dataset)
    xs = dataset[:,0]
    ys = dataset[:,1]
    x_prime = xs - np.min(xs)
    y_prime = ys - np.mean(ys)
    return np.column_stack((x_prime, y_prime))

def _partition_dataset(dataset, mode, num_test, seed):
    """Partition dataset into train/test split according to mode."""
    dataset = np.asarray(dataset)
    N = len(dataset)
    idx = np.arange(N)
    assert num_test < N
    rng = np.random.RandomState(seed)
    if mode == 'extrapolate':
        drop_idx = idx[N-num_test:]
    elif mode == 'interpolate_r':
        drop_idx = rng.choice(idx, size=num_test)
    elif mode == 'interpolate_c':
        start = rng.randint(0, N-num_test)
        drop_idx = idx[start:start+num_test]
    else:
        assert False, 'Unknown partition mode: %s' % (mode,)
    keep_idx = idx[~np.isin(idx, drop_idx)]
    return dataset[keep_idx], dataset[drop_idx]

def run_gp_model_hyperpriors(ripl, xs, ys):
    """Execute program setting data-dependent hyperpriors."""
    x_max = np.max(xs)
    y_max = np.max(ys)
    ripl.execute_program('''
    assume x_max = %1.10f;   // maximum of observed input
    assume y_max = %1.10f;   // maximum of observed output
    assume get_hyper_prior ~ mem((node_index) -> {
        uniform_continuous(0, 1) #hypers:node_index
        // if (node_index[0] == "WN" or node_index[0] == "C") {
        //     uniform_continuous(0, y_max) #hypers:node_index
        // } else {
        //     uniform_continuous(0, x_max) #hypers:node_index
        // }
    });
    ''' % (x_max, y_max))
    return ripl

def run_gp_model_synthesizer(ripl):
    """Execute the program defining synthesis model."""
    ripl.execute_program_from_file(PATH_GP_MODEL)
    return ripl

def get_particle_log_weight(ripl):
    """Return list of particle log weight."""
    return ripl.evaluate('particle_log_weights()')

def get_particle_log_joint(ripl):
    """Return list of log joint of each particle."""
    return ripl.evaluate('global_log_joint')

def get_particle_log_likelihood(ripl):
    """Return list of log likelihood of each particle."""
    return ripl.evaluate('global_log_likelihood')

def get_particle_log_prior(ripl):
    """Return list of log prior, likelihood, and joint of each particle."""
    log_joint = get_particle_log_joint(ripl)
    log_likelihood = get_particle_log_likelihood(ripl)
    return compute_particle_log_prior(log_joint, log_likelihood)

def get_particle_log_predictive(ripl, xs, ys):
    """Return list of log predictive on new data, for each particle."""
    xs_str = float_list_to_str(xs)
    ys_str = float_list_to_str(ys)
    print 'Computing log predictive on inputs with outputs:', xs_str, ys_str
    logps = ripl.evaluate('_tmp: observe gp(%s) = %s' % (xs_str, ys_str))
    ripl.forget('_tmp')
    return list(logps)

def get_particle_predictions(ripl, xs, num_replicates):
    # Format of array is: predictions_raw[replicate][xp]
    xs_str = float_list_to_str(xs)
    print 'Sampling predictions on input:', xs
    pred = [ripl.sample('gp(%s)' % (xs_str)) for _i in xrange(num_replicates)]
    return np.asarray(pred).tolist()

def compute_predictions_rmse(values, predictions):
    assert len(predictions) == len(values)
    predictions = np.asarray(predictions)
    sq_err = (predictions - values)**2
    rmse = np.sqrt(np.mean(sq_err))
    print 'Computed RMSE: ', rmse
    return rmse.tolist()

def observe_training_set(ripl, xs, ys):
    """Incorporate training set into the GP."""
    xs_str = float_list_to_str(xs)
    ys_str = float_list_to_str(ys)
    ripl.execute_program('observe gp(%s) = %s' % (xs_str, ys_str))
    for x, y in zip(xs, ys):
        print 'Observing training data point:', x, y
        # Do not uncomment this line, it causes the size of the trace to
        # explode and very slow inference!
        # ripl.observe('gp(%1.40f)' % x, y)
    return ripl

def get_synthesized_asts(ripl):
    return ripl.sample('ast')

def get_synthesized_programs(ripl):
    return ripl.sample('compile_ast_to_venturescript(ast)')

def run_mh_inference(ripl, steps):
    print 'Running MH for iterations: %d' % (steps,)
    start = time.time()
    ripl.infer('resimulation_mh(default, one, %d)' % (steps,))
    print 'Completed MH in %1.4f' % (time.time() - start)
    return ripl

def infer_and_predict(ripl, idx, iters, xs_test, ys_test, xs_probe,
        npred_in, npred_out):
    """Run MH inference and collect measurements and statistics."""
    print 'Starting epoch: %d' % (idx,)
    start = time.time()
    ripl = run_mh_inference(ripl, iters)
    runtime = time.time() - start
    log_weight = get_particle_log_weight(ripl)
    log_joint = get_particle_log_joint(ripl)
    log_likelihood = get_particle_log_likelihood(ripl)
    log_prior = compute_particle_log_prior(log_joint, log_likelihood)
    log_predictive = get_particle_log_predictive(ripl, xs_test, ys_test)
    predictions_held_in = get_particle_predictions(ripl, xs_probe, npred_in)
    predictions_held_out = get_particle_predictions(ripl, xs_test, npred_out)
    asts = get_synthesized_asts(ripl)
    programs = get_synthesized_programs(ripl)
    # Derived statistics.
    predictions_held_in_mean = np.mean(predictions_held_in, axis=0).tolist()
    predictions_held_out_mean = np.mean(predictions_held_out, axis=0).tolist()
    rmse_values = compute_predictions_rmse(ys_test, predictions_held_out_mean)
    print 'Finished epoch in seconds: %1.2f' % (runtime,)
    return {
        'iters'                    : iters,
        'log_weight'               : log_weight,
        'log_joint'                : log_joint,
        'log_likelihood'           : log_likelihood,
        'log_prior'                : log_prior,
        'log_predictive'           : log_predictive,
        'predictions_held_in'      : predictions_held_in,
        'predictions_held_out'     : predictions_held_out,
        'asts'                     : asts,
        'programs'                 : programs,
        # Derived statistics.
        'predictions_heldin_mean'  : predictions_held_in_mean,
        'predictions_heldout_mean' : predictions_held_out_mean,
        'rmse_values'              : rmse_values,
        'runtime'                  : runtime,
    }


# Command line interface.

@parsable
def generate_random_programs(count=1, seed=1):
    """Generate random noisy expressions from the AST prior."""
    ripl = make_new_ripl(seed)
    ripl = run_gp_model_hyperpriors(ripl, [0, 10], [0, 10])
    ripl = run_gp_model_synthesizer(ripl)
    def generate_random_program():
        return ripl.sample('''
            compile_ast_to_venturescript(
                generate_random_ast(normal(0,1)))
        ''')
    programs = [generate_random_program() for _i in xrange(count)]
    for program in programs:
        print >> sys.stdout, program

@parsable
def preprocess_dataset(fname):
    """Rescale dataset on x/y axis for GP learning."""
    dataset = np.loadtxt(fname, delimiter=',')
    dataset_processed = _preprocess_dataset(dataset)
    fname_processed = fname.replace('.csv', '.processed.csv')
    np.savetxt(fname_processed, dataset_processed, delimiter=',')
    print fname_processed

@parsable
def thin_dataset(fname, skip=2):
    """Select every `skip` observation from the given timeseries."""
    dataset = np.loadtxt(fname, delimiter=',')
    idxs = range(0, len(dataset), skip)
    dataset_thinned = dataset[idxs]
    fname_thinned = fname.replace('.csv', '.thinned@%d.csv' % (skip,))
    np.savetxt(fname_thinned, dataset_thinned, delimiter=',')
    print fname_thinned

@parsable
def partition_dataset(fname, mode, num_test=1, seed=1):
    """Partition the dataset into train/test splits."""
    dataset = np.loadtxt(fname, delimiter=',')
    dataset_train, dataset_test = \
        _partition_dataset(dataset, mode, num_test, seed)
    fname_train = fname.replace('.csv',
        '_mode@%s_seed@%d_train.csv' % (mode, seed,))
    fname_test = fname.replace('.csv',
        '_mode@%s_seed@%d_test.csv' % (mode, seed,))
    np.savetxt(fname_train, dataset_train, delimiter=',')
    np.savetxt(fname_test, dataset_test, delimiter=',')
    print fname_train, fname_test

def load_dataset_from_path(path_dataset, n_test):
    dataset = np.loadtxt(path_dataset, delimiter=',')
    dataset[:,0] = rescale_linear(dataset[:,0], 0, 1)
    dataset[:,1] = rescale_linear(dataset[:,1], -1, 1)
    xs_train = dataset[:-n_test, 0]
    ys_train = dataset[:-n_test, 1]
    xs_test = dataset[-n_test:, 0]
    ys_test = dataset[-n_test:, 1]
    return (xs_train, ys_train), (xs_test, ys_test)

def make_iteration_schedule(iters, epochs, schedule):
    if schedule == 'constant':
        return [iters*1 for i in xrange(epochs)]
    elif schedule == 'linear':
        return [iters*i for i in xrange(1, epochs+1)]
    elif schedule == 'doubling':
        return [iters*2**i for i in xrange(epochs)]
    else:
        assert False, 'Unknown schedule: %s' % (schedule,)

@parsable
def run_pipeline(
        path_dataset,
        n_test=1,
        shortname=None,
        iters=1,
        epochs=1,
        nprobe_held_in=10,
        npred_held_in=1,
        npred_held_out=1,
        schedule='constant',
        seed=-1,
    ):
    """Run synthesis pipeline and collect statistics during inference."""
    seed = np.random.randint(2**32-1) if seed < 0 else seed
    # Load and prepare datasets.
    dataset_train, dataset_test = load_dataset_from_path(path_dataset, n_test)
    xs_train, ys_train = dataset_train
    xs_test, ys_test = dataset_test
    # Make the probe points for interpolated data.
    xs_probe = np.linspace(min(xs_train)+1e-3, max(xs_train)-1e-3, nprobe_held_in)
    # Make iterations according to schedule.
    iterations = make_iteration_schedule(iters, epochs, schedule)
    print iterations
    # Create and prepare new RIPL.
    ripl = make_new_ripl(seed)
    ripl = run_gp_model_hyperpriors(ripl, xs_train, ys_train)
    ripl = run_gp_model_synthesizer(ripl)
    ripl = observe_training_set(ripl, xs_train, ys_train)
    # Run inference and collect statistics.
    statistics = [
        infer_and_predict(ripl, idx, iterations[idx], xs_test, ys_test,
            xs_probe, npred_held_in, npred_held_out,)
        for idx in xrange(epochs)
    ]
    filename = get_results_filename(shortname, n_test, iters, epochs, schedule, seed)
    filepath = '%s.json' % (os.path.join(PATH_RESULTS, filename),)
    with open(filepath, 'w') as fptr:
        json.dump({
            'path_dataset'          : path_dataset,
            'n_test'                : n_test,
            'xs_train'              : xs_train.tolist(),
            'ys_train'              : ys_train.tolist(),
            'xs_test'               : xs_test.tolist(),
            'ys_test'               : ys_test.tolist(),
            'xs_probe'              : xs_probe.tolist(),
            'num_iters'             : iters,
            'num_epochs'            : epochs,
            'nprobe_held_in'        : nprobe_held_in,
            'npred_held_in'         : npred_held_in,
            'npred_held_out'        : npred_held_out,
            'seed'                  : seed,
            'statistics'            : statistics,
            'schedule'              : schedule,
        }, fptr)
    print filepath

if __name__ == '__main__':
    parsable()
