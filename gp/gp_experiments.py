#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import base64
import json
import os
import sys
import time

import numpy as np

from datetime import datetime

import venture.lite.types as vt
import venture.shortcuts as vs

from parsable import parsable

PATH_PLUGINS = [
    'gp_synth_plugins.py',
]

PATH_GP_MODEL = './resources/gp_model_0.vnts'
PATH_RESULTS = '/tmp'

def timestamp():
    """Return current timestamp, up to the second."""
    return datetime.now().strftime('%Y%m%d-%H%M%S')

def float_list_to_str(items):
    """Return string representations of list of float items."""
    items_str = ', '.join('%1.5f' % (i,) for i in items)
    return '[%s]' % (items_str)

def get_results_filename(shortname, num_particles, num_iters, num_epochs,
        num_pred, seed):
    """Return filename to store results of given pipeline invocation."""
    parts = [
        ['stamp',       '%s' % (timestamp(),)],
        ['shortname',   '%s' % (shortname,)],
        ['particles',   '%d' % (num_particles,)],
        ['iters',       '%d' % (num_iters,)],
        ['epochs',      '%d' % (num_epochs,)],
        ['pred',        '%d' % (num_pred,)],
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

def load_serialized_ripl_string(binary, seed):
    """Load and prepare serialized RIPL from base64 string object."""
    ripl = make_new_ripl(seed)
    ripl.loads(base64.b64decode(binary))
    return ripl

def load_serialized_ripl_file(path, seed):
    """Load and prepare a serialized RIPL from a results file."""
    with open(path, 'r') as f:
        results = json.load(f)
    return load_serialized_ripl_string(results['ripl'], seed)

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
    x_min = np.min(xs)
    x_max = np.max(xs)
    y_max = np.max(ys)
    ripl.execute_program('''
    assume x_min = %1.10f;   // minimum of observed input
    assume x_max = %1.10f;   // maximum of observed input
    assume y_max = %1.10f;   // maximum of observed output
    assume get_hyper_prior ~ mem((node_index) -> {
        if (node_index[0] == "WN" or node_index[0] == "C") {
            // Sample hyper-priors ranging over the y axis.
            uniform_continuous(x_min, y_max) #hypers:node_index
        } else {
            // Sample hyper-priors ranging over the x axis.
            uniform_continuous(0, x_max) #hypers:node_index
        }
    });
    ''' % (x_min, x_max, y_max))
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
    logps = ripl.evaluate(
        '_tmp: observe gp(%s) = %s' % (xs_str, ys_str))
    ripl.forget('_tmp')
    return np.transpose(logps).tolist()

def get_particle_predictions(ripl, xs, num_replicates):
    # Format of array is: predictions_raw[replicate][particle][xp]
    xs_str = float_list_to_str(xs)
    print 'Sampling predictions on input:', xs
    predictions_raw = [
        ripl.sample_all('gp(%s)' % (xs_str))
        for _i in xrange(num_replicates)
    ]
    # Format of array is: predictions[particle][replicate][grid]
    predictions = np.swapaxes(predictions_raw, 0, 1)
    return predictions.tolist()

def compute_predictions_rmse(values, predictions):
    assert np.ndim(predictions) == 2
    num_particles, num_probes = np.shape(predictions)
    predictions = np.asarray(predictions)
    assert num_probes == len(values)
    sq_err = (predictions - values)**2
    mean_sq_err = np.mean(sq_err, axis=1)
    rmse = np.sqrt(mean_sq_err)
    assert np.shape(rmse) == (num_particles,)
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
    return ripl.sample_all('ast')

def get_synthesized_programs(ripl):
    return ripl.sample_all('compile_ast_to_venturescript(ast)')

def resample_particles(ripl, count, multiprocess):
    """Create a stochastic ensemble of particles."""
    print 'Resampling particles: %d' % (count,)
    if multiprocess:
        ripl.evaluate('resample_multiprocess(%d)' % (count,))
    else:
        ripl.evaluate('resample(%d)' % (count,))
    ripl.evaluate('reset_to_prior')
    return ripl

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
    log_weight = get_particle_log_weight(ripl)
    log_joint = get_particle_log_joint(ripl)
    log_likelihood = get_particle_log_likelihood(ripl)
    log_prior = compute_particle_log_prior(log_joint, log_likelihood)
    log_predictive = get_particle_log_predictive(ripl, xs_test, ys_test)
    predictions_heldin = get_particle_predictions(ripl, xs_probe, npred_in)
    predictions_heldout = get_particle_predictions(ripl, xs_test, npred_out)
    asts = get_synthesized_asts(ripl)
    programs = get_synthesized_programs(ripl)
    # Derived statistics.
    predictions_heldin_mean = np.mean(predictions_heldin, axis=1).tolist()
    predictions_heldout_mean = np.mean(predictions_heldout, axis=1).tolist()
    rmse_values = compute_predictions_rmse(ys_test, predictions_heldout_mean)
    runtime = time.time() - start
    print 'Finished epoch in seconds: %1.2f' % (runtime,)
    return {
        'log_weight'               : log_weight,
        'log_joint'                : log_joint,
        'log_likelihood'           : log_likelihood,
        'log_prior'                : log_prior,
        'log_predictive'           : log_predictive,
        'predictions_heldin'       : predictions_heldin,
        'predictions_heldout'      : predictions_heldout,
        'asts'                     : asts,
        'programs'                 : programs,
        # Derived statistics.
        'predictions_heldin_mean'  : predictions_heldin_mean,
        'predictions_heldout_mean' : predictions_heldout_mean,
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

def load_dataset_from_path(path_dataset):
    if path_dataset is None:
        # XXX WHAT A NIGHTMARE
        return np.asarray([1.0234]), np.asarray([1.0234])
    dataset = np.loadtxt(path_dataset, delimiter=',')
    xs = dataset[:,0]
    ys = dataset[:,1]
    return (xs, ys)

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
def run_pipeline(path_dataset_train, path_dataset_test=None, shortname=None,
        particles=1, iters=1, epochs=1, nprobe=10, npred_in=1, npred_out=1,
        seed=1, schedule='constant', multiprocess=False):
    """Run synthesis pipeline and collect statistics during inference."""
    # Load datasets from disk.
    xs_train, ys_train = load_dataset_from_path(path_dataset_train)
    xs_test, ys_test = load_dataset_from_path(path_dataset_test)
    xs_probe = np.linspace(min(xs_train)+1e-3, max(xs_train)-1e-3, nprobe)
    # Make iterations according to schedule.
    iterations = make_iteration_schedule(iters, epochs, schedule)
    print iterations
    # Create and prepare new RIPL.
    ripl = make_new_ripl(seed)
    ripl = run_gp_model_hyperpriors(ripl, xs_train, ys_train)
    ripl = run_gp_model_synthesizer(ripl)
    ripl = resample_particles(ripl, particles, multiprocess)
    ripl = observe_training_set(ripl, xs_train, ys_train)
    # Run inference and collect statistics.
    statistics = [
        infer_and_predict(ripl, idx, iterations[idx], xs_test, ys_test,
            xs_probe, npred_in, npred_out,)
        for idx in xrange(epochs)
    ]
    filename = get_results_filename(shortname, particles, iters, epochs,
        npred_out, seed)
    filepath = '%s.json' % (os.path.join(PATH_RESULTS, filename),)
    with open(filepath, 'w') as fptr:
        json.dump({
            'path_dataset_train'    : path_dataset_train,
            'path_dataset_test'     : path_dataset_test,
            'xs_train'              : xs_train.tolist(),
            'ys_train'              : ys_train.tolist(),
            'xs_test'               : xs_test.tolist(),
            'ys_test'               : ys_test.tolist(),
            'xs_probe'              : xs_probe.tolist(),
            'num_particles'         : particles,
            'num_iters'             : iters,
            'num_epochs'            : epochs,
            'num_probe'             : nprobe,
            'num_pred_in'           : npred_in,
            'num_pred_out'          : npred_out,
            'seed'                  : seed,
            'statistics'            : statistics,
            'ripl'                  : base64.b64encode(ripl.saves()),
            'schedule'              : schedule,
        }, fptr)
    print filepath

if __name__ == '__main__':
    parsable()
