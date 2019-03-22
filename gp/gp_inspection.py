#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import json
import os

from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from parsable import parsable

# from gp_plotting import plot_ast_depth_histogram
# from gp_plotting import plot_gp_predictive
# from gp_plotting import plot_performance_evolution
# from gp_plotting import plot_scatter_ast_depth_likelihood

# from gp_simplification import simplify_ast_binary
# from gp_simplification import simplify_ast_nary
# from gp_synth_plugins import count_ast_depth
# from gp_synth_plugins import count_base_kernels
# from gp_synth_plugins import count_operators

DEVNULL = open(os.devnull, 'w')

DIR_PLOTS = './resources/plots'
DIR_REPORTS = './resources/reports'

@parsable
def tabulate_experiment_statistics(path, particle=0):
    """Produce a CSV file containing statistics from experiment at path."""
    with open(path, 'r') as f:
        results = json.load(f)
    epochs = range(results['n_epochs'])
    statistics = results['statistics']
    unravel = lambda s, k: [s[e][k] for e in epochs]
    records = OrderedDict([
        ('particle'               , particle),
        ('epoch'                  , epochs),
        ('iters'                  , unravel(statistics, 'iters')),
        ('runtime'                , unravel(statistics, 'runtime')),
        ('rmse'                   , unravel(statistics, 'rmse')),
        ('log_likelihood'         , unravel(statistics, 'log_likelihood')),
        ('log_joint'              , unravel(statistics, 'log_joint')),
        ('log_predictive'         , unravel(statistics, 'log_predictive')),
    ])
    frame = pd.DataFrame(records)
    fname = path.split(os.sep)[-1]
    fname = os.path.join(DIR_REPORTS, fname.replace('.json', '.report.csv'))
    frame.to_csv(fname, index=False)
    print fname
    return frame

@parsable
def tabulate_experiments_statistics(*paths):
    """Produce a CSV containing aggregate statistics from experiments."""
    frames = [tabulate_experiment_statistics(p, i) for i, p in enumerate(paths)]
    frame = pd.concat(frames)
    fname = paths[0].split(os.sep)[-1]
    fname = os.path.join(DIR_REPORTS, fname.replace('.json', '.merged.report.csv'))
    frame.to_csv(fname, index=False)
    print fname

@parsable
def plot_predictions(path, epoch=-1):
    """Plot the observed data and predictions from a single run."""
    with open(path, 'r') as f:
        results = json.load(f)
    # Extract the dataset.
    xs_probe = results['xs_probe']
    xs_train = results['xs_train']
    ys_train = results['ys_train']
    xs_test = results['xs_test']
    ys_test = results['ys_test']
    # Extract the predictions.
    statistics = results['statistics']
    predictions_held_in = statistics[epoch]['predictions_held_in']
    predictions_held_out = statistics[epoch]['predictions_held_out']
    # Plot.
    fig, ax = plt.subplots()
    ax.scatter(xs_train, ys_train, marker='x', color='k', label='Observed Data')
    ax.scatter(xs_test, ys_test, marker='x', color='r', label='Test Data')
    # for ys in predictions_held_in:
    #     ax.plot(xs_probe, ys, color='g', alpha=0.2)
    # ax.plot(xs_probe, np.mean(ys, axis=0), color='g')
    # for ys in predictions_held_out:
    #     ax.plot(xs_test, ys, color='g', alpha=0.2)
    ax.plot(xs_probe, np.mean(predictions_held_in, axis=0), color='g')
    ax.plot(xs_test, np.mean(predictions_held_out, axis=0), color='g')
    fig.set_tight_layout(True)
    fname = path.split(os.sep)[-1]
    fname = os.path.join(DIR_PLOTS, fname.replace('.json', '.predictions.png'))
    fig.savefig(fname)
    print fname

@parsable
def plot_metric_evolution(path, metric):
    """Plot evolution of metric over time."""
    runtimes, metrics = extract_metric_evolution(path, 'runtime', metric)
    runtimes = runtimes[1:]
    metrics = metrics[1:]
    x_runtime = np.cumsum(np.median(runtimes, axis=1))
    y_metric_median = np.median(metrics, axis=1)
    y_metric_high = np.percentile(metrics, 75, axis=1)
    y_metric_low = np.percentile(metrics, 25, axis=1)
    fig, ax = plt.subplots()
    ax.errorbar(x_runtime, y_metric_median,
        yerr=[y_metric_median-y_metric_low, y_metric_high-y_metric_median],
        linewidth=.5, fmt='--.', label=metric, color='blue',
        )
    ax.set_ylim([0.10, 0.40])
    ax.grid()
    # Save to disk.
    fname = path.split(os.sep)[-1]
    fname = os.path.join(DIR_PLOTS,
        fname.replace('.csv', '.evolution.metric@%s.png' % (metric,)))
    ax.set_xlabel('Runtime')
    ax.set_ylabel(metric)
    fig.savefig(fname)
    print fname

def extract_metric_evolution(path, x_key, y_key):
    """Extract the series of values from file."""
    df = pd.read_csv(path)
    epochs = sorted(df['epoch'].unique())
    particles = sorted(df['particle'].unique())
    xs = [df[df['epoch']==epoch][x_key].values for epoch in epochs]
    ys = [df[df['epoch']==epoch][y_key].values for epoch in epochs]
    assert np.shape(xs) == (len(epochs), len(particles))
    assert np.shape(ys) == (len(epochs), len(particles))
    return np.asarray(xs), np.asarray(ys)

if __name__ == '__main__':
    parsable()
