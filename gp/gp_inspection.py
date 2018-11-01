#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import json
import os

from collections import OrderedDict

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
    """Product a CSV containing aggregate statistics from experiments."""
    frames = [tabulate_experiment_statistics(p, i) for i, p in enumerate(paths)]
    frame = pd.concat(frames)
    fname = paths[0].split(os.sep)[-1]
    fname = os.path.join(DIR_REPORTS, fname.replace('.json', '.merged.report.csv'))
    frame.to_csv(fname, index=False)
    print fname

if __name__ == '__main__':
    parsable()
