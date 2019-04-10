#!/usr/bin/env python

from math import exp
from math import log
from math import log1p
from math import pi
from math import sqrt

import pandas as pd

from parsable import parsable

PROB_OUTLIER = 0.5
TRUE_INLIER_NOISE = 0.5
TRUE_OUTLIER_NOISE = 5.0
TRUE_SLOPE = -1
TRUE_INTERCEPT = 2

HALF_LOG2PI = 0.5 * log(2 * pi)

def log_density_normal(x, mu, sigma):
    deviation = x - mu
    return - log(sigma) - HALF_LOG2PI \
        - (0.5 * deviation * deviation / (sigma * sigma))

def log_density_two_normal(x, mu, prob1, sigma1, sigma2):
    l1 = log(prob1) + log_density_normal(x, mu, sigma1)
    l2 = log1p(-prob1) + log_density_normal(x, mu, sigma2)
    m = max(l1, l2)
    return m + log(exp(l1 - m) + exp(l2 - m))

def assess_row(xs, ys, row):
    sigma1 = sqrt(exp(row['inlier_log_var']))
    sigma2 = sqrt(exp(row['outlier_log_var']))
    logps = [log_density_two_normal(y, row['slope']*x + row['intercept'],
        PROB_OUTLIER, sigma1, sigma2) for (x, y) in zip(xs, ys)]
    return sum(logps)

@parsable
def report_predictive_likelihood(path_test, path_results):
    """Report the predictive likelihood; over time."""
    df_test = pd.read_csv(path_test, header=0)
    df_stats = pd.read_csv(path_results, header=0)
    df_stats['predictive_likelihood'] = [
        assess_row(df_test['xs'], df_test['ys'], row)
        for (_i, row) in df_stats.iterrows()
    ]
    fname = path_results.replace('.csv', '.logps.csv')
    df_stats.to_csv(fname, index=None)
    print fname

if __name__ == '__main__':
    parsable()
