import pickle
import time

from collections import OrderedDict

import pandas as pd

# Uncomment this code to recompile the Stan model and save it to disk.
# import pystan
# model = pystan.StanModel(file='./model.stan')
# with open('./model.stan.pkl', 'wb') as f:
#     pickle.dump(model, f)

with open('./model.stan.pkl', 'rb') as f:
    model = pickle.load(f)

prob_outlier = 0.5

train_df = pd.read_csv('../train.csv')
train_xs = train_df['xs'].tolist()
train_ys = train_df['ys'].tolist()

data = {'N' : len(train_xs), 'x' : train_xs, 'y' : train_ys}

num_reps = 20
num_iters_col = []
slope_col = []
intercept_col = []
inlier_log_var_col = []
outlier_log_var_col = []
runtime_col = []
prob_outlier_col = []

steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for num_iters in steps:
    print 'num_iters: %s' % (num_iters,)
    for rep in range(num_reps):
        fname = '/tmp/samples_%s_%s.txt' % (num_iters, rep)
        start = time.time()
        model.sampling(data=data, iter=num_iters, chains=1, sample_file=fname, verbose=False)
        runtime = time.time() - start
        df = pd.read_csv(fname, sep=',', comment='#')
        num_rows = df.shape[0]
        slope_col.append(df['slope'][num_rows-1])
        intercept_col.append(df['intercept'][num_rows-1])
        inlier_log_var_col.append(df['inlier_log_var'][num_rows-1])
        outlier_log_var_col.append(df['outlier_log_var'][num_rows-1])
        runtime_col.append(runtime)
        prob_outlier_col.append(prob_outlier)
        num_iters_col.append(num_iters)

results = pd.DataFrame(OrderedDict([
    ('num_steps'       , num_iters_col),
    ('runtime'         , runtime_col),
    ('score'           , [0]*len(runtime_col)),
    ('slope'           , slope_col),
    ('intercept'       , intercept_col),
    ('inlier_log_var'  , inlier_log_var_col),
    ('outlier_log_var' , outlier_log_var_col),
]))
results.to_csv('./stan.results.csv', index=False, header=True)
