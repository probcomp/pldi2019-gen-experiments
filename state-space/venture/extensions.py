import time
import os
import random
import string
import numpy as np
import time
import cPickle as pickle
import tempfile

import venture.shortcuts as vs
from venture.ripl.utils import strip_types
import venture.value.dicts as v
from venture.lite import types as t
from venture.lite.sp_help import typed_nr, deterministic_typed

def make_name(sym, index):
    return sym + "_" + str(int(index))

def logsumexp(log_x_arr):
    max_log = np.max(log_x_arr)
    return max_log + np.log(np.sum(np.exp(log_x_arr - max_log)))

def concatenate(arr1, arr2):
    return np.concatenate([arr1, arr2])

def sum_sp(arr):
    return np.sum(arr)

def mean_sp(arr):
    return np.mean(arr)

def stderr(arr):
    return np.std(arr) / np.sqrt(len(arr))

def random_string(num_char):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(num_char))

def cat_string(str1, str2):
    return str1 + str2

def start_timer():
    return time.time()

def time_elapsed(start):
    return time.time() - start

def __venture_start__(ripl):
    start = time.time()
    # NOTE: these are all currently inference SPs
    ripl.bind_foreign_inference_sp("make_symbol", deterministic_typed(make_name, 
        [t.SymbolType(), t.NumberType()], t.SymbolType()))
    ripl.bind_foreign_inference_sp("logsumexp", deterministic_typed(logsumexp, 
        [t.ArrayUnboxedType(t.NumberType())], t.NumberType()))
    ripl.bind_foreign_inference_sp("concatenate", deterministic_typed(concatenate, 
        [t.ArrayUnboxedType(t.NumberType()), t.ArrayUnboxedType(t.NumberType())], t.ArrayUnboxedType(t.NumberType())))
    ripl.bind_foreign_inference_sp("sum", deterministic_typed(sum_sp, 
        [t.ArrayUnboxedType(t.NumberType())], t.NumberType()))
    ripl.bind_foreign_inference_sp("mean", deterministic_typed(mean_sp, 
        [t.ArrayUnboxedType(t.NumberType())], t.NumberType()))
    ripl.bind_foreign_inference_sp("stderr", deterministic_typed(stderr, 
        [t.ArrayUnboxedType(t.NumberType())], t.NumberType()))
    ripl.bind_foreign_inference_sp("random_string", deterministic_typed(random_string, 
        [t.IntegerType()], t.StringType()))
    ripl.bind_foreign_inference_sp("cat_string", deterministic_typed(cat_string, 
        [t.StringType(), t.StringType()], t.StringType()))
    ripl.bind_foreign_inference_sp("start_timer", deterministic_typed(start_timer,
        [], t.NumberType()))
    ripl.bind_foreign_inference_sp("time_elapsed", deterministic_typed(time_elapsed,
        [t.NumberType()], t.NumberType()))
    ripl.execute_program("define new_trace = proc() { run(new_model()) };")
    ripl.execute_program("define run_in_trace = proc(trace, program) { first(run(in_model(trace, program))) };")
    ripl.execute_program("define parallel_mapv = proc(f, l) { run(parallel_mapv_action(f, l, 16)) };")
