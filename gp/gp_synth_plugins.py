# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

import venture.lite.types as vt

from venture.lite import gp
from venture.lite.gp import GPCovarianceType
from venture.lite.gp import _cov_sp
from venture.lite.sp_help import deterministic_typed


from change_points import change_point
from gp_synth_compilers import compile_ast_to_embedded_dsl
from gp_synth_interpreter import interpret_embedded_dsl


def load_csv(path):
    return np.loadtxt(path)

def concatenate(arraylike1, arraylike2):
    if not isinstance(arraylike1, list):
        arraylike1 = arraylike1.tolist()
    if not isinstance(arraylike2, list):
        arraylike2 = arraylike2.tolist()
    return arraylike1 + arraylike2

def compute_rmse(iterations, actual, predictions_list):
    # Handle the nightmarish 4-dimensional array to index as follows:
    #   predictions[chain][iteration][sample][timepoint]
    predictions = np.asarray(predictions_list)
    (num_iterations, num_samples, num_particles, num_timepoints) \
        = np.shape(predictions)
    assert len(iterations) == num_iterations
    predictions = np.swapaxes(predictions, 0, 2)
    predictions = np.swapaxes(predictions, 1, 2)
    assert np.shape(predictions) \
        == (num_particles, num_iterations, num_samples, num_timepoints)
    # Compute the RMSE matrix.
    mean_predictions = np.mean(predictions, axis=2)
    assert np.shape(mean_predictions) \
        == (num_particles, num_iterations, num_timepoints)
    sqerr_predictions = (mean_predictions - actual)**2
    mean_sqerr_prediction = np.mean(sqerr_predictions, axis=2)
    rmse_prediction = np.sqrt(mean_sqerr_prediction)
    assert np.shape(rmse_prediction) == (num_particles, num_iterations)
    return rmse_prediction

def count_operators(ast, operator):
    if ast[0] in ['WN', 'C', 'LIN', 'SE', 'PER']:
        return 0
    elif ast[0] in ['+', '*']:
        count_current = int(ast[0] == operator)
        count_left = count_operators(ast[1], operator)
        count_right = count_operators(ast[2], operator)
        return count_current + count_left + count_right
    elif ast[0][0] in 'CP':
        count_current = int(ast[0][0] == operator)
        count_left = count_operators(ast[1], operator)
        count_right = count_operators(ast[2], operator)
        return count_current + count_left + count_right
    else:
        assert False, 'Unknown AST: %s' % (ast,)

def count_base_kernels(ast, kernel):
    if ast[0] in ['WN', 'C', 'LIN', 'SE', 'PER']:
        return int(ast[0] == kernel or kernel is None)
    elif ast[0] in ['+', '*']:
        count_left = count_base_kernels(ast[1], kernel)
        count_right = count_base_kernels(ast[2], kernel)
        return count_left + count_right
    elif ast[0][0] in ['CP']:
        count_left = count_base_kernels(ast[1], kernel)
        count_right = count_base_kernels(ast[2], kernel)
        return count_left + count_right
    else:
        assert False, 'Unknown AST: %s' % (ast,)

def count_ast_depth(ast):
    if ast[0] in ['WN', 'C', 'LIN', 'SE', 'PER']:
        return 1
    elif ast[0] in ['+', '*']:
        count_left = count_ast_depth(ast[1])
        count_right = count_ast_depth(ast[2])
        return 1 + max([count_left, count_right])
    elif ast[0][0] in ['CP']:
        count_left = count_ast_depth(ast[1])
        count_right = count_ast_depth(ast[2])
        return 1 + max([count_left, count_right])
    else:
        assert False, 'Unknown AST: %s' % (ast,)

def __venture_start__(ripl):
    ripl.execute_program('''
        define set_value_at_scope_block = (scope, block, value) -> {
            set_value_at2(scope, block, value)
        };
        assume gp_cov_wn = (c) -> {
            gp_cov_scale(c, gp_cov_bump(1e-9, 1e-11))
        };
        define gp_cov_wn = (c) -> {
            gp_cov_scale(c, gp_cov_bump(1e-9, 1e-11))
        };
    ''')
    ripl.bind_foreign_inference_sp(
        'sort',
        deterministic_typed(
            np.sort,
            [
                vt.ArrayUnboxedType(vt.NumberType()),
            ],
            vt.ArrayUnboxedType(vt.NumberType()),
            min_req_args=1
        )
    )
    ripl.bind_foreign_inference_sp(
        'get_mean',
        deterministic_typed(
            np.mean,
            [
                vt.ArrayUnboxedType(vt.NumberType()),
            ],
            vt.NumberType(),
            min_req_args=1
        )
    )
    ripl.bind_foreign_inference_sp(
        'get_predictive_mean',
        deterministic_typed(
            lambda x: np.mean(x, axis=0),
            [
                vt.ArrayUnboxedType(vt.ArrayUnboxedType(vt.NumberType())),
            ],
            vt.ArrayUnboxedType(vt.NumberType()),
            min_req_args=1
        )
    )
    ripl.bind_foreign_inference_sp(
        'compute_rmse',
        deterministic_typed(
            compute_rmse,
            [
                vt.ArrayUnboxedType(vt.NumberType()),
                vt.ArrayUnboxedType(vt.NumberType()),
                vt.ArrayUnboxedType(
                    vt.ArrayUnboxedType(
                        vt.ArrayUnboxedType(
                            vt.ArrayUnboxedType(vt.NumberType())))),
            ],
            vt.ArrayUnboxedType(vt.ArrayUnboxedType(vt.NumberType())),
            min_req_args=1
        )
    )
    ripl.bind_foreign_inference_sp(
        'load_csv',
        deterministic_typed(
            load_csv,
            [vt.StringType()],
            vt.ArrayUnboxedType(vt.NumberType()),
            min_req_args=1
        )
    )
    ripl.bind_foreign_inference_sp(
        'concatenate',
        deterministic_typed(
            concatenate,
            [
                vt.ArrayUnboxedType(vt.NumberType()),
                vt.ArrayUnboxedType(vt.NumberType()),
            ],
            vt.ArrayUnboxedType(vt.NumberType()),
            min_req_args=2
        )
    )
    ripl.bind_foreign_sp(
        'gp_cov_cp',
        _cov_sp(
            change_point,
            [
                vt.NumberType(),
                vt.NumberType(),
                GPCovarianceType('K'),
                GPCovarianceType('H')
            ]
        )
    )
    for sp_name in [
        'compile_ast_to_embedded_dsl',
        'compile_ast_to_venturescript'
        ]:
        ripl.bind_foreign_sp(
            sp_name,
            deterministic_typed(
                compile_ast_to_embedded_dsl,
                [vt.AnyType()],
                vt.StringType(),
                min_req_args=1
            )
        )
    for sp_name in ['vexec', 'eval_expr']:
        ripl.bind_foreign_sp(
            sp_name,
            deterministic_typed(
                interpret_embedded_dsl,
                [vt.StringType()],
                gp.gpType,
                min_req_args=1
            )
        )
