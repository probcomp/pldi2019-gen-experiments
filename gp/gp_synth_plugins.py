# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import numpy as np

import venture.lite.types as vt

from venture.lite import gp
from venture.lite.sp_help import deterministic_typed

from gp_synth_compilers import compile_ast_to_embedded_dsl
from gp_synth_interpreter import interpret_embedded_dsl

def load_csv(path):
    return np.loadtxt(path)

def __venture_start__(ripl):
    ripl.execute_program('''
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
        'load_csv',
        deterministic_typed(
            load_csv,
            [vt.StringType()],
            vt.ArrayUnboxedType(vt.NumberType()),
            min_req_args=1
        )
    )
    ripl.bind_foreign_sp(
        'compile_ast_to_venturescript',
        deterministic_typed(
            compile_ast_to_embedded_dsl,
            [vt.AnyType()],
            vt.StringType(),
            min_req_args=1
        )
    )
    ripl.bind_foreign_sp(
        'eval_expr',
        deterministic_typed(
            interpret_embedded_dsl,
            [vt.StringType()],
            gp.gpType,
            min_req_args=1
        )
    )
