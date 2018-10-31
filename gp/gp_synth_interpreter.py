# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

import venture.lite.types as vt
import venture.shortcuts as vs

from venture.lite import covariance
from venture.lite import gp
from venture.lite import sp

from venture.lite.sp_help import deterministic_typed

from venture.parser.venture_script.parse import VentureScriptParser

from change_points import change_point


def interpret_mean_kernel(ast):
    if ast[0]['value'] == 'gp_mean_const':
        c = ast[1]['value']
        return gp.mean_const(c)
    else:
        assert False, ''

def interpret_covariance_kernel(ast):
    if ast[0]['value'] == 'gp_cov_bump':
        min_tolerance = ast[1]['value']
        max_tolerance = ast[2]['value']
        return covariance.bump(min_tolerance, max_tolerance)
    elif ast[0]['value'] == 'gp_cov_scale':
        s2 = ast[1]['value']
        K = interpret_covariance_kernel(ast[2])
        return covariance.scale(s2, K)
    elif ast[0]['value'] == 'gp_cov_wn':
        s2 = ast[1]['value']
        K = covariance.bump(1e-9, 1e-11)
        return covariance.scale(s2, K)
    elif ast[0]['value'] == 'gp_cov_const':
        c = ast[1]['value']
        return covariance.const(c)
    elif ast[0]['value'] == 'gp_cov_linear':
        c = ast[1]['value']
        return covariance.linear(c)
    elif ast[0]['value'] == 'gp_cov_se':
        l2 = ast[1]['value']
        return covariance.se(l2)
    elif ast[0]['value'] == 'gp_cov_periodic':
        l2 = ast[1]['value']
        T = ast[2]['value']
        return covariance.periodic(l2, T)
    elif ast[0]['value'] == 'gp_cov_sum':
        K = interpret_covariance_kernel(ast[1])
        H = interpret_covariance_kernel(ast[2])
        return covariance.sum(K, H)
    elif ast[0]['value'] == 'gp_cov_product':
        K = interpret_covariance_kernel(ast[1])
        H = interpret_covariance_kernel(ast[2])
        return covariance.product(K, H)
    elif ast[0]['value'] == 'gp_cov_cp':
        location = ast[1]['value']
        scale = ast[2]['value']
        K = interpret_covariance_kernel(ast[3])
        H = interpret_covariance_kernel(ast[4])
        return change_point(location, scale, K, H)
    else:
        assert False, 'Failed to interpret AST'

def interpret_embedded_dsl(expr):
    parser = VentureScriptParser()
    ast = parser.parse_instruction(expr)['expression']
    assert len(ast) == 3
    assert ast[0]['value'] == 'make_gp'
    gp_mean = interpret_mean_kernel(ast[1])
    gp_cov = interpret_covariance_kernel(ast[2])
    return sp.VentureSPRecord(gp.GPSP(gp_mean, gp_cov))


if __name__ == '__main__':
    ripl = vs.make_lite_ripl()
    ripl.bind_foreign_sp(
        'interpret_embedded_dsl',
        deterministic_typed(
            interpret_embedded_dsl,
            [vt.StringType()],
            gp.gpType,
            min_req_args=1
        )
    )
    ripl.evaluate("""
        make_gp(gp_mean_const(0.), gp_cov_scale(0.1, gp_cov_bump(.1,.1)))
    """)
    ripl.execute_program("""
        assume gp = interpret_embedded_dsl(
            "make_gp(
                gp_mean_const(0.),
                gp_cov_sum(
                    gp_cov_scale(0.1, gp_cov_bump(.1,.1)),
                    gp_cov_se(0.1)))"
        )
    """)
