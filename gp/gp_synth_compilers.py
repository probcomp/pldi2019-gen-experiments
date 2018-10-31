# -*- coding: utf-8 -*-

# Copyright (c) 2018 MIT Probabilistic Computing Project.
# Released under Apache 2.0; refer to LICENSE.txt.

from cStringIO import StringIO

from iventure.magics import convert_from_venture_value

def compile_indent(stream, i):
    indentation = ' ' * i
    stream.write(indentation)

def compile_covariance_kernel(stream, i, ast):
    compile_indent(stream, i)
    if ast[0] == 'WN':
        # stream.write('gp_cov_scale(%1.4f, gp_cov_bump(1e-9, 1e-11))' % (ast[1],))
        stream.write('gp_cov_wn(%1.4f)' % (ast[1],))
    elif ast[0] == 'C':
        stream.write('gp_cov_const(%1.4f)' % (ast[1],))
    elif ast[0] == 'LIN':
        stream.write('gp_cov_linear(%1.4f)' % (ast[1],))
    elif ast[0] == 'SE':
        stream.write('gp_cov_se(%1.4f)' % (ast[1]**2,))
    elif ast[0] == 'PER':
        stream.write('gp_cov_periodic(%1.4f, %1.4f)' % (ast[1]**2, ast[2]))
    elif ast[0] == '+':
        stream.write('gp_cov_sum(\n')
        compile_covariance_kernel(stream, i+2, ast[1])
        stream.write(',\n')
        compile_covariance_kernel(stream, i+2, ast[2])
        stream.write(')')
    elif ast[0] == '*':
        stream.write('gp_cov_product(\n')
        compile_covariance_kernel(stream, i+2, ast[1])
        stream.write(',\n')
        compile_covariance_kernel(stream, i+2, ast[2])
        stream.write(')')
    elif ast[0][0] == 'CP':
        stream.write('gp_cov_cp(')
        # compile_indent(stream, i+2)
        stream.write('%1.4f, ' % (ast[0][1]))
        # compile_indent(stream, i+2)
        stream.write('%1.4f,\n' % (ast[0][2]))
        compile_covariance_kernel(stream, i+2, ast[1])
        stream.write(',\n')
        compile_covariance_kernel(stream, i+2, ast[2])
        stream.write(')')
    else:
        assert False, 'Unknown AST'

def compile_ast_to_embedded_dsl_assumes(ast, stream):
    stream.write('assume gp_mean = gp_mean_const(0.);\n')
    stream.write('assume gp_covariance_kernel =\n')
    compile_covariance_kernel(stream, 2, ast)
    stream.write(';\n')
    stream.write('assume gp = make_gp(gp_mean, gp_covariance_kernel);')

def compile_ast_to_embedded_dsl_expression(ast, stream=None):
    stream = stream or StringIO()
    stream.write('make_gp(\n')
    compile_indent(stream, 2)
    stream.write('gp_mean_const(0.),\n')
    compile_covariance_kernel(stream, 2, ast)
    stream.write(')')
    return stream

def compile_ast_to_embedded_dsl(ast):
    stream = StringIO()
    ast_python = convert_from_venture_value(ast)
    compile_ast_to_embedded_dsl_expression(ast_python, stream)
    return stream.getvalue()

# New compiler for functional interface.

def compile_covariance_kernel_proc(stream, i, ast):
    compile_indent(stream, i)
    if ast[0] == 'C':
        stream.write('((x1,x2) -> {%1.4f})' % (ast[1],))
    elif ast[0] == 'WN':
        stream.write('((x1,x2) -> if (x1==x2) {%1.4f} else {0})' % (ast[1],))
    elif ast[0] == 'LIN':
        stream.write('((x1,x2) -> {(x1-%1.4f)*(x2-%1.4f)})'
            % (ast[1], ast[1]))
    elif ast[0] == 'SE':
        stream.write('((x1,x2) -> {exp((x1-x2)**2/%1.4f)})' % (ast[1]**2,))
    elif ast[0] == 'PER':
        stream.write('((x1,x2) -> {-2/%1.4f*sin(2*pi/%1.4f*abs(x1-x2))**2})' % (ast[1]**2, ast[2]))
    elif ast[0] == '+':
        stream.write('((x1,x2) -> {')
        compile_covariance_kernel_proc(stream, i, ast[1])
        stream.write('(x1,x2) + ')
        compile_covariance_kernel_proc(stream, 0, ast[2])
        stream.write('(x1,x2)})')
    elif ast[0] == '*':
        stream.write('((x1,x2) -> {')
        compile_covariance_kernel_proc(stream, i, ast[1])
        stream.write('(x1,x2) * ')
        compile_covariance_kernel_proc(stream, 0, ast[2])
        stream.write('(x1,x2)})')
    elif ast[0][0] == 'CP':
        stream.write('((x1, x2) -> {')
        stream.write('sig1 = sigmoid(x1, %f, 0.1) * sigmoid(x2, %f, .1);'
            % (ast[0][1], ast[0][1]))
        stream.write('sig2 = (1-sigmoid(x1, %f, 0.1)) * (1-sigmoid(x2, %f, .1));'
            % (ast[0][1], ast[0][1]))
        stream.write('sig1 *')
        compile_covariance_kernel_proc(stream, 0, ast[1])
        stream.write('(x1,x2) +')
        stream.write('sig2 * ')
        compile_covariance_kernel_proc(stream, 0, ast[2])
        stream.write('(x1,x2)')
        stream.write('})')
    else:
        assert False, 'Unknown AST'

if __name__ == '__main__':
    ast = [
        '+',
            ['*',
                ['+', ['WN',49.5], ['C',250.9]],
                ['+',
                    ['PER', 13.2, 8.6],
                    ['+',
                        ['LIN', 1.2],
                        ['LIN', 4.9]]]],
            ['WN', 0.1]]
    stream = StringIO()
    compile_covariance_kernel_proc(stream, 0, ast)
    print stream.getvalue()
