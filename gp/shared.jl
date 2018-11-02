import LinearAlgebra
import Random
import Statistics

import CSV
import PyPlot

using Gen
using Gen: get_child


# ----------------------------
# Experiment harness utilities

function rescale_linear(xs::Vector{Float64}, yl::Float64, yh::Float64)
    xl = minimum(xs)
    xh = maximum(xs)
    slope = (yh - yl) / (xh - xl)
    intercept = yh - xh * slope
    return slope .* xs .+ intercept
end

function load_dataset_from_path(path::String, n_test::Int)
    df = CSV.read(path, header=0)
    xs = rescale_linear(Vector{Float64}(df[1]), 0., 1.)
    ys = rescale_linear(Vector{Float64}(df[2]), -1., 1.)
    xs_train = xs[1:end-n_test]
    ys_train = ys[1:end-n_test]
    xs_test = xs[end-n_test+1:end]
    ys_test = ys[end-n_test+1:end]
    return (xs_train, ys_train), (xs_test, ys_test)
end

function make_iteration_schedule(iters::Int, epochs::Int, sched::String)
    if sched == "constant"
        return [iters*1 for i in 1:epochs]
    elseif sched == "linear"
        return [iters*i for i in 1:epochs]
    elseif sched == "doubling"
        return [iters*2^i for i in 1:epochs]
    else
        @assert False "Unknown schedule: $(schedule)"
    end
end

function make_xs_probe(xs::Vector{Float64}, N::Int)
    start, stop = minimum(xs), maximum(xs)
    result = range(start, stop=stop, length=N)
    return collect(result)
end

function compute_rmse(vs::Vector{Float64}, us::Vector{Float64})
    @assert size(vs) == Base.size(us)
    sq_err = (vs .- us).^2
    sq_err_mean = Statistics.mean(sq_err)
    return sqrt(sq_err_mean)
end


# --------------------------------
# Gaussian process covariance tree

"""Node in a tree representing a covariance function"""
abstract type Node end
abstract type LeafNode <: Node end
abstract type BinaryOpNode <: Node end


"""
    size(::Node)

Number of nodes in the subtree rooted at this node.
"""
Base.size(::LeafNode) = 1
Base.size(node::BinaryOpNode) = node.size


"""Constant kernel"""
struct Constant <: LeafNode
    param::Float64
end


eval_cov(node::Constant, x1, x2) = node.param

function eval_cov_mat(node::Constant, xs::Vector{Float64})
    n = length(xs)
    fill(node.param, (n, n))
end


"""Linear kernel"""
struct Linear <: LeafNode
    param::Float64
end


eval_cov(node::Linear, x1, x2) = (x1 - node.param) * (x2 - node.param)

function eval_cov_mat(node::Linear, xs::Vector{Float64})
    xs_minus_param = xs .- node.param
    xs_minus_param * xs_minus_param'
end


"""Squared exponential kernel"""
struct SquaredExponential <: LeafNode
    length_scale::Float64
end


eval_cov(node::SquaredExponential, x1, x2) =
    exp(-0.5 * (x1 - x2) * (x1 - x2) / node.length_scale)

function eval_cov_mat(node::SquaredExponential, xs::Vector{Float64})
    diff = xs .- xs'
    exp.(-0.5 .* diff .* diff ./ node.length_scale)
end


"""Periodic kernel"""
struct Periodic <: LeafNode
    scale::Float64
    period::Float64
end


function eval_cov(node::Periodic, x1, x2)
    freq = 2 * pi / node.period
    exp((-1/node.scale) * (sin(freq * abs(x1 - x2)))^2)
end


function eval_cov_mat(node::Periodic, xs::Vector{Float64})
    freq = 2 * pi / node.period
    abs_diff = abs.(xs .- xs')
    exp.((-1/node.scale) .* (sin.(freq .* abs_diff)).^2)
end


"""Plus node"""
struct Plus <: BinaryOpNode
    left::Node
    right::Node
    size::Int
end


Plus(left, right) = Plus(left, right, size(left) + size(right) + 1)


function eval_cov(node::Plus, x1, x2)
    eval_cov(node.left, x1, x2) + eval_cov(node.right, x1, x2)
end


function eval_cov_mat(node::Plus, xs::Vector{Float64})
    eval_cov_mat(node.left, xs) .+ eval_cov_mat(node.right, xs)
end


"""Times node"""
struct Times <: BinaryOpNode
    left::Node
    right::Node
    size::Int
end


Times(left, right) = Times(left, right, size(left) + size(right) + 1)


function eval_cov(node::Times, x1, x2)
    eval_cov(node.left, x1, x2) * eval_cov(node.right, x1, x2)
end


function eval_cov_mat(node::Times, xs::Vector{Float64})
    eval_cov_mat(node.left, xs) .* eval_cov_mat(node.right, xs)
end


# -------------
# Node metadata

const CONSTANT = 1      # 0.2
const LINEAR = 2        # 0.2
const SQUARED_EXP = 3   # 0.2
const PERIODIC = 4      # 0.2
const PLUS = 5          # binary 0.1
const TIMES = 6         # binary 0.1

const node_type_to_num_children = Dict(
    CONSTANT => 0,
    LINEAR => 0,
    SQUARED_EXP => 0,
    PERIODIC => 0,
    PLUS => 2,
    TIMES => 2)

const max_branch = 2
const node_dist = Float64[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]


# ------------------------------------
# Sampling nodes in GP covariance tree


"""
    pick_random_node(::Node, cur::Int, max_branch::Int)

Return a random node in the subtree rooted at the given node, whose integer
index is given. The sampling is biased to choose nodes at higher indexes
in the tree.
"""
function pick_random_node end


pick_random_node(node::LeafNode, cur::Int, max_branch::Int) = cur


function pick_random_node(node::BinaryOpNode, cur::Int, max_branch::Int)
    if bernoulli(0.5)
        # pick this node
        cur
    else
        # recursively pick from the subtrees
        if bernoulli(0.5)
            pick_random_node(node.left, get_child(cur, 1, max_branch), max_branch)
        else
            pick_random_node(node.right, get_child(cur, 2, max_branch), max_branch)
        end
    end
end


"""
    pick_random_node_unbiased(::Node, cur::Int, max_branch::Int)

Return a random node in the subtree rooted at the given node, whose integer
index is given. The sampling is uniform at random over all nodes in the
tree.
"""
function pick_random_node_unbiased end


pick_random_node_unbiased(node::LeafNode, cur::Int, max_branch::Int) = cur


function pick_random_node_unbiased(node::BinaryOpNode, cur::Int, max_branch::Int)
    probs = [1, size(node.left), size(node.right)] ./ size(node)
    choice = categorical(probs)
    if choice == 1
        return cur
    elseif choice == 2
        n_child = get_child(cur, 1, max_branch)
        return pick_random_node_unbiased(node.left, n_child, max_branch)
    elseif choice == 3
        n_child = get_child(cur, 2, max_branch)
        return pick_random_node_unbiased(node.right, n_child, max_branch)
    else
        @assert false "Unexpected child node $(choice)"
    end
end


"""Compute covariance matrix by evaluating function on each pair of inputs."""
function compute_cov_matrix(covariance_fn::Node, noise, xs)
    n = length(xs)
    cov_matrix = Matrix{Float64}(undef, n, n)
    for i=1:n
        for j=1:n
            cov_matrix[i, j] = eval_cov(covariance_fn, xs[i], xs[j])
        end
        cov_matrix[i, i] += noise
    end
    return cov_matrix
end


"""Compute covariance function by recursively computing covariance matrices."""
function compute_cov_matrix_vectorized(covariance_fn, noise, xs)
    n = length(xs)
    eval_cov_mat(covariance_fn, xs) + Matrix(noise * LinearAlgebra.I, n, n)
end


"""Obtain conditional multivariate normal predictive distribution."""
function get_conditional_mu_cov(covariance_fn::Node, noise::Float64,
        xs::Vector{Float64}, ys::Vector{Float64}, new_xs::Vector{Float64})
    n_prev = length(xs)
    n_new = length(new_xs)
    means = zeros(n_prev + n_new)
    cov_matrix = compute_cov_matrix(covariance_fn, noise, vcat(xs, new_xs))
    cov_matrix_11 = cov_matrix[1:n_prev, 1:n_prev]
    cov_matrix_22 = cov_matrix[n_prev+1:n_prev+n_new, n_prev+1:n_prev+n_new]
    cov_matrix_12 = cov_matrix[1:n_prev, n_prev+1:n_prev+n_new]
    cov_matrix_21 = cov_matrix[n_prev+1:n_prev+n_new, 1:n_prev]
    @assert cov_matrix_12 == cov_matrix_21'
    mu1 = means[1:n_prev]
    mu2 = means[n_prev+1:n_prev+n_new]
    conditional_mu = mu2 + cov_matrix_21 * (cov_matrix_11 \ (ys - mu1))
    conditional_cov_matrix = cov_matrix_22 - cov_matrix_21 * (cov_matrix_11 \ cov_matrix_12)
    conditional_cov_matrix = 0.5 * conditional_cov_matrix + 0.5 * conditional_cov_matrix'
    return conditional_mu, conditional_cov_matrix
end


"""Return predictive log likelihood of new input/output values."""
function compute_log_likelihood_predictive(covariance_fn::Node, noise::Float64,
        xs::Vector{Float64}, ys::Vector{Float64}, new_xs::Vector{Float64},
        new_ys::Vector{Float64})
    mu, cov = get_conditional_mu_cov(covariance_fn, noise, xs, ys, new_xs)
    return logpdf(mvnormal, new_ys, mu, cov)
end


"""Return predictive samples of output values for new inputs."""
function gp_predictive_samples(covariance_fn::Node, noise::Float64,
        xs::Vector{Float64}, ys::Vector{Float64}, new_xs::Vector{Float64})
    mu, cov = get_conditional_mu_cov(covariance_fn, noise, xs, ys, new_xs)
    return mvnormal(mu, cov)
end


function gp_predictive_samples(covariance_fn::Node, noise::Float64,
        xs::Vector{Float64}, ys::Vector{Float64}, new_xs::Vector{Float64},
        N::Int)
    return [
        gp_predictive_samples(covariance_fn, noise, xs, ys, new_xs)
        for _i in 1:N
    ]
end


"""Return mean of predictive distribution on output values for new inputs."""
function gp_predictive_mean(covariance_fn::Node, noise::Float64,
        xs::Vector{Float64}, ys::Vector{Float64}, new_xs::Vector{Float64})
    mu, cov = get_conditional_mu_cov(covariance_fn, noise, xs, ys, new_xs)
    return mu
end
