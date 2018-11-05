using LinearAlgebra
using Random
using Statistics

using Distributions


################################################################################
# cov_tree.jl

"""Sample a categorical variable with given weights."""
function sample_categorical(probs::Vector{Float64})
    u = rand()
    cdf = cumsum(probs)
    for (i, c) in enumerate(cdf)
        if u < c return i end
    end
end


"""Return index of child node in a tree."""
function get_child(parent::Int, child_num::Int, max_branch::Int)
    @assert child_num >= 1 && child_num <= max_branch
    (parent - 1) * max_branch + child_num + 1
end


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

const MAX_BRANCH_GP = 2
const node_dist = Float64[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]


"""
    pick_random_node(::Node, cur::Int, max_branch::Int)

Return a random node in the subtree rooted at the given node, whose integer
index is given. The sampling is biased to choose nodes at higher indexes
in the tree.
"""
function pick_random_node end


pick_random_node(node::LeafNode, cur::Int, max_branch::Int) = cur


function pick_random_node(node::BinaryOpNode, cur::Int, max_branch::Int)
    probs = [.5, .25, .25]
    choice = sample_categorical(probs)
    if choice == 1
        return cur
    elseif choice == 2
        n_child = get_child(cur, 1, max_branch)
        return pick_random_node(node.left, n_child, max_branch)
    elseif choice == 3
        n_child = get_child(cur, 2, max_branch)
        return pick_random_node(node.right, n_child, max_branch)
    else
        @assert false "Unexpected child node $(choice)"
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
    choice = sample_categorical(probs)
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


################################################################################
# gp_predict.jl


"""Return log likelihood given input/output values."""
function compute_log_likelihood(covariance_fn::Node, noise::Float64,
        xs::Vector{Float64}, ys::Vector{Float64})
    mu = zeros(length(xs))
    cov = compute_cov_matrix_vectorized(covariance_fn, noise, xs)
    mvn = Distributions.MvNormal(mu, cov)
    return Distributions.logpdf(mvn, ys)
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
    mvn = Distributions.MvNormal(mu, cov)
    return Distributions.logpdf(mvn, new_ys)
end

"""Return predictive samples of output values for new inputs."""
function gp_predictive_samples(covariance_fn::Node, noise::Float64,
        xs::Vector{Float64}, ys::Vector{Float64}, new_xs::Vector{Float64})
    mu, cov = get_conditional_mu_cov(covariance_fn, noise, xs, ys, new_xs)
    mvn = Distributions.MvNormal(mu, cov)
    return rand(mvn)
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


################################################################################
# modeling code

struct Trace
    cov_fn::Node
    noise::Float64
    xs::Vector{Float64}
    ys::Vector{Float64}
    log_likelihood::Float64
end


function covariance_prior(cur::Int)
    node_type = sample_categorical(node_dist)

    if node_type == CONSTANT
        param = rand()
        node = Constant(param)

    # linear kernel
    elseif node_type == LINEAR
        param = rand()
        node = Linear(param)

    # squared exponential kernel
    elseif node_type == SQUARED_EXP
        length_scale= rand()
        node = SquaredExponential(length_scale)

    # periodic kernel
    elseif node_type == PERIODIC
        scale = rand()
        period = rand()
        node = Periodic(scale, period)

    # plus combinator
    elseif node_type == PLUS
        child1 = get_child(cur, 1, MAX_BRANCH_GP)
        child2 = get_child(cur, 2, MAX_BRANCH_GP)
        left = covariance_prior(child1)
        right = covariance_prior(child2)
        node = Plus(left, right)

    # times combinator
    elseif node_type == TIMES
        child1 = get_child(cur, 1, MAX_BRANCH_GP)
        child2 = get_child(cur, 2, MAX_BRANCH_GP)
        left = covariance_prior(child1)
        right = covariance_prior(child2)
        node = Times(left, right)

    # unknown node type
    else
        error("Unknown node type: $node_type")
    end

    return node
end

# Inference

function replace_subtree(cov_fn::LeafNode, cur::Int, cov_fn2::Node, cur2::Int)
    return cur == cur2 ? cov_fn2 : cov_fn
end


function replace_subtree(cov_fn::BinaryOpNode, cur::Int, cov_fn2::Node, cur2::Int)
    if cur == cur2
        return cov_fn2
    end
    child_l = get_child(cur, 1, MAX_BRANCH_GP)
    child_r = get_child(cur, 2, MAX_BRANCH_GP)
    subtree_left = child_l == cur2 ? cov_fn2 :
        replace_subtree(cov_fn.left, child_l, cov_fn2, cur2)
    subtree_right = child_r == cur2 ? cov_fn2 :
        replace_subtree(cov_fn.right, child_r, cov_fn2, cur2)
    return typeof(cov_fn)(subtree_left, subtree_right)
end


function propose_new_subtree(prev_trace)
    loc_delta = pick_random_node_unbiased(prev_trace.cov_fn, 1, MAX_BRANCH_GP)
    subtree = covariance_prior(loc_delta)
    cov_fn_new = replace_subtree(prev_trace.cov_fn, 1, subtree, loc_delta)
    log_likelihood = compute_log_likelihood(cov_fn_new, prev_trace.noise,
        prev_trace.xs, prev_trace.ys)
    return Trace(cov_fn_new, prev_trace.noise, prev_trace.xs, prev_trace.ys,
        log_likelihood)
end


function propose_new_noise(prev_trace)
    noise_new = rand(Distributions.Gamma(1, 1)) + 0.01
    log_likelihood = compute_log_likelihood(prev_trace.cov_fn, noise_new,
        prev_trace.xs, prev_trace.ys)
    return Trace(prev_trace.cov_fn, noise_new, prev_trace.xs, prev_trace.ys,
        log_likelihood)
end

function mh_resample(prev_trace, f_propose::Function)
    new_trace = f_propose(prev_trace)
    alpha_size = log(size(prev_trace.cov_fn)) - log(size(new_trace.cov_fn))
    alpha_ll = new_trace.log_likelihood - prev_trace.log_likelihood
    alpha = alpha_ll + alpha_size
    return log(rand()) < alpha ? new_trace : prev_trace
end

# Pipeline API functions.

function initialize_trace(xs::Vector{Float64}, ys::Vector{Float64})
    cov_fn::Node = covariance_prior(1)
    noise = rand(Distributions.Gamma(1, 1)) + 0.01
    log_likelihood = compute_log_likelihood(cov_fn, noise, xs, ys)
    return Trace(cov_fn, noise, xs, ys, log_likelihood)
end

function run_mcmc(prev_trace, iters::Int)
    new_trace = prev_trace
    for iter=1:iters
        new_trace = mh_resample(new_trace, propose_new_subtree)
        new_trace = mh_resample(new_trace, propose_new_noise)
    end
    return new_trace
end

extract_cov_noise(trace::Trace) = trace.cov_fn, trace.noise
