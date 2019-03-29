include("cov_tree.jl")

using Distributions
using Statistics

################################################################################
# inference utilities

const MAX_BRANCH_GP = 2

"""Return log likelihood given input/output values."""
function compute_log_likelihood(covariance_fn::Node, noise::Float64,
        xs::Vector{Float64}, ys::Vector{Float64})
    mu = zeros(length(xs))
    cov = compute_cov_matrix_vectorized(covariance_fn, noise, xs)
    mvn = Distributions.MvNormal(mu, cov)
    return Distributions.logpdf(mvn, ys)
end


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

"""
    pick_random_node_biased(::Node, cur::Int, max_branch::Int)

Return a random node in the subtree rooted at the given node, whose integer
index is given. The sampling is biased to choose nodes at higher indexes
in the tree.
"""
function pick_random_node_biased end


function pick_random_node_biased(node::LeafNode, cur::Int, max_branch::Int)
    return (cur, node)
end


function pick_random_node_biased(node::BinaryOpNode, cur::Int, max_branch::Int)
    if bernoulli(0.5)
        # pick this node
        (cur, node)
    else
        # recursively pick from the subtrees
        if bernoulli(0.5)
            n_child = get_child(cur, 1, max_branch)
            pick_random_node_biased(node.left, n_child, max_branch)
        else
            n_child = get_child(cur, 2, max_branch)
            pick_random_node_biased(node.right, n_child, max_branch)
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


function pick_random_node_unbiased(node::LeafNode, cur::Int, max_branch::Int)
    return (cur, node)
end


function pick_random_node_unbiased(node::BinaryOpNode, cur::Int, max_branch::Int)
    probs = [1, size(node.left), size(node.right)] ./ size(node)
    choice = sample_categorical(probs)
    if choice == 1
        return (cur, node)
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



################################################################################
# modeling code

struct Trace
    cov_fn::Node
    noise::Float64
    xs::Vector{Float64}
    ys::Vector{Float64}
    log_likelihood::Float64
end


function covariance_prior()
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
        left = covariance_prior()
        right = covariance_prior()
        node = Plus(left, right)

    # times combinator
    elseif node_type == TIMES
        left = covariance_prior()
        right = covariance_prior()
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

# MH correction for biased sampling.
function get_alpha_subtree_biased end
get_alpha_subtree_biased(::LeafNode, ::LeafNode) = 0
get_alpha_subtree_biased(::BinaryOpNode, ::BinaryOpNode) = 0
get_alpha_subtree_biased(::LeafNode, ::BinaryOpNode) = -log(2)
get_alpha_subtree_biased(::BinaryOpNode, ::LeafNode) = log(2)

# MH correction for unbiased sampling.
function get_alpha_subtree_unbiased(cov_fn_prev, cov_fn_prop)
    return log(size(cov_fn_prev)) - log(size(cov_fn_prop))
end

function mh_resample_subtree_biased(prev_trace)
    (loc_delta, node_delta) = pick_random_node_biased(prev_trace.cov_fn, 1, MAX_BRANCH_GP)
    subtree = covariance_prior()
    cov_fn_new = replace_subtree(prev_trace.cov_fn, 1, subtree, loc_delta)
    log_likelihood = compute_log_likelihood(cov_fn_new, prev_trace.noise,
        prev_trace.xs, prev_trace.ys)
    new_trace = Trace(cov_fn_new, prev_trace.noise, prev_trace.xs,
        prev_trace.ys, log_likelihood)
    alpha_size = get_alpha_subtree_biased(node_delta, subtree)
    alpha_ll = new_trace.log_likelihood - prev_trace.log_likelihood
    alpha = alpha_ll + alpha_size
    return log(rand()) < alpha ? new_trace : prev_trace
end

function mh_resample_subtree_unbiased(prev_trace)
    (loc_delta, node_delta) = pick_random_node_unbiased(prev_trace.cov_fn, 1, MAX_BRANCH_GP)
    subtree = covariance_prior()
    cov_fn_new = replace_subtree(prev_trace.cov_fn, 1, subtree, loc_delta)
    log_likelihood = compute_log_likelihood(cov_fn_new, prev_trace.noise,
        prev_trace.xs, prev_trace.ys)
    new_trace = Trace(cov_fn_new, prev_trace.noise, prev_trace.xs,
        prev_trace.ys, log_likelihood)
    alpha_size = get_alpha_subtree_unbiased(node_delta, subtree)
    alpha_ll = new_trace.log_likelihood - prev_trace.log_likelihood
    alpha = alpha_ll + alpha_size
    return log(rand()) < alpha ? new_trace : prev_trace
end

function mh_resample_subtree_root(prev_trace)
    cov_fn_new = covariance_prior()
    log_likelihood = compute_log_likelihood(cov_fn_new, prev_trace.noise,
        prev_trace.xs, prev_trace.ys)
    new_trace = Trace(cov_fn_new, prev_trace.noise, prev_trace.xs,
        prev_trace.ys, log_likelihood)
    alpha = new_trace.log_likelihood - prev_trace.log_likelihood
    return log(rand()) < alpha ? new_trace : prev_trace
end

function propose_new_noise(prev_trace)
    noise_new = rand(Distributions.Gamma(1, 1)) + 0.01
    log_likelihood = compute_log_likelihood(prev_trace.cov_fn, noise_new,
        prev_trace.xs, prev_trace.ys)
    return Trace(prev_trace.cov_fn, noise_new, prev_trace.xs, prev_trace.ys,
        log_likelihood)
end

function mh_resample_noise(prev_trace)
    new_trace = propose_new_noise(prev_trace)
    alpha = new_trace.log_likelihood - prev_trace.log_likelihood
    return log(rand()) < alpha ? new_trace : prev_trace
end


# Pipeline API functions.

function initialize_trace(xs::Vector{Float64}, ys::Vector{Float64})
    cov_fn::Node = covariance_prior()
    noise = rand(Distributions.Gamma(1, 1)) + 0.01
    log_likelihood = compute_log_likelihood(cov_fn, noise, xs, ys)
    return Trace(cov_fn, noise, xs, ys, log_likelihood)
end

function run_mcmc(prev_trace, iters::Int)
    new_trace = prev_trace
    for iter=1:iters
        new_trace = mh_resample_subtree_unbiased(new_trace)
        new_trace = mh_resample_noise(new_trace)
    end
    return new_trace
end

extract_cov_noise(trace::Trace) = trace.cov_fn, trace.noise
