import LinearAlgebra
import Random

using Gen

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

const node_dist = Float64[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]

const MAX_BRANCH = 2


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
