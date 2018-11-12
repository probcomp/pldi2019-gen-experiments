include("../shared.jl")

import Random
Random.seed!(432)

import Distributions
using FunctionalCollections

using ReverseDiff

import Gen.get_static_argument_types
import Gen.has_argument_grads
import Gen.has_output_grad
import Gen.logpdf
import Gen.logpdf_grad
import Gen.random

############################
# reverse mode AD for fill #
############################

function Base.fill(x::ReverseDiff.TrackedReal{V,D,O}, n::Integer) where {V,D,O}
    tp = ReverseDiff.tape(x)
    out = ReverseDiff.track(fill(ReverseDiff.value(x), n), V, tp)
    ReverseDiff.record!(tp, ReverseDiff.SpecialInstruction, fill, (x, n), out)
    return out
end

@noinline function ReverseDiff.special_reverse_exec!(
        instruction::ReverseDiff.SpecialInstruction{typeof(fill)})
    x, n = instruction.input
    output = instruction.output
    ReverseDiff.istracked(x) &&
        ReverseDiff.increment_deriv!(x, sum(ReverseDiff.deriv(output)))
    ReverseDiff.unseed!(output)
    return nothing
end

@noinline function ReverseDiff.special_forward_exec!(
        instruction::ReverseDiff.SpecialInstruction{typeof(fill)})
    x, n = instruction.input
    ReverseDiff.value!(instruction.output, fill(ReverseDiff.value(x), n))
    return nothing
end

###################
# collapsed model #
###################

struct TwoNormals <: Distribution{Float64} end
const two_normals = TwoNormals()

function logpdf(::TwoNormals, x, mu, sigma1, sigma2)
    if sigma1 < 0 || sigma2 < 0
        return -Inf
    end
    l1 = Distributions.logpdf(Distributions.Normal(mu, sigma1), x) + log(.5)
    l2 = Distributions.logpdf(Distributions.Normal(mu, sigma2), x) + log(.5)
    m = max(l1, l2)
    return m + log(exp(l1 - m) + exp(l2 - m))
end

function logpdf_grad(::TwoNormals, x, mu, sigma1, sigma2)
    l1 = Distributions.logpdf(Distributions.Normal(mu, sigma1), x) + log(.5)
    l2 = Distributions.logpdf(Distributions.Normal(mu, sigma2), x) + log(.5)
    (deriv_x_1, deriv_mu_1, deriv_std_1) = logpdf_grad(normal, x, mu, sigma1)
    (deriv_x_2, deriv_mu_2, deriv_std_2) = logpdf_grad(normal, x, mu, sigma2)
    w1 = 1.0 / (1.0 + exp(l1 - l2))
    w2 = 1.0 / (1.0 + exp(l2 - l1))
    @assert isapprox(w1 + w2, 1.0)
    deriv_x = deriv_x_1 * w1 + deriv_x_2 * w2
    return (
        deriv_x,
        w1 * deriv_mu_1 + w2 * deriv_mu_2,
        w1 * deriv_std_1,
        w2 * deriv_std_2,
    )
end

function random(::TwoNormals, mu, sigma1, sigma2)
    if rand() < .5
        return mu + sigma1 * randn()
    else
        return mu + sigma2 * randn()
    end
end

has_output_grad(::TwoNormals) = true
has_argument_grads(::TwoNormals) = (true, true, true)
get_static_argument_types(::TwoNormals) = (Float64, Float64, Float64)

data = plate(two_normals)

@compiled @gen function model(xs::Vector{Float64})
    n::Int = length(xs)
    # inlier_std::Float64 = @addr(normal(0,2), :inlier_std)
    # outlier_std::Float64 = @addr(normal(0,2), :outlier_std)
    slope::Float64 = @addr(normal(0,2), :slope)
    intercept::Float64 = @addr(normal(0,2), :intercept)

    # inlier_std::Float64 = .5
    # outlier_std::Float64 = .5
    # slope::Float64 = -1
    # intercept::Float64 = 2

    means::Vector{Float64} = broadcast(+, slope * xs, intercept)
    ys::PersistentVector{Float64} = @addr(
        data(means, fill(.5, n), fill(5, n)),
        :data)
    return ys
end

# Quick debugging function for computing the objective function.

function compute_objective(xs::Vector{Float64}, ys::Vector{Float64},
        sigma1::Float64, sigma2::Float64)
    means = broadcast(+, -1 * xs, 2)
    return sum([
        logpdf(two_normals, y, mean, sigma1, sigma2)
        for (y, mean) in zip(ys, means)
    ])
end

#######################
# inference operators #
#######################

@gen function observer(ys::Vector{Float64})
    for (i, y) in enumerate(ys)
        @addr(dirac(y), :data => i)
    end
end

Gen.load_generated_functions()

##################
# run experiment #
##################

slope_intercept_selection = let
    s = DynamicAddressSet()
    push!(s, :slope)
    push!(s, :intercept)
    StaticAddressSet(s)
end

std_selection = let
    s = DynamicAddressSet()
    push!(s, :inlier_std)
    push!(s, :outlier_std)
    StaticAddressSet(s)
end

function do_inference(n)
    # prepare dataset
    xs, ys = generate_dataset()
    observations = get_assignment(simulate(observer, (ys,)))

    # initial trace
    (trace, _) = generate(model, (xs,), observations)

    score = get_call_record(trace).score
    assignment = get_assignment(trace)
    println((
        score,
        assignment[:slope],
        assignment[:intercept],
        # assignment[:inlier_std],
        # assignment[:outlier_std],
    ))

    for i=1:n

        # step on the parameters
        for j=1:5
            trace = map_optimize(model, slope_intercept_selection,
                trace, max_step_size=1e-1, min_step_size=1e-10)
            # trace = map_optimize(model, std_selection,
            #     trace, max_step_size=1e-1, min_step_size=1e-10)

            # report loop stats
            score = get_call_record(trace).score
            assignment = get_assignment(trace)
            println((
                score,
                assignment[:slope],
                assignment[:intercept],
                # assignment[:inlier_std],
                # assignment[:outlier_std],
            ))
        end

    end
end

do_inference(100)
