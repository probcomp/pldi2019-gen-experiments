include("../shared.jl")

import Random
import Distributions
using FunctionalCollections

import Gen: random, logpdf, get_static_argument_types

###################
# collapsed model #
###################

struct TwoNormals <: Distribution{Float64} end
const two_normals = TwoNormals()

function logpdf(::TwoNormals, x, mu, sigma1, sigma2)
    if sigma1 < 0 || sigma2 < 0
        return -Inf
    end
    l1 = Distributions.logpdf(Distributions.Normal(mu, sigma1), x) + log(0.5)
    l2 = Distributions.logpdf(Distributions.Normal(mu, sigma2), x) + log(0.5)
    m = max(l1, l2)
    m + log(exp(l1 - m) + exp(l2 - m))
end

function random(::TwoNormals, mu, sigma1, sigma2)
    mu + (rand() < 0.5 ? sigma1 : sigma2) * randn()
end

get_static_argument_types(::TwoNormals) = [Float64, Float64, Float64]

data = plate(two_normals)

@compiled @gen function model(xs::Vector{Float64})
    n::Int = length(xs)
    inlier_std::Float64 = @addr(gamma(1, 1), :inlier_std)
    outlier_std::Float64 = @addr(gamma(1, 1), :outlier_std)
    slope::Float64 = @addr(normal(0, 2), :slope)
    intercept::Float64 = @addr(normal(0, 2), :intercept)
    means::Vector{Float64} = broadcast(+, slope * xs, intercept)
    ys::PersistentVector{Float64} = @addr(
        data(means, fill(inlier_std, n), fill(outlier_std, n)), :data)
    return ys
end

#######################
# inference operators #
#######################

@compiled @gen function slope_proposal(prev)
    slope::Float64 = get_assignment(prev)[:slope]
    @addr(normal(slope, .5), :slope)
end

@compiled @gen function intercept_proposal(prev)
    intercept::Float64 = get_assignment(prev)[:intercept]
    @addr(normal(intercept, .5), :intercept)
end

@compiled @gen function inlier_std_proposal(prev)
    inlier_std::Float64 = get_assignment(prev)[:inlier_std]
    @addr(normal(inlier_std, .5), :inlier_std)
end

@compiled @gen function outlier_std_proposal(prev)
    outlier_std::Float64 = get_assignment(prev)[:outlier_std]
    @addr(normal(outlier_std, .5), :outlier_std)
end

function logsumexp(arr)
    min_arr = maximum(arr)
    min_arr + log(sum(exp.(arr .- min_arr)))
end

@gen function observer(ys::Vector{Float64})
    for (i, y) in enumerate(ys)
        @addr(dirac(y), :data => i)
    end
end

Gen.load_generated_functions()

##################
# run experiment #
##################

function do_inference(n)

    # prepare dataset
    xs, ys = generate_dataset()
    observations = get_assignment(simulate(observer, (ys,)))

    # initial trace
    (trace, weight) = generate(model, (xs,), observations)

    runtime = 0
    for i=1:n
        start = time()
        # steps on the parameters
        trace = mh(model, slope_proposal, (), trace)
        trace = mh(model, intercept_proposal, (), trace)
        trace = mh(model, inlier_std_proposal, (), trace)
        trace = mh(model, outlier_std_proposal, (), trace)
        elapsed = time() - start
        runtime += elapsed

		assignment = get_assignment(trace)
        score = get_call_record(trace).score
		println((score, assignment[:inlier_std], assignment[:outlier_std],
            assignment[:slope], assignment[:intercept]))
    end

	assignment = get_assignment(trace)
    score = get_call_record(trace).score
    return (
        n,
        runtime,
        score,
        assignment[:slope],
        assignment[:intercept],
        assignment[:inlier_std],
        assignment[:outlier_std],
        )
end

#################
# run inference #
#################

do_inference(10)

results = do_inference(200)
fname = "compiled_mh.results.csv"
open(fname, "a") do f
    write(f, join(results, ',') * '\n')
end
