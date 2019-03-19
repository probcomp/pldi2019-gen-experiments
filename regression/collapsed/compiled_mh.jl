include("../shared.jl")

import Random
import Distributions

import Gen: random, logpdf

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

@gen (static) function dummy_two_normal(mu, inlier_std, outlier_std)
    @trace(two_normals(mu, inlier_std, outlier_std), :y)
end

data = Map(dummy_two_normal)

@gen (static) function model(xs::Vector{Float64})
    n = length(xs)
    inlier_std = @trace(gamma(1, 1), :inlier_std)
    outlier_std = @trace(gamma(1, 1), :outlier_std)
    slope = @trace(normal(0, 2), :slope)
    intercept = @trace(normal(0, 2), :intercept)
    means = broadcast(+, slope * xs, intercept)
    ys = @trace(
        data(means, fill(inlier_std, n), fill(outlier_std, n)), :data)
    return ys
end

#######################
# inference operators #
#######################

@gen (static) function slope_proposal(prev)
    slope = prev[:slope]
    @trace(normal(slope, .5), :slope)
end

@gen (static) function intercept_proposal(prev)
    intercept = prev[:intercept]
    @trace(normal(intercept, .5), :intercept)
end

@gen (static) function inlier_std_proposal(prev)
    inlier_std = prev[:inlier_std]
    @trace(normal(inlier_std, .5), :inlier_std)
end

@gen (static) function outlier_std_proposal(prev)
    outlier_std = prev[:outlier_std]
    @trace(normal(outlier_std, .5), :outlier_std)
end

function logsumexp(arr)
    min_arr = maximum(arr)
    min_arr + log(sum(exp.(arr .- min_arr)))
end

Gen.load_generated_functions()

##################
# run experiment #
##################

function do_inference(n)
    (xs, ys) = load_dataset("../train.csv")
    observations = choicemap()
    for (i, y) in enumerate(ys)
        observations[:data => i => :y] = y
    end

    # initial trace
    (trace, weight) = generate(model, (xs,), observations)

    runtime = 0
    for i=1:n
        start = time()
        # steps on the parameters
        (trace, _accept) = mh(trace, slope_proposal, ())
        (trace, _accept) = mh(trace, intercept_proposal, ())
        (trace, _accept) = mh(trace, inlier_std_proposal, ())
        (trace, _accept) = mh(trace, outlier_std_proposal, ())
        elapsed = time() - start
        runtime += elapsed

        score = get_score(trace)
		println((score, trace[:inlier_std], trace[:outlier_std],
            trace[:slope], trace[:intercept]))
    end

    score = get_score(trace)
    return (
        n,
        runtime,
        score,
        trace[:slope],
        trace[:intercept],
        trace[:inlier_std],
        trace[:outlier_std],
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
