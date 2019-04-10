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
    inlier_log_var = @trace(normal(0, 1), :inlier_log_var)
    outlier_log_var = @trace(normal(0, 1), :outlier_log_var)
    slope = @trace(normal(0, 2), :slope)
    intercept = @trace(normal(0, 2), :intercept)
    means = broadcast(+, slope * xs, intercept)
    inlier_stds = fill(sqrt(exp(inlier_log_var)), n)
    outlier_stds = fill(sqrt(exp(outlier_log_var)), n)
    ys = @trace(data(means, inlier_stds, outlier_stds), :data)
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

@gen (static) function inlier_log_var_proposal(prev)
    inlier_log_var = prev[:inlier_log_var]
    @trace(normal(inlier_log_var, .5), :inlier_log_var)
end

@gen (static) function outlier_log_var_proposal(prev)
    outlier_log_var = prev[:outlier_log_var]
    @trace(normal(outlier_log_var, .5), :outlier_log_var)
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

    results = []

    runtime = 0
    for i=1:n
        start = time()
        # steps on the parameters
        (trace, _accept) = mh(trace, slope_proposal, ())
        (trace, _accept) = mh(trace, intercept_proposal, ())
        (trace, _accept) = mh(trace, inlier_log_var_proposal, ())
        (trace, _accept) = mh(trace, outlier_log_var_proposal, ())
        # record runtime
        elapsed = time() - start
        runtime += elapsed
        # update stats
        stats = (
            i,
            runtime,
            get_score(trace),
            trace[:slope],
            trace[:intercept],
            trace[:inlier_log_var],
            trace[:outlier_log_var],)
        push!(results, stats)
		println(stats)
    end

    return results
end

#################
# run inference #
#################

do_inference(10)

fname = "static_mh.results.csv"
header=["num_steps", "runtime", "score", "slope",
    "intercept", "inlier_log_var", "outlier_log_var"]
open(fname, "w") do f
    write(f, join(header, ',') * '\n')
end

num_reps = 5
for i in 1:num_reps
    results = do_inference(1000)
    open(fname, "a") do f
        for row in results
            write(f, join(row, ',') * '\n')
        end
    end
end
