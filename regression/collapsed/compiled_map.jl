include("../shared.jl")

import Random
Random.seed!(43)

import Distributions

import Gen.has_argument_grads
import Gen.has_output_grad
import Gen.logpdf
import Gen.logpdf_grad
import Gen.random

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
        NaN,
        NaN,
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
has_argument_grads(::TwoNormals) = (true, false, false)
get_static_argument_types(::TwoNormals) = (Float64, Float64, Float64)

@gen (static) function dummy_two_normal(
        (grad)(mu::Float64),
        inlier_std::Float64,
        outlier_std::Float64)
    y = @trace(two_normals(mu, inlier_std, outlier_std), :y)
    return y
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
    return n
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

@gen (static) function inlier_std_proposal(prev)
    inlier_std = prev[:inlier_std]
    @trace(normal(inlier_std, .5), :inlier_std)
end

@gen (static) function outlier_std_proposal(prev)
    outlier_std = prev[:outlier_std]
    @trace(normal(outlier_std, .5), :outlier_std)
end

Gen.load_generated_functions()

##################
# run experiment #
##################

slope_intercept_selection = select(:slope, :intercept)
std_selection = select(:inlier_std, :outlier_std)

function do_inference(method, n)
    # prepare dataset
    (xs, ys) = load_dataset("../train.csv")
    observations = choicemap()
    for (i, y) in enumerate(ys)
        observations[:data => i => :y] = y
    end

    # initial trace
    (trace, _) = generate(model, (xs,), observations)

    score = get_score(trace)

    runtime = 0
    for i=1:n
        start = time()
        if method == "mala"
            (trace, _accept) = mala(trace, slope_intercept_selection, 0.0001)
        elseif method == "map"
            trace = map_optimize(trace, slope_intercept_selection,
                max_step_size=1e-1, min_step_size=1e-10)
        else
            @assert false "Unknown method: $(method)"
        end

        (trace, _) = mh(trace, inlier_std_proposal, ())
        (trace, _) = mh(trace, outlier_std_proposal, ())

        elapsed = time() - start
        runtime += elapsed

        # report loop stats
        score = get_score(trace)
        println((
            i, score,
            trace[:slope],
            trace[:intercept],
            trace[:inlier_std],
            trace[:outlier_std]
        ))
    end

    score = get_score(trace)
    return ((
        n,
        runtime,
        score,
        trace[:slope],
        trace[:intercept],
        trace[:inlier_std],
        trace[:outlier_std]
        ))
end

#################
# run inference #
#################

method = "map"
do_inference(method, 10)

results = do_inference(method, 500)
fname = "compiled_$(method).results.csv"
open(fname, "a") do f
    write(f, join(results, ',') * '\n')
end
