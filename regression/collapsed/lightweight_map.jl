include("../shared.jl")

import Random
import Distributions
using FunctionalCollections

using ReverseDiff

import Gen.get_static_argument_types
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

function logpdf(::TwoNormals, x, w1, mu1, mu2, sigma1, sigma2)
    if sigma1 < 0 || sigma2 < 0
        return -Inf
    end
    l1 = Distributions.logpdf(Distributions.Normal(mu1, sigma1), x) + log(w1)
    l2 = Distributions.logpdf(Distributions.Normal(mu2, sigma2), x) + log(1 - w1)
    m = max(l1, l2)
    m + log(exp(l1 - m) + exp(l2 - m))
end

function logpdf_grad(::TwoNormals, x, w1, mu1, mu2, sigma1, sigma2)
    l1 = Distributions.logpdf(Distributions.Normal(mu1, sigma1), x) + log(w1)
    l2 = Distributions.logpdf(Distributions.Normal(mu2, sigma2), x) + log(1 - w1)
    (deriv_x_1, deriv_mu_1, deriv_sigma_1) = logpdf_grad(normal, x, mu1, sigma1)
    (deriv_x_2, deriv_mu_2, deriv_sigma_2) = logpdf_grad(normal, x, mu2, sigma2)
    w1 = 1.0 / (1.0 + exp(l1 - l2))
    w2 = 1.0 / (1.0 + exp(l2 - l1))
    @assert isapprox(w1 + w2, 1.0)
    deriv_x = deriv_x_1 * w1 + deriv_x_2 * w2
    return (deriv_x,
        NaN,
        w1 * deriv_mu_1,
        w2 * deriv_mu_2,
        w1 * deriv_sigma_1,
        w2 * deriv_sigma_2,)
end

function random(::TwoNormals, w1, mu1, mu2, sigma1, sigma2)
    if rand() < w1
        mu1 + sigma1 * randn()
    else
        mu2 + sigma2 * randn()
    end
end

has_output_grad(::TwoNormals) = true
has_argument_grads(::TwoNormals) = (false, true, true, true, true)
get_static_argument_types(::TwoNormals) = [Float64, Float64, Float64, Float64, Float64]

data = plate(two_normals)

@compiled @gen function model(xs::Vector{Float64})
    n::Int = length(xs)
    inlier_std::Float64 = @addr(gamma(1, 1), :inlier_std)
    outlier_std::Float64 = @addr(gamma(1, 1), :outlier_std)
    slope::Float64 = @addr(normal(0, 2), :slope)
    intercept::Float64 = @addr(normal(0, 2), :intercept)
    means::Vector{Float64} = broadcast(+, slope * xs, intercept)
    ys::PersistentVector{Float64} = @addr(
        data(
            fill(.5, n),
            means,
            means,
            fill(inlier_std, n),
            fill(outlier_std, n)),
        :data)
    return ys
end

#######################
# inference operators #
#######################

@gen function observer(ys::Vector{Float64})
    for (i, y) in enumerate(ys)
        @addr(dirac(y), :data => i => :y)
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

    # Cannot use `get_assignment` to make observations,
    observations = get_assignment(simulate(observer, (ys,)))
    # observations = DynamicAssignment()
    # for (i, y) in enumerate(ys)
    #     observations[:data => i => :y] = y
    # end

    # initial trace
    (trace, _) = generate(model, (xs,), observations)

    for i=1:n

        # step on the parameters
        for j=1:5
            trace = map_optimize(model, slope_intercept_selection,
                trace, max_step_size=0.1, min_step_size=1e-5)
            trace = map_optimize(model, std_selection,
                trace, max_step_size=0.1, min_step_size=1e-10)
        end

        score = get_call_record(trace).score

        # report loop stats
        score = get_call_record(trace).score
        assignment = get_assignment(trace)
        score = get_call_record(trace).score
        println((score, assignment[:inlier_std], assignment[:outlier_std],
            assignment[:slope], assignment[:intercept]))
    end

    assignment = get_assignment(trace)
    score = get_call_record(trace).score
    return (score, assignment[:inlier_std], assignment[:outlier_std],
        assignment[:slope], assignment[:intercept])

end

(score, inlier_std, outlier_std, slope, intercept) = do_inference(500)
