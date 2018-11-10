include("../shared.jl")

#########
# model #
#########

@gen function datum(x::Float64, params::Params)
    is_outlier = @addr(bernoulli(params.prob_outlier), :z)
    std = is_outlier ? params.inlier_std : params.outlier_std
    y = @addr(normal(x * params.slope + params.intercept, std), :y)
    return y
end

# This is a higher-order Gen function returning a new gen function which is
# multiple independent applications.
data = plate(datum)

@gen function model(xs::Vector{Float64})
    inlier_std = @addr(Gen.gamma(1, 1), :inlier_std)
    outlier_std = @addr(Gen.gamma(1, 1), :outlier_std)
    slope = @addr(normal(0, 2), :slope)
    intercept = @addr(normal(0, 2), :intercept)
    params = Params(0.5, inlier_std, outlier_std, slope, intercept)
    @diff begin
        argdiff = noargdiff
        for addr in [:slope, :intercept, :inlier_std, :outlier_std]
            if !isnodiff(@choicediff(addr))
                argdiff = unknownargdiff
            end
        end
    end
    ys = @addr(data(xs, fill(params, length(xs))), :data)
    return ys
end

# Three options for @choicediff
# Check lightweight/update.jl
# 1. isnodiff
# 2. isnewdiff
# 3. prev
# Also check plate/update.jl for the plate-specific implementations.
# You cannot use :z since that address is not owned by @gen model.
# This whole thing is to ensure that when a Bernoulli choice is switched
# only visit the single data point corresponding to the flipped outlier
# indicator.

#######################
# inference operators #
#######################

@gen function slope_proposal(prev)
    slope = get_assignment(prev)[:slope]
    @addr(normal(slope, 0.5), :slope)
end

@gen function intercept_proposal(prev)
    intercept = get_assignment(prev)[:intercept]
    @addr(normal(intercept, 0.5), :intercept)
end

@gen function inlier_std_proposal(prev)
    inlier_std = get_assignment(prev)[:inlier_std]
    @addr(normal(inlier_std, 0.5), :inlier_std)
end

@gen function outlier_std_proposal(prev)
    outlier_std = get_assignment(prev)[:outlier_std]
    @addr(normal(outlier_std, 0.5), :outlier_std)
end

@gen function is_outlier_proposal(prev, i::Int)
    prev = get_assignment(prev)[:data => i => :z]
    @addr(bernoulli(prev ? 0.0 : 1.0), :data => i => :z)
end

@gen function observer(ys::Vector{Float64})
    for (i, y) in enumerate(ys)
        @addr(dirac(y), :data => i => :y)
    end
end

Gen.load_generated_functions()

##################
# run experiment #
##################

function do_inference(n)

    # prepare dataset
    xs, ys = generate_dataset()
    # observations = get_assignment(simulate(observer, (ys,)))
    observations = DynamicAssignment()
    for (i, y) in enumerate(ys)
        observations[:data => i => :y] = y
    end

    # initial trace
    (trace, _) = generate(model, (xs,), observations)

    for i=1:n

        # steps on the parameters
        for j=1:5
            trace = mh(model, slope_proposal, (), trace)
            trace = mh(model, intercept_proposal, (), trace)
            trace = mh(model, inlier_std_proposal, (), trace)
            trace = mh(model, outlier_std_proposal, (), trace)
        end

        # step on the outliers
        for j=1:length(xs)
            trace = mh(model, is_outlier_proposal, (j,), trace)
        end

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

#################
# run inference #
#################

(score, inlier_std, outlier_std, slope, intercept) = do_inference(500)

using Test
max_std = max(inlier_std, outlier_std)
min_std = min(inlier_std, outlier_std)
# @test isapprox(min_std, 0.5, atol=1e-1)
# @test isapprox(max_std, 5.0, atol=1e-0)
# @test isapprox(slope, -1, atol=1e-1)
# @test isapprox(intercept, 2, atol=2e-1)
