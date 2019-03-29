include("../shared.jl")

using Gen
import Random

#########
# model #
#########

@gen (static) function datum(x::Float64, (grad)(inlier_std::Float64),
        (grad)(outlier_std::Float64), (grad)(slope::Float64),
        (grad)(intercept::Float64))
    is_outlier = @trace(bernoulli(0.5), :z)
    std = is_outlier ? inlier_std : outlier_std
    y = @trace(normal(x * slope + intercept, sqrt(exp(std))), :y)
    return y
end

data = Map(datum)

@gen (static) function model(xs::Vector{Float64})
    n = length(xs)
    inlier_std = @trace(normal(0, 2), :inlier_std)
    outlier_std = @trace(normal(0, 2), :outlier_std)
    slope = @trace(normal(0, 2), :slope)
    intercept = @trace(normal(0, 2), :intercept)
    @trace(data(xs, fill(inlier_std, n), fill(outlier_std, n),
            fill(slope, n), fill(intercept, n)),
        :data)
end

#######################
# inference operators #
#######################

@gen (static) function slope_proposal(prev)
    slope::Float64 = prev[:slope]
    @trace(normal(slope, .5), :slope)
end

@gen (static) function intercept_proposal(prev)
    intercept::Float64 = prev[:intercept]
    @trace(normal(intercept, .5), :intercept)
end

@gen (static) function inlier_std_proposal(prev)
    inlier_std::Float64 = prev[:inlier_std]
    @trace(normal(inlier_std, .5), :inlier_std)
end

@gen (static) function outlier_std_proposal(prev)
    outlier_std::Float64 = prev[:outlier_std]
    @trace(normal(outlier_std, .5), :outlier_std)
end

@gen (static) function is_outlier_proposal(prev, i::Int)
    prev_z = prev[:data => i => :z]
    @trace(bernoulli(prev_z ? 0.0 : 1.0), :data => i => :z)
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
    (trace, _) = generate(model, (xs,), observations)

    scores = Vector{Float64}(undef, n)

    runtime = 0
    for i=1:n

        start = time()
        # steps on the parameters
        (trace, _accept) = mh(trace, slope_proposal, ())
        (trace, _accept) = mh(trace, intercept_proposal, ())
        (trace, _accept) = mh(trace, inlier_std_proposal, ())
        (trace, _accept) = mh(trace, outlier_std_proposal, ())

        # step on the outliers
        for j=1:length(xs)
            (trace, _accept) = mh(trace, is_outlier_proposal, (j,))
        end
        elapsed = time() - start
        runtime += elapsed

        # report loop stats
        score = get_score(trace)
        println((
            i, score,
            trace[:slope],
            trace[:intercept],
            sqrt(exp(trace[:inlier_std])),
            sqrt(exp(trace[:outlier_std]))))
    end

    score = get_score(trace)
    return (
        n,
        runtime,
        score,
        trace[:slope],
        trace[:intercept],
        trace[:inlier_std],
        trace[:outlier_std])
end

#################
# run inference #
#################

do_inference(10)

results = do_inference(200)
fname = "static_mh.results.csv"
open(fname, "a") do f
    write(f, join(results, ',') * '\n')
end
