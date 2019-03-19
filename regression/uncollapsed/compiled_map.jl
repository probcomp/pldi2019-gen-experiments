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

        # XXX To get this to work:
        # 1. Use adaptive gradients 1e-1 to 1e-10
        # 2. Make selection for slope/intercept and inlier_std/outlier_std.
        # 3. Make 10 steps instead of 1 step.
        trace = map_optimize(trace,
            select(:slope, :intercept, :inlier_std, :outlier_std),
            max_step_size=1e-6, min_step_size=1e-6)

        # step on the outliers
        for j=1:length(xs)
            (trace, _accept) = mh(trace, is_outlier_proposal, (j,))
        end
        elapsed = time() - start
        runtime += elapsed

        # report loop stats
        score = get_score(trace)
        println((score,
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
fname = "compiled_map.results.csv"
open(fname, "a") do f
    write(f, join(results, ',') * '\n')
end
