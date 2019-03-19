include("../shared.jl")

#########
# model #
#########

# This is a higher-order Gen function returning a new gen function which is
# multiple independent applications.

@gen function model(xs::Vector{Float64})
    inlier_std = @trace(normal(0, 2), :inlier_std)
    outlier_std = @trace(normal(0, 2), :outlier_std)
    slope = @trace(normal(0, 2), :slope)
    intercept = @trace(normal(0, 2), :intercept)
    ys = Float64[]
    for (i, x) in enumerate(xs)
        is_outlier = @trace(bernoulli(.5), :data => i => :z)
        std = is_outlier ? inlier_std : outlier_std
        mean = x * slope + intercept
        y = @trace(normal(mean, sqrt(exp(std))), :data => i => :y)
        push!(ys, y)
    end
    return ys
end

#######################
# inference operators #
#######################

@gen function slope_proposal(prev)
    slope = prev[:slope]
    @trace(normal(slope, 0.5), :slope)
end

@gen function intercept_proposal(prev)
    intercept = prev[:intercept]
    @trace(normal(intercept, 0.5), :intercept)
end

@gen function inlier_std_proposal(prev)
    inlier_std = prev[:inlier_std]
    @trace(normal(inlier_std, 0.5), :inlier_std)
end

@gen function outlier_std_proposal(prev)
    outlier_std = prev[:outlier_std]
    @trace(normal(outlier_std, 0.5), :outlier_std)
end

@gen function is_outlier_proposal(prev, i::Int)
    z = prev[:data => i => :z]
    @trace(bernoulli(z ? 0.0 : 1.0), :data => i => :z)
end

##################
# run experiment #
##################

function do_inference(n)

    # prepare dataset
    xs, ys = load_dataset("../train.csv")
    observations = choicemap()
    for (i, y) in enumerate(ys)
        observations[:data => i => :y] = y
    end

    # initial trace
    (trace, _) = generate(model, (xs,), observations)

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
            (trace, _bool) = mh(trace, is_outlier_proposal, (j,))
        end
        elapsed = time() - start
        runtime += elapsed

        # report loop stats
        score = get_score(trace)
        println((
            score,
            trace[:slope],
            trace[:intercept],
            sqrt(exp(trace[:inlier_std])),
            sqrt(exp(trace[:outlier_std]))),
        )
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
fname = "lightweight_mh.results.csv"
open(fname, "a") do f
    write(f, join(results, ',') * '\n')
end
