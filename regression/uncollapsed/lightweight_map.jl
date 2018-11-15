include("../shared.jl")

#########
# model #
#########

@gen function datum(x::Float64,
        @ad(inlier_std), @ad(outlier_std), @ad(slope), @ad(intercept))
    is_outlier = @addr(bernoulli(0.5), :z)
    std = is_outlier ? inlier_std : outlier_std
    y = @addr(normal(x * slope + intercept, sqrt(exp(std))), :y)
    return y
end

data = plate(datum)

@gen function model(xs::Vector{Float64})
    n = length(xs)
    inlier_std = @addr(normal(0, 2), :inlier_std)
    outlier_std = @addr(normal(0, 2), :outlier_std)
    slope = @addr(normal(0, 2), :slope)
    intercept = @addr(normal(0, 2), :intercept)
    @diff begin
        argdiff = noargdiff
        for addr in [:slope, :intercept, :inlier_std, :outlier_std]
            if !isnodiff(@choicediff(addr))
                argdiff = unknownargdiff
            end
        end
    end
    ys = @addr(
            data(
                xs,
                fill(inlier_std, n),
                fill(outlier_std, n),
                fill(slope, n),
                fill(intercept, n)),
            :data,
            argdiff)
    return ys
end

#######################
# inference operators #
#######################

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

selection = DynamicAddressSet()
push!(selection, :slope)
push!(selection, :intercept)
push!(selection, :inlier_std)
push!(selection, :outlier_std)

function do_inference(n)

    # prepare dataset
    xs, ys = load_dataset("../train.csv")
    observations = get_assignment(simulate(observer, (ys,)))

    # initial trace
    (trace, _) = generate(model, (xs,), observations)

    runtime = 0
    for i=1:n

        start = time()

        # XXX To get this to work:
        # 1. Use adaptive gradients 1e-1 to 1e-10
        # 2. Make selection for slope/intercept and inlier_std/outlier_std.
        # 3. Make 10 steps instead of 1 step.
        trace = map_optimize(model, selection,
            trace, max_step_size=1e-6, min_step_size=1e-6)

        # step on the outliers
        for j=1:length(xs)
            trace = mh(model, is_outlier_proposal, (j,), trace)
        end
        elapsed = time() - start
        runtime += elapsed

        # report loop stats
        score = get_call_record(trace).score
        assignment = get_assignment(trace)
        println((i, score,
            assignment[:slope],
            assignment[:intercept],
            sqrt(exp(assignment[:inlier_std])),
            sqrt(exp(assignment[:outlier_std]))))
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

do_inference(10)

results = do_inference(100)
fname = "lightweight_map.results.csv"
open(fname, "a") do f
    write(f, join(results, ',') * '\n')
end