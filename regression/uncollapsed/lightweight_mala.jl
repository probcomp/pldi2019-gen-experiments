include("../shared.jl")

#########
# model #
#########

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

@gen function is_outlier_proposal(prev, i::Int)
    z = prev[:data => i => :z]
    @trace(bernoulli(z ? 0.0 : 1.0), :data => i => :z)
end

##################
# run experiment #
##################

function do_inference(method, n)
    (xs, ys) = load_dataset("../train.csv")
    observations = choicemap()
    for (i, y) in enumerate(ys)
        observations[:data => i => :y] = y
    end

    # initial trace
    (trace, _) = generate(model, (xs,), observations)

    runtime = 0
    for i=1:n
        start  = time()
        # step on the numeric parameters.
        selection = select(:slope, :intercept, :inlier_std, :outlier_std)
        if method == "mala"
            (trace, _accept) = mala(trace, selection, 0.0001)
        elseif method == "hmc"
            (trace, _accept) = hmc(trace, selection)
        else
            @assert false "Unknown method: $(method)"
        end

        # step on the outliers
        for j=1:length(xs)
            (trace, _accept) = mh(trace, is_outlier_proposal, (j,))
        end
        elapsed = time() - start
        runtime += elapsed

        score = get_score(trace)
        println((
            ("score", score),
            ("slope", trace[:slope]),
            ("intercept", trace[:intercept]),
            ("inlier_std", sqrt(exp(trace[:inlier_std]))),
            ("outlier_std", sqrt(exp(trace[:outlier_std]))),
        ))
    end
    score = get_score(trace)
    return (
        n,
        runtime,
        score,
        trace[:slope],
        trace[:intercept],
        trace[:inlier_std],
        trace[:outlier_std]
    )
end

#################
# run inference #
#################

do_inference("mala", 10)

results = do_inference("mala", 1000)
fname = "lightweight_mala.results.csv"
open(fname, "a") do f
    write(f, join(results, ',') * '\n')
end
