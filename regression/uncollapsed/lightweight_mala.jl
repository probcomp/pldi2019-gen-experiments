include("../shared.jl")

#########
# model #
#########

@gen function datum(x::Float64, @ad(inlier_std), @ad(outlier_std), @ad(slope), @ad(intercept))
    is_outlier = @addr(bernoulli(0.5), :z)
    std = is_outlier ? inlier_std : outlier_std
    y = @addr(normal(x * slope + intercept, sqrt(exp(std))), :y)
    return y
end

data = plate(datum)

@gen function model(xs::Vector{Float64})
    n = length(xs)
    inlier_std = exp(@addr(normal(0, 2), :inlier_std))
    outlier_std = exp(@addr(normal(0, 2), :outlier_std))
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
        :data, argdiff)
    return ys
end

#######################
# inference operators #
#######################

@gen function is_outlier_proposal(prev, i::Int)
    prev_z = get_assignment(prev)[:data => i => :z]
    @addr(bernoulli(prev_z ? 0.0 : 1.0), :data => i => :z)
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
push_leaf_node!(selection, :slope)
push_leaf_node!(selection, :intercept)
push_leaf_node!(selection, :inlier_std)
push_leaf_node!(selection, :outlier_std)

function do_inference(method, n)
    xs, ys = generate_dataset()
    observations = get_assignment(simulate(observer, (ys,)))

    # initial trace
    (trace, _) = generate(model, (xs,), observations)

    for i=1:n

        # step on the numeric parameters.
        if method == "mala"
            trace = mala(model, selection, trace, 0.0001)
        elseif method == "hmc"
            trace = hmc(model, selection, trace)
        else
            @assert false "Unknown method: $(method)"
        end

        # step on the outliers
        for j=1:length(xs)
            trace = mh(model, is_outlier_proposal, (j,), trace)
        end

        score = get_call_record(trace).score
        assignment = get_assignment(trace)
        println((
            ("score", score),
            ("slope", assignment[:slope]),
            ("intercept", assignment[:intercept]),
            ("inlier_std", sqrt(exp(assignment[:inlier_std]))),
            ("outlier_std", sqrt(exp(assignment[:outlier_std]))),
        ))
    end
end

do_inference("mala", 1000)
