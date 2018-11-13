include("../shared.jl")

using Gen
import Random

using FunctionalCollections
using ReverseDiff

############################
# reverse mode AD for fill #
############################

function Base.fill(x::ReverseDiff.TrackedReal{V,D,O}, n::Integer) where {V,D,O}
    tp = ReverseDiff.tape(x)
    out = ReverseDiff.track(fill(ReverseDiff.value(x), n), V, tp)
    ReverseDiff.record!(tp, ReverseDiff.SpecialInstruction, fill, (x, n), out)
    return out
end

@noinline function ReverseDiff.special_reverse_exec!(
        instruction::ReverseDiff.SpecialInstruction{typeof(fill)})
    x, n = instruction.input
    output = instruction.output
    ReverseDiff.istracked(x) &&
        ReverseDiff.increment_deriv!(x, sum(ReverseDiff.deriv(output)))
    ReverseDiff.unseed!(output)
    return nothing
end

@noinline function ReverseDiff.special_forward_exec!(
        instruction::ReverseDiff.SpecialInstruction{typeof(fill)})
    x, n = instruction.input
    ReverseDiff.value!(instruction.output, fill(ReverseDiff.value(x), n))
    return nothing
end

#########
# model #
#########

@compiled @gen function datum(x::Float64, @ad(inlier_std::Float64),
        @ad(outlier_std::Float64), @ad(slope::Float64), @ad(intercept::Float64))
    is_outlier::Bool = @addr(bernoulli(0.5), :z)
    std::Float64 = is_outlier ? inlier_std : outlier_std
    y::Float64 = @addr(normal(x * slope + intercept, sqrt(exp(std))), :y)
    return y
end

data = plate(datum)

function compute_argdiff(inlier_std_diff, outlier_std_diff, slope_diff, intercept_diff)
    if all([c == NoChoiceDiff() for c in [
            inlier_std_diff, outlier_std_diff, slope_diff, intercept_diff]])
        noargdiff
    else
        unknownargdiff
    end
end

@compiled @gen function model(xs::Vector{Float64})
    n::Int = length(xs)
    inlier_std::Float64 = @addr(normal(0, 2), :inlier_std)
    outlier_std::Float64 = @addr(normal(0, 2), :outlier_std)
    slope::Float64 = @addr(normal(0, 2), :slope)
    intercept::Float64 = @addr(normal(0, 2), :intercept)
    inlier_std_diff::Union{PrevChoiceDiff{Float64},NoChoiceDiff} = @change(:inlier_std)
    outlier_std_diff::Union{PrevChoiceDiff{Float64},NoChoiceDiff} = @change(:outlier_std)
    slope_diff::Union{PrevChoiceDiff{Float64},NoChoiceDiff} = @change(:slope)
    intercept_diff::Union{PrevChoiceDiff{Float64},NoChoiceDiff} = @change(:intercept)
    argdiff::Union{NoArgDiff,UnknownArgDiff} = compute_argdiff(
        inlier_std_diff, outlier_std_diff, slope_diff, intercept_diff)
    @addr(data(xs, fill(inlier_std, n), fill(outlier_std, n),
               fill(slope, n), fill(intercept, n)),
          :data, argdiff)
end

#######################
# inference operators #
#######################

@compiled @gen function slope_proposal(prev)
    slope::Float64 = get_assignment(prev)[:slope]
    @addr(normal(slope, .5), :slope)
end

@compiled @gen function intercept_proposal(prev)
    intercept::Float64 = get_assignment(prev)[:intercept]
    @addr(normal(intercept, .5), :intercept)
end

@compiled @gen function inlier_std_proposal(prev)
    inlier_std::Float64 = get_assignment(prev)[:inlier_std]
    @addr(normal(inlier_std, .5), :inlier_std)
end

@compiled @gen function outlier_std_proposal(prev)
    outlier_std::Float64 = get_assignment(prev)[:outlier_std]
    @addr(normal(outlier_std, .5), :outlier_std)
end

@compiled @gen function flip_z(z::Bool)
    @addr(bernoulli(z ? 0.0 : 1.0), :z)
end

data_proposal = at_dynamic(flip_z, Int)

@compiled @gen function is_outlier_proposal(prev, i::Int)
    prev_z::Bool = get_assignment(prev)[:data => i => :z]
    @addr(data_proposal(i, (prev_z,)), :data)
end

Gen.load_generated_functions()

##################
# run experiment #
##################


function do_inference(n)
    (xs, ys) = generate_dataset()
    observations = DynamicAssignment()
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
        elapsed = time() - start
        runtime += elapsed

        # report loop stats
        score = get_call_record(trace).score
        assignment = get_assignment(trace)
        println((score,
            assignment[:slope],
            assignment[:intercept],
            sqrt(exp(assignment[:inlier_std])),
            sqrt(exp(assignment[:outlier_std]))))
    end

    score = get_call_record(trace).score
    assignment = get_assignment(trace)
    return (
        n,
        runtime,
        score,
        assignment[:slope],
        assignment[:intercept],
        assignment[:inlier_std],
        assignment[:outlier_std])
end

#################
# run inference #
#################

do_inference(10)

results = do_inference(200)
fname = "compiled_mh.results.csv"
open(fname, "a") do f
    write(f, join(results, ',') * '\n')
end
