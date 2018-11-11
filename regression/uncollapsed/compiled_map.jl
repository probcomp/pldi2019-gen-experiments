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
    (xs, ys) = generate_dataset()
    observations = DynamicAssignment()
    for (i, y) in enumerate(ys)
        observations[:data => i => :y] = y
    end

    # initial trace
    (trace, _) = generate(model, (xs,), observations)

    scores = Vector{Float64}(undef, n)

    for i=1:n
        for j=1:5
            trace = map_optimize(model, slope_intercept_selection, trace,
                max_step_size=0.1, min_step_size=1e-10)
            trace = map_optimize(model, std_selection, trace,
                max_step_size=0.1, min_step_size=1e-10)
        end

        # step on the outliers
        for j=1:length(xs)
            trace = mh(model, is_outlier_proposal, (j,), trace)
        end

        # report loop stats
        score = get_call_record(trace).score
        assignment = get_assignment(trace)
        println((score,
            assignment[:slope],
            assignment[:intercept],
            sqrt(exp(assignment[:inlier_std])),
            sqrt(exp(assignment[:outlier_std]))))
    end
    return scores
end

scores = do_inference(500)
