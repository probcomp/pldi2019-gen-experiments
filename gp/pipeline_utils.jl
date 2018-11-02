using CSV
using Dates

using Statistics: mean

timestamp() = Dates.format(Dates.now(), "Ymdd-HMS")

"""Obtain filename to save results."""
function get_results_filename(shortname::String, n_test::Int, n_iters::Int,
        n_epochs::Int, sched::String, seed::Int)
    parts = [
        ("stamp",       "$(timestamp())"),
        ("shortname",   "$(shortname)"),
        ("ntest",       "$(n_test)"),
        ("iters",       "$(n_iters)"),
        ("epochs",      "$(n_epochs)"),
        ("schedule",    "$(sched)"),
        ("seed",        "$(seed)"),
    ]
    return join((join(part, "@") for part in parts), "_")
end

"""Rescale data linearly between [yl, yh]."""
function rescale_linear(xs::Vector{Float64}, yl::Float64, yh::Float64)
    xl = minimum(xs)
    xh = maximum(xs)
    slope = (yh - yl) / (xh - xl)
    intercept = yh - xh * slope
    return slope .* xs .+ intercept
end

"""Rescale dataset from path, linearly rescale, and split into test/."""
function load_dataset_from_path(path::String, n_test::Int)
    df = CSV.read(path, header=0)
    xs = rescale_linear(Vector{Float64}(df[1]), 0., 1.)
    ys = rescale_linear(Vector{Float64}(df[2]), -1., 1.)
    xs_train = xs[1:end-n_test]
    ys_train = ys[1:end-n_test]
    xs_test = xs[end-n_test+1:end]
    ys_test = ys[end-n_test+1:end]
    return (xs_train, ys_train), (xs_test, ys_test)
end

"""Return iteration schedule over given number of epochs."""
function make_iteration_schedule(iters::Int, epochs::Int, sched::String)
    if sched == "constant"
        return [iters*1 for i in 1:epochs]
    elseif sched == "linear"
        return [iters*i for i in 1:epochs]
    elseif sched == "doubling"
        return [iters*2^i for i in 1:epochs]
    else
        @assert False "Unknown schedule: $(schedule)"
    end
end

"""Return the list of xs at which to probe."""
function make_xs_probe(xs::Vector{Float64}, N::Int)
    start, stop = minimum(xs), maximum(xs)
    result = range(start, stop=stop, length=N)
    return collect(result)
end

"""Compute root mean-squared difference."""
function compute_rmse(vs::Vector{Float64}, us::Vector{Float64})
    @assert size(vs) == Base.size(us)
    sq_err = (vs .- us).^2
    sq_err_mean = mean(sq_err)
    return sqrt(sq_err_mean)
end
