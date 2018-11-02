#!/usr/bin/env julia

using Dates

using ArgParse
using CSV
using JSON

using Statistics: mean


const PATH_RESULTS = joinpath(".", "resources", "results")

"""Return current timestamp."""
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

function infer_and_predict(trace, epoch::Int, iters::Int,
        xs_train::Vector{Float64}, ys_train::Vector{Float64},
        xs_test::Vector{Float64}, ys_test::Vector{Float64},
        xs_probe::Vector{Float64}, npred_in::Int, npred_out::Int)
    # Run MCMC inference and collect measurements and statistics.
    start = time()
    trace = run_mcmc(trace, iters)
    runtime = time() - start
    println("Completed $(iters) iterations in $(runtime) seconds")
    # Collect statistics.
    cov = get_call_record(trace).retval
    noise = get_assignment(trace)[:noise]
    # Run predictions.
    predictions_held_in = gp_predictive_samples(
        cov, noise, xs_train, ys_train, xs_probe, npred_in)
    predictions_held_out = gp_predictive_samples(
        cov, noise, xs_train, ys_train, xs_test, npred_out)
    log_predictive = compute_log_likelihood_predictive(
        cov, noise, xs_train, ys_train, xs_test, ys_test)
    predictions_held_in_mean = gp_predictive_samples(
        cov, noise, xs_train, ys_train, xs_probe)
    predictions_held_out_mean =gp_predictive_samples(
        cov, noise, xs_train, ys_train, xs_test)
    rmse = compute_rmse(ys_test, predictions_held_out_mean)
    results = Dict(
        "iters"                     => iters,
        "log_weight"                => 0,
        "log_joint"                 => 0,
        "log_likelihood"            => 0,
        "log_prior"                 => 0,
        "log_predictive"            => log_predictive,
        "predictions_held_in"       => predictions_held_in,
        "predictions_held_out"      => predictions_held_out,
        "predictions_held_in_mean"  => predictions_held_in_mean,
        "predictions_held_out_mean" => predictions_held_out_mean,
        "rmse"                      => rmse,
        "runtime"                   => runtime,
    )
    return trace, results
end

function run_pipeline(
        path_dataset::String,
        n_test::Int,
        shortname::String,
        iters::Int,
        epochs::Int,
        sched::String,
        nprobe_held_in::Int,
        npred_held_in::Int,
        npred_held_out::Int,
        chains::Int,
        seed::Int,
    )
    # Set experiment configuration parameters
    # Prepare the held-in and held-out data.
    dataset = load_dataset_from_path(path_dataset, n_test)
    xs_train, ys_train = dataset[1]
    xs_test, ys_test = dataset[2]
    xs_probe = make_xs_probe(xs_train, nprobe_held_in)

    # Prepare the iteration schedule.
    iterations = make_iteration_schedule(iters, epochs, sched)

    # XXX Major hack, add 1 epoch with 1 iteration for JIT.
    iterations = vcat([1], iterations)

    # Each chain will have a separate seed, and each iteration of the loop
    # shall correspond to an independent run. We use an inner-loop in Julia
    # as opposed to invoking independent processes to ensure that compilation
    # occurs only once.
    main_seed = (seed == -1) ? rand(1:2^32-1) : seed
    Random.seed!(main_seed)
    seeds = rand(1:2^32-1, chains)

    for chain_seed in seeds
        # Run the experiment.
        Random.seed!(chain_seed)
        trace = initialize_trace(xs_train, ys_train)
        statistics = []
        for (epoch, iter) in enumerate(iterations)
            trace, results = infer_and_predict(
                trace, epoch, iter, xs_train, ys_train, xs_test, ys_test,
                xs_probe, npred_held_in, npred_held_out,)
            append!(statistics, results)
        end
        # Save results to disk.
        fname = get_results_filename(
            shortname, n_test, iters, epochs, sched, chain_seed)
        fpath = joinpath(PATH_RESULTS, "$(fname).json")
        result = Dict(
                "path_dataset"          => path_dataset,
                "n_test"                => n_test,
                "xs_train"              => xs_train,
                "ys_train"              => ys_train,
                "xs_test"               => xs_test,
                "ys_test"               => ys_test,
                "xs_probe"              => xs_probe,
                "n_iters"               => iters,
                "n_epochs"              => epochs,
                "nprobe_held_in"        => nprobe_held_in,
                "npred_held_in"         => npred_held_in,
                "npred_held_out"        => npred_held_out,
                "seed"                  => seed,
                "statistics"            => statistics,
                "schedule"              => sched,
        )
        open(fpath, "w") do f
            write(f, JSON.json(result))
            println(fpath)
        end
    end
end


# Main runner.

settings = ArgParseSettings()
@add_arg_table settings begin
    "--n-test"
        arg_type = Int
        default = 1
    "--shortname"
        arg_type = String
        default = "gen"
    "--iters"
        arg_type = Int
        default = 1
    "--epochs"
        arg_type = Int
        default = 1
    "--nprobe-held-in"
        arg_type = Int
        default = 100
    "--npred-held-in"
        arg_type = Int
        default = 10
    "--npred-held-out"
        arg_type = Int
        default = 10
    "--sched"
        arg_type = String
        default = "constant"
    "--chains"
        arg_type = Int
        default = 4
    "--seed"
        arg_type = Int
        default = -1
    "path_dataset"
        arg_type = String
        required = true
    "mode"
        arg_type = String
        required = true
end

args = parse_args(settings)
println("Running experiment with args: $(args)")

# XXX There must be a better way to do this!
# This technique works because include operates on global scope:
# https://stackoverflow.com/questions/41288626/what-exactly-does-include-do?
#   > include always runs at the current global scope even when called from
#   > inside of a function.
if args["mode"] == "lightweight"
    include("lightweight.jl")
elseif args["mode"] == "incremental"
    include("incremental.jl")
else
    @assert false "Unknown mode: $(args["mode"])"
end
include("gp_predict.jl")

run_pipeline(
    args["path_dataset"],
    args["n-test"],
    "$(args["mode"])-$(args["shortname"])",
    args["iters"],
    args["epochs"],
    args["sched"],
    args["nprobe-held-in"],
    args["npred-held-in"],
    args["npred-held-out"],
    args["chains"],
    args["seed"],
    )
