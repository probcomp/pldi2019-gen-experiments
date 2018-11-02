include("shared.jl")

@gen function covariance_prior(cur::Int)
    node_type = @addr(categorical(node_dist), (cur, :type))

    if node_type == CONSTANT
        param = @addr(uniform_continuous(0, 1), (cur, :param))
        node = Constant(param)

    # linear kernel
    elseif node_type == LINEAR
        param = @addr(uniform_continuous(0, 1), (cur, :param))
        node = Linear(param)

    # squared exponential kernel
    elseif node_type == SQUARED_EXP
        length_scale= @addr(uniform_continuous(0, 1), (cur, :length_scale))
        node = SquaredExponential(length_scale)

    # periodic kernel
    elseif node_type == PERIODIC
        scale = @addr(uniform_continuous(0, 1), (cur, :scale))
        period = @addr(uniform_continuous(0, 1), (cur, :period))
        node = Periodic(scale, period)

    # plus combinator
    elseif node_type == PLUS
        child1 = get_child(cur, 1, max_branch)
        child2 = get_child(cur, 2, max_branch)
        left = @splice(covariance_prior(child1))
        right = @splice(covariance_prior(child2))
        node = Plus(left, right)

    # times combinator
    elseif node_type == TIMES
        child1 = get_child(cur, 1, max_branch)
        child2 = get_child(cur, 2, max_branch)
        left = @splice(covariance_prior(child1))
        right = @splice(covariance_prior(child2))
        node = Times(left, right)

    # unknown node type
    else
        error("Unknown node type: $node_type")
    end

    return node
end

@gen function model(xs::Vector{Float64})
    n = length(xs)
    covariance_fn::Node = @addr(covariance_prior(1), :tree)
    noise = @addr(gamma(1, 1), :noise) + 0.01
    cov_matrix = compute_cov_matrix_vectorized(covariance_fn, noise, xs)
    @addr(mvnormal(zeros(n), cov_matrix), :ys)
    return covariance_fn
end

@gen function subtree_proposal(prev_trace, root::Int)
    @addr(covariance_prior(root), :tree)
end

@gen function noise_proposal(prev_trace)
    @addr(gamma(1, 1), :noise)
end

function correction(prev_trace, new_trace)
    prev_size = size(get_call_record(prev_trace).retval)
    new_size = size(get_call_record(new_trace).retval)
    log(prev_size) - log(new_size)
end

function initialize_trace(xs::Vector{Float64}, ys::Vector{Float64})
    constraints = DynamicAssignment()
    constraints[:ys] = ys
    (trace, _) = generate(model, (xs,), constraints)
    return trace
end

function run_mcmc(trace, iters::Int)
    for iter=1:iters
        covariance_fn = get_call_record(trace).retval
        root = pick_random_node_unbiased(covariance_fn, 1, max_branch)
        trace = mh(model, subtree_proposal, (root,), trace, correction)
        trace = mh(model, noise_proposal, (), trace)
    end
    return trace
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
    return Dict(
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
end

function run_pipeline()
    path_dataset = "resources/matlab_timeseries/01-airline.csv"
    n_test = 20
    shortname = "nothing"
    iters = 1000
    epochs = 1
    sched = "constant"
    nprobe_held_in = 100
    npred_held_in = 10
    npred_held_out = 10
    iterations = make_iteration_schedule(iters, epochs, sched)
    chains = 4

    dataset = load_dataset_from_path(path_dataset, n_test)
    xs_train, ys_train = dataset[1]
    xs_test, ys_test = dataset[2]

    xs_probe = make_xs_probe(xs_train, nprobe_held_in)
    seeds = rand(1:2^32-1, chains)

    for seed in seeds
        Random.seed!(seed)
        trace = initialize_trace(xs_train, ys_train)
        statistics = [
            infer_and_predict(trace, epoch, iter, xs_train, ys_train,
                xs_test, ys_test, xs_probe, npred_held_in, npred_held_out)
            for (epoch, iter) in enumerate(iterations)
        ]
        for stats in statistics
            println(stats)
        end

        ys_pred_held_in = statistics[end]["predictions_held_in_mean"]
        ys_pred_held_out = statistics[end]["predictions_held_out_mean"]

        fig, ax = PyPlot.subplots()
        ax[:scatter](xs_train, ys_train, marker="x", color="k", label="Observed Data")
        ax[:scatter](xs_test, ys_test, marker="x", color="r", label="Test Data")
        ax[:plot](xs_probe, ys_pred_held_in, color="g")
        ax[:plot](xs_test, ys_pred_held_out, color="g")
        ax[:set_xlim]((0, 1.5))
        ax[:set_ylim]((-1.5, 1.5))
        ax[:legend](true, loc="upper left")
        fig[:set_tight_layout](true)
        fig[:set_size_inches](6,6)
        fig[:savefig]("resources/lightweight_seed@$(seed).png")
    end
end

run_pipeline()
