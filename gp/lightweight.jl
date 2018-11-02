include("cov_tree.jl")

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
        child1 = Gen.get_child(cur, 1, max_branch)
        child2 = Gen.get_child(cur, 2, max_branch)
        left = @splice(covariance_prior(child1))
        right = @splice(covariance_prior(child2))
        node = Plus(left, right)

    # times combinator
    elseif node_type == TIMES
        child1 = Gen.get_child(cur, 1, max_branch)
        child2 = Gen.get_child(cur, 2, max_branch)
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
