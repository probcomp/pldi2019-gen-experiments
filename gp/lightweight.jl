include("cov_tree.jl")

# Model.

@gen function covariance_prior()
    node_type = @trace(categorical(node_dist), :type)

    if node_type == CONSTANT
        param = @trace(uniform_continuous(0, 1), :param)
        node = Constant(param)

    # linear kernel
    elseif node_type == LINEAR
        param = @trace(uniform_continuous(0, 1), :param)
        node = Linear(param)

    # squared exponential kernel
    elseif node_type == SQUARED_EXP
        length_scale= @trace(uniform_continuous(0, 1), :length_scale)
        node = SquaredExponential(length_scale)

    # periodic kernel
    elseif node_type == PERIODIC
        scale = @trace(uniform_continuous(0, 1), :scale)
        period = @trace(uniform_continuous(0, 1), :period)
        node = Periodic(scale, period)

    # plus combinator
    elseif node_type == PLUS
        left = @trace(covariance_prior(), :left)
        right = @trace(covariance_prior(), :right)
        node = Plus(left, right)

    # times combinator
    elseif node_type == TIMES
        left = @trace(covariance_prior(), :left)
        right = @trace(covariance_prior(), :right)
        node = Times(left, right)

    # unknown node type
    else
        error("Unknown node type: $node_type")
    end

    return node
end

@gen function model(xs::Vector{Float64})
    n = length(xs)
    covariance_fn::Node = @trace(covariance_prior(), :tree)
    noise = @trace(gamma(1, 1), :noise) + 0.01
    cov_matrix = compute_cov_matrix_vectorized(covariance_fn, noise, xs)
    @trace(mvnormal(zeros(n), cov_matrix), :ys)
    return covariance_fn
end

# Proposals and inference.

@gen function random_node_path_biased(node::Node)
    p_stop = isa(node, LeafNode) ? 1.0 : 0.5
    if @trace(bernoulli(p_stop), :stop)
        return :tree
    else
        (next_node, direction) = @trace(bernoulli(0.5), :left) ? (node.left, :left) : (node.right, :right)
        rest_of_path = @trace(random_node_path_biased(next_node), :rest_of_path)
        if isa(rest_of_path, Pair)
            return :tree => direction => rest_of_path[2]
        else
            return :tree => direction
        end
    end
end

@gen function random_node_path_unbiased(node::Node)
    p_stop = isa(node, LeafNode) ? 1.0 : 1/size(node)
    if @trace(bernoulli(p_stop), :stop)
        return :tree
    else
        p_left = size(node.left) / (size(node) - 1)
        (next_node, direction) = @trace(bernoulli(p_left), :left) ? (node.left, :left) : (node.right, :right)
        rest_of_path = @trace(random_node_path_unbiased(next_node), :rest_of_path)
        if isa(rest_of_path, Pair)
            return :tree => direction => rest_of_path[2]
        else
            return :tree => direction
        end
    end
end

@gen function random_node_path_root(node::Node)
    return :tree
end

@gen function regen_random_subtree(prev_trace)
    @trace(covariance_prior(), :new_subtree)
    @trace(random_node_path_unbiased(get_retval(prev_trace)), :path)
end

function subtree_involution(trace, fwd_assmt::ChoiceMap, path_to_subtree, proposal_args::Tuple)
    # Need to return a new trace, a bwd_assmt, and a weight.
    model_assmt = get_choices(trace)
    bwd_assmt = choicemap()
    set_submap!(bwd_assmt, :path, get_submap(fwd_assmt, :path))
    set_submap!(bwd_assmt, :new_subtree, get_submap(model_assmt, path_to_subtree))
    new_trace_update = choicemap()
    set_submap!(new_trace_update, path_to_subtree, get_submap(fwd_assmt, :new_subtree))
    (new_trace, weight, _, _) =
        update(trace, get_args(trace), (NoChange(),), new_trace_update)
    (new_trace, bwd_assmt, weight)
end

function run_mcmc(trace, iters::Int)
    for iter=1:iters
        (trace, _) = mh(trace, regen_random_subtree, (), subtree_involution)
        (trace, _) = mh(trace, select(:noise))
    end
    return trace
end

# Initialize trace and extract variables.

function initialize_trace(xs::Vector{Float64}, ys::Vector{Float64})
    constraints = choicemap()
    constraints[:ys] = ys
    (trace, _) = generate(model, (xs,), constraints)
    return trace
end

function extract_cov_noise(trace)
    cov = get_retval(trace)
    noise = trace[:noise] + 0.01
    return (cov, noise)
end
