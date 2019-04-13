include("../pose.jl")
include("../renderer.jl")
include("model.jl")
include("proposal.jl")

import Random

function train_inference_network(num_epoch::Int, epoch_size::Int,
                                 num_minibatch::Int, minibatch_size::Int)
    
    for epoch=1:num_epoch

        # generate training examples (choice maps) for this epoch
        println("generating training data...")
        training_examples = Vector{Any}(undef, epoch_size)
        for i=1:epoch_size
            trace = simulate(generative_model, ())
            training_examples[i] = get_choices(trace)
        end

        # estimate objective function value, on this epoch (using unbatched)
        println("estimating objective value...")
        total_weight = 0.
        for example in training_examples
            image = example[:image]
            constraints = choicemap()
            set_submap!(constraints, :pose, get_submap(example, :pose))
            (_, weight) = generate(proposal, (image,), constraints)
            total_weight += weight
        end
        est_objective = total_weight / epoch_size
        println("estimated objective: $est_objective")

        # do gradient updates for this epoch (for batch proposal)
        for minibatch=1:num_minibatch
            minibatch_idx = Random.randperm(epoch_size)[1:minibatch_size]
            minibatch_examples = training_examples[minibatch_idx]
            images = Matrix{Float64}[cm[:image] for cm in minibatch_examples]
            constraints = choicemap()
            set_submap!(constraints, :poses, vectorize_internal([get_submap(cm, :pose) for cm in minibatch_examples]))
            (trace, weight) = generate(proposal_batched, (images,), constraints)
            
            # increments gradient accumulators for batched proposal
            accumulate_param_gradients!(trace, nothing)
    
             # performs ADAM update and then resets gradient accumulators
            apply!(update)
        end

        # save TF parameters for the network to disk
        sess = get_session(net)
        saver = train.Saver()
        saver.save(sess, "./net.ckpt")
    end
end

# train it
train_inference_network(100000, 1000, 300, 100)
