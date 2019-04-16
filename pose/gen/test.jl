include("../pose.jl")
include("../renderer.jl")
include("model.jl")
include("proposal.jl")

# load the TF neural net parameters from disk
sess = get_session(net)
saver = train.Saver()
saver.restore(sess, "./net.ckpt")


# generate test examples (choice maps)
epoch_size = 100
println("generating test data...")
training_examples = Vector{Any}(undef, epoch_size)
for i=1:epoch_size
    trace = simulate(generative_model, ())
    training_examples[i] = get_choices(trace)
end

# estimate objective function value, on this epoch (using unbatched)
println("estimating objective value...")
total_weight = 0.
for example in training_examples
    global total_weight
    image = example[:image]
    constraints = choicemap()
    set_submap!(constraints, :pose, get_submap(example, :pose))
    (_, weight) = generate(proposal, (image,), constraints)
    total_weight += weight
end
est_objective = total_weight / epoch_size
println("estimated objective: $est_objective")
