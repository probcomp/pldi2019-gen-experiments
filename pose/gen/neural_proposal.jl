using Gen
using GenTF
using PyCall
using Printf
using FileIO
import Random

@pyimport tensorflow as tf
@pyimport tensorflow.nn as nn
@pyimport tensorflow.train as train

function conv2d(x, W)
    nn.conv2d(x, W, (1, 1, 1, 1), "SAME")
end

function max_pool_2x2(x)
    nn.max_pool(x, (1, 2, 2, 1), (1, 2, 2, 1), "SAME")
end

function weight_variable(shape)
    initial = 0.001 * randn(shape...)
    tf.Variable(initial)
end

function bias_variable(shape)
    initial = fill(.1, shape...)
    tf.Variable(initial)
end

const num_output = 32

struct NetworkArchitecture
    num_conv1::Int
    num_conv2::Int
    num_conv3::Int
    num_fc::Int
end

const arch = NetworkArchitecture(32, 32, 64, 1024)

image_flat = tf.placeholder(tf.float)
image = tf.reshape(image_flat, [-1, width, height, 1])

# convolution + max pooling
W_conv1 = weight_variable([5, 5, 1, arch.num_conv1])
b_conv1 = bias_variable([arch.num_conv1])
h_conv1 = nn.relu(tf.add(conv2d(image, W_conv1), b_conv1))
h_pool1 = max_pool_2x2(h_conv1)

# convolution + max pooling
W_conv2 = weight_variable([5, 5, arch.num_conv1, arch.num_conv2])
b_conv2 = bias_variable([arch.num_conv2])
h_conv2 = nn.relu(tf.add(conv2d(h_pool1, W_conv2), b_conv2))
h_pool2 = max_pool_2x2(h_conv2)

# convolution + max pooling
W_conv3 = weight_variable([5, 5, arch.num_conv1, arch.num_conv2])
b_conv3 = bias_variable([arch.num_conv2])
h_conv3 = nn.relu(tf.add(conv2d(h_pool1, W_conv2), b_conv2))
h_pool3 = max_pool_2x2(h_conv2)
h_pool3_flat = tf.reshape(h_pool3, [-1, div(width, 8) * div(height, 8) * arch.num_conv3])

# fully connected layer
W_fc1 = weight_variable([div(width, 8) * div(height, 8) * arch.num_conv3, arch.num_fc])
b_fc1 = bias_variable([arch.num_fc])
h_fc1 = nn.relu(tf.add(tf.matmul(h_pool3_flat, W_fc1), b_fc1))

# output layer
W_fc2 = weight_variable([arch.num_fc, num_output])
b_fc2 = bias_variable([num_output])
output = nn.softmax(tf.add(tf.matmul(h_fc1, W_fc2), b_fc2), axis=1) # N x num_output

const sess = tf.Session()
const params = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2]
const net = TFFunction(params, [xs], output, sess)
const update = ParamUpdate(ADAM(1e-4, 0.9, 0.999, 1e-08), net)

@gen function neural_proposal_predict_beta((grad)(outputs::Vector{Float64}))

    # global rotation
    @trace(beta(exp(outputs[1]), exp(outputs[2])), :rot_z)

    # right elbow location
    @trace(beta(exp(outputs[3]), exp(outputs[4])), :elbow_r_loc_x)
    @trace(beta(exp(outputs[5]), exp(outputs[6])), :elbow_r_loc_y)
    @trace(beta(exp(outputs[7]), exp(outputs[8])), :elbow_r_loc_z)

    # left elbow location
    @trace(beta(exp(outputs[11]), exp(outputs[12])), :elbow_l_loc_x)
    @trace(beta(exp(outputs[13]), exp(outputs[14])), :elbow_l_loc_y)
    @trace(beta(exp(outputs[15]), exp(outputs[16])), :elbow_l_loc_z)

    # right elbow rotation
    @trace(beta(exp(outputs[9]), exp(outputs[10])), :elbow_r_rot_z)

    # left elbow rotation
    @trace(beta(exp(outputs[17]), exp(outputs[18])), :elbow_l_rot_z)

    # hip
    @trace(beta(exp(outputs[19]), exp(outputs[20])), :hip_loc_z)

    # right heel
    @trace(beta(exp(outputs[21]), exp(outputs[22])), :heel_r_loc_x)
    @trace(beta(exp(outputs[23]), exp(outputs[24])), :heel_r_loc_y)
    @trace(beta(exp(outputs[25]), exp(outputs[26])), :heel_r_loc_z)

    # left heel
    @trace(beta(exp(outputs[27]), exp(outputs[28])), :heel_l_loc_x)
    @trace(beta(exp(outputs[29]), exp(outputs[30])), :heel_l_loc_y)
    @trace(beta(exp(outputs[31]), exp(outputs[32])), :heel_l_loc_z)
end

@gen function proposal(image::Matrix{Float64})

    # run inference network
    image_flat = reshape(image, 1, width * height)
    outputs = @trace(net(image_flat), :network)

    # make prediction given inference network outputs
    @trace(neural_proposal_predict_beta(outputs[1,:]), :pose)
end

@gen function proposal_batched(images::Vector{Matrix{Float64}})

    # get images from input trace
    batch_size = length(images)
    images_flat = zeros(Float32, batch_size, width * height)
    for i=1:batch_size
        images_flat[i,:] = images[i][:]
    end

    # run inference network in batch
    outputs = @trace(net(images_flat), :network)

    # make prediction for each image given inference network outputs
    for i=1:batch_size
        @trace(neural_proposal_predict_beta(outputs[i,:]), :poses => i)
    end
end

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
            constraints = get_submap(example, :pose)
            (_, weight) = generate(proposal, (image,), constraints)
            total_weight += weight
        end
        est_objective = total_weight / epoch_size
        println("estimated objective: $est_objective")

        # do gradient updates for this epoch (for batch proposal)
        for minibatch=1:num_minibatch
            images = Matrix{Float64}[cm[:image] for cm in training_examples]
            constraints = vectorize_internal([get_submap(cm, :pose) for cm in training_examples])
            (trace, weight) = generate(proposal_batched, (images,), constraints)
            
            # increments gradient accumulators for batched proposal
            accumulate_param_gradients!(trace, nothing)
    
             # performs ADAM update and then resets gradient accumulators
            apply!(update)
        end

    end
end

function train_inference_network(num_batch::Int, batch_size::Int,
                                 num_minibatch::Int, minibatch_size::Int, 
                                 proposal::NeuralProposal, params_fname_trunk,
                                 session::Session, renderer; verbose=false)

    function input_extractor(teacher_choices_arr::Vector{Any})
        @assert length(teacher_choices_arr) == minibatch_size
        images = Matrix{Float64}[choices[:image] for choices in teacher_choices_arr]
        (images,)
    end

    function constraint_extractor(teacher_choices_arr::Vector{Any})
        @assert length(teacher_choices_arr) == minibatch_size
        poses = vectorize_internal([get_internal_node(c, :pose) for c in teacher_choices_arr])
        constraints = DynamicChoiceTrie()
        set_internal_node!(constraints, :poses, poses)
        constraints
    end

    function minibatch_callback(batch::Int, minibatch::Int, avg_score::Float64, verbose::Bool)
        if verbose
            println("batch $batch of $num_batch, minibatch $minibatch of $num_minibatch, avg score: $avg_score")
        end
        tf.run(session, proposal.network_update)
    end
    
    function batch_callback(batch::Int, verbose::Bool)
        as_default(GenTF.get_graph(proposal.network)) do
            saver = tf.train.Saver()
            params_fname = "$(params_fname_trunk)-$batch.jld"
            println("finished batch $batch, saving params to $params_fname...")
            save(saver, session, params_fname)
        end
    end

    conf = SGDTrainConf(
                num_batch,
                batch_size,
                num_minibatch,
                minibatch_size,
                input_extractor,
                constraint_extractor,
                minibatch_callback,
                batch_callback)
    sgd_train_batch(generative_model, (renderer,), proposal.neural_proposal_batched, conf, verbose)
end

