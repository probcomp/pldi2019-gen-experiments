include("../pose.jl")
include("../renderer.jl")
include("model.jl")
include("proposal.jl")

import FileIO
using Images: ImageCore
import Random
using JLD
import JSON

# load the TF neural net parameters from disk
sess = get_session(net)
saver = train.Saver()
saver.restore(sess, "net.ckpt")

Random.seed!(1)

# generate ground truth image
const pose_addrs = [
    :rot_z,
    :elbow_r_loc_x, :elbow_r_loc_y, :elbow_r_loc_z,
    :elbow_l_loc_x, :elbow_l_loc_y, :elbow_l_loc_z,
    :elbow_r_rot_z, :elbow_l_rot_z,
    :hip_loc_z,
    :heel_r_loc_x, :heel_r_loc_y, :heel_r_loc_z,
    :heel_l_loc_x, :heel_l_loc_y, :heel_l_loc_z]

function simulate_test_image()
    trace = simulate(generative_model, ())
    image = trace[:image]
    (ground_truth_latent_image, _, _) = get_retval(trace)
    pose_dict = Dict()
    for addr in pose_addrs
        pose_dict[addr] = trace[:pose => addr]
    end
    open("ground_truth_pose.json", "w") do f
        write(f, JSON.json(pose_dict))
    end
    ground_truth_wireframe = render(wireframe_renderer, BodyPose(pose_dict))
    save("image.jld",
        "image", image,
        "ground_truth_latent_image", ground_truth_latent_image,
        "ground_truth_wireframe", ground_truth_wireframe)
end
#simulate_test_image()

function read_test_image()
    pose_dict = JSON.parsefile("ground_truth_pose.json")
    ground_truth_wireframe = load("image.jld", "ground_truth_wireframe")
    ground_truth_latent_image = load("image.jld", "ground_truth_latent_image")
    image = load("image.jld", "image")
    (pose_dict, ground_truth_latent_image, ground_truth_wireframe, image)
end
(ground_truth_pose_dict, ground_truth_latent_image, ground_truth_wireframe, image) = read_test_image()

const n = 10

# run sampling importance resampling with the trained proposal
function custom_proposal_importance_resampling()
    constraints = choicemap((:image, image))
    (inferred_trace, _) = importance_resampling(
        generative_model, (), constraints,
        proposal, (image,), 10)
    pose = BodyPose(get_submap(get_choices(inferred_trace), :pose))
    render(wireframe_renderer, pose)
end

custom_proposal_wireframes = [custom_proposal_importance_resampling() for _=1:n]

# run sampling importance resampling with the prior proposal
function generic_proposal_importance_resampling()
    constraints = choicemap((:image, image))
    (inferred_trace, _) = importance_resampling(
        generative_model, (), constraints, 10)
    pose = BodyPose(get_submap(get_choices(inferred_trace), :pose))
    render(wireframe_renderer, pose)
end

generic_proposal_wireframes = [generic_proposal_importance_resampling() for _=1:n]

# show the observed image and the reconstruction
top_row = hcat(ground_truth_wireframe, ground_truth_latent_image, image, [zero(image) for i=1:(n-3)]...)
middle_row = hcat(generic_proposal_wireframes...)
bottom_row = hcat(custom_proposal_wireframes...)
combined = vcat(top_row, middle_row, bottom_row)
FileIO.save("importance.png", map(ImageCore.clamp01, combined))
