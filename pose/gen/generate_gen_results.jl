include("../pose.jl")
include("../renderer.jl")
include("model.jl")
include("proposal.jl")

import FileIO
using Images: ImageCore
import Random
using JLD
import JSON
using Statistics: median, mean, std, quantile

# load the TF neural net parameters from disk
sess = get_session(net)
saver = train.Saver()
saver.restore(sess, "net.ckpt")

const pose_addrs = [
    :rot_z,
    :elbow_r_loc_x, :elbow_r_loc_y, :elbow_r_loc_z,
    :elbow_l_loc_x, :elbow_l_loc_y, :elbow_l_loc_z,
    :elbow_r_rot_z, :elbow_l_rot_z,
    :hip_loc_z,
    :heel_r_loc_x, :heel_r_loc_y, :heel_r_loc_z,
    :heel_l_loc_x, :heel_l_loc_y, :heel_l_loc_z]

Random.seed!(1)

# generate ground truth image
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
    FileIO.save("image.png", map(ImageCore.clamp01, image))
    FileIO.save("ground_truth.png", map(ImageCore.clamp01, ground_truth_wireframe))
end
simulate_test_image()

# do inference experiment
Random.seed!(2)

function read_test_image()
    pose_dict = JSON.parsefile("ground_truth_pose.json")
    ground_truth_wireframe = load("image.jld", "ground_truth_wireframe")
    ground_truth_latent_image = load("image.jld", "ground_truth_latent_image")
    image = load("image.jld", "image")
    (pose_dict, ground_truth_latent_image, ground_truth_wireframe, image)
end
(ground_truth_pose_dict, ground_truth_latent_image, ground_truth_wireframe, image) = read_test_image()

# run sampling importance resampling with the trained proposal
function custom_proposal_importance_resampling()
    constraints = choicemap((:image, image))
    (trace, _) = importance_resampling(
        generative_model, (), constraints,
        proposal, (image,), 10)
    trace
end

# run sampling importance resampling with the prior proposal
function generic_proposal_importance_resampling()
    constraints = choicemap((:image, image))
    (trace, _) = importance_resampling(
        generative_model, (), constraints, 12)
    trace
end

function do_experiment(n::Int)

    # do custom proposal experiment
    custom_proposal_runtimes = []
    custom_proposal_wireframes = []
    custom_proposal_poses = []
    for i=1:n
        start = time_ns()
        trace = custom_proposal_importance_resampling()
        elapsed = (time_ns() - start) / 1e9
        pose = BodyPose(get_submap(get_choices(trace), :pose))
        push!(custom_proposal_poses, pose)
        wireframe = render(wireframe_renderer, pose)
        push!(custom_proposal_runtimes, elapsed)
        push!(custom_proposal_wireframes, wireframe)
    end

    # do generic proposal experiment
    generic_proposal_runtimes = []
    generic_proposal_wireframes = []
    generic_proposal_poses = []
    for i=1:n
        start = time_ns()
        trace = generic_proposal_importance_resampling()
        elapsed = (time_ns() - start) / 1e9
        pose = BodyPose(get_submap(get_choices(trace), :pose))
        push!(generic_proposal_poses, pose)
        wireframe = render(wireframe_renderer, pose)
        push!(generic_proposal_runtimes, elapsed)
        push!(generic_proposal_wireframes, wireframe)
    end

    # save results
    results = Dict(:generic => Dict(:times => generic_proposal_runtimes,
                                    :poses => generic_proposal_poses),
                   :custom => Dict(:times => custom_proposal_runtimes,
                                   :poses => custom_proposal_poses))
    open("results.json", "w") do f
        write(f, JSON.json(results))
    end

    # print timing results summary
    l = quantile(custom_proposal_runtimes, 0.25)
    m = median(custom_proposal_runtimes)
    u = quantile(custom_proposal_runtimes, 0.75)
    println("custom proposal: median: $m, lower quartile: $l, upper quartile: $u")
    l = quantile(generic_proposal_runtimes, 0.25)
    m = median(generic_proposal_runtimes)
    u = quantile(generic_proposal_runtimes, 0.75)
    println("generic proposal: median: $m, lower quartile: $l, upper quartile: $u")

    # save the inferred wireframes
    for i=1:n
        FileIO.save("generic-proposal-wireframe-$i.png", map(ImageCore.clamp01, generic_proposal_wireframes[i]))
        FileIO.save("custom-proposal-wireframe-$i.png", map(ImageCore.clamp01, custom_proposal_wireframes[i]))
    end
end

# first run, to trigger precompilation
println("first run..")
do_experiment(4)

# second run
println("second run..")
do_experiment(4)
