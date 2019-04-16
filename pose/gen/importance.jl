include("../pose.jl")
include("../renderer.jl")
include("model.jl")
include("proposal.jl")

import FileIO
using Images: ImageCore
import Random
using JLD

# load the TF neural net parameters from disk
sess = get_session(net)
saver = train.Saver()
saver.restore(sess, "net.ckpt")

Random.seed!(1)

# generate ground truth image
#trace = simulate(generative_model, ())
#image = trace[:image]
#(ground_truth_latent_image, _, _) = get_retval(trace)
#save("image.jld", "image", image, "ground_truth_latent_image", ground_truth_latent_image)
image = load("image.jld", "image")
ground_truth_latent_image = load("image.jld", "ground_truth_latent_image")

const n = 10

# run sampling importance resampling with the trained proposal
function custom_proposal_importance_resampling()
    constraints = choicemap((:image, image))
    (inferred_trace, _) = importance_resampling(
        generative_model, (), constraints,
        proposal, (image,), 10)
    (latent_image, _, _) = get_retval(inferred_trace)
    latent_image
end

custom_proposal_latent_images = [custom_proposal_importance_resampling() for _=1:n]

# run sampling importance resampling with the prior proposal
function generic_proposal_importance_resampling()
    constraints = choicemap((:image, image))
    (inferred_trace, _) = importance_resampling(
        generative_model, (), constraints, 10)
    (latent_image, _, _) = get_retval(inferred_trace)
    latent_image
end

generic_proposal_latent_images = [generic_proposal_importance_resampling() for _=1:n]

# show the observed image and the reconstruction
top_row = hcat(ground_truth_latent_image, image, [zero(image) for i=1:(n-2)]...)
middle_row = hcat(generic_proposal_latent_images...)
bottom_row = hcat(custom_proposal_latent_images...)
combined = vcat(top_row, middle_row, bottom_row)
FileIO.save("importance.png", map(ImageCore.clamp01, combined))
