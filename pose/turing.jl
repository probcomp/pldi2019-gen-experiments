using Turing
import Random
import Distributions
import ImageCore
import FileIO

Random.seed!(1)

################
# noisy matrix #
################

import Distributions: logpdf
import Random: rand

struct NoisyMatrix <: ContinuousMatrixDistribution
    mu::Matrix{Float64}
    noise::Float64
end

function logpdf(dist::NoisyMatrix, x::Matrix{Float64})
    var = dist.noise * dist.noise
    diff = x .- dist.mu
    vec = diff[:]
    return -(vec' * vec)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end

function rand(dist::NoisyMatrix)
    mat = copy(dist.mu)
    (w, h) = size(dist.mu)
    for i=1:w
        for j=1:h
            mat[i, j] = dist.mu[i, j] + randn() * dist.noise
        end
    end
    return mat
end

#########
# model #
#########


struct Point3
    x::Float64
    y::Float64
    z::Float64
end

Point3(tup::Tuple{U,U,U}) where {U<:Real} = Point3(tup[1], tup[2], tup[3])

Base.:+(a::Point3, b::Point3) = Point3(a.x + b.x, a.y + b.y, a.z + b.z)
Base.:-(a::Point3, b::Point3) = Point3(a.x - b.x, a.y - b.y, a.z - b.z)

tup(point::Point3) = (point.x, point.y, point.z)

struct BodyPose
    rotation::Point3
    elbow_r_loc::Point3
    elbow_l_loc::Point3
    elbow_r_rot::Point3
    elbow_l_rot::Point3
    hip_loc::Point3
    heel_r_loc::Point3
    heel_l_loc::Point3
end

function Base.:+(a::BodyPose, b::BodyPose)
    BodyPose(
        a.rotation + b.rotation,
        a.elbow_r_loc + b.elbow_r_loc,
        a.elbow_l_loc + b.elbow_l_loc,
        a.elbow_r_rot + b.elbow_r_rot,
        a.elbow_l_rot + b.elbow_l_rot,
        a.hip_loc + b.hip_loc,
        a.heel_r_loc + b.heel_r_loc,
        a.heel_l_loc + b.heel_l_loc)
end

include("renderer.jl")

###############
# scene prior #
###############

# rescale values from [0, 1] to another interval
scale(value, min, max) = min + (max - min) * value
unscale(scaled, min, max) = (scaled - min) / (max - min)

scale_rot(z) = Point3(0., 0., scale(z, -pi/4, pi/4))
unscale_rot(pt::Point3) = unscale(pt.z, -pi/4, pi/4)

scale_elbow_r_loc(x, y, z) = Point3(scale(x, -1, 0), scale(y, -1, 1), scale(z, -1, 1))
unscale_elbow_r_loc(pt::Point3) = (unscale(pt.x, -1, 0), unscale(pt.y, -1, 1), unscale(pt.z, -1, 1))

scale_elbow_r_rot(z) = Point3(0., 0., scale(z, 0, 2*pi))
unscale_elbow_r_rot(pt::Point3) = unscale(pt.z, 0, 2*pi)

scale_elbow_l_loc(x, y, z) = Point3(scale(x, 0, 1), scale(y, -1, 1), scale(z, -1, 1))
unscale_elbow_l_loc(pt::Point3) = (unscale(pt.x, 0, 1), unscale(pt.y, -1, 1), unscale(pt.z, -1, 1))

scale_elbow_l_rot(z) = Point3(0., 0., scale(z, 0, 2*pi))
unscale_elbow_l_rot(pt::Point3) = unscale(pt.z, 0, 2*pi)

scale_hip_loc(z) = Point3(0., 0., scale(z, -0.35, 0))
unscale_hip_loc(pt::Point3) = unscale(pt.z, -0.35, 0)

scale_heel_r_loc(x, y, z) = Point3(scale(x, -0.45, 0.1), scale(y, -1, 0.5), scale(z, -0.2, 0.2))
unscale_heel_r_loc(pt::Point3) = (unscale(pt.x, -0.45, 0.1), unscale(pt.y, -1, 0.5), unscale(pt.z, -0.2, 0.2))

scale_heel_l_loc(x, y, z) = Point3(scale(x, -0.1, 0.45), scale(y, -1, 0.5), scale(z, -0.2, 0.2))
unscale_heel_l_loc(pt::Point3) = (unscale(pt.x, -0.1, 0.45), unscale(pt.y, -1, 0.5), unscale(pt.z, -0.2, 0.2))

const blender = "blender"
const width = 128
const height = 128
const renderer = BodyPoseDepthRenderer(width, height, blender, "HumanKTH.decimated.blend", 59897)
const wireframe_renderer = BodyPoseWireframeRenderer(400, 400, blender, "HumanKTH.decimated.blend", 59898)

@model model(observed_image) = begin

    # global rotation
    rotation_x ~ Uniform(0, 1)
    rotation::Point3 = scale_rot(rotation_x)

    # right elbow location
    elbow_r_loc_x ~ Uniform(0, 1)
    elbow_r_loc_y ~ Uniform(0, 1)
    elbow_r_loc_z ~ Uniform(0, 1)
    elbow_r_loc::Point3 = scale_elbow_r_loc(elbow_r_loc_x, elbow_r_loc_y, elbow_r_loc_z)
    
    # left elbow location
    elbow_l_loc_x ~ Uniform(0, 1)
    elbow_l_loc_y ~ Uniform(0, 1)
    elbow_l_loc_z ~ Uniform(0, 1)
    elbow_l_loc::Point3 = scale_elbow_l_loc(elbow_l_loc_x, elbow_l_loc_y, elbow_l_loc_z)

    # right elbow rotation
    elbow_r_rot_z ~ Uniform(0, 1)
    elbow_r_rot::Point3 = scale_elbow_r_rot(elbow_r_rot_z)

    # left elbow rotation
    elbow_l_rot_z ~ Uniform(0, 1)
    elbow_l_rot::Point3 = scale_elbow_l_rot(elbow_l_rot_z)

    # hip
    hip_loc_z ~ Uniform(0, 1)
    hip_loc::Point3 = scale_hip_loc(hip_loc_z)

    # right heel
    heel_r_loc_x ~ Uniform(0, 1)
    heel_r_loc_y ~ Uniform(0, 1)
    heel_r_loc_z ~ Uniform(0, 1)
    heel_r_loc::Point3 = scale_heel_r_loc(heel_r_loc_x, heel_r_loc_y, heel_r_loc_z)

    # left heel
    heel_l_loc_x ~ Uniform(0, 1)
    heel_l_loc_y ~ Uniform(0, 1)
    heel_l_loc_z ~ Uniform(0, 1)
    heel_l_loc::Point3 = scale_heel_l_loc(heel_l_loc_x, heel_l_loc_y, heel_l_loc_z)

    pose = BodyPose(
        rotation,
        elbow_r_loc,
        elbow_l_loc,
        elbow_r_rot,
        elbow_l_rot,
        hip_loc,
        heel_r_loc,
        heel_l_loc)

    image::Matrix{Float64} = render(renderer, pose)
    blurred::Matrix{Float64} = imfilter(image, Kernel.gaussian(1))
    observed_image ~ NoisyMatrix(blurred, 0.1)

    return pose
end

function logsumexp(arr::AbstractArray{T}) where {T <: Real}
    max_arr = maximum(arr)
    max_arr + log(sum(exp.(arr .- max_arr)))
end

# load the test image
observed_image = convert(Matrix{Float64}, FileIO.load("observed.png"))
something = model(observed_image)

function get_pose(sample::Turing.Sample)
    s = sample.value
    rotation = scale_rot(s[:rotation_x])
    elbow_r_loc = scale_elbow_r_loc(s[:elbow_r_loc_x], s[:elbow_r_loc_y], s[:elbow_r_loc_z])
    elbow_l_loc = scale_elbow_l_loc(s[:elbow_l_loc_x], s[:elbow_l_loc_y], s[:elbow_l_loc_z])
    elbow_r_rot = scale_elbow_r_rot(s[:elbow_r_rot_z])
    elbow_l_rot = scale_elbow_l_rot(s[:elbow_l_rot_z])
    hip_loc = scale_hip_loc(s[:hip_loc_z])
    heel_r_loc = scale_heel_r_loc(s[:heel_r_loc_x], s[:heel_r_loc_y], s[:heel_r_loc_z])
    heel_l_loc = scale_heel_l_loc(s[:heel_l_loc_x], s[:heel_l_loc_y], s[:heel_l_loc_z])
    
    return BodyPose(
            rotation,
            elbow_r_loc,
            elbow_l_loc,
            elbow_r_rot,
            elbow_l_rot,
            hip_loc,
            heel_r_loc,
            heel_l_loc)
end

function get_log_score(sample::Turing.Sample)
    @assert sample.weight == 0.
    return sample[:lp]
end

function do_inference(n)
    chain = Turing.sample(something, IS(n))
    samples = chain.value2
    log_weights = map(get_log_score, samples)
    log_total_weight = logsumexp(log_weights)
    log_normalized_weights = log_weights .- log_total_weight
    weights = exp.(log_normalized_weights)
    dist = Distributions.Categorical(weights)
    idx = rand(dist)
    println("picked idx: $idx")
    sample = samples[idx]
    pose = get_pose(sample)
    return (pose, samples)
end

for i=1:4
    println("i: $i")
    start = time_ns() / 1e9
    (pose, _) = do_inference(5000)
    elapsed = (time_ns() /1e9 - start)
    wireframe = render(wireframe_renderer, pose)
    FileIO.save("wireframe-$i.png", map(ImageCore.clamp01, wireframe))
    println("elapsed (sec.): $elapsed")
end

# results:
#i: 1
#picked idx: 666
#elapsed (sec.): 98.89742918100092
#i: 2
#picked idx: 386
#elapsed (sec.): 89.83002206901438
#i: 3
#picked idx: 46
#elapsed (sec.): 91.25445953200688
#i: 4
#picked idx: 250
#elapsed (sec.): 88.41515733400593

# median elapsed: 90.54224080051063
