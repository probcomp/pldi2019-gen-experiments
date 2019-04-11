using Gen
using DataFrames: DataFrame

#############
# body pose #
#############

struct Point3
    x::Float64
    y::Float64
    z::Float64
end

Point3(tup::Tuple{U,U,U}) where {U<:Real} = Point3(tup[1], tup[2], tup[3])

Base.:+(a::Point3, b::Point3) = Point3(a.x + b.x, a.y + b.y, a.z + b.z)
Base.:-(a::Point3, b::Point3) = Point3(a.x - b.x, a.y - b.y, a.z - b.z)
Base.norm(a::Point3) = sqrt(a.x * a.x + a.y * a.y + a.z * a.z)

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

function BodyPose(choices::ChoiceTrie)
    rot_z = choices[:rot_z]
    rotation = scale_rot(rot_z)
    elbow_r_loc_x = choices[:elbow_r_loc_x]
    elbow_r_loc_y = choices[:elbow_r_loc_y]
    elbow_r_loc_z = choices[:elbow_r_loc_z]
    elbow_r_loc = scale_elbow_r_loc(elbow_r_loc_x, elbow_r_loc_y, elbow_r_loc_z)
    elbow_l_loc_x = choices[:elbow_l_loc_x]
    elbow_l_loc_y = choices[:elbow_l_loc_y]
    elbow_l_loc_z = choices[:elbow_l_loc_z]
    elbow_l_loc = scale_elbow_l_loc(elbow_l_loc_x, elbow_l_loc_y, elbow_l_loc_z)
    elbow_r_rot_z = choices[:elbow_r_rot_z]
    elbow_r_rot = scale_elbow_r_rot(elbow_r_rot_z)
    elbow_l_rot_z = choices[:elbow_l_rot_z]
    elbow_l_rot = scale_elbow_l_rot(elbow_l_rot_z)
    hip_loc_z = choices[:hip_loc_z]
    hip_loc = scale_hip_loc(hip_loc_z)
    heel_r_loc_x = choices[:heel_r_loc_x]
    heel_r_loc_y = choices[:heel_r_loc_y]
    heel_r_loc_z = choices[:heel_r_loc_z]
    heel_r_loc = scale_heel_r_loc(heel_r_loc_x, heel_r_loc_y, heel_r_loc_z)
    heel_l_loc_x = choices[:heel_l_loc_x]
    heel_l_loc_y = choices[:heel_l_loc_y]
    heel_l_loc_z = choices[:heel_l_loc_z]
    heel_l_loc = scale_heel_l_loc(heel_l_loc_x, heel_l_loc_y, heel_l_loc_z)
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

@gen function body_pose_model()

    # global rotation
    rotation_x::Float64 = @trace(uniform(0, 1), :rot_z)
    rotation::Point3 = scale_rot(rotation_x)

    # right elbow location
    elbow_r_loc_x::Float64 = @trace(uniform(0, 1), :elbow_r_loc_x)
    elbow_r_loc_y::Float64 = @trace(uniform(0, 1), :elbow_r_loc_y)
    elbow_r_loc_z::Float64 = @trace(uniform(0, 1), :elbow_r_loc_z)
    elbow_r_loc::Point3 = scale_elbow_r_loc(elbow_r_loc_x, elbow_r_loc_y, elbow_r_loc_z)
    
    # left elbow location
    elbow_l_loc_x::Float64 = @trace(uniform(0, 1), :elbow_l_loc_x)
    elbow_l_loc_y::Float64 = @trace(uniform(0, 1), :elbow_l_loc_y)
    elbow_l_loc_z::Float64 = @trace(uniform(0, 1), :elbow_l_loc_z)
    elbow_l_loc::Point3 = scale_elbow_l_loc(elbow_l_loc_x, elbow_l_loc_y, elbow_l_loc_z)

    # right elbow rotation
    elbow_r_rot_z::Float64 = @trace(uniform(0, 1), :elbow_r_rot_z)
    elbow_r_rot::Point3 = scale_elbow_r_rot(elbow_r_rot_z)

    # left elbow rotation
    elbow_l_rot_z::Float64 = @trace(uniform(0, 1), :elbow_l_rot_z)
    elbow_l_rot::Point3 = scale_elbow_l_rot(elbow_l_rot_z)

    # hip
    hip_loc_z::Float64 = @trace(uniform(0, 1), :hip_loc_z)
    hip_loc::Point3 = scale_hip_loc(hip_loc_z)

    # right heel
    heel_r_loc_x::Float64 = @trace(uniform(0, 1), :heel_r_loc_x)
    heel_r_loc_y::Float64 = @trace(uniform(0, 1), :heel_r_loc_y)
    heel_r_loc_z::Float64 = @trace(uniform(0, 1), :heel_r_loc_z)
    heel_r_loc::Point3 = scale_heel_r_loc(heel_r_loc_x, heel_r_loc_y, heel_r_loc_z)

    # left heel
    heel_l_loc_x::Float64 = @trace(uniform(0, 1), :heel_l_loc_x)
    heel_l_loc_y::Float64 = @trace(uniform(0, 1), :heel_l_loc_y)
    heel_l_loc_z::Float64 = @trace(uniform(0, 1), :heel_l_loc_z)
    heel_l_loc::Point3 = scale_heel_l_loc(heel_l_loc_x, heel_l_loc_y, heel_l_loc_z)

    return BodyPose(
        rotation,
        elbow_r_loc,
        elbow_l_loc,
        elbow_r_rot,
        elbow_l_rot,
        hip_loc,
        heel_r_loc,
        heel_l_loc)::BodyPose
end

struct BodyPoseSceneModel end

function sample(::BodyPoseSceneModel)
    trace = simulate(body_pose_model, ())
    return get_call_record(trace).retval::BodyPose
end

#############################
# combined generative model #
#############################

include("../renderer.jl")

struct NoisyMatrix <: Gen.Distribution{Matrix{Float64}} end

const noisy_matrix = NoisyMatrix()

function Gen.logpdf(::NoisyMatrix, x::Matrix{Float64}, mu::Matrix{U}, noise::T) where {U<:Real,T<:Real}
    var = noise * noise
    diff = x - mu
    vec = diff[:]
    return -(vec' * vec)/ (2.0 * var) - 0.5 * log(2.0 * pi * var)
end

function Gen.random(::NoisyMatrix, mu::Matrix{U}, noise::T) where {U<:Real,T<:Real}
    mat = copy(mu)
    (w, h) = size(mu)
    for i=1:w
        for j=1:h
            mat[i, j] = mu[i, j] + randn() * noise
        end
    end
    return mat
end

(::NoisyMatrix)(mu, noise) = Gen.random(NoisyMatrix(), mu, noise)

const blender = "blender"
const blender_model = "HumanKTH.decimated.blend"
const renderer = BodyPoseDepthRenderer(width, height, blender, model, port)

@gen function generative_model()
    pose = @trace(body_pose_model(), :pose)
    image = render(renderer, pose)
    blurred = imfilter(image, Kernel.gaussian(1))
    observable = @trace(noisy_matrix(blurred, 0.1), :image)
    return (image, blurred, observable)
end

const width = 128
const height = 128
