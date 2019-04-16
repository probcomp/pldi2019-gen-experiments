using Gen

# depends on: pose.jl, renderer.jl

function BodyPose(choices)
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
    rotation_x = @trace(uniform(0, 1), :rot_z)
    rotation = scale_rot(rotation_x)

    # right elbow location
    elbow_r_loc_x = @trace(uniform(0, 1), :elbow_r_loc_x)
    elbow_r_loc_y = @trace(uniform(0, 1), :elbow_r_loc_y)
    elbow_r_loc_z = @trace(uniform(0, 1), :elbow_r_loc_z)
    elbow_r_loc = scale_elbow_r_loc(elbow_r_loc_x, elbow_r_loc_y, elbow_r_loc_z)
    
    # left elbow location
    elbow_l_loc_x = @trace(uniform(0, 1), :elbow_l_loc_x)
    elbow_l_loc_y = @trace(uniform(0, 1), :elbow_l_loc_y)
    elbow_l_loc_z = @trace(uniform(0, 1), :elbow_l_loc_z)
    elbow_l_loc = scale_elbow_l_loc(elbow_l_loc_x, elbow_l_loc_y, elbow_l_loc_z)

    # right elbow rotation
    elbow_r_rot_z = @trace(uniform(0, 1), :elbow_r_rot_z)
    elbow_r_rot = scale_elbow_r_rot(elbow_r_rot_z)

    # left elbow rotation
    elbow_l_rot_z = @trace(uniform(0, 1), :elbow_l_rot_z)
    elbow_l_rot = scale_elbow_l_rot(elbow_l_rot_z)

    # hip
    hip_loc_z = @trace(uniform(0, 1), :hip_loc_z)
    hip_loc = scale_hip_loc(hip_loc_z)

    # right heel
    heel_r_loc_x = @trace(uniform(0, 1), :heel_r_loc_x)
    heel_r_loc_y = @trace(uniform(0, 1), :heel_r_loc_y)
    heel_r_loc_z = @trace(uniform(0, 1), :heel_r_loc_z)
    heel_r_loc = scale_heel_r_loc(heel_r_loc_x, heel_r_loc_y, heel_r_loc_z)

    # left heel
    heel_l_loc_x = @trace(uniform(0, 1), :heel_l_loc_x)
    heel_l_loc_y = @trace(uniform(0, 1), :heel_l_loc_y)
    heel_l_loc_z = @trace(uniform(0, 1), :heel_l_loc_z)
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

# dimension of the image
const width = 128
const height = 128

struct NoisyMatrix <: Gen.Distribution{Matrix{Float64}} end

const pixel_noise = NoisyMatrix()

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
const renderer = BodyPoseDepthRenderer(width, height, blender, blender_model, 62000)
const wireframe_renderer = BodyPoseWireframeRenderer(400, 400, blender, blender_model, 62001)

function render_depth_image(pose)
    render(renderer, pose)
end

function gaussian_blur(image)
    imfilter(image, Kernel.gaussian(1))
end

@gen function generative_model()
    pose = @trace(body_pose_model(), :pose)
    image = render_depth_image(pose)
    blurred = gaussian_blur(image)
    observable = @trace(pixel_noise(blurred, 0.1), :image)
    return (image, blurred, observable)
end
