using FileIO
using ImageFiltering: imfilter, Kernel

include("blender_client.jl")

##################
# depth renderer #
##################

struct BodyPoseDepthRenderer
    width::Int
    height::Int
    blender_client::BlenderClient
end

function BodyPoseDepthRenderer(width, height, blender::String, model::String, port)
    client = BlenderClient(blender, model, port)
    setup_for_depth!(client)
    set_resolution!(client, width, height)
    set_object_location!(client, "Camera", Point3(0, -8.5, 5))
    set_object_rotation_euler!(client, "Camera", Point3(pi/3., 0, 0))
    add_plane!(client, "background", Point3(0,4,0), Point3(pi/3.,0,0), Point3(20,20,20))
    set_object_location!(client, RIG, Point3(0, 0, 0))
    set_object_rotation_euler!(client, RIG, Point3(0, 0, 0))
    set_object_scale!(client, RIG, Point3(3, 3, 3))
    add_plane!(client, "nearplane", Point3(-2,-4,0), Point3(pi/3.,0,0), Point3(0.1,0.1,0.1))
    setup_for_depth!(client)
    set_resolution!(client, width, height)
    return BodyPoseDepthRenderer(width, height, client)
end

Base.close(renderer::BodyPoseDepthRenderer) = close(renderer.blender_client)

function render(renderer::BodyPoseDepthRenderer, pose::BodyPose)
    tmp = tempname() * ".png"
    set_body_pose!(renderer.blender_client, pose)
    render(renderer.blender_client, tmp)
    img = FileIO.load(tmp)
    rm(tmp)
    return convert(Matrix{Float64}, img)
end

######################
# wireframe renderer #
######################

struct BodyPoseWireframeRenderer
    width::Int
    height::Int
    blender_client::BlenderClient
end

function BodyPoseWireframeRenderer(width, height, blender::String, model::String, port)
    client = BlenderClient(blender, model, port)
    setup_for_wireframe!(client)
    set_resolution!(client, width, height)
    set_object_location!(client, "Camera", Point3(0, -8.5, 5))
    set_object_rotation_euler!(client, "Camera", Point3(pi/3., 0, 0))
    add_plane!(client, "background", Point3(0,4,0), Point3(pi/3.,0,0), Point3(20,20,20))
    set_object_location!(client, RIG, Point3(0, 0, 0))
    set_object_rotation_euler!(client, RIG, Point3(0, 0, 0))
    set_object_scale!(client, RIG, Point3(3, 3, 3))
    add_plane!(client, "nearplane", Point3(-2,-4,0), Point3(pi/3.,0,0), Point3(0.1,0.1,0.1))
    setup_for_wireframe!(client)
    set_resolution!(client, width, height)
    return BodyPoseWireframeRenderer(width, height, client)
end

Base.close(renderer::BodyPoseWireframeRenderer) = close(renderer.blender_client)

function render(renderer::BodyPoseWireframeRenderer, pose::BodyPose)
    tmp = tempname() * ".png"
    set_body_pose!(renderer.blender_client, pose)
    render(renderer.blender_client, tmp)
    img = FileIO.load(tmp)
    rm(tmp)
    return convert(Matrix{Float64}, img)
end
