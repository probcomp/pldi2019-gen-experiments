using PyCall

rpyc = pyimport("rpyc")

const RIG = "rig"
const ARM_ELBOW_R = "arm elbow_R"
const ARM_ELBOW_L = "arm elbow_L"
const HIP = "hip"
const HEEL_R = "heel_R"
const HEEL_L = "heel_L"

##################
# blender client #
##################

mutable struct BlenderClient
    conn::PyObject
    root::PyObject
end

function BlenderClient(blender_path::String, model_path::String, port::Int)
    host = "localhost"
    conn = rpyc.connect(host, port)
    root = conn.root
    BlenderClient(conn, root)
end

function setup_for_depth!(client::BlenderClient)
    client.root.setup_for_depth()
    nothing
end

function setup_for_wireframe!(client::BlenderClient)
    client.root.setup_for_wireframe()
    nothing
end

function set_resolution!(client::BlenderClient, x, y)
    client.root.set_resolution(x, y)
end

function add_plane!(client::BlenderClient, object::String, loc::Point3, rot::Point3, scale::Point3)
    client.root.add_plane(object, tup(loc), tup(rot), tup(scale))
end

function set_object_location!(client::BlenderClient, object::String, point::Point3)
    client.root.set_object_location(object, tup(point))
    nothing
end

function set_object_rotation_euler!(client::BlenderClient, object::String, point::Point3)
    client.root.set_object_rotation_euler(object, tup(point))
    nothing
end

function set_object_scale!(client::BlenderClient, object::String, point::Point3)
    client.root.set_object_scale(object, tup(point))
    nothing
end

function get_object_location!(client::BlenderClient, object::String)
    Point3(client.root.get_object_location(object))
end

function get_object_rotation_euler(client::BlenderClient, object::String)
    Point3(client.root.get_object_rotation_euler(object))
end

function get_object_scale(client::BlenderClient, object::String)
    Point3(client.root.get_object_scale(object))
end

function set_bone_location(client::BlenderClient, object::String, bone::String, location::Point3)
    client.root.set_bone_location(object, bone, tup(location))
    nothing
end

function set_bone_rotation_euler!(client::BlenderClient, object::String, bone::String, rotation_euler::Point3)
    client.root.set_bone_rotation_euler(object, bone, tup(rotation_euler))
    nothing
end

function get_bone_location(client::BlenderClient, object::String, bone::String)
    Point3(client.root.get_bone_location(object, bone))
end

function get_bone_rotation_euler(client::BlenderClient, object::String, bone::String)
    Point3(client.root.get_bone_rotation_euler(object, bone))
end

function render(client::BlenderClient, filepath)
    client.root.render(filepath)
end

function get_body_pose(client::BlenderClient)
    BodyPose(
        get_object_rotation_euler(client, RIG),
        get_bone_location(client, RIG, ARM_ELBOW_R),
        get_bone_location(client, RIG, ARM_ELBOW_L),
        get_bone_rotation_euler(client, RIG, ARM_ELBOW_R),
        get_bone_rotation_euler(client, RIG, ARM_ELBOW_L),
        get_bone_location(client, RIG, HIP),
        get_bone_location(client, RIG, HEEL_R),
        get_bone_location(client, RIG, HEEL_L))
end

function set_body_pose!(client::BlenderClient, pose::BodyPose)
    pose_dict = Dict(
        "rotation" => tup(pose.rotation),
        "elbow_r_loc" => tup(pose.elbow_r_loc),
        "elbow_l_loc" => tup(pose.elbow_l_loc),
        "elbow_r_rot" => tup(pose.elbow_r_rot),
        "elbow_l_rot" => tup(pose.elbow_l_rot),
        "hip_loc" => tup(pose.hip_loc),
        "heel_r_loc" => tup(pose.heel_r_loc),
        "heel_l_loc" => tup(pose.heel_l_loc))
    client.root.set_body_pose(pose_dict)
end
