import rpyc
import bpy
import time
import sys

class BlockingServer(rpyc.utils.server.Server):

    def _accept_method(self, sock):
        self._authenticate_and_serve_client(sock)


def addNode(tree, typ):
    node = tree.nodes.new(typ)
    return node

def linkNodes(tree, node1, node2, out=0, inp=0):
    tree = bpy.context.scene.node_tree
    tree.links.new(node1.outputs[out], node2.inputs[inp])
   
class BlenderService(rpyc.Service):

    def __init__(self):
        super().__init__()
        self.scene = bpy.data.scenes['Scene']

    def on_connect(self, conn):
        pass

    def on_disconnect(self, conn):
        pass

    def exposed_setup_for_depth(self):
        self.scene.render.engine = "CYCLES"
        self.scene.cycles.samples=1
        self.scene.render.tile_x=25
        self.scene.render.tile_y=25
        self.scene.cycles.max_bounces=0
        self.scene.cycles.caustics_reflective=False
        self.scene.cycles.caustics_refractive=False
        self.scene.render.layers[0].use_pass_normal = False
        self.scene.render.layers[0].use_pass_combined = False
        self.scene.render.use_compositing = True
        self.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        addNode(tree, 'CompositorNodeNormalize')
        render = tree.nodes['Render Layers']
        composite = tree.nodes['Composite']
        norm = tree.nodes['Normalize']
        linkNodes(tree, render, norm, out=2)
        linkNodes(tree, norm, composite)
        bpy.data.worlds['World'].horizon_color = (0, 0, 0)
        self.scene.render.resolution_percentage = 100

    def exposed_setup_for_wireframe(self):
        self.scene.render.engine = "BLENDER_RENDER"
        bpy.data.materials["sss"].type = "WIRE"

    def exposed_set_resolution(self, x, y):
        self.scene.render.resolution_x = x
        self.scene.render.resolution_y = y
        self.scene.render.resolution_percentage = 100

    def exposed_add_plane(self, name, location, rotation_euler, scale):
        bpy.ops.mesh.primitive_plane_add()
        obj = bpy.context.object
        obj.location = location
        obj.rotation_euler = rotation_euler
        obj.scale = scale
        obj.name = name
   
    def exposed_set_object_location(self, name, location):
        obj = bpy.data.objects[name]
        obj.location = location

    def exposed_set_object_rotation_euler(self, name, rotation_euler):
        obj = bpy.data.objects[name]
        obj.rotation_euler = rotation_euler

    def exposed_set_object_scale(self, name, scale):
        obj = bpy.data.objects[name]
        obj.scale = scale

    def exposed_get_object_location(self, name):
        return tuple(bpy.data.objects[name].location)

    def exposed_get_object_rotation_euler(self, name):
        return tuple(bpy.data.objects[name].rotation_euler)

    def exposed_get_object_scale(self, name):
        return tuple(bpy.data.objects[name].scale)

    def exposed_render(self, filepath):
        self.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)

    def exposed_set_bone_location(self, object_name, bone_name, location):
        bone = bpy.data.objects[object_name].pose.bones[bone_name]
        bone.location = location

    def exposed_set_bone_rotation_euler(self, object_name, bone_name, rotation_euler):
        bone = bpy.data.objects[object_name].pose.bones[bone_name]
        bone.rotation_mode = 'XYZ'
        bone.rotation_euler = rotation_euler

    def exposed_get_bone_location(self, object_name, bone_name):
        bone = bpy.data.objects[object_name].pose.bones[bone_name]
        return tuple(bone.location)

    def exposed_get_bone_rotation_euler(self, object_name, bone_name):
        bone = bpy.data.objects[object_name].pose.bones[bone_name]
        return tuple(bone.rotation_euler)

    def exposed_set_body_pose(self, pose):
        self.exposed_set_object_rotation_euler("rig", pose["rotation"])
        self.exposed_set_bone_location("rig", "arm elbow_R", pose["elbow_r_loc"])
        self.exposed_set_bone_location("rig", "arm elbow_L", pose["elbow_l_loc"])
        self.exposed_set_bone_rotation_euler("rig", "arm elbow_R", pose["elbow_r_rot"])
        self.exposed_set_bone_rotation_euler("rig", "arm elbow_L", pose["elbow_l_rot"])
        self.exposed_set_bone_location("rig", "hip", pose["hip_loc"])
        self.exposed_set_bone_location("rig", "heel_R", pose["heel_r_loc"])
        self.exposed_set_bone_location("rig", "heel_L", pose["heel_l_loc"])


# note: the model should already have been loaded
argv = sys.argv
argv = argv[argv.index("--") + 1:]
port = int(argv[0])
t = BlockingServer(BlenderService, port=port)
t.start()
