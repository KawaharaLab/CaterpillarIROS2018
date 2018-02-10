import bpy
import pickle
import json
import math
import sys

# ---------------------------
# This script can only be inside Blender.
# ---------------------------


class Renderer:
    def __init__(self):
        self._scene = bpy.context.scene

        # Clean
        self.delete_objects()

        self._objects = {}
        self._on_animation = False

    def load_simulation(self, file_path: str, file_type: str):
        """
        run simulation saved in file_path

        file_path: where simulation result is saved
        file_type: simulation type. pickle and json is supported
        """
        if file_type == 'json':
            simulation_protocol = self._load_json(file_path)
        elif file_type == 'pickle':
            simulation_protocol = self._load_pickle(file_path)
        else:
            raise Exception("invalid file type")

        # Draw objects
        objects = simulation_protocol['objects']
        for (r, pos, name) in objects:
            self.add_shpere(r, pos, name)

        # Run steps
        frames = simulation_protocol['frames']
        # self.start_simulation(1)
        self.start_simulation(1)
        for f, motions in frames.items():
            self.next_frame()
            for (n, pos) in motions:
                self.move_object(n, pos)
        self.end_simulation()

    def _load_json(self, file_path:str) -> dict:
        with open(file_path, 'r') as f:
            raw_sim_proc = json.load(f)
        sim_proc = {}
        sim_proc['objects'] = [(o['rad'], tuple(o['pos']), o['id']) for o in raw_sim_proc['objects']]
        sim_proc['frames'] = {int(i): [(o['id'], tuple(o['pos'])) for o in v] for i, v in  raw_sim_proc['frames'].items()}
        return sim_proc

    def _load_pickle(self, file_path:str) -> dict:
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def delete_objects(self):
        for item in self._scene.objects:
            self._scene.objects.unlink(item)
            bpy.data.objects.remove(item)

    def add_shpere(self, radius: float, location: tuple, name):
        assert name not in self._objects.keys()
        bpy.ops.mesh.primitive_uv_sphere_add(location=location, size=radius)
        self._objects[name] = bpy.context.object

    @property
    def objects_names(self):
        return self._objects.keys()

    def _add_plane(self):
        bpy.ops.mesh.primitive_plane_add(location=(0, 0, 0))
        self._plane = bpy.context.object
        self._plane.dimensions = (1000, 1000, 0)

    def _add_mesh(self):
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=100, y_subdivisions=100, radius=500)
        # mesh = bpy.context.active_object
        # mat = bpy.data.materials.new(name="MeshMaterial")
        # mesh.data.materials.append(mat)
        # bpy.context.object.active_material.diffuse_color = (255, 1, 1)

    def _add_light(self):
        lamp_data = bpy.data.lamps.new(name="lamp_data", type='POINT')
        lamp_obj = bpy.data.objects.new(name="lamp_obj", object_data=lamp_data)
        self._scene.objects.link(lamp_obj)
        lamp_obj.location = (0, 0, 10)

    def _add_camera(self):
        cam_data = bpy.data.cameras.new(name="cam")
        cam_ob = bpy.data.objects.new(name="Camera", object_data=cam_data)
        self._scene.objects.link(cam_ob)
        cam_ob.location = (0, 0, 10)
        cam_ob.rotation_euler = (0, 0, -math.pi)
        cam = bpy.data.cameras[cam_data.name]
        cam.lens = 10
        self._scene.camera = bpy.context.object

    def start_simulation(self, frames_step=50):
        self._add_plane()
        # self._add_mesh()
        # self._add_light()
        self._add_camera()
        self._on_simulation = True
        self._scene.frame_start = 0
        # self._scene.frame_end = 0
        self._frame_num = 0
        self._frames_step = frames_step

    def end_simulation(self):
        if not self._on_simulation:
            raise Exception("Simulation hasn't been started yet.")
        self._scene.frame_end = self._scene.frame_start + self._frame_num
        self._on_animation = False

    def move_object(self, obj_name, location: tuple):
        if not self._on_simulation:
            raise Exception("Simulation hasn't been started yet.")
        assert obj_name in self._objects.keys()
        self._objects[obj_name].location = location
        self._objects[obj_name].keyframe_insert(data_path="location", index=-1)

    def next_frame(self):
        """
        This method should be callled before any motion in the next frame.
        """
        if not self._on_simulation:
            raise Exception("Simulation hasn't been started yet.")
        self._scene.frame_set(self._frame_num)
        self._frame_num += self._frames_step


renderer = Renderer()
file_path = sys.argv[1]
file_type = sys.argv[2] if len(sys.argv) >= 3 else 'json'
print("file type {}".format(file_type))
renderer.load_simulation(file_path, file_type)
# renderer.start_simulation()
# renderer.end_simulation()
