import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/src")
import taichi as ti

# ti.init(arch=ti.cuda, device_memory_GB=4, offline_cache=False, debug=False, kernel_profiler=True)
ti.init(arch=ti.cpu, device_memory_GB=10, offline_cache=False, debug=False, kernel_profiler=True)

from src.vdb_grid import *
from src.tools.particle_to_sdf import *
from src.vdb_viewer import *
from src.tools.volume_to_mesh import *

voxel_size = 0.01
particle_radius = 0.005

# This is a cube of size 100 * 100 * 100
cube_point_clouds = ti.Vector.field(3, ti.f32, 100 * 100 * 100)

@ti.kernel
def make_cube(point_cloud : ti.template()):
    base_coord = ti.Vector([0.25, 0.25, 0.25])
    for i, j, k in ti.ndrange(100, 100, 100):
        index = i + j * 100 + k * 100 * 100
        cube_point_clouds[index] = base_coord + ti.Vector([i, j, k]) * particle_radius * 2

vdb_default_levels = [4, 4, 2]
bounding_box = [
    ti.Vector([0.0, 0.0, 0.0]),
    ti.Vector([10.0, 10.0, 10.0])
]
vdb_grid = VdbGrid(bounding_box, vdb_default_levels)

num_vertices = ti.field(dtype=ti.i32, shape=())
num_indices = ti.field(dtype=ti.i32, shape=())

vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=1000000)
indices = ti.field(dtype=ti.i32, shape=3000000)




if __name__ == "__main__":

    sdf_tool = ParticleToSdf(bounding_box, vdb_default_levels, 1000000)
    make_cube(cube_point_clouds)
    sdf_tool.particle_to_sdf_anisotropic(cube_point_clouds, 1000000, particle_radius)
    VolumeToMesh.marching_cube(sdf_tool.sdf, vdb_grid, 0.5, num_vertices, vertices, num_indices, indices)
    print(f"Generated {num_vertices[None]} vertices and {num_indices[None]} indices")

    window = ti.ui.Window("Vdb Viewer", (1280, 720))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(0, 0.5, -0.5)

    while True:
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.LMB)
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
        scene.mesh(vertices=vertices, indices=indices, vertex_count=num_vertices[None], index_count=num_indices[None], show_wireframe=True)

        canvas.scene(scene)
        window.show()


    
    

    # vdb_viewer = VdbViewer(sdf_tool.sdf, bounding_box, num_max_vertices=10000000, num_max_indices=30000000)

    # while True:
    #     vdb_viewer.run_viewer_frame(sdf_tool.sdf, True)

    ti.profiler.print_kernel_profiler_info()
