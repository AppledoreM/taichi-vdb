import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/src")
import taichi as ti

ti.init(arch=ti.cuda, device_memory_GB=4, offline_cache=True, debug=False, kernel_profiler=True)
# ti.init(arch=ti.cpu, device_memory_GB=10, offline_cache=False, debug=False, kernel_profiler=True)

from src.vdb_grid import *
from src.tools.particle_to_sdf import *
from src.vdb_viewer import *
from src.tools.volume_to_mesh import *

particle_radius = 0.005
voxel_dim = ti.Vector([particle_radius * 2, particle_radius * 2, particle_radius * 2])

# This is a cube of size 100 * 100 * 100
max_num_particles = 10000000
point_cloud = ti.Vector.field(3, ti.f32, max_num_particles)

shape_cube = 0
shape_sphere = 1
@ti.kernel
def make_shape(point_cloud : ti.template(), shape_id: ti.template()) -> ti.i32:
    counter = 0
    if ti.static(shape_id == shape_sphere):
        base_coord = ti.Vector([1., 1, 1])
        center = ti.Vector([2, 2, 2])
        for i, j, k in ti.ndrange(200, 200, 200):
            pos = base_coord + ti.Vector([i, j, k]) * particle_radius * 2
            if (pos - center).norm() < 1:
                index = ti.atomic_add(counter, 1)
                point_cloud[index] = pos
    elif ti.static(shape_id == shape_cube):
        base_coord = ti.Vector([0.25, 0.25, 0.25])
        for i, j, k in ti.ndrange(100, 100, 100):
            index = ti.atomic_add(counter, 1)
            pos = base_coord + ti.Vector([i, j, k]) * particle_radius * 2
            point_cloud[index] = pos

    return counter


vdb_default_levels = [4, 4, 4]
vdb_grid = VdbGrid(voxel_dim, vdb_default_levels)

num_vertices = ti.field(dtype=ti.i32, shape=())
num_indices = ti.field(dtype=ti.i32, shape=())

vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=3000000)
normal_buffer = ti.Vector.field(n=4, dtype=ti.f32, shape=3000000)
indices = ti.field(dtype=ti.i32, shape=5000000)


show_mesh = False
profile_epoch = 10
export_mesh = False

if __name__ == "__main__":

    sdf_tool = ParticleToSdf(voxel_dim, vdb_default_levels, max_num_particles)
    num_particles = make_shape(point_cloud, shape_sphere)
    for i in range(profile_epoch):
        sdf_tool.vdb.clear()
        sdf_tool.sdf.clear()
        print(f"{num_particles} in total.")
        sdf_tool.particle_to_sdf_anisotropic(point_cloud, num_particles, particle_radius)
        num_indices[None] = 0
        num_vertices[None] = 0
        vdb_grid.clear()
        VolumeToMesh.marching_cube(sdf_tool.sdf, vdb_grid, 0.01, num_vertices, vertices, num_indices, indices, normal_buffer)
        print(f"Generated {num_vertices[None]} vertices and {num_indices[None]} indices")

    if export_mesh:
        writer = ti.tools.PLYWriter(num_vertices=num_vertices[None], num_faces=num_indices[None]//3, face_type="tri")
        arr_vertices = vertices.to_numpy()[:num_vertices[None]]
        arr_indices = indices.to_numpy()[:num_indices[None]]
        arr_normals = normal_buffer.to_numpy()[:num_vertices[None]]
        writer.add_vertex_pos(arr_vertices[:, 0], arr_vertices[:, 1], arr_vertices[:, 2])
        writer.add_vertex_normal(arr_normals[:, 0], arr_normals[:, 1], arr_normals[:, 2])
        writer.add_faces(arr_indices)
        writer.export("mesh.ply")

    if show_mesh:
        window = ti.ui.Window("Vdb Viewer", (1280, 720))
        canvas = window.get_canvas()
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()
        camera.position(0, 0.5, -0.5)

        while True:
            vdb_grid.clear()
            camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.LMB)
            scene.set_camera(camera)
            scene.ambient_light((0.8, 0.8, 0.8))
            scene.point_light(pos=(1.5, 1.5, 1.5), color=(1, 1, 1))
            scene.point_light(pos=(3.5, 3, 3.5), color=(0.2, 0.2, 0.2))
            scene.point_light(pos=(0.5, 3, 0.5), color=(0.2, 0.2, 0.2))
            scene.mesh(vertices=vertices, indices=indices, vertex_count=num_vertices[None], index_count=num_indices[None],
                       normals=normal_buffer[:,:3])

            canvas.scene(scene)
            window.show()

    # vdb_viewer = VdbViewer(sdf_tool.sdf, bounding_box, num_max_vertices=10000000, num_max_indices=30000000)

    # while True:
    #     vdb_viewer.run_viewer_frame(sdf_tool.sdf, True)

    ti.profiler.print_kernel_profiler_info()
