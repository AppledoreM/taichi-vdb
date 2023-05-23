import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/src")
from src.vdb_grid import *
from src.tools.particle_to_sdf import *
from src.vdb_viewer import *

ti.init(arch=ti.cuda, device_memory_GB=4, offline_cache=False, debug=False, kernel_profiler=True)
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




if __name__ == "__main__":
    # unittest.main()
    sdf_tool = ParticleToSdf(bounding_box, vdb_default_levels, 1000000)
    make_cube(cube_point_clouds)
    sdf_tool.particle_to_sdf_anisotropic(cube_point_clouds, 1000000, particle_radius)
    vdb_viewer = VdbViewer(sdf_tool.sdf, bounding_box, num_max_vertices=10000000, num_max_indices=30000000)

    while True:
        vdb_viewer.run_viewer_frame(sdf_tool.sdf, True)

    ti.profiler.print_kernel_profiler_info()
