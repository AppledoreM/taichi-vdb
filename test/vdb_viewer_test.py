
import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/src")
import unittest
import taichi as ti
from src.vdb_grid import *
from src.vdb_viewer import *
import random

ti.init(arch=ti.cuda, device_memory_GB=4, offline_cache=False, debug=False, kernel_profiler=False)


sparse_grid_default_levels = [4, 4, 2]
bounding_box = [ti.Vector([0.0, 0.0, 0.0]),
                ti.Vector([8.0, 8.0, 8.0])
               ]
voxel_bounding_box = [
                    ti.Vector([-1, -1, -1]) ,
                    ti.Vector([1024, 1024, 1024])
                ]
vdb_grid = VdbGrid(bounding_box, sparse_grid_default_levels)



@ti.kernel
def test_basic_write(offset: ti.template()):
    for i, j, k in ti.ndrange(2, 2, 2):
        if i + offset[0] < 1024 and j + offset[1] < 1024 and offset[2] + k < 1024:
            if i + offset[0] >= 0 and j + offset[1] >= 0 and k + offset[2] >= 0:
                vdb_grid.set_value(offset[0] + i, offset[1] + j, offset[2] + k, 1)


def has_equal(a: ti.template(), b: ti.template()):
    return a[0] == b[0] or a[1] == b[1] or a[2] == b[2]

if __name__ == "__main__":
    # unittest.main()

    vdb_viewer = VdbViewer(vdb_grid, bounding_box, num_max_vertices=100000, num_max_indices=300000)
    counter = 0
    offset = bounding_box[0]
    speed = 1
    direction = ti.Vector([1, 1, 1])
    prev_offset = offset
    while True:
        vdb_viewer.run_viewer_frame(vdb_grid)
        counter = counter + 1

        if counter % 300 == 0:
            direction = ti.Vector([random.randint(-2, 2), random.randint(-2, 2), random.randint(-2, 2)])
            counter = 0

        if counter % 10 == 0:
            prev_offset = offset
            offset += direction * speed
            offset = ti.min(offset, voxel_bounding_box[1])
            offset = ti.max(offset, voxel_bounding_box[0])
            while prev_offset[0] == offset[0] and prev_offset[1] == offset[1] and prev_offset[2] == offset[2] or \
                has_equal(offset, voxel_bounding_box[1]) or has_equal(offset, voxel_bounding_box[0]):
                direction = ti.Vector([random.randint(-2, 2), random.randint(-2, 2), random.randint(-2, 2)])
                offset += direction * speed
                offset = ti.min(offset, voxel_bounding_box[1])
                offset = ti.max(offset, voxel_bounding_box[0])
            ti.deactivate_all_snodes()
            test_basic_write(ti.Vector([int(offset[0]), int(offset[1]), int(offset[2])]))
            print(f"Offset: {offset}")

            print(vdb_grid.data_wrapper.value2[16, 16, 16])
            # TODO: DEBUG THIS PART USING SNODE
            for i,
            vdb_grid.prune(1e-7)


    # ti.profiler.print_kernel_profiler_info()

