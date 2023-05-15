import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/src")
import unittest
import taichi as ti
from src.vdb_grid import *

ti.init(arch=ti.cuda, device_memory_GB=4, offline_cache=True, debug=False, kernel_profiler=True)


# @ti.data_oriented
# class VdbGridTest(unittest.TestCase):
#     def setUp(self) -> None:
#         self.dtype = ti.f32
#
#         # Sparse Grid Provided by Taichi Infrastructure
#         self.sparse_grid_default_levels = [2, 2, 2, 2, 2]
#         self.sg = ti.field(self.dtype)
#         self.sg_root = ti.root
#         cur_node = self.sg_root
#         for i in range(len(self.sparse_grid_default_levels)):
#             level_size = self.sparse_grid_default_levels[i]
#             if i + 1 == len(self.sparse_grid_default_levels):
#                 cur_node = cur_node.dense(ti.ijk, (level_size, level_size, level_size))
#                 cur_node.place(self.sg)
#             else:
#                 cur_node = cur_node.pointer(ti.ijk, (level_size, level_size, level_size))
#
#         # Implemented VDB Grid
#         self.vdb_grid = VdbGrid(self.sparse_grid_default_levels)
#
#
#     @ti.kernel
#     def test_basic_read_write(self):
#         n = 1000
#         for i, j, k in ti.ndrange(n, n, n):
#             self.vdb_grid.set_value(i, j, k, i * j * k)
#
#         for i, j, k in ti.ndrange(n, n, n):
#             value = self.vdb_grid.get_value(i, j, k)
#             assert value == i * j * k, "Value differs at ({}, {}, {}). Expected: {}, But Got: {}".format(i, j, k, i * j * k, value)


sparse_grid_default_levels = [2, 2, 2, 2, 2]
vdb_grid = VdbGrid(sparse_grid_default_levels)

@ti.kernel
def test_basic_read_write():
    fill_dim = ti.Vector([1000, 1000, 500])
    query_dim = ti.Vector([1000, 1000, 1000])

    ti.loop_config(block_dim=512)
    for i, j, k in ti.ndrange(fill_dim[0], fill_dim[1], fill_dim[2]):
        vdb_grid.set_value(i, j, k, 4, i * j * k)

    ti.loop_config(block_dim=512)
    for i, j, k in ti.ndrange(query_dim[0], query_dim[1], query_dim[2]):
        value = vdb_grid.get_value(i, j, k)
        expected = i * j * k if k < fill_dim[2] else 0
        assert value == expected, "Value differs at ({}, {}, {}). Expected: {}, But Got: {}".format(i, j, k, i * j * k, value)

# @ti.kernel
# def simple_test():
#     for i in range(1):
#         vdb_grid.set_value(20, 42, 1, 4, 3)
#
#     for i in range(1):
#         for j in range(64):
#             print("{} {}".format(j, ti.is_active(vdb_grid.data_wrapper.node0, j)))
#         print(ti.is_active(vdb_grid.data_wrapper.node1, 1))
#
#     for i in range(1):
#         print(vdb_grid.get_value(20, 42, 1))



if __name__ == "__main__":
    # unittest.main()
    test_basic_read_write()
    ti.profiler.print_kernel_profiler_info()
