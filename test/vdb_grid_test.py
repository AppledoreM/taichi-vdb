import os.path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/src")
import unittest
import taichi as ti
from src.vdb_grid import *

ti.init(arch=ti.cuda, device_memory_GB=8, offline_cache=False, debug=True)


@ti.data_oriented
class VdbGridTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dtype = ti.f32

        # Sparse Grid Provided by Taichi Infrastructure
        self.sparse_grid_default_levels = [2, 2, 2, 2, 2]
        self.sg = ti.field(self.dtype)
        self.sg_root = ti.root
        cur_node = self.sg_root
        for i in range(len(self.sparse_grid_default_levels)):
            level_size = self.sparse_grid_default_levels[i]
            if i + 1 == len(self.sparse_grid_default_levels):
                cur_node = cur_node.dense(ti.ijk, (level_size, level_size, level_size))
                cur_node.place(self.sg)
            else:
                cur_node = cur_node.pointer(ti.ijk, (level_size, level_size, level_size))

        # Implemented VDB Grid
        self.vdb_grid = VdbGrid(self.sparse_grid_default_levels)

    @ti.kernel
    def test_basic_read_write(self):
        for i, j, k in ti.ndrange(100, 100, 100):
            self.vdb_grid.set_value(i, j, k, i * j * k)

        for i, j, k in ti.ndrange(100, 100, 100):
            value = self.vdb_grid.get_value(i, j, k)
            assert value == i * j * k, "Value differs at ({}, {}, {}). Expected: {}, But Got: {}".format(i, j, k, i * j * k, value)


if __name__ == "__main__":
    unittest.main()
