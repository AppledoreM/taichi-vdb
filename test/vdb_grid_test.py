import sys

sys.path.append("../src/")
import unittest
import taichi as ti
from src.vdb_grid import *

ti.init(arch=ti.cuda, device_memory_GB=4)


@ti.data_oriented
class VdbGridTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dtype = ti.f32

        # Sparse Grid Provided by Taichi Infrastructure
        self.sparse_grid_default_levels = [2 ** 5, 2 ** 5, 2 ** 5, 2 ** 5, 2 ** 3]
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
        self.vdb_grid = VdbGrid(None, self.dtype)

    @ti.kernel
    def test_root_node(self):
        # 1. We set if root node is initialized correctly
        for i in range(VdbRootNode.root_max_size):
            assert self.vdb_grid.root_node.get_root_value_at(i) == self.vdb_grid.root_node.background_value
            assert self.vdb_grid.root_node.get_root_state_at(i) == False

        # 2. Test simple get & set functions
        root_node = self.vdb_grid.root_node
        for i in range(VdbRootNode.root_max_size):
            root_node.set_root_value_at(i, i)
        # Without setting the active bits, they should all be the background value
        for i in range(VdbRootNode.root_max_size):
            assert root_node.get_root_state_at(i) == root_node.background_value

        for i in range(VdbRootNode.root_max_size):
            if i & 1:
                root_node.set_root_state_at(i, 1)
        for i in range(VdbRootNode.root_max_size):
            if i & 1:
                assert root_node.get_root_state_at(i) == True
                assert root_node.get_root_value_at(i) == i
            else:
                assert root_node.get_root_state_at(i) == False
                assert root_node.get_root_value_at(i) == 0


if __name__ == "__main__":
    unittest.main()
