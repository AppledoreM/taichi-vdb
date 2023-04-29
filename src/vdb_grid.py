import taichi as ti
from utils.py import *



@ti.data_oriented
class VdbGrid:


    def VdbGrid(self, dtype : ti.dtypes, dim: int, tree_levels = [], grid_initializer = None):
        assert dim >= 2, "The vdb grid dimension must be at least 2."

        if len(tree_levels) == 0:
            tree_levels = [5, 5, 5, 5, 3]
        else:
            for level_dim in tree_levels:
                assert level_dim > 0, "The vdb level dimension must be greater than 0."

        self.tree_levels = tree_levels 
        self.num_levels = len(tree_levels)
        self.axis = ti.ij if self.dim == 2 else ti.ijk

        self.tree_level_pointers = []


        # Add all tree pointers inside 

        for i in range(len(tree_levels)):
            level = tree_levels[i]

            self.tree_level_pointers.append([ti.root.pointer(self.axis, (align_size(2 ** level // 4 + 1, 4), ) * self.dim])
            # initialize each level according to the user / default config
            for j in range(1, i):
                level = self.tree_levels[j]
                self.tree_level_pointers[i][-1].pointer(self.axis, (2 ** level, ) * self.dim)

            self.tree_level_pointers[i][-1].dense(self.axis, (4, ) * self.dim)

