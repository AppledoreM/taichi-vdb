import taichi as ti
from utils import *


class VdbNodeConfig:
    def __init__(self, log2x: int, log2y: int, log2z: int, dtype=ti.f32, child_node=None):
        vdb_assert(log2x > 0 and log2y > 0 and log2z > 0, "VdbNodeConfig needs all log2x, log2y, log2z to be positive.")
        self.dtype = dtype
        self.log2x = log2x
        self.log2y = log2y
        self.log2z = log2z
        self.slog2x = log2x
        self.slog2y = log2y
        self.slog2z = log2z
        if child_node is not None:
            self.slog2x += child_node.slog2x
            self.slog2y += child_node.slog2y
            self.slog2z += child_node.slog2z



@ti.data_oriented
class VdbInternalNode:

    def __init__(self, parent_node: ti.template(), node_config: VdbNodeConfig, dtype=ti.f32):
        snode_size = 1 << (node_config.log2x + node_config.log2y + node_config.log2z)
        mask_size = snode_size // 32
        vdb_assert(mask_size * 32 == snode_size, "Snode size is too small for an internal.")

        vdb_log("Vdb Grid Initialized with internal node shape ({}, {}, {}) and size: {}".format(node_config.log2x,
                                                                                        node_config.log2y,
                                                                                        node_config.log2z, snode_size))

        self.node_config = node_config
        self.child_node = parent_node.pointer(ti.i, snode_size)
        self.value_mask_node = parent_node.dense(ti.i, mask_size)
        self.child_mask_node = parent_node.dense(ti.i, mask_size)
        self.value_node = parent_node.dense(ti.i, snode_size)

        self.value_mask = ti.field(ti.i32)
        self.child_mask = ti.field(ti.i32)
        self.value = ti.field(dtype)

        self.value_mask_node.place(self.value_mask)
        self.child_mask_node.place(self.child_mask)
        self.value_node.place(self.value)

    @ti.func
    def is_child_active(self, index: ti.i32) -> bool:
        return ti.is_active(self.child_node, index)



@ti.data_oriented
class VdbRootNode:

    def __init__(self, node_config: VdbNodeConfig, dtype=ti.f32, background_value=0.0):
        self.dtype = dtype
        self.background_value = background_value
        self.root_snode_size = 1 << (node_config.log2x + node_config.log2y + node_config.log2z)
        self.node_config = node_config

        self.root = ti.root
        self.child_node = self.root.pointer(ti.i, self.root_snode_size)
        self.root_data_value = ti.field(self.dtype, shape=self.root_snode_size)

        root_mask_size = VdbRootNode.root_max_size // 32
        self.root_data_state = ti.field(ti.i32, shape=self.root_snode_size)

        vdb_log("Vdb Grid Initialized with root node shape ({}, {}, {}) and size: {}".format(node_config.log2x,
                                                                                                 node_config.log2y,
                                                                                                 node_config.log2z,
                                                                                             self.root_snode_size))

    @ti.func
    def get_root_state_at(self, index: ti.i32) -> bool:

        mapped_index = index // 32
        bit_index = index & 31

        return self.root_data_state[mapped_index] & (1 << bit_index)

    @ti.func
    def get_root_value_at(self, index: ti.i32):
        res = self.background_value
        if self.get_root_state_at(index):
            res = self.root_data_value[index]
        return res

    @ti.func
    def set_root_state_at(self, index: ti.i32, value: ti.i32):

        mapped_index = index // 32
        bit_index = index & 31

        if value:
            self.root_data_state[mapped_index] |= (1 << bit_index)
        else:
            self.root_data_state[mapped_index] &= ~(1 << bit_index)

    @ti.func
    def set_root_value_at(self, index: ti.i32, value):
        self.root_data_value[index] = value


class VdbLeafNode:

    @staticmethod
    def read_leaf_node_flag(leaf_node: ti.template()) -> ti.i64:
        return leaf_node

    def __init__(self, parent_node: ti.template(), node_config: VdbNodeConfig, dtype=ti.f32):
        self.value = ti.field(dtype)
        self.mask = ti.field(ti.i32)
        self.flag = ti.field(ti.i64)

        snode_size = 1 << (node_config.log2x + node_config.log2y + node_config.log2z)
        mask_size = snode_size // 32
        vdb_assert(mask_size * 32 == snode_size, "The size of the leaf nodes needs to be greater than 32.")
        vdb_log("Vdb Grid Initialized with leaf shape ({}, {}, {}) and size: {}".format(node_config.log2x,
                                                                                        node_config.log2y,
                                                                                        node_config.log2z, snode_size))

        self.node_config = node_config
        self.value_node = parent_node.dense(ti.i, snode_size)
        self.mask_node = parent_node.dense(ti.i, mask_size)
        self.flag_node = parent_node.dense(ti.i, 1)

        self.mask_node.place(self.mask)
        self.value_node.place(self.value)
        self.flag_node.place(self.flag)



@ti.data_oriented
class VdbGrid:

    def __init__(self, tree_levels=None, dtype=ti.f32, background_value=0.0):

        if tree_levels is None:
            tree_levels = [5, 5, 5, 5, 3]
        else:
            vdb_assert(isinstance(tree_levels, type([])), "Tree levels should be an array type.")
            for level_dim in tree_levels:
                vdb_assert(level_dim > 0, "The vdb level dimension must be greater than 0.")

        self.dtype = dtype
        node_config_list = []
        for i in range(len(tree_levels) - 1, -1, -1):
            if i + 1 == len(tree_levels):
                node_config_list.append(VdbNodeConfig(tree_levels[i], tree_levels[i], tree_levels[i], dtype))

            else:
                node_config_list.append(
                    VdbNodeConfig(tree_levels[i], tree_levels[i], tree_levels[i], dtype, node_config_list[-1]))

        node_config_list.reverse()
        self.num_tree_level = len(self.node_config_list)

        self.root_node = VdbRootNode(self.node_config_list[0], dtype, background_value)
        self.node_list = []

        cur_node = self.root_node.root_data_child_node
        for i in range(1, len(self.node_config_list)):
            if i + 1 == len(self.node_config_list):
                self.leaf_node = VdbLeafNode(cur_node, node_config_list[i], dtype)
                self.node_list.append(self.leaf_node)
            else:
                self.node_list.append(VdbInternalNode(cur_node, self.node_config_list[i], dtype))

    @ti.func
    def get_value(self, x: int, y: int, z: int):
        cur_extent = [ 1 << self.root_node.node_config.slog2x, 1 << self.root_node.node_config.slog2y, 1 << self.root_node.node_config.slog2z]

        assert 0 <= x < cur_extent[0], "X needs to be in range of [0, {}) when getting value".format(1 << self.root_node.node_config.slog2x)
        assert 0 <= y < cur_extent[1], "Y needs to be in range of [0, {}) when getting value".format(1 << self.root_node.node_config.slog2y)
        assert 0 <= z < cur_extent[2], "Z needs to be in range of [0, {}) when getting value".format(1 << self.root_node.node_config.slog2z)

        res = self.root_node.background_value
        for i in ti.static(range(self.num_tree_level)):
            if ti.static(i + 1 < self.num_tree_level):
                node_config = self.node_list[i].node_config
                child_node_config = self.node_list[i + 1].node_config
                internal_offset = ((( x & (1 << node_config.slog2x) - 1) >> child_node_config.slog2x) << (node_config.log2y + node_config.log2z)) + \
                        ((( y & (1 << node_config.slog2y) - 1) >> child_node_config.slog2y) << node_config.log2z) + \
                        ((z & (1 << node_config.slog2z) - 1) >> child_node_config.log2z)
                if ti.static(i == 0):
                    # Root node
                    if not self.root_node.is_child_active(internal_offset):
                        if self.root_node.get_root_state_at(internal_offset):
                            res = self.root_node.get_root_value_at(internal_offset)
                        break
                else:
                    pass
            else:

        return res


