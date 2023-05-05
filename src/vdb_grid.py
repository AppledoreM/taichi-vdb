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

    @staticmethod
    def append_internal_node(parent_node: ti.template(), node_config: VdbNodeConfig, dtype=ti.f32):
        snode_size = 1 << (node_config.log2x + node_config.log2y + node_config.log2z)
        mask_size = snode_size // 32
        vdb_assert(mask_size * 32 == snode_size, "Snode size is too small for an internal.")

        vdb_log("Vdb Grid Initialized with internal node shape ({}, {}, {}) and size: {}".format(node_config.log2x,
                                                                                        node_config.log2y,
                                                                                        node_config.log2z, snode_size))

        child_node = parent_node.pointer(ti.i, snode_size)
        value_mask_node = parent_node.dense(ti.i, mask_size)
        child_mask_node = parent_node.dense(ti.i, mask_size)
        value_node = parent_node.dense(ti.i, snode_size)

        value_mask = ti.field(ti.i32)
        child_mask = ti.field(ti.i32)
        value = ti.field(dtype)

        value_mask_node.place(value_mask)
        child_mask_node.place(child_mask)
        value_node.place(value)

        return value_mask, child_mask, value, child_node


@ti.data_oriented
class VdbRootNode:
    root_max_size = 4096

    def __init__(self, dtype=ti.f32, background_value=0.0):
        self.dtype = dtype
        self.background_value = background_value

        self.root = ti.root
        self.root_data_child_node = self.root.pointer(ti.i, VdbRootNode.root_max_size)
        self.root_data_value = ti.field(self.dtype, shape=VdbRootNode.root_max_size)

        root_mask_size = VdbRootNode.root_max_size // 32
        self.root_data_state = ti.field(ti.i32, shape=root_mask_size)

    @ti.func
    def is_data_at_index_child(self, index: ti.i32) -> bool:
        return ti.is_active(self.root_data_child_node, index)

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

    @staticmethod
    def append_leaf_node(parent_node: ti.template(), node_config: VdbNodeConfig, dtype=ti.f32):
        value = ti.field(dtype)
        mask = ti.field(ti.i32)
        flag = ti.field(ti.i64)

        snode_size = 1 << (node_config.log2x + node_config.log2y + node_config.log2z)
        mask_size = snode_size // 32
        vdb_assert(mask_size * 32 == snode_size, "The size of the leaf nodes needs to be greater than 32.")
        vdb_log("Vdb Grid Initialized with leaf shape ({}, {}, {}) and size: {}".format(node_config.log2x,
                                                                                        node_config.log2y,
                                                                                        node_config.log2z, snode_size))

        value_node = parent_node.dense(ti.i, snode_size)
        mask_node = parent_node.dense(ti.i, mask_size)
        flag_node = parent_node.dense(ti.i, 1)

        mask_node.place(mask)
        value_node.place(value)
        flag_node.place(flag)

        return value, mask, flag


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
        self.node_config_list = []
        for i in range(len(tree_levels) - 1, -1, -1):
            if i + 1 == len(tree_levels):
                self.node_config_list.append(VdbNodeConfig(tree_levels[i], tree_levels[i], tree_levels[i], dtype))
            else:
                self.node_config_list.append(
                    VdbNodeConfig(tree_levels[i], tree_levels[i], tree_levels[i], dtype, self.node_config_list[-1]))

        self.node_config_list.reverse()

        self.root_node = VdbRootNode(dtype, background_value)
        self.internal_value_mask_list = []
        self.internal_child_mask_list = []
        self.internal_value_list = []

        cur_node = self.root_node.root_data_child_node
        for i in range(len(self.node_config_list)):
            if i + 1 == len(self.node_config_list):
                value, value_mask, flag = VdbLeafNode.append_leaf_node(cur_node, self.node_config_list[i], dtype)
                self.leaf_value = value
                self.leaf_value_mask = value_mask
                self.leaf_flag = flag
            else:
                value_mask, child_mask, value, cur_node = VdbInternalNode.append_internal_node(cur_node,
                                                                                               self.node_config_list[i],
                                                                                               dtype)
                self.internal_value_mask_list.append(value_mask)
                self.internal_child_mask_list.append(child_mask)
                self.internal_value_list.append(value)
