import taichi as ti
from src.utils import *


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
                                                                                                 node_config.log2z,
                                                                                                 snode_size))

        self.snode_size = snode_size
        self.node_config = node_config
        self.child_node = parent_node.pointer(ti.i, snode_size)
        self.value_mask_node = parent_node.dense(ti.i, mask_size)
        self.child_mask_node = parent_node.dense(ti.i, mask_size)
        self.value_node = parent_node.dense(ti.i, snode_size)

        self.value_mask = ti.field(ti.u32)
        self.child_mask = ti.field(ti.u32)
        self.value = ti.field(dtype)

        self.value_mask_node.place(self.value_mask)
        self.child_mask_node.place(self.child_mask)
        self.value_node.place(self.value)

@ti.data_oriented
class VdbRootNode:

    def __init__(self, node_config: VdbNodeConfig, dtype=ti.f32, background_value=0.0):
        self.dtype = dtype
        self.background_value = background_value
        self.root_snode_size = 1 << (node_config.log2x + node_config.log2y + node_config.log2z)
        self.node_config = node_config

        self.root = ti.root
        self.child_node = self.root.pointer(ti.i, self.root_snode_size)

        self.value = ti.field(self.dtype, shape=self.root_snode_size)

        self.value_mask = ti.field(ti.u32, shape=self.root_snode_size // 32)
        self.child_mask = ti.field(ti.u32, shape=self.root_snode_size // 32)

        vdb_log("Vdb Grid Initialized with root node shape ({}, {}, {}) and size: {}".format(node_config.log2x,
                                                                                             node_config.log2y,
                                                                                             node_config.log2z,
                                                                                             self.root_snode_size))


@ti.data_oriented
class VdbLeafNode:

    @staticmethod
    def read_leaf_node_flag(leaf_node: ti.template()) -> ti.i64:
        return leaf_node

    def __init__(self, parent_node: ti.template(), node_config: VdbNodeConfig, dtype=ti.f32):
        self.value = ti.field(dtype)
        self.value_mask = ti.field(ti.u32)
        self.flag = ti.field(ti.i64)

        snode_size = 1 << (node_config.log2x + node_config.log2y + node_config.log2z)
        mask_size = snode_size // 32
        vdb_assert(mask_size * 32 == snode_size, "The size of the leaf nodes needs to be greater than 32.")
        vdb_log("Vdb Grid Initialized with leaf shape ({}, {}, {}) and size: {}".format(node_config.log2x,
                                                                                        node_config.log2y,
                                                                                        node_config.log2z, snode_size))

        self.node_config = node_config
        self.value_node = parent_node.dense(ti.i, snode_size)
        self.value_mask_node = parent_node.dense(ti.i, mask_size)
        self.flag_node = parent_node.dense(ti.i, 1)

        self.value_mask_node.place(self.value_mask)
        self.value_node.place(self.value)
        self.flag_node.place(self.flag)


class VdbFieldId:
    value = 0
    child_mask = 1
    value_mask = 2
@ti.data_oriented
class VdbDataWrapper:
    max_vdb_level = 5

    def __init__(self, node_list, value_list, child_mask_list, value_mask_list):
        get_field = lambda field_list, index: field_list[index] if len(field_list) > index else None
        get_node = lambda node_list_, index: node_list_[index].child_node if len(node_list_) > index else None

        self.node0 = get_node(node_list, 0)
        self.node1 = get_node(node_list, 1)
        self.node2 = get_node(node_list, 2)
        self.node3 = get_node(node_list, 3)

        self.value0 = get_field(value_list, 0)
        self.value1 = get_field(value_list, 1)
        self.value2 = get_field(value_list, 2)
        self.value3 = get_field(value_list, 3)
        self.value4 = get_field(value_list, 4)

        self.child_mask0 = get_field(child_mask_list, 0)
        self.child_mask1 = get_field(child_mask_list, 1)
        self.child_mask2 = get_field(child_mask_list, 2)
        self.child_mask3 = get_field(child_mask_list, 3)
        self.child_mask4 = get_field(child_mask_list, 4)

        self.value_mask0 = get_field(value_mask_list, 0)
        self.value_mask1 = get_field(value_mask_list, 1)
        self.value_mask2 = get_field(value_mask_list, 2)
        self.value_mask3 = get_field(value_mask_list, 3)
        self.value_mask4 = get_field(value_mask_list, 4)

    @ti.func
    def set_value(self, level, index, value):
        assert level < VdbDataWrapper.max_vdb_level, "Level exceeds maximum vdb level {}".format(VdbDataWrapper.max_vdb_level)
        if level == 0:
            self.value0[index] = value
        elif level == 1:
            self.value1[index] = value
        elif level == 2:
            self.value2[index] = value
        elif level == 3:
            self.value3[index] = value
        elif level == 4:
            self.value4[index] = value

    @ti.func
    def get_value(self, level, index):
        assert level < VdbDataWrapper.max_vdb_level, "Level exceeds maximum vdb level {}".format(VdbDataWrapper.max_vdb_level)
        res = 0.0
        if level == 0:
            res = self.value0[index]
        elif level == 1:
            res = self.value1[index]
        elif level == 2:
            res = self.value2[index]
        elif level == 3:
            res = self.value3[index]
        elif level == 4:
            res = self.value4[index]
        return res

    @ti.func
    def set_value_mask(self, level, index, value: bool):
        assert level < VdbDataWrapper.max_vdb_level, "Level exceeds maximum vdb level {}".format(VdbDataWrapper.max_vdb_level)
        mapped_index = index // 32
        bit_index = ti.u32(index & 31)
        op_byte = ti.u32(1) << bit_index

        if value:
            if level == 0:
                self.value_mask0[mapped_index] |= op_byte
            if level == 1:
                self.value_mask1[mapped_index] |= op_byte
            if level == 2:
                self.value_mask2[mapped_index] |= op_byte
            if level == 3:
                self.value_mask3[mapped_index] |= op_byte
            if level == 4:
                self.value_mask4[mapped_index] |= op_byte
        else:
            if level == 0:
                self.value_mask0[mapped_index] &= ~op_byte
            if level == 1:
                self.value_mask1[mapped_index] &= ~op_byte
            if level == 2:
                self.value_mask2[mapped_index] &= ~op_byte
            if level == 3:
                self.value_mask3[mapped_index] &= ~op_byte
            if level == 4:
                self.value_mask4[mapped_index] &= ~op_byte

    @ti.func
    def get_value_mask(self, level, index) -> bool:
        assert level < VdbDataWrapper.max_vdb_level, "Level exceeds maximum vdb level {}".format(VdbDataWrapper.max_vdb_level)
        mapped_index = index // 32
        bit_index = ti.u32(index & 31)
        target_byte = ti.u32(1) << bit_index

        res = ti.u32(0)
        if level == 0:
            res = (self.value_mask0[mapped_index] & target_byte) >> bit_index
        if level == 1:
            res = (self.value_mask1[mapped_index] & target_byte) >> bit_index
        if level == 2:
            res = (self.value_mask2[mapped_index] & target_byte) >> bit_index
        if level == 3:
            res = (self.value_mask3[mapped_index] & target_byte) >> bit_index
        if level == 4:
            res = (self.value_mask4[mapped_index] & target_byte) >> bit_index

        return bool(res)


    @ti.func
    def set_child_mask(self, level, index, value: bool):
        assert level + 1 < VdbDataWrapper.max_vdb_level, "There is no child mask at leaf level "
        mapped_index = index // 32
        bit_index = ti.u32(index & 31)
        op_byte = ti.u32(1) << bit_index

        if value:
            if level == 0:
                self.child_mask0[mapped_index] |= op_byte
            elif level == 1:
                self.child_mask1[mapped_index] |= op_byte
            elif level == 2:
                self.child_mask2[mapped_index] |= op_byte
            elif level == 3:
                self.child_mask3[mapped_index] |= op_byte
        else:
            if level == 0:
                self.child_mask0[mapped_index] &= ~op_byte
            elif level == 1:
                self.child_mask1[mapped_index] &= ~op_byte
            elif level == 2:
                self.child_mask2[mapped_index] &= ~op_byte
            elif level == 3:
                self.child_mask3[mapped_index] &= ~op_byte


    @ti.func
    def is_child_active(self, level, index) -> bool:
        res = False
        if level == 0:
            res = ti.is_active(self.node0, index)
        elif level == 1:
            res = ti.is_active(self.node1, index)
        elif level == 2:
            res = ti.is_active(self.node2, index)
        elif level == 3:
            res = ti.is_active(self.node3, index)

        return res

    @ti.func
    def get_child_mask(self, level, index) -> bool:
        assert 0 < level, "There is no child mask at root level"
        assert level + 1 < VdbDataWrapper.max_vdb_level, "There is no child mask at leaf level "
        mapped_index = index // 32
        bit_index = ti.u32(index & 31)
        target_byte = ti.u32(1) << bit_index

        res = ti.u32(0)

        if level == 0:
            res = (self.child_mask1[mapped_index] & target_byte) >> bit_index
        elif level == 1:
            res = (self.child_mask1[mapped_index] & target_byte) >> bit_index
        elif level == 2:
            res = (self.child_mask2[mapped_index] & target_byte) >> bit_index
        elif level == 3:
            res = (self.child_mask3[mapped_index] & target_byte) >> bit_index

        return bool(res)

class VdbGrid:

    def __init__(self, tree_levels=None, dtype=ti.f32, background_value=0.0):

        if tree_levels is None:
            tree_levels = [5, 5, 5, 5, 3]
        else:
            vdb_assert(isinstance(tree_levels, type([])), "Tree levels should be an array type.")
            for level_dim in tree_levels:
                vdb_assert(level_dim > 0, "The vdb level dimension must be greater than 0.")

        self.dtype = dtype
        self.num_tree_level = len(tree_levels)

        self.node_config_list = []
        for i in range(self.num_tree_level - 1, -1, -1):
            if i + 1 == self.num_tree_level:
                self.node_config_list.append(VdbNodeConfig(tree_levels[i], tree_levels[i], tree_levels[i], dtype))

            else:
                self.node_config_list.append(
                    VdbNodeConfig(tree_levels[i], tree_levels[i], tree_levels[i], dtype, self.node_config_list[-1]))

        self.node_config_list.reverse()

        self.root_node = VdbRootNode(self.node_config_list[0], dtype, background_value)
        self.node_list = [self.root_node]

        cur_node = self.root_node.child_node
        for i in range(1, len(self.node_config_list)):
            if i + 1 == len(self.node_config_list):
                self.leaf_node = VdbLeafNode(cur_node, self.node_config_list[i], dtype)
                self.node_list.append(self.leaf_node)
            else:
                self.node_list.append(VdbInternalNode(cur_node, self.node_config_list[i], dtype))
                cur_node = self.node_list[-1].child_node

        self.config = ti.Vector.field(n=3, dtype=ti.i32, shape=self.num_tree_level)
        self.sconfig = ti.Vector.field(n=3, dtype=ti.i32, shape=self.num_tree_level)
        self.tpdconfig = ti.Vector.field(n=3, dtype=ti.i32, shape=self.num_tree_level)

        for i in range(self.num_tree_level):
            self.config[i] = ti.Vector([self.node_config_list[i].log2x, self.node_config_list[i].log2y, self.node_config_list[i].log2z])
            self.sconfig[i] = ti.Vector([self.node_config_list[i].slog2x, self.node_config_list[i].slog2y, self.node_config_list[i].slog2z])
            self.tpdconfig[i] = self.config[i]
            if i >= 1:
                self.tpdconfig[i] += self.tpdconfig[i - 1]

        value_list = []
        child_mask_list = []
        value_mask_list = []

        for i in range(0, self.num_tree_level):
            cur_node = self.node_list[i]
            value_list.append(cur_node.value)
            value_mask_list.append(cur_node.value_mask)
            if i + 1 < self.num_tree_level:
                child_mask_list.append(cur_node.child_mask)

        self.data_wrapper = VdbDataWrapper(self.node_list, value_list, child_mask_list, value_mask_list)

    @ti.func
    def extent_checker(self, x: int, y: int, z: int):
        if __debug__:
            cur_extent = [1 << self.root_node.node_config.slog2x, 1 << self.root_node.node_config.slog2y,
                          1 << self.root_node.node_config.slog2z]

            assert 0 <= x < cur_extent[0], "X needs to be in range of [0, {}) when getting value".format(
                1 << self.root_node.node_config.slog2x)
            assert 0 <= y < cur_extent[1], "Y needs to be in range of [0, {}) when getting value".format(
                1 << self.root_node.node_config.slog2y)
            assert 0 <= z < cur_extent[2], "Z needs to be in range of [0, {}) when getting value".format(
                1 << self.root_node.node_config.slog2z)
        else:
            pass


    @ti.func
    def calc_offset(self, x: int, y: int, z: int, level):
        leaf_config = self.tpdconfig[self.num_tree_level - 1]
        voxel_offset = x * (1 << (leaf_config[1] + leaf_config[2])) + y * (1 << leaf_config[2]) + z
        config = self.config[level]
        sconfig = self.sconfig[level]
        cconfig = sconfig - config

        block_size = 1 << (cconfig[0] + cconfig[1] + cconfig[2])
        return voxel_offset // block_size


    @ti.func
    def set_value(self, x: int, y: int, z: int, level, value):
        self.extent_checker(x, y, z)
        offset = self.calc_offset(x, y, z, level)

        self.data_wrapper.set_value(level, offset, value)
        self.data_wrapper.set_value_mask(level, offset, True)

    @ti.func
    def get_value(self, x: int, y: int, z: int):
        self.extent_checker(x, y, z)

        res = self.root_node.background_value
        found = False

        for i in range(self.num_tree_level):
            if found:
                continue

            offset = self.calc_offset(x, y, z, i)

            if not self.data_wrapper.is_child_active(i, offset):
                found = True
                res = self.data_wrapper.get_value(i, offset)

        return res
