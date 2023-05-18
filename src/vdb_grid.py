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

        if ti.static(child_node is not None):
            self.slog2x += child_node.slog2x
            self.slog2y += child_node.slog2y
            self.slog2z += child_node.slog2z

class VdbOpId:
    set_op = 0
    add_op = 1
    sub_op = 2
    mul_op = 3
    div_op = 4


@ti.data_oriented
class VdbDataWrapper:
    max_vdb_levels = 5

    def __init__(self, num_vdb_levels, config):
        self.pointer_list = [ ti.root ]
        self.bitmasked_list = []
        self.value_list = []

        self.num_vdb_levels = ti.static(num_vdb_levels)

        # Compile time determinantion of fields
        if ti.static(self.num_vdb_levels > 0):
            self.value0 = ti.field(ti.f32)
            self.value_list.append(self.value0)
            self.leaf_value = self.value0
        if ti.static(self.num_vdb_levels > 1):
            self.value1 = ti.field(ti.f32)
            self.value_list.append(self.value1)
            self.leaf_value = self.value1
        if ti.static(self.num_vdb_levels > 2):
            self.value2 = ti.field(ti.f32)
            self.value_list.append(self.value2)
            self.leaf_value = self.value2
        if ti.static(self.num_vdb_levels > 3):
            self.value3 = ti.field(ti.f32)
            self.value_list.append(self.value3)
            self.leaf_value = self.value3
        if ti.static(self.num_vdb_levels > 4):
            self.value4 = ti.field(ti.f32)
            self.value_list.append(self.value4)
            self.leaf_value = self.value4


        for i in ti.static(range(self.num_vdb_levels)):
            self.bitmasked_list.append(self.pointer_list[-1].dense(ti.ijk, 1 << config[i]))
            self.bitmasked_list[-1].place(self.value_list[i])
            if i + 1 < self.num_vdb_levels:
                self.pointer_list.append(self.pointer_list[-1].pointer(ti.ijk, 1 << config[i]))

        if ti.static(self.num_vdb_levels > 1):
            self.child0 = self.pointer_list[1]
        if ti.static(self.num_vdb_levels > 2):
            self.child1 = self.pointer_list[2]
        if ti.static(self.num_vdb_levels > 3):
            self.child2 = self.pointer_list[3]
        if ti.static(self.num_vdb_levels > 4):
            self.child3 = self.pointer_list[4]

    
    @ti.func
    def set_op(self, i, j, k, value, f: ti.template()):
        f[i, j, k] = value

    @ti.func
    def add_op(self, i, j, k, value, f: ti.template()):
        f[i, j, k] += value

    @ti.func
    def sub_op(self, i, j, k, value, f: ti.template()):
        f[i, j, k] -= value

    @ti.func
    def mul_op(self, i, j, k, value, f: ti.template()):
        f[i, j, k] *= value

    @ti.func
    def div_op(self, i, j, k, value, f: ti.template()):
        f[i, j, k] /= value


    @ti.func
    def apply_op(self, i, j, k, value, op: ti.template(), f: ti.template(), fixed_level: ti.template(), level: ti.template()):
        if ti.static(level == fixed_level):
            op(i, j, k, value, f)


    @ti.func
    def modify_value(self, level: ti.template(), i, j, k, value: ti.template(), op: ti.template()):
        assert level >= 0, "Level needs to be non-negative."
        assert level < self.num_vdb_levels, "Level exceeds maximum vdb level {}".format(self.num_vdb_levels)

        if ti.static(self.num_vdb_levels > 0):
            self.apply_op(i, j, k, value, op, self.value0, 0, level)
        if ti.static(self.num_vdb_levels > 1):
            self.apply_op(i, j, k, value, op, self.value1, 1, level)
        if ti.static(self.num_vdb_levels > 2):
            self.apply_op(i, j, k, value, op, self.value2, 2, level)
        if ti.static(self.num_vdb_levels > 3):
            self.apply_op(i, j, k, value, op, self.value3, 3, level)
        if ti.static(self.num_vdb_levels > 4):
            self.apply_op(i, j, k, value, op, self.value4, 4, level)


    @ti.func
    def set_value(self, level: ti.template(), i, j, k, value):
        self.modify_value(level, i, j, k, value, self.set_op)


    @ti.func
    def read_value(self, level: ti.template(), i, j, k):
        assert level < self.num_vdb_levels, "Level exceeds maximum vdb level {}".format(self.num_vdb_levels)
        
        res = 0.0
        if ti.static(level + 1 == self.num_vdb_levels):
            res = self.leaf_value[i, j, k]
        else:
            i, j, k = self.rescale_index_from_world_voxel(level, i, j, k)
            if ti.static(level == 0 and self.num_vdb_levels > 0):
                res = self.value0[i, j, k]
            elif ti.static(level == 1 and self.num_vdb_levels > 1):
                res = self.value1[i, j, k]
            elif ti.static(level == 2 and self.num_vdb_levels > 2):
                res = self.value2[i, j, k]
            elif ti.static(level == 3 and self.num_vdb_levels > 3):
                res = self.value3[i, j, k]
            elif ti.static(level == 4 and self.num_vdb_levels > 4):
                res = self.value4[i, j, k]

        return res
    


    #@param: level   - denotes the target child snode to scale to
    #        i, j, k - denotes the coordinates of world voxel to rescale
    @ti.func
    def rescale_index_from_world_voxel(self, level: ti.template(), i, j, k):
        assert level + 1 < self.num_vdb_levels, "Should never rescale index at or beyond the leaf level"
        index = ti.Vector.zero(ti.i32, n=3)

        if ti.static(level == 0 and self.num_vdb_levels > 1):
            index = ti.rescale_index(self.leaf_value, self.child0, ti.Vector([i, j, k]))
        elif ti.static(level == 1 and self.num_vdb_levels > 2):
            index = ti.rescale_index(self.leaf_value, self.child1, ti.Vector([i, j, k]))
        elif ti.static(level == 2 and self.num_vdb_levels > 3):
            index = ti.rescale_index(self.leaf_value, self.child2, ti.Vector([i, j, k]))
        elif ti.static(level == 3 and self.num_vdb_levels > 4):
            index = ti.rescale_index(self.leaf_value, self.child3, ti.Vector([i, j, k]))

        return index




    #@param: level   - denotes the level of sparse grid to check 
    #        i, j, k - denotes the coordinate of world voxel 
    #@detail: Return if a child is active
    @ti.func
    def is_child_active(self, level: ti.template(), i, j, k) -> bool:
        i, j, k = self.rescale_index_from_world_voxel(level, i, j, k)
        res = False

        if ti.static(level == 0 and self.num_vdb_levels > 0):
            res = ti.is_active(self.child0, [i, j, k])
        elif ti.static(level == 1 and self.num_vdb_levels > 1):
            res = ti.is_active(self.child1, [i, j, k])
        elif ti.static(level == 2 and self.num_vdb_levels > 2):
            res = ti.is_active(self.child2, [i, j, k])
        elif ti.static(level == 3 and self.num_vdb_levels > 3):
            res = ti.is_active(self.child3, [i, j, k])

        return res


class VdbGrid:

    def __init__(self, level_configs=None, dtype=ti.f32, background_value=0.0, origin = ti.Vector([0.0, 0.0, 0.0])):

        if level_configs is None:
            level_configs = [5, 5, 5, 5, 3]
        else:
            vdb_assert(isinstance(level_configs, type([])), "Tree levels should be an array type.")
            for level_dim in level_configs:
                vdb_assert(level_dim > 0, "The vdb level dimension must be greater than 0.")

        self.dtype = dtype
        self.num_vdb_levels = ti.static(len(level_configs))
        self.leaf_level = self.num_vdb_levels - 1
        self.origin = origin

        # Build configuration of each level
        config_list = []
        for i in range(self.num_vdb_levels - 1, -1, -1):
            if i + 1 == self.num_vdb_levels:
                config_list.append(VdbNodeConfig(level_configs[i], level_configs[i], level_configs[i], dtype))

            else:
                config_list.append(VdbNodeConfig(level_configs[i], level_configs[i], level_configs[i], dtype, config_list[-1]))

        config_list.reverse()

        # Store configurations in fields
        self.config = ti.Vector.field(n=3, dtype=ti.i32, shape=self.num_vdb_levels + 1)
        self.sconfig = ti.Vector.field(n=3, dtype=ti.i32, shape=self.num_vdb_levels + 1)
        self.ssize = ti.field(ti.f32, shape=self.num_vdb_levels + 1)


        for i in range(self.num_vdb_levels):
            self.config[i] = ti.Vector([config_list[i].log2x, config_list[i].log2y, config_list[i].log2z])
            self.sconfig[i] = ti.Vector([config_list[i].slog2x, config_list[i].slog2y, config_list[i].slog2z])
            sconfig = self.sconfig[i]
            self.ssize[i] = 1 << (sconfig[0] + sconfig[1] + sconfig[2])

        self.data_wrapper = VdbDataWrapper(self.num_vdb_levels, self.config)

    

    #@param: i, j, k denotes the coordinates in the voxel space
    #        value denotes the target value to set
    #@detail: change the value of voxel to specificed
    @ti.func
    def modify_value(self, i, j, k, value, op: ti.template()): 
        if ti.static(op == VdbOpId.set_op):
            self.data_wrapper.set_value(self.leaf_level, i, j, k, value)
        else:
            print("Unrecognized operation to modify vdb value")
            pass

    @ti.func
    def set_value(self, i, j, k, value):
        self.modify_value(i, j, k, value, VdbOpId.set_op)


    @ti.func
    def read_value_impl(self, level: ti.template(), i, j, k):
        res = 0.0
        if ti.static(level + 1 == self.num_vdb_levels):
            res = self.data_wrapper.read_value(self.leaf_level, i, j, k)
        else:
            if not self.data_wrapper.is_child_active(level, i, j, k):
                res = self.data_wrapper.read_value(level, i, j, k) 
            else:
                res = self.read_value_impl(level + 1, i, j, k)

        return res


    @ti.func
    def read_value(self, i, j, k):
        return self.read_value_impl(self.leaf_level, i, j, k)


            
        

