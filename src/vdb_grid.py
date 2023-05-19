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

    def __init__(self, num_vdb_levels, config: ti.template(), sconfig: ti.template(), voxel_dim):
        self.pointer_list = [ ti.root ]
        self.bitmasked_list = []
        self.value_list = []
        self.voxel_dim = voxel_dim
        self.config = config
        self.sconfig = sconfig

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
            self.bitmasked_list.append(self.pointer_list[-1].bitmasked(ti.ijk, 1 << config[i]))
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
    def add_value(self, level: ti.template(), i, j, k, value):
        self.modify_value(level, i, j, k, value, self.add_op)

    @ti.func
    def sub_value(self, level: ti.template(), i, j, k, value):
        self.modify_value(level, i, j, k, value, self.sub_op)

    @ti.func
    def mul_value(self, level: ti.template(), i, j, k, value):
        self.modify_value(level, i, j, k, value, self.mul_op)

    @ti.func
    def div_value(self, level: ti.template(), i, j, k, value):
        self.modify_value(level, i, j, k, value, self.div_op)

    @ti.func
    def read_value_local(self, level: ti.template(), i, j, k):
        assert level < self.num_vdb_levels, "Level exceeds maximum vdb level {}".format(self.num_vdb_levels)

        res = 0.0
        if ti.static(level + 1 == self.num_vdb_levels):
            res = self.leaf_value[i, j, k]
        else:
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

    @ti.func
    def read_value_world(self, level: ti.template(), i, j, k):
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
        assert level + 1 <= self.num_vdb_levels, "Should never rescale index beyond the leaf level"
        index = ti.Vector.zero(ti.i32, n=3)

        if ti.static(level + 1 == self.num_vdb_levels):
            index = ti.Vector([i, j, k])
        elif ti.static(level == 0 and self.num_vdb_levels > 1):
            index = ti.rescale_index(self.leaf_value, self.child0, ti.Vector([i, j, k]))
        elif ti.static(level == 1 and self.num_vdb_levels > 2):
            index = ti.rescale_index(self.leaf_value, self.child1, ti.Vector([i, j, k]))
        elif ti.static(level == 2 and self.num_vdb_levels > 3):
            index = ti.rescale_index(self.leaf_value, self.child2, ti.Vector([i, j, k]))
        elif ti.static(level == 3 and self.num_vdb_levels > 4):
            index = ti.rescale_index(self.leaf_value, self.child3, ti.Vector([i, j, k]))
        else:
            assert False

        return index




    #@param: level   - denotes the level of sparse grid to check 
    #        i, j, k - denotes the coordinate of world voxel 
    #@detail: Return if a child is active
    @ti.func
    def is_child_active(self, level: ti.template(), i, j, k) -> bool:
        i, j, k = self.rescale_index_from_world_voxel(level, i, j, k)
        res = False

        if ti.static(level == 0 and self.num_vdb_levels > 1):
            res = ti.is_active(self.child0, [i, j, k])
        elif ti.static(level == 1 and self.num_vdb_levels > 2):
            res = ti.is_active(self.child1, [i, j, k])
        elif ti.static(level == 2 and self.num_vdb_levels > 3):
            res = ti.is_active(self.child2, [i, j, k])
        elif ti.static(level == 3 and self.num_vdb_levels > 4):
            res = ti.is_active(self.child3, [i, j, k])

        return res


    #@param: vertices - denotes the vertex field that will be field with 1 cube of vertices
    #        vertex_offset   - denotes offset of in the vertex field to start filling the vertices
    #        i, j, k  - denotes index of the cube in corresponding cube_dim
    #        cube_dim - denotes the length of 3 sides of a cube
    @ti.func
    def fill_vdb_bbox_cube_vertices(self, vertices:ti.template(), vertex_offset, indices: ti.template(), index_offset, i, j, k, cube_dim: ti.template()):
        position_offset = ti.Vector([i, j, k]) * cube_dim
        dx = ti.Vector([1, 0, 0])
        dy = ti.Vector([0, 1, 0])
        dz = ti.Vector([0, 0, 1])

        vertices[vertex_offset] = position_offset
        vertices[vertex_offset + 1] = position_offset + dx * cube_dim
        vertices[vertex_offset + 2] = position_offset + (dx + dy) * cube_dim
        vertices[vertex_offset + 3] = position_offset + dy * cube_dim
        vertices[vertex_offset + 4] = position_offset + dz * cube_dim
        vertices[vertex_offset + 5] = position_offset + (dx + dz) * cube_dim
        vertices[vertex_offset + 6] = position_offset + (dx + dy + dz) * cube_dim
        vertices[vertex_offset + 7] = position_offset + (dy + dz) * cube_dim

        indices[index_offset + 0] = vertex_offset + 0
        indices[index_offset + 1] = vertex_offset + 1

        indices[index_offset + 2] = vertex_offset + 1
        indices[index_offset + 3] = vertex_offset + 2

        indices[index_offset + 4] = vertex_offset + 2
        indices[index_offset + 5] = vertex_offset + 3

        indices[index_offset + 6] = vertex_offset + 3
        indices[index_offset + 7] = vertex_offset + 0

        indices[index_offset + 8] = vertex_offset + 0
        indices[index_offset + 9] = vertex_offset + 4

        indices[index_offset + 10] = vertex_offset + 4
        indices[index_offset + 11] = vertex_offset + 5

        indices[index_offset + 12] = vertex_offset + 5
        indices[index_offset + 13] = vertex_offset + 6

        indices[index_offset + 14] = vertex_offset + 6
        indices[index_offset + 15] = vertex_offset + 2

        indices[index_offset + 16] = vertex_offset + 7
        indices[index_offset + 17] = vertex_offset + 4

        indices[index_offset + 18] = vertex_offset + 6
        indices[index_offset + 19] = vertex_offset + 7

        indices[index_offset + 20] = vertex_offset + 1
        indices[index_offset + 21] = vertex_offset + 5

        indices[index_offset + 22] = vertex_offset + 3
        indices[index_offset + 23] = vertex_offset + 7



    #@param: vertices  - denotes the vertex field that will be filled with vertex information
    #@detail: Fill the vertices with formatted order of points that will be used by VdbViewer
    @ti.kernel
    def generate_vdb_bbox_vertices_impl(self, vertices: ti.template(), indices: ti.template(), per_vertex_color: ti.template()):
        vertex_count = 0
        index_count = 0
        if ti.static(self.num_vdb_levels > 1):
            for i, j, k in self.child0:
                vertex_offset = ti.atomic_add(vertex_count, 8)
                index_offset = ti.atomic_add(index_count, 24)
                self.fill_vdb_bbox_cube_vertices(vertices, vertex_offset, indices, index_offset, i, j, k, self.voxel_dim * (1 << self.sconfig[1]))
                for w in ti.static(range(8)):
                    per_vertex_color[w + vertex_offset] = (0.16, 0.5, 0.73)

        if ti.static(self.num_vdb_levels > 0):
            for i, j, k in self.value0:
                vertex_offset = ti.atomic_add(vertex_count, 8)
                index_offset = ti.atomic_add(index_count, 24)
                self.fill_vdb_bbox_cube_vertices(vertices, vertex_offset, indices, index_offset, i, j, k, self.voxel_dim * (1 << self.sconfig[1]))

                for w in ti.static(range(8)):
                    per_vertex_color[w + vertex_offset] = (0.16, 0.5, 0.73)


        if ti.static(self.num_vdb_levels > 2):

            for i, j, k in self.child1:
                vertex_offset = ti.atomic_add(vertex_count, 8)
                index_offset = ti.atomic_add(index_count, 24)
                self.fill_vdb_bbox_cube_vertices(vertices, vertex_offset, indices, index_offset, i, j, k, self.voxel_dim * (1 << self.sconfig[2]))

                for w in ti.static(range(8)):
                    per_vertex_color[w + vertex_offset] = (0.906, 0.298, 0.235)

        if ti.static(self.num_vdb_levels > 1):
            for i, j, k in self.value1:
                vertex_offset = ti.atomic_add(vertex_count, 8)
                index_offset = ti.atomic_add(index_count, 24)
                self.fill_vdb_bbox_cube_vertices(vertices, vertex_offset, indices, index_offset, i, j, k, self.voxel_dim * (1 << self.sconfig[2]))

                for w in ti.static(range(8)):
                    per_vertex_color[w + vertex_offset] = (0.906, 0.298, 0.235)

        if ti.static(self.num_vdb_levels > 3):
            for i, j, k in self.child2:
                vertex_offset = ti.atomic_add(vertex_count, 8)
                index_offset = ti.atomic_add(index_count, 24)
                self.fill_vdb_bbox_cube_vertices(vertices, vertex_offset, indices, index_offset, i, j, k, self.voxel_dim * (1 << self.sconfig[3]))

                for w in ti.static(range(8)):
                    per_vertex_color[w + vertex_offset] = (0.153, 0.682, 0.376)

        if ti.static(self.num_vdb_levels > 2):
            for i, j, k in self.value2:
                vertex_offset = ti.atomic_add(vertex_count, 8)
                index_offset = ti.atomic_add(index_count, 24)
                self.fill_vdb_bbox_cube_vertices(vertices, vertex_offset, indices, index_offset, i, j, k,
                                                 self.voxel_dim * (1 << self.sconfig[3]))

                for w in ti.static(range(8)):
                    per_vertex_color[w + vertex_offset] = (0.153, 0.682, 0.376)

        if ti.static(self.num_vdb_levels > 4):
            for i, j, k in self.child3:
                vertex_offset = ti.atomic_add(vertex_count, 8)
                index_offset = ti.atomic_add(index_count, 24)
                self.fill_vdb_bbox_cube_vertices(vertices, vertex_offset, indices, index_offset, i, j, k, self.voxel_dim * (1 << self.sconfig[4]))

                for w in ti.static(range(8)):
                    per_vertex_color[w + vertex_offset] = (0.557, 0.267, 0.678)

        if ti.static(self.num_vdb_levels > 3):
            for i, j, k in self.value3:
                vertex_offset = ti.atomic_add(vertex_count, 8)
                index_offset = ti.atomic_add(index_count, 24)
                self.fill_vdb_bbox_cube_vertices(vertices, vertex_offset, indices, index_offset, i, j, k,
                                                 self.voxel_dim * (1 << self.sconfig[4]))

                for w in ti.static(range(8)):
                    per_vertex_color[w + vertex_offset] = (0.557, 0.267, 0.678)

        if ti.static(self.num_vdb_levels > 4):
            for i, j, k in self.value4:
                vertex_offset = ti.atomic_add(vertex_count, 8)
                index_offset = ti.atomic_add(index_count, 24)
                self.fill_vdb_bbox_cube_vertices(vertices, vertex_offset, indices, index_offset, i, j, k, self.voxel_dim)

                for w in ti.static(range(8)):
                    per_vertex_color[w + vertex_offset] = (0.902, 0.494, 0.133)


    @ti.func
    def prune_level_tolerance(self, level: ti.template(), snode: ti.template(), f: ti.template(), tolerance: ti.template()):
        assert level + 1 < self.num_vdb_levels, "Must not prune level at or beyond the leaf level."
        config = self.config[level + 1]
        print(f"Pruning Level: {level}")
        for i, j, k in snode:
            is_equal = True
            ni, nj, nk = ti.rescale_index(snode, f, [i, j, k])
            value = f[ni, nj, nk]
            for x, y, z in ti.ndrange(1 << config[0], 1 << config[1], 1 << config[2]):
                nx = ni + x
                ny = nj + y
                nz = nk + z
                # if nx < 5 and ny < 5 and nz < 5:
                    # print(f"Level {level}; Coord: ({nx}, {ny}, {nz}); Child Active: {self.is_child_active(level + 1, nx, ny, nz)}; Read Value: {self.read_value_local(level + 1, nx, ny, nz)}")
                if ti.static(level + 1 == self.num_vdb_levels) or not self.is_child_active(level + 1, nx, ny, nz):
                    if ti.static(tolerance != 0):
                        is_equal &= approx_equal(self.read_value_local(level + 1, nx, ny, nz), value, tolerance)
                    else:
                        is_equal &= (self.read_value_local(level + 1, nx, ny, nz) == value)
                else:
                    is_equal &= False

                if not is_equal:
                    break

            print(f"Is all equal at ({i}, {j}, {k}): {is_equal}")
            if is_equal:
                ti.deactivate(snode, [i, j, k])
                if value != 0.0:
                    self.set_value(level, i, j, k, value)

    @ti.kernel
    def prune(self, tolerance:ti.template()):
        # Prune the tree from bottom up
        if ti.static(self.num_vdb_levels > 4):
            self.prune_level_tolerance(3, self.child3, self.value4, tolerance)
        if ti.static(self.num_vdb_levels > 3):
            self.prune_level_tolerance(2, self.child2, self.value3, tolerance)
        if ti.static(self.num_vdb_levels > 2):
            self.prune_level_tolerance(1, self.child1, self.value2, tolerance)
        if ti.static(self.num_vdb_levels > 1):
            self.prune_level_tolerance(0, self.child0, self.value1, tolerance)

@ti.data_oriented
class VdbGrid:

    def __init__(self, bounding_box: ti.template(), level_configs=None, dtype=ti.f32, background_value=0.0, origin = ti.Vector([0.0, 0.0, 0.0])):


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


        for i in ti.static(range(self.num_vdb_levels)):
            self.config[i] = ti.Vector([config_list[i].log2x, config_list[i].log2y, config_list[i].log2z])
            self.sconfig[i] = ti.Vector([config_list[i].slog2x, config_list[i].slog2y, config_list[i].slog2z])
            sconfig = self.sconfig[i]
            self.ssize[i] = 1 << (sconfig[0] + sconfig[1] + sconfig[2])

        self.bounding_box = bounding_box
        self.voxel_dim = (bounding_box[1] - bounding_box[0]) / (1 << self.sconfig[0][0])

        self.data_wrapper = VdbDataWrapper(self.num_vdb_levels, self.config, self.sconfig, self.voxel_dim)

    

    #@param: i, j, k denotes the coordinates in the voxel space
    #        value denotes the target value to set
    #@detail: change the value of voxel to specificed
    @ti.func
    def modify_value(self, i, j, k, value, op: ti.template()): 
        if ti.static(op == VdbOpId.set_op):
            self.data_wrapper.set_value(self.leaf_level, i, j, k, value)
        elif ti.static(op == VdbOpId.add_op):
            self.data_wrapper.add_value(self.leaf_level, i, j, k, value)
        elif ti.static(op == VdbOpId.sub_op):
            self.data_wrapper.sub_value(self.leaf_level, i, j, k, value)
        elif ti.static(op == VdbOpId.mul_op):
            self.data_wrapper.mul_value(self.leaf_level, i, j, k, value)
        elif ti.static(op == VdbOpId.div_op):
            self.data_wrapper.div_value(self.leaf_level, i, j, k, value)
        else:
            print("Unrecognized operation to modify vdb value")
            pass

    @ti.func
    def set_value(self, i, j, k, value):
        self.modify_value(i, j, k, value, VdbOpId.set_op)

    @ti.func
    def add_value(self, i, j, k, value):
        self.modify_value(i, j, k, value, VdbOpId.add_op)

    @ti.func
    def sub_value(self, i, j, k, value):
        self.modify_value(i, j, k, value, VdbOpId.sub_op)

    @ti.func
    def mul_value(self, i, j, k, value):
        self.modify_value(i, j, k, value, VdbOpId.mul_op)

    @ti.func
    def div_value(self, i, j, k, value):
        self.modify_value(i, j, k, value, VdbOpId.div_op)

    @ti.func
    def read_value_impl(self, level: ti.template(), i, j, k):
        res = 0.0
        if ti.static(level + 1 == self.num_vdb_levels):
            res = self.data_wrapper.read_value_world(self.leaf_level, i, j, k)
        else:
            if not self.data_wrapper.is_child_active(level, i, j, k):
                res = self.data_wrapper.read_value_world(level, i, j, k)
            else:
                res = self.read_value_impl(level + 1, i, j, k)

        return res

    @ti.func
    def read_value(self, i, j, k):
        return self.read_value_impl(self.leaf_level, i, j, k)

    def prune(self, tolerance: ti.template()):
        self.data_wrapper.prune(tolerance)




            
        

