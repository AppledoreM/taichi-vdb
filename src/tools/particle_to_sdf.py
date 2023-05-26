## @package ParticleToSdf
# Converting particles to sdf with method:
# "Reconstructing Surfaces of Particle-Based Fluids Using Anisotropic Kernels" by JinHun Yu and Greg Turk


import taichi as ti
import numpy as np
from src.vdb_grid import *


@ti.data_oriented
class ParticleToSdf:

    # Implementation of anisotropic sampling of particles

    def __init__(self, voxel_dim, level_configs, max_num_particles):
        assert level_configs is not None, "Level configuration must not be None."
        self.sdf = VdbGrid(voxel_dim, level_configs)
        self.max_num_particles = max_num_particles
        self.vdb = VdbGrid(voxel_dim, level_configs, ti.f32)

        # Anisotropic kernel
        self.G = ti.Matrix.field(3, 3, ti.f32, shape=max_num_particles)
        self.x_bar = ti.Vector.field(3, ti.f32, shape=max_num_particles)

        self.kr = 4
        self.ks = 1400
        self.kn = 0.5
        self.Neps = 25
        self.akLambda = 0.9

    @ti.func
    def isotropic_weight(self, x_i: ti.template(), x_j: ti.template(), r_i: ti.template()):
        norm_dx = (x_i - x_j).norm()
        res = 0.0
        if norm_dx < r_i:
            norm_dx /= r_i
            res = 1.0 - norm_dx * norm_dx * norm_dx

        return res

    @ti.kernel
    def compute_anisotropic_kernel_with_grid(self, particle_pos: ti.template(), num_particles: ti.template(),
                                             smoothing_radius: ti.f32):
        smoothing_voxel_radius = ti.ceil(smoothing_radius * self.vdb.data_wrapper.inv_voxel_dim, ti.i32) - 1
        for i, j, k in self.vdb.data_wrapper.leaf_value:
            id = int(self.vdb.read_value_world(i, j, k))

            if id > 0:
                x_i = particle_pos[id - 1]

                sum_weight = 0.0
                x_wi = ti.Vector.zero(n=3, dt=ti.f32)
                for dx, dy, dz in ti.ndrange(
                        (-2 * smoothing_voxel_radius[0], 2 * smoothing_voxel_radius[0] + 1),
                        (-2 * smoothing_voxel_radius[1], 2 * smoothing_voxel_radius[1] + 1),
                        (-2 * smoothing_voxel_radius[2], 2 * smoothing_voxel_radius[2] + 1)
                ):
                    nx = i + dx
                    ny = j + dy
                    nz = k + dz

                    new_id = int(self.vdb.read_value_world(nx, ny, nz))

                    if new_id > 0:
                        x_j = particle_pos[new_id - 1]
                        w_ij = self.isotropic_weight(x_i, x_j, 2 * smoothing_radius)
                        sum_weight += w_ij
                        x_wi += w_ij * x_j

                if sum_weight == 0.0:
                    self.G[id - 1] = ti.Matrix.zero(ti.f32, 3, 3)
                else:
                    # Only consider non-isolated particles
                    x_wi /= sum_weight
                    self.x_bar[id - 1] = (1 - self.akLambda) * x_i + self.akLambda * x_wi

                    C_i = ti.Matrix.zero(ti.f32, 3, 3)
                    num_neighbor_particles = 0
                    for dx, dy, dz in ti.ndrange(
                            (-2 * smoothing_voxel_radius[0], 2 * smoothing_voxel_radius[0] + 1),
                            (-2 * smoothing_voxel_radius[1], 2 * smoothing_voxel_radius[1] + 1),
                            (-2 * smoothing_voxel_radius[2], 2 * smoothing_voxel_radius[2] + 1)
                    ):
                        nx = i + dx
                        ny = j + dy
                        nz = k + dz

                        new_id = int(self.vdb.read_value_world(nx, ny, nz))

                        if new_id > 0:
                            x_j = particle_pos[new_id - 1]
                            w_ij = self.isotropic_weight(x_i, x_j, 2 * smoothing_radius)
                            if w_ij > 0.0:
                                num_neighbor_particles += 1
                                dp = x_j - x_wi
                                C_i += w_ij * dp.outer_product(dp)

                    C_i /= sum_weight

                    R, Sigma, RT = ti.svd(C_i)

                    # Then, we see if we need to modify C_i
                    # condition is  sigma_1 >= kr * sigma_d
                    if num_neighbor_particles > self.Neps:
                        Sigma[1, 1] = ti.max(Sigma[1, 1], Sigma[0, 0] / self.kr)
                        Sigma[2, 2] = ti.max(Sigma[2, 2], Sigma[0, 0] / self.kr)
                        Sigma *= self.ks
                    else:
                        Sigma[0, 0] = self.kn
                        Sigma[1, 1] = self.kn
                        Sigma[2, 2] = self.kn

                    Sigma[0, 0] = 1.0 / Sigma[0, 0]
                    Sigma[1, 1] = 1.0 / Sigma[1, 1]
                    Sigma[2, 2] = 1.0 / Sigma[2, 2]

                    self.G[id - 1] = (1 / smoothing_radius) * R @ Sigma @ R.transpose()
                    # print(f"{1/smoothing_radius}, {R.determinant()}, {Sigma.determinant()}, {R. determinant()}, {self.G[id - 1].determinant()}")

    @ti.kernel
    def fill_vdb_grid(self, particle_pos: ti.template(), num_particles: ti.template(), smoothing_radius: ti.f32):
        ti.loop_config(block_dim=512)
        prune_search_radius = ti.ceil(smoothing_radius * self.vdb.data_wrapper.inv_voxel_dim, ti.i32) - 1
        prune_threshold = 0.1
        Ns = (2 * prune_search_radius[0] - 1) * (2 * prune_search_radius[1] - 1) * (2 * prune_search_radius[2] - 1)

        for id in range(num_particles):
            self.G[id] = ti.Matrix.identity(dt=ti.f32, n=3)
            # i, j, k = self.vdb.transform.coord_to_voxel_packed(particle_pos[id])
            self.vdb.set_value_packed(particle_pos[id], id + 1)

        # used_particle_count = 0
        # for id in range(num_particles):
        #     particle_count = 0
        #     i, j, k = self.vdb.transform.coord_to_voxel_packed(particle_pos[id])
        #     for di, dj, dk in ti.ndrange((-prune_search_radius[0], prune_search_radius[0] + 1),
        #                                  (-prune_search_radius[1], prune_search_radius[1] + 1),
        #                                  (-prune_search_radius[2], prune_search_radius[2] + 1)
        #                                  ):
        #         ni = i + di
        #         nj = j + dj
        #         nk = k + dk
        #         if self.vdb.read_value_world(ni, nj, nk) > 0:
        #             particle_count += 1
        #
        #     if ti.abs(Ns - particle_count) > prune_threshold * Ns:
        #         used_particle_count += 1
        #         self.sdf.set_value_packed(particle_pos[id], id + 1)
        # print(f"{used_particle_count}")

    @ti.kernel
    def mark_surface_vertex(self):
        for i, j, k in self.vdb.data_wrapper.leaf_value:
            for dx, dy, dz in ti.static(ti.ndrange((0, 2), (0, 2), (0, 2))):
                self.sdf.add_value_world(i + dx, j + dy, k + dz, 1)

    @ti.func
    def anisotropic_kernel(self, dx, G):
        q = (G @ dx).norm()
        res = 0.0
        # Wayland C6
        if 0 <= q <= 2:
            res = ti.static(1365 / (512 * np.pi)) * ti.pow(1 - q / 2, 8) * (
                    4 * ti.pow(q, 3) + 6.25 * q * q + 4 * q + 1) * G.determinant()

        return res

    @ti.kernel
    def compute_sdf_fixed_volume(self, smoothing_radius: ti.f32, volume: ti.f32):
        smoothing_voxel_radius = ti.ceil(smoothing_radius * self.vdb.data_wrapper.inv_voxel_dim, ti.i32) - 1
        for i, j, k in self.sdf.data_wrapper.leaf_value:
            sdf_value = 0.0

            vertex_pos = self.sdf.transform.voxel_to_coord(i, j, k)
            for dx, dy, dz in ti.ndrange(
                    (-2 * smoothing_voxel_radius[0], 2 * smoothing_voxel_radius[0] + 1),
                    (-2 * smoothing_voxel_radius[1], 2 * smoothing_voxel_radius[1] + 1),
                    (-2 * smoothing_voxel_radius[2], 2 * smoothing_voxel_radius[2] + 1)
            ):
                nx = i + dx
                ny = j + dy
                nz = k + dz
                id = int(self.vdb.read_value_world(nx, ny, nz))

                if id > 0:
                    sdf_value -= volume * self.anisotropic_kernel(self.x_bar[id - 1] - vertex_pos, self.G[id - 1])
            self.sdf.set_value_world(i, j, k, sdf_value)

    @ti.kernel
    def rasterize_particles(self, particle_pos: ti.template(), num_particles: ti.template(),
                            particle_radius: ti.template()):
        particle_radius_voxel = 1 + ti.ceil(particle_radius / self.sdf.transform.voxel_dim, ti.i32)

        for id in range(num_particles):
            pos = particle_pos[id]
            pos_voxel_coord = self.sdf.transform.coord_to_voxel_packed(pos)
            for i, j, k in ti.ndrange(
                    (-particle_radius_voxel[0], particle_radius_voxel[0] + 1),
                    (-particle_radius_voxel[1], particle_radius_voxel[1] + 1),
                    (-particle_radius_voxel[2], particle_radius_voxel[2] + 1)
            ):
                adjacent_voxel_coord = pos_voxel_coord + ti.Vector([i, j, k])
                center = self.sdf.transform.voxel_to_coord_packed(adjacent_voxel_coord + ti.Vector([0.5, 0.5, 0.5]))
                value = (center - pos).norm()
                if value - particle_radius < 0 or self.sdf.read_value_world(adjacent_voxel_coord[0], adjacent_voxel_coord[1], adjacent_voxel_coord[2]) >= 0.0:
                    self.vdb.max_value_world(adjacent_voxel_coord[0], adjacent_voxel_coord[1], adjacent_voxel_coord[2],
                                             value - particle_radius)
                    # print(f"Value {value} radius: {particle_radius}")


    @ti.func
    def gaussian_filter(self, x: ti.template(), y: ti.template(), z: ti.template()):
        x2y2z2 = ti.static(x * x + y * y + z * z)
        coeff = ti.static(1 / ti.pow(ti.sqrt(2 * np.pi), 3))
        return ti.static(coeff * ti.exp(-x2y2z2 / 2))

    @ti.kernel
    def gaussian_kernel(self, kernel_width: ti.template()):

        for i, j, k in self.sdf.data_wrapper.leaf_value:
            filtered_value = 0.0
            sample_count = 0
            for di, dj, dk in ti.static(ti.ndrange(
                    (-kernel_width, kernel_width + 1),
                    (-kernel_width, kernel_width + 1),
                    (-kernel_width, kernel_width + 1)
            )):
                ni = i + di
                nj = j + dj
                nk = k + dk
                filtered_value += self.sdf.read_value_world(ni, nj, nk) * ti.static(self.gaussian_filter(di, dj, dk))
            self.vdb.set_value_world(i, j, k, filtered_value)

    @ti.func
    def can_dilate(self, i, j, k):
        return self.sdf.read_value_world(i, j, k) >= 0.0

    @ti.kernel
    def dilate_kernel(self):

        for i, j, k in self.sdf.data_wrapper.leaf_value:
            value = self.sdf.read_value_world(i, j, k)
            if value < 0.0:
                if self.can_dilate(i + 1, j, k):
                    self.vdb.max_value_world(i + 1, j, k, 0.0)
                if self.can_dilate(i - 1, j, k):
                    self.vdb.max_value_world(i - 1, j, k, 0.0)
                if self.can_dilate(i, j + 1, k):
                    self.vdb.max_value_world(i, j + 1, k, 0.0)
                if self.can_dilate(i, j - 1, k):
                    self.vdb.max_value_world(i, j - 1, k, 0.0)
                if self.can_dilate(i, j, k + 1):
                    self.vdb.max_value_world(i, j, k + 1, 0.0)
                if self.can_dilate(i, j, k - 1):
                    self.vdb.max_value_world(i, j, k - 1, 0.0)

    @ti.func
    def can_erode(self, i, j, k):
        return self.sdf.read_value_world(i, j, k) >= 0.0
    @ti.kernel
    def erode_kernel(self):

        for i, j, k in self.sdf.data_wrapper.leaf_value:
            value = self.sdf.read_value_world(i, j, k)
            if value < 0.0:
                is_erode = self.can_erode(i + 1, j, k)
                is_erode |= self.can_erode(i - 1, j, k)
                is_erode |= self.can_erode(i, j + 1, k)
                is_erode |= self.can_erode(i, j - 1, k)
                is_erode |= self.can_erode(i, j, k + 1)
                is_erode |= self.can_erode(i, j, k - 1)

                if not is_erode:
                    self.vdb.set_value_world(i, j, k, value)





    ## @brief Implementation of anisotropic kernel sampling of particles
    def particle_to_sdf_anisotropic(self, particle_pos: ti.template(), num_particles: ti.template(),
                                    particle_radius: ti.f32, smoothing_radius=0.03):
        particle_volume = ti.static(4 * np.pi / 3) * particle_radius * particle_radius * particle_radius
        # Step 1: Fill particle grid
        self.fill_vdb_grid(particle_pos, num_particles, smoothing_radius)
        # field_copy(self.vdb.data_wrapper.leaf_value, self.sdf.data_wrapper.leaf_value)
        # self.vdb.prune(0)
        # self.sdf.clear()
        # Step 2: Compute anisotropic kernel
        self.compute_anisotropic_kernel_with_grid(particle_pos, num_particles, smoothing_radius)
        # Step 3: Mark Surface Vertices
        self.mark_surface_vertex()

        # Step 4: Compute sdf with fixed volume
        self.compute_sdf_fixed_volume(smoothing_radius, particle_volume)
        # Step 5: Rasterize particles
        self.vdb.clear()
        field_copy(self.vdb.data_wrapper.leaf_value, self.sdf.data_wrapper.leaf_value)
        self.rasterize_particles(particle_pos, num_particles, particle_radius)
        self.sdf.clear()
        field_copy(self.sdf.data_wrapper.leaf_value, self.vdb.data_wrapper.leaf_value)

        # # Dilate
        # self.vdb.clear()
        # field_copy(self.vdb.data_wrapper.leaf_value, self.sdf.data_wrapper.leaf_value)
        # self.dilate_kernel()
        # self.sdf.clear()
        # field_copy(self.sdf.data_wrapper.leaf_value, self.vdb.data_wrapper.leaf_value)
        #
        # self.vdb.clear()
        # self.gaussian_kernel(1)
        # self.sdf.clear()
        # field_copy(self.sdf.data_wrapper.leaf_value, self.vdb.data_wrapper.leaf_value)
        #
        # # Erode
        # self.vdb.clear()
        # self.erode_kernel()
        # self.sdf.clear()
        # field_copy(self.sdf.data_wrapper.leaf_value, self.vdb.data_wrapper.leaf_value)
        #
        # self.vdb.clear()
        # self.erode_kernel()
        # self.sdf.clear()
        # field_copy(self.sdf.data_wrapper.leaf_value, self.vdb.data_wrapper.leaf_value)

