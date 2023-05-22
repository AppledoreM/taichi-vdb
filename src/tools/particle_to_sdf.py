## @package ParticleToSdf
# Converting particles to sdf with method:
# "Reconstructing Surfaces of Particle-Based Fluids Using Anisotropic Kernels" by JinHun Yu and Greg Turk


import taichi as ti
from src.vdb_grid import *

@ti.data_oriented
class ParticleToSdf:

    # Implementation of anisotropic sampling of particles

    def __init__(self, bounding_box, level_configs, max_num_particles):
        assert level_configs is not None, "Level configuration must not be None."
        self.sdf = VdbGrid(bounding_box, level_configs)
        self.max_num_particles = max_num_particles
        self.vdb = VdbGrid(bounding_box, level_configs, ti.i32)

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

                self.G[id - 1] = (1 / smoothing_radius) * R @ Sigma @ RT.transpose()

    @ti.kernel
    def fill_vdb_grid(self, particle_pos: ti.template(), num_particles: ti.template()):
        ti.loop_config(block_dim=512)
        for i in range(num_particles):
            self.vdb.set_value_coord(particle_pos[i], i + 1)


    @ti.kernel
    def mark_surface_vertex(self, smoothing_radius: ti.f32):
        smoothing_voxel_radius = ti.ceil(smoothing_radius * self.vdb.data_wrapper.inv_voxel_dim, ti.i32) - 1
        for i, j, k in self.vdb.data_wrapper.leaf_value:
            is_surface_cell = False
            for dx, dy, dz in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
                if self.vdb.read_value_world(i + dx, j + dy, k + dz) == 0:
                    is_surface_cell = True
                    break

            if is_surface_cell:
                for dx, dy, dz in ti.static(ti.ndrange((0, 2), (0, 2), (0, 2))):
                    self.sdf.set_value_world(i + dx, j + dy, k + dz, 1)


    @ti.func
    def anisotropic_kernel(self, dx, G):
        res = 0.0
        q = (G @ dx).norm()

        if 0 < q <= 0.5:
            res = G.determinant() * 8 / 3.14 * (6 * (q * q * q - q * q) + 1)
        elif 0.5 < q <= 1:
            res = 16 / 3.14 * G.determinant() * (1 - q) * (1 - q) * (1 - q)
        return res


    @ti.kernel
    def compute_sdf_fixed_volume(self,smoothing_radius: ti.f32, volume: ti.f32):
        smoothing_voxel_radius = ti.ceil(smoothing_radius * self.vdb.data_wrapper.inv_voxel_dim, ti.i32) - 1
        for i, j, k in self.sdf.data_wrapper.leaf_value:
            sdf_value = 0.0

            vertex_pos = ti.Vector([i, j, k]) * self.sdf.voxel_dim
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
                    sdf_value += volume * self.anisotropic_kernel(self.x_bar[id - 1] - vertex_pos, self.G[id - 1])
            self.sdf.set_value_world(i, j, k, sdf_value)



    ## @brief Implementation of anisotropic kernel sampling of particles
    def particle_to_sdf_anisotropic(self, particle_pos: ti.template(), num_particles: ti.template(),
                                    particle_radius: ti.f32, smoothing_radius=0.03):
        particle_volume = 4 / 3 * 3.14 * particle_radius * particle_radius * particle_radius
        # Step 1: Fill particle grid
        self.fill_vdb_grid(particle_pos, num_particles)
        # Step 2: Compute anisotropic kernel
        self.compute_anisotropic_kernel_with_grid(particle_pos, num_particles, smoothing_radius)
        # Step 3: Mark Surface Vertices
        self.mark_surface_vertex(smoothing_radius)
        # Step 4: Compute sdf with fixed volume
        self.compute_sdf_fixed_volume(smoothing_radius, particle_volume)











