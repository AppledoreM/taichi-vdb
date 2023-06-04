import taichi as ti
import numpy as np
from src.vdb_grid import *
from src.vdb_math import poly6_kernel_3d


class SPHFluid:
    class SPHSamplingType:
        Default = 0
        Pressure = 1
        Viscosity = 2

    @staticmethod
    @ti.func
    def spiky_kernel_3d(h: ti.template(), r: ti.template()):
        res = 0.0
        if 0 <= r <= h:
            res = ti.static(15 / np.pi) * ti.pow(h - r, 3) / ti.pow(h, 6)

        return res

    @staticmethod
    @ti.func
    def grad_spiky_kernel_3d(h: ti.template(), r: ti.template()):
        res = 0.0

        if 0 <= r <= h:
            res = ti.static(-45 / np.pi) * (h - r) * (h - r) / ti.pow(h, 6)

        return res

    @staticmethod
    @ti.func
    def viscosity_kernel_3d(h: ti.template(), r: ti.template()):
        res = 0.0
        if 0 <= r <= h:
            res = ti.static(15 / (2 * np.pi)) * (-ti.pow(r, 3) / (2 * ti.pow(h, 3)) + r * r / (h * h) + h / (2 * r) - 1) / ti.pow(h, 3)

        return res

    @staticmethod
    @ti.func
    def laplacian_viscosity_kernel_3d(h: ti.template(), r: ti.template()):
        res = 0.0
        if 0 <= r <= h:
            res = ti.static(45 / np.pi) * (h - r) / ti.pow(h, 6)

        return res

    def __init__(self, k, rho0, mu, voxel_size, max_particle_count: ti.template(),
                 bounding_box: ti.types.ndarray(dtype=ti.f32, ndim=1, element_dim=3),
                 dtype=ti.f32, is_uniform_mass: ti.template() = True, is_uniform_density: ti.template() = True):

        self.max_particle_count = max_particle_count
        self.particle_count = ti.field(dtype=ti.i32, shape=())
        self.is_uniform_mass = ti.static(is_uniform_mass)
        self.is_uniform_density = ti.static(is_uniform_density)

        self.voxel_size = voxel_size

        self.k = k
        self.rho0 = rho0
        self.mu = mu

        if ti.static(is_uniform_mass):
            self.particle_mass = 0
            self.particle_density = 0
        else:
            self.particle_mass = ti.field(dtype=dtype, shape=max_particle_count)
            self.particle_density = ti.field(dtype=dtype, shape=max_particle_count)

        self.particle_pos = ti.Vector.field(n=3, dtype=dtype, shape=max_particle_count)
        self.particle_vel = ti.Vector.field(n=3, dtype=dtype, shape=max_particle_count)
        self.particle_force = ti.field(dtype=dtype, shape=max_particle_count)

        self.bounding_box = bounding_box
        self.particle_marker = VdbGrid(voxel_size, dtype=ti.i32, origin=bounding_box[0])

    @ti.kernel 
    def add_particles(self, particle_count: ti.i32, particle_pos: ti.template(), particle_mass: ti.template(), 
                      particle_density: ti.template()):

        self.particle_count[None] += particle_count
        assert self.particle_count[None] < self.max_particle_count, f"Cannot add more particles than {self.max_particle_count}"

        if ti.static(self.is_uniform_mass):
            self.particle_mass = particle_mass
        if ti.static(self.is_uniform_density):
            self.particle_density = particle_density

        for i in range(particle_count):
            self.particle_pos[i] = particle_pos[i]

            if not ti.static(self.is_uniform_mass):
                self.particle_mass[i] = particle_mass[i]
            if not ti.static(self.is_uniform_density):
                self.particle_density[i] = particle_density[i]


    @ti.func
    def sample_quantity(self, i: ti.i32, j: ti.i32, k: ti.i32, 
                        A: ti.template(), B: ti.template(),
                        h: ti.f32, voxel_support_radius: ti.i32, 
                        is_sample_density: ti.template(), sph_sampling_type: ti.template() 
                       ):
        target_id = self.particle_marker.read_value_world(i, j, k) - 1
        if target_id >= 0:
            r_i = self.particle_pos[target_id]

            support_w = (-voxel_support_radius, voxel_support_radius)
            sampled_A = 0.0
            for di, dj, dk in ti.ndrange(support_w, support_w, support_w):
                particle_id = self.particle_marker.read_value_world(i + di, j + dj, k + dk)

                if particle_id > 0:
                    particle_id -= 1
                    sample = 0.0

                    r_j = self.particle_pos[particle_id]

                    if ti.static(self.is_uniform_mass):
                        sample = self.particle_mass
                    else:
                        sample = self.particle_mass[particle_id]

                    if not ti.static(is_sample_density):
                        sample *= B[particle_id]
                        if ti.static(self.is_uniform_density):
                            sample /= self.particle_density
                        else:
                            sample /= self.particle_density[particle_id]

                    weight = 0.0
                    if ti.static(sph_sampling_type == SPHFluid.SPHSamplingType.Default):
                        weight = poly6_kernel_3d((r_i - r_j).norm(), h)
                    elif ti.static(sph_sampling_type == SPHFluid.SPHSamplingType.Pressure):
                        weight = self.spiky_kernel_3d((r_i - r_j).norm(), h)
                    elif ti.static(sph_sampling_type == SPHFluid.SPHSamplingType.Viscosity):
                        weight = self.viscosity_kernel_3d((r_i - r_j).norm(), h)

                    sampled_A += sample * weight

            A[target_id] = sampled_A

    @ti.kernel
    def step(self, dt: ti.f32):
        h = self.voxel_size
        voxel_h = 1
        voxel_support = (-voxel_h, voxel_h)

        # Sample Density
        for i, j, k in self.particle_marker:
            self.sample_quantity(i, j, k, self.particle_density, None, h, voxel_h, True, SPHFluid.SPHSamplingType.Default)

        # Compute Pressure
        for i in range(self.particle_count[None]):
            self.pressure[i] = self.k * (self.particle_density[i] - self.rho0)

        # Compute Forces
        for i, j, k in self.particle_marker:
            target_id = self.particle_marker.read_value_world(i, j, k) - 1

            if target_id > 0:
                r_i = self.particle_pos[target_id]
                p_i = self.k * (self.particle_density[target_id] - self.rho0)
                v_i = self.particle_vel[i]
                
                ext_force = 0.0
                for di, dj, dk in ti.ndrange(voxel_support, voxel_support, voxel_support):
                    ni = i + di
                    nj = j + dj
                    nk = k + dk

                    neighbor_id = self.particle_marker.read_value_world(ni, nj, nk) - 1
                    if neighbor_id > 0:
                        r_j = self.particle_pos[neighbor_id]
                        m_j = self.particle_mass[neighbor_id]
                        p_j = self.k * (self.particle_density[neighbor_id] - self.rho0)
                        rho_j = self.particle_density[neighbor_id]
                        
                        # Compute pressure force
                        pressure_force = -m_j * (p_i + p_j) / (2 * rho_j) * SPHFluid.spiky_kernel_3d(voxel_h, (r_i - r_j).norm())
                        viscosity_force = self.mu * m_j * (self.particle_vel[j] - v_i) / rho_j  * SPHFluid.laplacian_viscosity_kernel_3d(voxel_h, (r_i - r_j).norm())
                        ext_force += pressure_force + viscosity_force

                self.particle_force[target_id] = ext_force

        # Update Velocity
        for i in range(self.particle_count[None]):
            accel = self.particle_force[i] / self.particle_density[i] - 9.8
            self.particle_vel[i] += accel * dt
            self.particle_pos[i] += dt * self.particle_vel[i]

