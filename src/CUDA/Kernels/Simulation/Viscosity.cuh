#pragma once

#include <cuda.h>

#include "Defines.h"

#include "CUDA/Math.cuh"
#include "CUDA/Parameters.cuh"
#include "CUDA/Simulation/Particles.cuh"
#include "CUDA/Spatial.cuh"

namespace pbf::kernels::simulation {
__global__ auto compute_velocity_and_vorticity(
    Particles::DevicePointers const particles,
    unsigned const* const d_neighbours,
    float const h,
    float const support_radius,
    float const viscosity_correction,
    u32 const max_neighbours) -> void
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= particles.size)
        return;

    auto const v_i = (particles.d_position_star[idx] - particles.d_position[idx]) / h;
    float3 v_new { 0., 0., 0. };
    float3 omega { 0., 0., 0. };

    auto const offset = idx * max_neighbours;
    auto const num_neighbours = d_neighbours[offset];
    for (auto i = offset + 1; i < offset + num_neighbours + 1; i++) {
        auto const j = d_neighbours[i];

        auto const v_j = (particles.d_position_star[j] - particles.d_position[j]) / h;
        auto const v_ij = v_j - v_i;
        auto const p_ij = particles.d_position_star[idx] - particles.d_position_star[j];
        auto const rho_j = particles.d_density[j];

        auto const W = pbf::math::poly6(p_ij, support_radius);

        if constexpr (pbf::parameters::debug::apply_vorticity_confinement) {
            auto const nablaW = pbf::math::grad_spiky(p_ij, support_radius);
            omega = omega + cross(v_ij, nablaW) / (rho_j + 1.f);
        }

        v_new = v_new + v_ij * W / (rho_j + 1.f);
    }

    if constexpr (pbf::parameters::debug::apply_vorticity_confinement) {
        particles.d_vorticity[idx] = omega;
    }

    if constexpr (pbf::parameters::debug::apply_xsph_viscosity) {
        particles.d_velocity[idx] = v_i + viscosity_correction * v_new;
    } else {
        particles.d_velocity[idx] = v_i;
    }
}
}
