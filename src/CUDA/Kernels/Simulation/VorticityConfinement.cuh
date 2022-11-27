#pragma once

#include <cuda.h>

#include "Defines.h"

#include "CUDA/Math.cuh"
#include "CUDA/Parameters.cuh"
#include "CUDA/Simulation/Particles.cuh"
#include "CUDA/Spatial.cuh"

namespace pbf::kernels::simulation {
__global__ auto confine_vorticity_then_update(
    Particles::DevicePointers const particles,
    unsigned const* const d_neighbours,
    float const vorticity_strength,
    float const h,
    float const support_radius,
    u32 const max_neighbours) -> void
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= particles.size)
        return;

    auto const p_i = particles.d_position_star[idx];
    particles.d_position[idx] = p_i;

    if constexpr (pbf::parameters::debug::apply_vorticity_confinement) {
        auto const offset = idx * max_neighbours;
        auto const num_neighbours = d_neighbours[offset];

        auto const omega_i = particles.d_vorticity[idx];
        auto const length_omega_i = L2(omega_i);
        float3 eta { 0., 0., 0. };
        for (auto i = offset + 1; i < offset + num_neighbours + 1; i++) {
            auto const j = d_neighbours[i];
            auto const length_omega_j = L2(particles.d_vorticity[j]);
            auto const p_j = particles.d_position_star[j];
            auto const nablaW = pbf::math::grad_spiky(p_i - p_j, support_radius);

            eta = eta + nablaW * (length_omega_j - length_omega_i);
        }

        float3 F_vorticity = vorticity_strength * cross(normalize(eta), omega_i);
        particles.d_velocity[idx] = particles.d_velocity[idx] + F_vorticity * h;
    }
}
}
