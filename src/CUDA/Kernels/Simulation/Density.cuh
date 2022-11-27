#pragma once

#include <cuda.h>

#include "Defines.h"

#include "CUDA/Math.cuh"
#include "CUDA/Parameters.cuh"
#include "CUDA/Simulation/Particles.cuh"
#include "CUDA/Spatial.cuh"

namespace pbf::kernels::simulation {
__global__ auto compute_density(
    Particles::DevicePointers const particles,
    u32 const* const d_neighbours,
    float const support_radius,
    u32 const max_neighbours) -> void
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= particles.size)
        return;

    auto rho_i = 0.f;
    auto const p_i = particles.d_position_star[idx];

    auto const offset = idx * max_neighbours;
    auto const num_neighbours = d_neighbours[offset];
    for (auto i = offset + 1; i < offset + num_neighbours + 1; i++) {
        auto const j = d_neighbours[i];
        auto const p_j = particles.d_position_star[j];

        rho_i += pbf::math::poly6(p_i - p_j, support_radius);
    }

    particles.d_density[idx] = rho_i;
}
}
