#pragma once

#include <cuda.h>

#include "Defines.h"

#include "CUDA/Math.cuh"
#include "CUDA/Parameters.cuh"
#include "CUDA/Simulation/Particles.cuh"
#include "CUDA/Spatial.cuh"

namespace pbf::kernels::simulation {
__global__ auto laplacian_smoothing(
    Particles::DevicePointers const particles,
    u32 const* const d_neighbours,
    float const support_radius,
    float const smoothing,
    u32 const max_neighbours) -> void
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= particles.size)
        return;

    auto denoised = float3 { 0., 0., 0. };
    auto const p_i = particles.d_position[idx];

    auto const offset = idx * max_neighbours;
    auto const num_neighbours = d_neighbours[offset];
    for (auto i = offset + 1; i < offset + num_neighbours + 1; i++) {
        auto const j = d_neighbours[i];
        auto const p_j = particles.d_position[j];
        auto const rho_j = particles.d_density[j];

        auto const W = pbf::math::poly6(p_i - p_j, support_radius);

        denoised = denoised + W * p_j / (rho_j + 1.f);
    }

    particles.d_position_star[idx] = (1. - smoothing) * p_i + smoothing * denoised;
}
}
