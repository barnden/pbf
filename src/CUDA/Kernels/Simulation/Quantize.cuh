#pragma once

#include <cuda.h>

#include "Defines.h"

#include "CUDA/Math.cuh"
#include "CUDA/Parameters.cuh"

namespace pbf::kernels::simulation {

__global__ auto quantize(
    float3* d_position_star,
    u32* d_grid,
    u32 grid_size,
    u32 num_particles,
    float spacing) -> void
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_particles)
        return;

    auto position = d_position_star[idx];
    auto hash = pbf::spatial::hash(position, spacing, grid_size);

    atomicAdd(&d_grid[hash], 1);
}

}