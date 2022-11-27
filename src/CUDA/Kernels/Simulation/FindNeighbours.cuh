#pragma once

#include <cuda.h>

#include "Defines.h"

#include "CUDA/Math.cuh"
#include "CUDA/Spatial.cuh"
#include "CUDA/Simulation/Particles.cuh"

namespace pbf::kernels::simulation {

__global__ auto find_neighbours(
    Particles::DevicePointers const particles,
    u32 const* const d_grid,
    u32 const* const d_entries,
    u32* const d_neighbours,
    float const spacing,
    float const support_radius,
    u32 const grid_size,
    u32 const max_neighbours) -> void
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= particles.size)
        return;

    auto const p_i = particles.d_position[idx];
    auto const lb = to_int3((p_i - spacing) / spacing);
    auto const ub = to_int3((p_i + spacing) / spacing);
    auto const offset = idx * max_neighbours;
    float const max_norm = support_radius * support_radius;

    auto num_neighbours = 0;

    for (auto x = lb.x; x <= ub.x; x++) {
        for (auto y = lb.y; y <= ub.y; y++) {
            for (auto z = lb.z; z <= ub.z; z++) {
                auto hash = pbf::spatial::hash({ x, y, z }, grid_size);

                auto start = d_grid[hash];
                auto end = d_grid[hash + 1];

                for (auto i = start; i < end; i++) {
                    auto const j = d_entries[i];

                    if (j == idx)
                        continue;

                    auto const p_j = particles.d_position[j];
                    float distance = L2_squared(p_i - p_j);

                    if (distance > max_norm)
                        continue;

                    d_neighbours[offset + num_neighbours + 1] = j;
                    num_neighbours++;

                    if (num_neighbours >= max_neighbours - 1) {
                        d_neighbours[offset] = max_neighbours - 1;
                        return;
                    }
                }
            }
        }
    }

    d_neighbours[offset] = num_neighbours;
}
}
