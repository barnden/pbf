#pragma once

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include "CUDA/Kernels.cuh"
#include "CUDA/Parameters.cuh"
#include "CUDA/Spatial.cuh"
#include "CUDA/Simulation/Particles.cuh"

#include "Defines.h"

namespace pbf::simulation {
__host__ auto sort(
    Particles::DevicePointers const particles,
    thrust::device_vector<u32>& grid,
    thrust::device_vector<u32>& output,
    float spacing) -> void
{
    namespace kernels = pbf::kernels::simulation;
    using pbf::parameters::CUDA::elements_per_thread;
    using pbf::parameters::CUDA::threads_per_block;

    // (0) Clear grid
    cudaMemset(raw_pointer_cast(grid.data()), 0, grid.size() * sizeof(u32));

    {
        // (1) Bin particles into flattened grid cells via hashing
        auto num_blocks = (particles.size + threads_per_block - 1u) / threads_per_block;

        kernels::quantize<<<num_blocks, threads_per_block>>>(
            particles.d_position,
            thrust::raw_pointer_cast(grid.data()),
            static_cast<u32>(grid.size()),
            static_cast<u32>(particles.size),
            spacing);
    }

    {
        // (2) Prefix sum flatted grid
        thrust::inclusive_scan(thrust::device,
                               grid.begin(), grid.end(),
                               grid.begin());
    }

    {
        // (3) Sort elements by grid
        auto num_threads = (particles.size + elements_per_thread - 1u) / elements_per_thread;
        auto num_blocks = (num_threads + threads_per_block - 1u) / threads_per_block;

        auto keyof_position =
            [spacing, grid_size = grid.size()] __device__(float3 position) -> unsigned {
            return pbf::spatial::hash(position, spacing, grid_size);
        };

        pbf::kernels::sort::placement<<<num_blocks, threads_per_block>>>(
            keyof_position,
            thrust::raw_pointer_cast(particles.d_position),
            thrust::raw_pointer_cast(grid.data()),
            thrust::raw_pointer_cast(output.data()),
            static_cast<unsigned>(particles.size),
            elements_per_thread);
    }
}
}
