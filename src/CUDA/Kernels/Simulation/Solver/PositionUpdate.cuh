#pragma once

#include <cuda.h>

#include "Defines.h"

#include "CUDA/Math.cuh"
#include "CUDA/Simulation/Particles.cuh"
#include "CUDA/Spatial.cuh"

namespace pbf::kernels::solver {

template <class Box>
__global__ auto update_positions(
    Particles::DevicePointers const particles,
    Box const domain) -> void
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= particles.size)
        return;

    auto p = particles.d_position_star[idx] + particles.d_position_delta[idx];

    p.x = max(p.x, domain.min.x);
    p.y = max(p.y, domain.min.y);
    p.z = max(p.z, domain.min.z);

    p.x = min(p.x, domain.max.x);
    p.y = min(p.y, domain.max.y);
    p.z = min(p.z, domain.max.z);

    particles.d_position_star[idx] = p;
}

}
