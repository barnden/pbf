#pragma once

#include <cuda.h>

#include "Defines.h"

#include "CUDA/Math.cuh"
#include "CUDA/Parameters.cuh"
#include "CUDA/Simulation/Particles.cuh"

namespace pbf::kernels::simulation {

__global__ void euler_integration(
    Particles::DevicePointers const particles,
    float3 const F_gravity,
    float const h)
{
    /**
     * NOTE: I opted to not have a external force vector since I only intend on
     *       having gravity act on the fluid. A trivial modification to the
     *       kernel can be done to accept a vector of external forces.
     */
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= particles.size)
        return;

    particles.d_velocity[idx] = particles.d_velocity[idx] + h * F_gravity;
    particles.d_position_star[idx] = particles.d_position[idx] + h * particles.d_velocity[idx];
}

}
