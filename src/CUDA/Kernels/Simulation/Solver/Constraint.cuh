#pragma once

#include <cuda.h>

#include "Defines.h"

#include "CUDA/Math.cuh"
#include "CUDA/Spatial.cuh"
#include "CUDA/Simulation/Particles.cuh"

namespace pbf::kernels::solver {

__global__ auto compute_constraints(
    Particles::DevicePointers const particles,
    unsigned const* const d_neighbours,
    float const rest_density,
    float const relaxation,
    float const support_radius,
    u32 max_neighbours) -> void
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= particles.size)
        return;

    auto const offset = idx * max_neighbours;
    auto const num_neighbours = d_neighbours[offset];

    auto p_i = particles.d_position_star[idx];
    // Calculate lambda_i = C_i(p_1, ..., p_n) / { sum_k | nabla_p_k C_i |^2 + epsilon } (eq. 9)

    // (1) calculate rho_i and nabla_p_k C_i
    auto rho_i = 0.f;
    float kj_grad = 0.f;
    float3 ki_grad { 0., 0., 0. };

    for (auto j = offset + 1; j < offset + num_neighbours + 1; j++) {
        auto p_j = particles.d_position_star[d_neighbours[j]];
        auto nablaW = pbf::math::grad_spiky(p_i - p_j, support_radius) / rest_density;
        auto W = pbf::math::poly6(p_i - p_j, support_radius);

        rho_i += W;

        // for k == j, nabla_p_k C_i = - nabla_p_k W(p_i - p_j, h)
        kj_grad += L2_squared(nablaW);

        // for k == i, nabla_p_k C_i = 1 / rho_0 sum_j nabla_p_k W(p_i - p_j, h)
        ki_grad = ki_grad + nablaW;
    }
    // (2) C_i(p_1, ..., p_n) = rho_i / rho_0 - 1
    float C_i = rho_i / rest_density - 1.f;

    // (3) Compute lambda
    float denominator = relaxation + kj_grad + L2_squared(ki_grad);

    particles.d_constraint[idx] = -C_i / denominator;
}

}
