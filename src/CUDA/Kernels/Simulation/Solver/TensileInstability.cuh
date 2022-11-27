#pragma once

#include <cuda.h>

#include "Defines.h"

#include "CUDA/Math.cuh"
#include "CUDA/Simulation/Particles.cuh"
#include "CUDA/Spatial.cuh"

namespace pbf::kernels::solver {

template <class ArtificialPressure>
__global__ auto compute_tensile_instability(
    Particles::DevicePointers particles,
    unsigned const* const d_neighbours,
    float const rest_density,
    float const support_radius,
    ArtificialPressure const artificial_pressure,
    u32 const max_neighbours) -> void
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= particles.size)
        return;

    auto const offset = idx * max_neighbours;
    auto const num_neighbours = d_neighbours[offset];

    if (num_neighbours == 0) {
        particles.d_position_delta[idx] = float3 { 0., 0., 0. };
        return;
    }

    // delta P_i := 1/rho_0 sum_j (lambda_i + lambda_j + s_corr) nablaW(p_i - p_j, h)
    auto const lambda_i = particles.d_constraint[idx];
    auto const p_i = particles.d_position_star[idx];

    // Denominator of (13): W(delta q, h)
    auto WdeltaQ = pbf::math::poly6(
        float3 { 0., 0., artificial_pressure.deltaQ * support_radius },
        support_radius);

    auto deltaP = float3 { 0., 0., 0. };

    for (auto i = offset + 1; i < offset + num_neighbours + 1; i++) {
        auto const j = d_neighbours[i];
        auto const lambda_j = particles.d_constraint[j];
        auto const p_j = particles.d_position_star[j];

        auto const W = pbf::math::poly6(p_i - p_j, support_radius);
        auto const nablaW = pbf::math::grad_spiky(p_i - p_j, support_radius);

        auto const s_corr = -artificial_pressure.k * powf(W / WdeltaQ, artificial_pressure.n);

        deltaP = deltaP + nablaW * (lambda_i + lambda_j + s_corr);
    }

    particles.d_position_delta[idx] = deltaP / rest_density;
}
}
