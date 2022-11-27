#pragma once

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "CUDA/Kernels.cuh"
#include "CUDA/Simulation/Sort.cuh"
#include "CUDA/Simulation/Particles.cuh"

#include "Defines.h"

namespace pbf {
namespace kernels::simulation {
    __global__ auto dampen(
        float damping,
        float3* const d_velocity,
        u32 const num_particles) -> void
    {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= num_particles)
            return;

        d_velocity[idx] = damping * d_velocity[idx];
    }
};

class PBF {
public:
    Particles m_particles;

    thrust::device_vector<u32> m_grid;
    thrust::device_vector<u32> m_sorted_indices;
    thrust::device_ptr<u32> m_neighbours;

    u32 m_solver_iterations = 4;
    u32 m_neighbours_size;

    __host__ auto __debug_print_statistics() -> void
    {
        auto sum = 0u;
        auto min_neighbours = std::numeric_limits<u32>::max();
        auto max_neighbours = std::numeric_limits<u32>::min();
        for (auto i = 0uz; i < size(); i++) {
            auto const offset = i * parameters::hashing::max_neighbours;
            u32 const num_neighbours = m_neighbours[offset];

            min_neighbours = std::min(min_neighbours, num_neighbours);
            max_neighbours = std::max(max_neighbours, num_neighbours);
            sum += num_neighbours;
        }

        std::cout << "[Debug] avg neighbours: " << (sum / static_cast<float>(size()))
                  << " min: " << min_neighbours
                  << " max: " << max_neighbours << std::endl;
    }

    __host__ auto __debug_print_neighbours() -> void
    {
        for (auto i = 0uz; i < size(); i++) {
            auto const offset = i * parameters::hashing::max_neighbours;
            auto const num_neighbours = m_neighbours[offset];

            std::cout << "particle " << i << " has " << num_neighbours << " neighbours [";
            for (auto j = offset + 1; j < offset + num_neighbours + 1; j++) {
                auto distance = std::sqrtf(L2_squared(m_particles.position_star()[i] - m_particles.position_star()[m_neighbours[j]]));
                std::cout << m_neighbours[j] << " (dist: " << distance << ")" << ", ";
            }
            std::cout << "], actual: [";

            for (auto j = 0uz; j < size(); j++) {
                if (i == j)
                    continue;

                auto distance = std::sqrtf(L2_squared(m_particles.position_star()[i] - m_particles.position_star()[j]));
                if (distance < 0.1) {
                    std::cout << j << " (dist: " << distance << ")" << ", ";
                }
            }

            std::cout << "]" << std::endl;
        }
    }

    auto relax(size_t N = 20) -> void
    {
        auto const num_blocks = (size() + 255) / 256;
        for (auto i = 0uz; i < N; i++) {
            auto damping_constant = i / (2. * static_cast<float>(N));
            step(1e-4 * pbf::parameters::physical::timestep);
            pbf::kernels::simulation::dampen<<<num_blocks, 256>>>(
                damping_constant,
                thrust::raw_pointer_cast(m_particles.velocity()),
                size());
        }

        cudaMemset(
            thrust::raw_pointer_cast(m_particles.velocity()),
            0,
            sizeof(float3) * size());
    }

public:
    PBF(size_t N)
        : m_particles(Particles::initialize(N))
        , m_grid(pbf::parameters::hashing::grid_size)
        , m_sorted_indices(m_particles.size())
    {
        using parameters::hashing::max_neighbours;

        if (std::numeric_limits<u32>::max() / max_neighbours <= m_particles.size()) {
            throw "Neighbours array would overflow.";
        }

        m_neighbours_size = m_particles.size() * max_neighbours;
        m_neighbours = thrust::device_malloc<u32>(m_neighbours_size);

        relax(360);
    }

    ~PBF()
    {
        thrust::device_free(m_neighbours);
        m_neighbours = NULL;
    }

    __host__ auto reset() -> void {
        
    }

    __host__ auto step(float timestep = parameters::physical::timestep) -> void
    {
        using thrust::raw_pointer_cast;
        // The PBF algorithm [Position Based Fluids, SIGGRAPH 2013]
        // Line numbers correspond to Algorithm 1 in the paper.
        using pbf::parameters::CUDA::elements_per_thread;
        using pbf::parameters::CUDA::threads_per_block;

        auto num_blocks = (size() + threads_per_block - 1u) / threads_per_block;
        auto particles = m_particles.raw();

        // (1) Integrate [lines 1-4]
        kernels::simulation::euler_integration<<<num_blocks, threads_per_block>>>(
            particles,
            pbf::parameters::physical::gravity,
            timestep);

        // (2) Spatial Hashing to Find Neighbours [lines 5-7]
        simulation::sort(
            particles,
            m_grid,
            m_sorted_indices,
            parameters::hashing::spacing);

        kernels::simulation::find_neighbours<<<num_blocks, threads_per_block>>>(
            particles,
            raw_pointer_cast(m_grid.data()),
            raw_pointer_cast(m_sorted_indices.data()),
            raw_pointer_cast(m_neighbours),
            parameters::hashing::spacing,
            parameters::physical::support_radius,
            m_grid.size(),
            parameters::hashing::max_neighbours);

        if constexpr (parameters::debug::print_neighbours)
            __debug_print_neighbours();

        if constexpr (parameters::debug::print_statistics)
            __debug_print_statistics();

        // (3) Gauss-Seidel iterations [lines 8-19]
        for (auto i = 0u; i < m_solver_iterations; i++) {
            // (3.1) Compute constraints, i.e. lambdas [lines 9-11]
            kernels::solver::compute_constraints<<<num_blocks, threads_per_block>>>(
                particles,
                raw_pointer_cast(m_neighbours),
                parameters::physical::rest_density,
                parameters::physical::relaxation,
                parameters::physical::support_radius,
                parameters::hashing::max_neighbours);

            // (3.2) Tensile instability [lines 12-15]*
            //         * Collision detection will be done in the next kernel
            kernels::solver::compute_tensile_instability<<<num_blocks, threads_per_block>>>(
                particles,
                raw_pointer_cast(m_neighbours),
                parameters::physical::rest_density,
                parameters::physical::support_radius,
                parameters::physical::artificial_pressure,
                parameters::hashing::max_neighbours);

            // (3.3) Collision detection and Positional Update [lines 16-18]
            kernels::solver::update_positions<<<num_blocks, threads_per_block>>>(
                particles,
                parameters::physical::domain);
        }

        // (4) Update particle information [lines 20-24]
        // (4.0) Compute densities
        if constexpr (pbf::parameters::debug::apply_vorticity_confinement || pbf::parameters::debug::apply_xsph_viscosity) {
            kernels::simulation::compute_density<<<num_blocks, threads_per_block>>>(
                particles,
                raw_pointer_cast(m_neighbours),
                parameters::physical::support_radius,
                parameters::hashing::max_neighbours);
        }

        // (4.1) Compute velocity, vorticity, apply XSPH viscosity [lines 21-22]
        kernels::simulation::compute_velocity_and_vorticity<<<num_blocks, threads_per_block>>>(
            particles,
            raw_pointer_cast(m_neighbours),
            timestep,
            parameters::physical::support_radius,
            parameters::physical::viscosity_correction,
            parameters::hashing::max_neighbours);

        // (4.2) Vorticity Confinement and Position Update [lines 22-23]
        kernels::simulation::confine_vorticity_then_update<<<num_blocks, threads_per_block>>>(
            particles,
            raw_pointer_cast(m_neighbours),
            parameters::physical::vorticity_strength,
            timestep,
            parameters::physical::support_radius,
            parameters::hashing::max_neighbours);

        m_particles.release();

        cudaDeviceSynchronize();
    }

    [[nodiscard]] auto size() const noexcept -> u32
    {
        return static_cast<u32>(m_particles.size());
    }
};
}
