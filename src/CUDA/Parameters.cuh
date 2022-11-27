#pragma once

#include "Defines.h"

#include "CUDA/Math.cuh"

/**
 * STYLE: In this file...
 *         - DO NOT use auto.
 *         - DO NOT use double precision floats.
 * 
 * FIXME: Make config updatable at runtime.
 */

namespace pbf::parameters {
namespace debug {
    constexpr bool print_neighbours = false;
    constexpr bool print_statistics = false;
    constexpr bool apply_xsph_viscosity = true;
    constexpr bool apply_vorticity_confinement = true;
}

namespace CUDA {
    constexpr u32 threads_per_block = 256u;
    constexpr u32 elements_per_thread = 256u;
}

namespace physical {
    constexpr float particle_radius = 0.05f;
    constexpr float3 gravity { 0., 0., -9.81f };
    constexpr float relaxation = 3'500.f;
    constexpr float rest_density = 5'500.f;
    constexpr float support_radius = 0.1f;
    constexpr float viscosity_correction = 0.25f;
    constexpr float vorticity_strength = 0.001f;
    constexpr float timestep = 1.f / 360.f;

    struct ArtificialPressure {
        float deltaQ = 0.3f;
        float n = 4.f;
        float k = 0.001f;
    } artificial_pressure;

    struct Box {
        float3 min = { -3., -3., 0. };
        float3 max = { 3., 3., 12. };
    } domain;
}

namespace hashing {
    constexpr float spacing = physical::support_radius;
    float3 grid_dimensions = physical::domain.max - physical::domain.min;
    u32 grid_size = static_cast<u32>(
        0.5f + (grid_dimensions.x / spacing) * (grid_dimensions.y / spacing) * (grid_dimensions.z / spacing));

    // Implicit max number of particles is (2^32-1) / max_neighbours
    // This is because all data passed to and integer types within kernels
    // are always u32s. The types can be upgraded to u64s easily if needed.
    constexpr u32 max_neighbours = 256u;
}
};
