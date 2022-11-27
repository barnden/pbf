#pragma once

#include "Math.cuh"
#include "Defines.h"

namespace pbf::spatial {

[[nodiscard]] __host__ __device__ auto hash(
    int3 position,
    u32 cells) noexcept -> unsigned
{
    auto h = (position.x * 92837111) ^ (position.y * 689287499) ^ (position.z * 283923481);

    return std::abs(h) % cells;
}

[[nodiscard]] __host__ __device__ auto hash(
    float3 position,
    float spacing,
    u32 cells) noexcept -> unsigned
{
    position = position / spacing;
    return hash(int3 { static_cast<int>(position.x),
                       static_cast<int>(position.y),
                       static_cast<int>(position.z) },
                cells);
}

}