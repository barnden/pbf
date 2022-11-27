#pragma once

#include "Defines.h"

#include "CUDA/Math.cuh"

namespace pbf::kernels::sort {

template <typename KeyFunction, typename T, typename U, typename V>
__global__ auto placement(
    KeyFunction keyof,
    T const* const d_data,
    U* const d_count,
    V* const d_output,
    u32 num_elements,
    u32 chunk_size) -> void
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    auto const ai = idx * chunk_size;
    auto const bi = std::min(ai + chunk_size, num_elements);

    if (ai >= num_elements || bi < ai)
        return;

    for (auto i = ai; i < bi; i++) {
        auto j = keyof(d_data[i]);
        auto position = atomicSub(&d_count[j], 1) - 1;

        d_output[position] = i;
    }
}

}