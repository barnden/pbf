#pragma once

#include <ostream>

[[nodiscard]] __host__ __device__ auto to_int3(float3 x) noexcept -> int3
{
    return {
        static_cast<int>(x.x),
        static_cast<int>(x.y),
        static_cast<int>(x.z)
    };
}

[[nodiscard]] __host__ __device__ auto clamp(float x, float a, float b) noexcept -> float
{
    return max(a, min(b, x));
}

[[nodiscard]] __host__ __device__ auto clamp(float3 x, float3 a, float3 b) noexcept -> float3
{
    return make_float3(
        clamp(x.x, a.x, b.x),
        clamp(x.y, a.y, b.y),
        clamp(x.z, a.z, b.z));
}

#define CUDA_VEC_OPERATOR(T, op)                                                                                   \
    [[nodiscard]] static __inline__ __host__ __device__ auto operator op(T const& lhs, T const& rhs) noexcept -> T \
    {                                                                                                              \
        return make_##T(lhs.x op rhs.x, lhs.y op rhs.y, lhs.z op rhs.z);                                           \
    }

#define CUDA_RIGHT_SCALAR_OPERATOR(T, op, R)                                                                       \
    [[nodiscard]] static __inline__ __host__ __device__ auto operator op(T const& lhs, R const& rhs) noexcept -> T \
    {                                                                                                              \
        return make_##T(lhs.x op rhs, lhs.y op rhs, lhs.z op rhs);                                                 \
    }

#define CUDA_LEFT_SCALAR_OPERATOR(T, op, L)                                                                        \
    [[nodiscard]] static __inline__ __host__ __device__ auto operator op(L const& lhs, T const& rhs) noexcept -> T \
    {                                                                                                              \
        return make_##T(lhs op rhs.x, lhs op rhs.y, lhs op rhs.z);                                                 \
    }

#define CUDA_VEC_OSTREAM(T)                                   \
    std::ostream& operator<<(std::ostream& out, T const& vec) \
    {                                                         \
        out << vec.x << ", " << vec.y << ", " << vec.z;       \
                                                              \
        return out;                                           \
    }

#define CUDA_SCALAR_OPERATOR(T, S, op)   \
    CUDA_RIGHT_SCALAR_OPERATOR(T, op, S) \
    CUDA_LEFT_SCALAR_OPERATOR(T, op, S)

#define CUDA_CROSS_PRODUCT(T)                                                                                \
    [[nodiscard]] static __inline__ __host__ __device__ auto cross(T const& lhs, T const& rhs) noexcept -> T \
    {                                                                                                        \
        T result;                                                                                            \
                                                                                                             \
        result.x = lhs.y * rhs.z - lhs.z * rhs.y;                                                            \
        result.y = lhs.z * rhs.x - lhs.x * rhs.z;                                                            \
        result.z = lhs.x * rhs.y - lhs.y * rhs.x;                                                            \
                                                                                                             \
        return result;                                                                                       \
    }

CUDA_VEC_OPERATOR(float3, -);
CUDA_VEC_OPERATOR(float3, +);

CUDA_VEC_OPERATOR(uint3, -);
CUDA_VEC_OPERATOR(uint3, +);

CUDA_SCALAR_OPERATOR(float3, float, +);
CUDA_SCALAR_OPERATOR(float3, float, -);
CUDA_SCALAR_OPERATOR(float3, float, *);
CUDA_SCALAR_OPERATOR(float3, float, /);

CUDA_VEC_OSTREAM(float3);
CUDA_CROSS_PRODUCT(float3);

[[nodiscard]] __host__ __device__ auto dot(float3 const& a, float3 const& b) noexcept -> float
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] __host__ __device__ auto L2_squared(float3 const& vec) noexcept -> float
{
    return dot(vec, vec);
}

[[nodiscard]] __host__ __device__ auto L2(float3 const& vec) noexcept -> float
{
    return sqrtf(dot(vec, vec));
}

[[nodiscard]] __host__ __device__ auto normalize(float3 const& vec) noexcept -> float3
{
    auto magnitude = L2(vec);

    if (magnitude == 0.f)
        return vec;

    return vec / magnitude;
}

namespace pbf::math {
// From Macklin's PBF slides: http://mmacklin.com/pbf_slides.pdf

[[nodiscard]] __host__ __device__ float poly6(float3 r, float h) noexcept
{
    static constexpr float coeff = 315.f / (64.f * 3.14159265f);
    auto const r2 = L2_squared(r);
    auto const h2 = h * h;

    if (r2 > h2 || r2 < 1e-10)
        return 0.;

    auto const h4 = h2 * h2;
    auto const h9 = h4 * h4 * h;
    auto const d = h2 - r2;
    auto const d3 = d * d * d;

    return (coeff / h9) * d3;
}

[[nodiscard]] __host__ __device__ float3 grad_spiky(float3 r, float h) noexcept
{
    static constexpr auto coeff = -45.f / 3.14159265f;

    auto r1 = L2(r);

    if (r1 > h || r1 < 1e-5)
        return { 0., 0., 0. };

    auto h3 = h * h * h;
    auto h6 = h3 * h3;
    auto d = h - r1;
    auto d2 = d * d;

    return (coeff / h6) * d2 * r / max(r1, 1e-24f);
}
}