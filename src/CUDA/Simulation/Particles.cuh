#pragma once

#include <GL/glew.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/device_ptr.h>

#include "Defines.h"

namespace pbf {

class Particles {
public:
    struct DevicePointers {
        float3* d_position;
        float3* d_position_star;
        float3* d_position_delta;
        float3* d_velocity;

        float* d_density;
        float* d_constraint;
        float3* d_vorticity;

        size_t size;
    };

private:
    template <typename T>
    using ptr = thrust::device_ptr<T>;

    size_t m_size;

    mutable struct {
        GLuint gl;
        mutable struct cudaGraphicsResource* cuda;
    } m_position_buffer;
    ptr<float3> m_position_star;
    ptr<float3> m_position_delta;
    ptr<float3> m_velocity;

    ptr<float> m_density; // rho
    ptr<float> m_constraint; // lambda
    ptr<float3> m_vorticity; // omega

    template <typename T>
    [[nodiscard]] auto allocate() const noexcept -> ptr<T>
    {
        return thrust::device_malloc<T>(size());
    }

    template <typename T>
    void allocate(ptr<T> Particles::* member)
    {
        this->*member = allocate<T>();
    }

    void allocate_position_buffer()
    {
        /**
         * CUDA -- OpenGL interoperability is one way.
         * 
         * Therefore, we must allocate an OpenGL VBO, map it to a CUDA graphics
         * resource, then obtain a device pointer whenever we want to send the
         * particle positions over to a CUDA kernel.
         * 
         * I decided to create the OpenGL buffers here rather than inside the
         * renderer object as it makes more sense _to me_.
         * IMPORTANT: The constructor can be called ONLY AFTER ALL the following
         *              - glfwInit()
         *              - glfwCreateWindow()
         *              - glewInit()
         *            have finished executing, without error. Basically this
         *            means we must have the renderer already setup before
         *            creating an instance of this class.
         * 
         * IMPORTANT: We must UNMAP the CUDA graphics resource whenever we are
         *            done using the buffer.
         * 
         * IMPORTANT: We must NOT reuse the device pointer obtained from this
         *            across multiple rendering passes as OpenGL may decide to
         *            move the VBO to another address.
         */
        glGenBuffers(1, &m_position_buffer.gl);
        glBindBuffer(GL_ARRAY_BUFFER, m_position_buffer.gl);
        glBufferData(
            GL_ARRAY_BUFFER,
            static_cast<GLuint>(size() * sizeof(float3)),
            0,
            GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, 0);

        if (auto error = cudaGraphicsGLRegisterBuffer(
                &m_position_buffer.cuda,
                m_position_buffer.gl,
                cudaGraphicsMapFlagsNone)) {

            std::cerr
                << "cudaGraphicsGLRegisterBuffer() "
                << "(" << typeid(decltype(error)).name() << ": " << error << ": " << cudaGetErrorString(error) << ")"
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;

            exit(EXIT_FAILURE);
        };

    }

    template <typename T>
    auto deallocate(ptr<T> pointer) -> void
    {
        thrust::device_free(pointer);
    }

    void deallocate(auto Particles::* member)
    {
        thrust::device_free(this->*member);
    }

    Particles(size_t num_particles)
        : m_size(num_particles)
    {
        allocate_position_buffer();
        allocate(&Particles::m_position_star);
        allocate(&Particles::m_position_delta);
        allocate(&Particles::m_velocity);

        allocate(&Particles::m_density);
        allocate(&Particles::m_constraint);
        allocate(&Particles::m_vorticity);

        using thrust::raw_pointer_cast;
    }

public:
    [[nodiscard]] static auto initialize(size_t N) -> Particles
    {
        Particles particles(N * N * N);

        auto count = 0;
        auto position = particles.position();

        for (auto x = 0uz; x < N; x++) {
            for (auto y = 0uz; y < N; y++) {
                for (auto z = 0uz; z < N; z++) {
                    position[count] = float3 {
                        x / 20.f - 2.f,
                        y / 20.f - 2.f,
                        z / 20.f + 2.f
                    };

                    count++;
                }
            }
        }

        particles.release();

        return particles;
    }

    ~Particles()
    {
        deallocate(&Particles::m_position_star);
        deallocate(&Particles::m_position_delta);
        deallocate(&Particles::m_velocity);

        deallocate(&Particles::m_density);
        deallocate(&Particles::m_constraint);
        deallocate(&Particles::m_vorticity);

        cudaGraphicsUnregisterResource(m_position_buffer.cuda);

        glBindBuffer(1, m_position_buffer.gl);
        glDeleteBuffers(1, &m_position_buffer.gl);
    }

    [[nodiscard]] auto position() const noexcept -> ptr<float3>
    {
        float3* d_position;

        if (auto error = cudaGraphicsMapResources(1, &m_position_buffer.cuda, 0)) {
            std::cerr
                << "cudaGraphicsMapResources() "
                << "(" << typeid(decltype(error)).name() << ": " << error << ": " << cudaGetErrorString(error) << ")"
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;

            exit(EXIT_FAILURE);
        };

        size_t num_bytes;
        if (auto error = cudaGraphicsResourceGetMappedPointer((void**)&d_position, &num_bytes, m_position_buffer.cuda)) {
            std::cerr
                << "cudaGraphicsResourceGetMappedPointer() "
                << "(" << typeid(decltype(error)).name() << ": " << error << ": " << cudaGetErrorString(error) << ")"
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;

            exit(EXIT_FAILURE);
        }

        return thrust::device_pointer_cast(d_position);
    }

    auto release() const noexcept -> void
    {
        if (auto error = cudaGraphicsUnmapResources(1, &m_position_buffer.cuda, 0)) {
            std::cerr
                << "cudaGraphicsUnmapResources() "
                << "(" << typeid(decltype(error)).name() << ": " << error << ": " << cudaGetErrorString(error) << ")"
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;

            exit(EXIT_FAILURE);
        }
    }

    [[nodiscard]] auto position_star() const noexcept -> ptr<float3>
    {
        return m_position_star;
    }

    [[nodiscard]] auto position_delta() const noexcept -> ptr<float3>
    {
        return m_position_delta;
    }

    [[nodiscard]] auto velocity() const noexcept -> ptr<float3>
    {
        return m_velocity;
    }

    [[nodiscard]] auto density() const noexcept -> ptr<float>
    {
        return m_density;
    }

    [[nodiscard]] auto constraint() const noexcept -> ptr<float>
    {
        return m_constraint;
    }

    [[nodiscard]] auto vorticity() const noexcept -> ptr<float3>
    {
        return m_vorticity;
    }

    [[nodiscard]] auto raw() const noexcept -> DevicePointers
    {
        return DevicePointers {
            .d_position = raw_pointer_cast(position()),
            .d_position_star = raw_pointer_cast(m_position_star),
            .d_position_delta = raw_pointer_cast(m_position_delta),
            .d_velocity = raw_pointer_cast(m_velocity),

            .d_density = raw_pointer_cast(m_density),
            .d_constraint = raw_pointer_cast(m_constraint),
            .d_vorticity = raw_pointer_cast(m_vorticity),

            .size = size()
        };
    };

    [[nodiscard]] auto size() const noexcept -> size_t
    {
        return m_size;
    }

    [[nodiscard]] auto position_vbo() const noexcept -> GLuint
    {
        return m_position_buffer.gl;
    }
};
}
