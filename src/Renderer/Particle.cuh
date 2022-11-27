#pragma once

#include <memory>
#include <vector>

#include "Program.h"

#include <GL/glew.h>

namespace renderer {

class Particle {
    struct {
        GLuint position;
        GLuint color;
    } m_buffers;

    size_t m_num_particles;
    std::vector<float3> m_color;

public:
    Particle(GLuint position_buffer, size_t num_particles)
        : m_num_particles(num_particles)
        , m_color(num_particles, float3 { 0.2, 0.2, 0.6 })
    {
        m_buffers.position = position_buffer;
        glGenBuffers(1, &m_buffers.color);

        // TODO: Add ability to dynamically change colors
        glBindBuffer(GL_ARRAY_BUFFER, m_buffers.color);
        glBufferData(GL_ARRAY_BUFFER, m_num_particles * sizeof(float3), m_color.data(), GL_STATIC_DRAW);
    }

    auto draw(std::shared_ptr<Program> const& program) -> void
    {
        glEnableVertexAttribArray(program->get_attribute("i_Position"));
        glBindBuffer(GL_ARRAY_BUFFER, m_buffers.position);
        glVertexAttribPointer(program->get_attribute("i_Position"), 3, GL_FLOAT, GL_FALSE, 0, 0);

        {
            auto attrib = program->get_attribute("i_Color");
            if (attrib != -1) {
                glEnableVertexAttribArray(program->get_attribute("i_Color"));
                glBindBuffer(GL_ARRAY_BUFFER, m_buffers.color);
                glVertexAttribPointer(program->get_attribute("i_Color"), 3, GL_FLOAT, GL_FALSE, 0, 0);
            }
        }

        glDrawArrays(GL_POINTS, 0, m_num_particles);

        glDisableVertexAttribArray(program->get_attribute("i_Color"));
        glDisableVertexAttribArray(program->get_attribute("i_Position"));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
};

}
