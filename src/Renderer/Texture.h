#pragma once

#include <GL/glew.h>

#include "Renderer/Utils.h"

namespace renderer {

class Texture {
    GLuint m_texture_id;
    GLuint m_type;

public:
    template <typename... Args>
    Texture(GLuint type, Args... args)
        : m_type(type)
    {
        glGenTextures(1, &m_texture_id);
        glBindTexture(m_type, m_texture_id);

        glTexImage2D(type, args...);
        glBindTexture(m_type, 0);
    }

    template <typename... Args>
    void parameter(Args... args)
    {
        utils::Lock lock(*this);
        glTexParameteri(m_type, args...);
    }

    void bind()
    {
        glBindTexture(m_type, m_texture_id);
    }

    void unbind()
    {
        glBindTexture(m_type, 0);
    }

    void attach(GLuint attachment, GLuint uniform)
    {
        glActiveTexture(attachment);
        glBindTexture(m_type, m_texture_id);
        glUniform1i(uniform, attachment - GL_TEXTURE0);
    }

    auto id() const -> GLuint
    {
        return m_texture_id;
    }
};

}
