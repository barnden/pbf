#pragma once

#include <GL/glew.h>

#include "Renderer/Texture.h"
#include "Renderer/Utils.h"

namespace renderer {

class Framebuffer {
    GLuint m_framebuffer_id;

public:
    Framebuffer()
    {
        glGenFramebuffers(1, &m_framebuffer_id);
    }

    void bind()
    {
        glBindFramebuffer(GL_FRAMEBUFFER, m_framebuffer_id);
    }

    void unbind()
    {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void attach_texture(Texture& texture, GLuint attachment)
    {
        utils::Lock self_lock(*this);
        utils::Lock tex_lock(texture);

        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            attachment,
            GL_TEXTURE_2D,
            texture.id(),
            0);
    }
};

}
