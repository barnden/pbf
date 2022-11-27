#pragma once

#include <string>
#include <unordered_map>

#include <GL/glew.h>

#include "Renderer/Utils.h"

namespace renderer {

class Program {
    GLuint m_pid;
    std::unordered_map<std::string, GLint> m_attributes;
    std::unordered_map<std::string, GLint> m_uniforms;

protected:
    std::string m_vertex_shader;
    std::string m_fragment_shader;

public:
    Program()
        : m_pid(0)
        , m_vertex_shader("")
        , m_fragment_shader("") { };

    Program(
        std::string const& vertex_shader,
        std::string const& fragment_shader)
        : m_pid(0)
        , m_vertex_shader(vertex_shader)
        , m_fragment_shader(fragment_shader)
    {
        init();
    }

    ~Program() = default;

    auto init() noexcept -> bool
    {
        GLint rc;

        auto const vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        auto const fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);

        auto const* vertex_shader_code = renderer::utils::read_file(m_vertex_shader);
        auto const* fragment_shader_code = renderer::utils::read_file(m_fragment_shader);

        glShaderSource(vertex_shader, 1, &vertex_shader_code, NULL);
        glShaderSource(fragment_shader, 1, &fragment_shader_code, NULL);

        for (auto [shader, name] : {
                 std::pair(vertex_shader, m_vertex_shader),
                 std::pair(fragment_shader, m_fragment_shader) }) {
            glCompileShader(shader);
            glGetShaderiv(shader, GL_COMPILE_STATUS, &rc);

            if (!rc) {
                GLchar InfoLog[1024];
                glGetShaderInfoLog(shader, sizeof(InfoLog) - 1, NULL, InfoLog);

                std::cerr << "Failed to compile shader: " << name << std::endl
                          << InfoLog << std::endl;
                return false;
            }
        }

        m_pid = glCreateProgram();
        glAttachShader(m_pid, vertex_shader);
        glAttachShader(m_pid, fragment_shader);

        glLinkProgram(m_pid);
        glGetProgramiv(m_pid, GL_LINK_STATUS, &rc);

        free(const_cast<char*>(vertex_shader_code));
        free(const_cast<char*>(fragment_shader_code));

        vertex_shader_code = NULL;
        fragment_shader_code = NULL;

        if (!rc) {
            std::cerr << "failed link program" << std::endl;
            return false;
        }

        return true;
    }
    void bind()
    {
        glUseProgram(m_pid);
    }
    void unbind()
    {
        glUseProgram(0);
    }

    void add_attribute(std::initializer_list<std::string> names)
    {
        for (auto&& name : names) {
            add_attribute(name);
        }
    }

    void add_attribute(std::string const& name)
    {
        m_attributes[name] = glGetAttribLocation(m_pid, name.c_str());
    }

    void add_uniform(std::initializer_list<std::string> names)
    {
        for (auto&& name : names) {
            add_uniform(name);
        }
    }

    void add_uniform(std::string const& name)
    {
        m_uniforms[name] = glGetUniformLocation(m_pid, name.c_str());
    }

    auto get_attribute(std::string const& name) -> GLint
    {
        // NOTE: C++20 added unordered_set::contains but linting with clangd
        //       on CUDA with the C++ standard set still counts this as an error
        if (!m_attributes.contains(name)) {
            std::cerr << "attempted to get invalid attribute " << name << std::endl;
            return -1;
        }

        return m_attributes[name];
    }

    auto get_uniform(std::string const& name) -> GLint
    {
        if (!m_uniforms.contains(name)) {
            std::cerr << "attempted to get invalid uniform " << name << std::endl;
            return -1;
        }

        return m_uniforms[name];
    }
};

}
