#include <format>
#include <fstream>
#include <iostream>
#include <memory>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <bitset>

#include "Renderer/Camera.h"
#include "Renderer/Framebuffer.h"
#include "Renderer/MatrixStack.h"
#include "Renderer/Particle.cuh"
#include "Renderer/Program.h"
#include "Renderer/Texture.h"

#include "CUDA/Simulation/Simulation.cuh"

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>

using namespace pbf;
using namespace renderer;

GLFWwindow* g_window;

std::shared_ptr<Camera> g_camera;
std::shared_ptr<Program> g_program;
std::shared_ptr<Program> g_point_program;
std::shared_ptr<renderer::Particle> g_particle;
std::bitset<256> g_key_toggles;

GLuint fullscreen_vbo;

std::shared_ptr<Framebuffer> g_framebuffer;
std::shared_ptr<Texture> g_color_texture;
std::shared_ptr<Texture> g_depth_texture;

void initialize_cuda()
{
    auto num_devices = 0;
    cudaGetDeviceCount(&num_devices);

    if (num_devices < 1)
        exit(EXIT_FAILURE);

    cudaSetDevice(0);
}

// This function is called when the mouse is clicked
static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    (void)button;

    // Get the current mouse position.
    double xmouse, ymouse;
    glfwGetCursorPos(window, &xmouse, &ymouse);
    // Get current window size.
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    if (action == GLFW_PRESS) {
        bool shift = (mods & GLFW_MOD_SHIFT) != 0;
        bool ctrl = (mods & GLFW_MOD_CONTROL) != 0;
        bool alt = (mods & GLFW_MOD_ALT) != 0;
        g_camera->mouse_clicked((float)xmouse, (float)ymouse, shift, ctrl, alt);
    }
}

// This function is called when the mouse moves
static void cursor_position_callback(GLFWwindow* window, double xmouse, double ymouse)
{
    int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
    if (state == GLFW_PRESS) {
        g_camera->mouse_moved((float)xmouse, (float)ymouse);
    }
}

static void char_callback(GLFWwindow* window, unsigned int key)
{
    (void)window;

    g_key_toggles.flip(key);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    (void)mods;
    (void)scancode;

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GL_TRUE);
    }
}

void initialize_opengl()
{
    static constexpr auto error_callback = [](auto code, auto desc) -> void {
        std::cerr << "GLFW Error " << code << ": " << desc << std::endl;
    };

    glfwSetErrorCallback(error_callback);

    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }

    g_window = glfwCreateWindow(640, 480, "Position Based Fluids", NULL, NULL);

    if (!g_window) {
        std::cerr << "glfwCreateWindow() NOT OK" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(g_window);
    glewExperimental = true;

    if (glewInit() != GLEW_OK) {
        std::cerr << "glewInit() NOT OK" << std::endl;
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glGetError();

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "  GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

    glfwSwapInterval(1);

    glfwSetCursorPosCallback(g_window, cursor_position_callback);
    glfwSetMouseButtonCallback(g_window, mouse_button_callback);
    glfwSetCharCallback(g_window, char_callback);
    glfwSetKeyCallback(g_window, key_callback);

    glGenVertexArrays(1, &fullscreen_vbo);
}

void initialize_renderer()
{
    glfwSetTime(0.);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);

    g_point_program = std::make_shared<Program>(
        "../resources/Point.vertex.glsl",
        "../resources/Point.fragment.glsl");

    g_point_program->add_attribute({ "i_Position",
                                     "i_Color" });

    g_point_program->add_uniform({ "u_Perspective",
                                   "u_ModelView" });

    g_program = std::make_shared<Program>(
        "../resources/FullScreenTriangle.vertex.glsl",
        "../resources/Texture.fragment.glsl");

    g_program->add_uniform({ "u_Perspective",
                             "u_ModelView",
                             "u_Resolution",
                             "u_Texture",
                             "u_Depth" });

    g_camera = std::make_shared<Camera>();
    g_camera->set_init_distance(12.f);
}

auto P = std::make_shared<MatrixStack>();
auto MV = std::make_shared<MatrixStack>();

void render()
{
    int width = 0;
    int height = 0;

    glfwGetFramebufferSize(g_window, &width, &height);
    g_camera->set_aspect_ratio(width / static_cast<float>(height));

    P->push();
    g_camera->apply_projection_matrix(P);

    MV->push();
    g_camera->apply_view_matrix(MV);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    {
        utils::Lock fb_lock(*g_framebuffer);
        utils::Lock prog_lock(*g_point_program);

        glEnable(GL_DEPTH_TEST);
        glClearColor(22.f / 255.f, 22.f / 255.f, 29.f / 255.f, 1.0f);
        glViewport(0, 0, 1024, 768);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUniformMatrix4fv(g_point_program->get_uniform("u_Perspective"), 1, GL_FALSE, glm::value_ptr(P->top()));
        glUniformMatrix4fv(g_point_program->get_uniform("u_ModelView"), 1, GL_FALSE, glm::value_ptr(MV->top()));
        g_particle->draw(g_point_program);
    }
    {
        utils::Lock lock(*g_program);

        glDisable(GL_DEPTH_TEST);
        glClearColor(1.f, 1.f, 1.f, 1.f);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT);

        glUniformMatrix4fv(g_program->get_uniform("u_Perspective"), 1, GL_FALSE, glm::value_ptr(P->top()));
        glUniformMatrix4fv(g_program->get_uniform("u_ModelView"), 1, GL_FALSE, glm::value_ptr(MV->top()));
        glUniform2i(g_program->get_uniform("u_Resolution"), width, height);

        g_color_texture->attach(GL_TEXTURE0, g_program->get_uniform("u_Texture"));
        g_color_texture->attach(GL_TEXTURE1, g_program->get_uniform("u_Depth"));

        glBindVertexArray(fullscreen_vbo);
        glDrawArrays(GL_TRIANGLES, 0, 3);

        glFlush();
    }

    MV->pop();
    P->pop();
}

int main()
{
    initialize_cuda();
    initialize_opengl();
    initialize_renderer();

    auto sim = PBF(40);

    g_particle = std::make_shared<renderer::Particle>(
        sim.m_particles.position_vbo(),
        sim.size());

    std::cout << std::endl;
    auto step = [&]() -> float {
        cudaEvent_t start;
        cudaEvent_t end;
        float time;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        cudaEventRecord(start, 0);
        sim.step();
        cudaEventRecord(end, 0);

        cudaEventSynchronize(end);
        cudaEventElapsedTime(&time, start, end);

        std::cout << std::format("\e[1A\e[K[simulation] {:0.3f} ms ({:0.0f} FPS)", time, 1000.f / time) << std::endl;
        return time;
    };

    g_framebuffer = std::make_shared<Framebuffer>();
    g_color_texture = std::make_shared<Texture>(GL_TEXTURE_2D, 0, GL_RGBA, 1024, 768, 0, GL_RGBA, GL_FLOAT, nullptr);
    g_depth_texture = std::make_shared<Texture>(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, 1024, 768, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

    for (auto&& texture : { g_color_texture, g_depth_texture }) {
        texture->parameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        texture->parameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }

    g_framebuffer->attach_texture(*g_color_texture, GL_COLOR_ATTACHMENT0);
    g_framebuffer->attach_texture(*g_depth_texture, GL_DEPTH_ATTACHMENT);

    while (!glfwWindowShouldClose(g_window)) {
        if (!glfwGetWindowAttrib(g_window, GLFW_ICONIFIED)) {
            if (g_key_toggles[(unsigned)' '])
                step();

            render();
            glfwSwapBuffers(g_window);
        }

        glfwPollEvents();
    }

    glfwDestroyWindow(g_window);
    glfwTerminate();
}
