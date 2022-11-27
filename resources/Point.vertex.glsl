#version 330

uniform mat4 u_Perspective;
uniform mat4 u_ModelView;
uniform vec2 u_Resolution;

in vec3 i_Color;
in vec3 i_Position;

out vec3 v_Color;

void main()
{
    vec4 view = u_ModelView * vec4(i_Position, 1.0);
    v_Color = i_Color;

    gl_Position = u_Perspective * view;
    gl_PointSize = 56.0 / gl_Position.w;
}
