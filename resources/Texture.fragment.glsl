#version 330

uniform mat4 u_Perspective;
uniform mat4 u_ModelView;
uniform sampler2D u_Texture;
uniform sampler2D u_Depth;

in vec2 v_TexCoords;

out vec4 o_Color;

vec3 get_normal(vec2 p) {
    const vec2 dx = vec2(0.001, 0.0);
    const vec2 dy = vec2(0.0, 0.001);

    float depth = texture2D(u_Depth, p).r;
    float depth_y = texture2D(u_Depth, p + dy).r;

    vec3 dpx = vec3(p + dx, texture2D(u_Depth, p + dx).r);
    vec3 dpy = vec3(p + dy, texture2D(u_Depth, p + dy).r);

    vec3 normal = cross(dpx, dpy);
    return normalize(normal);
}

void main()
{
    // o_Color = vec4(get_normal(v_TexCoords) * 0.5 + 0.5, 1.0);
    o_Color = vec4(texture2D(u_Texture, v_TexCoords).rgb, 1.0);
}
