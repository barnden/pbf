#version 330

uniform mat4 u_Perspective;
uniform mat4 u_ModelView;
uniform vec2 u_Resolution;

in vec3 v_Color;

out vec4 o_Color;

float linearizeDepth(float d, float near, float far)
{
    float z_n = 2.0 * d - 1.0;

    return (2.0 * near * far) / (far + near - z_n * (far - near));
}

void main()
{
    vec2 p = gl_PointCoord * 2. - 1.;
    float xy = dot(p, p);

    if (xy > 1.0)
        discard;
    o_Color = vec4(v_Color * sqrt(1.0 - xy), 1.0);
}
