#version 330

out vec2 v_TexCoords;
void main()
{
    const vec2 vertices[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2(3.0, -1.0),
        vec2(-1.0, 3.0));

    gl_Position = vec4(vertices[gl_VertexID], 0, 1);
    v_TexCoords = 0.5 * gl_Position.xy + vec2(0.5);
}
