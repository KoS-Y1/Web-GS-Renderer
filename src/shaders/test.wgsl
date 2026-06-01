@vertex
fn vertexMain(
    @builtin(vertex_index) vertexIndex: u32
) -> @builtin(position) vec4f {
    let pos = array(
        vec2f(0.0f, 0.5f),
        vec2f(-0.5f, -0.5f),
        vec2f(0.5f, -0.5f)
    );
    return vec4f(pos[vertexIndex], 0.0f, 1.0f);
}

@fragment
fn fragmentMain() -> @location(0) vec4f {
    return vec4f(1.0f, 0.0f, 0.0f, 1.0f);
}