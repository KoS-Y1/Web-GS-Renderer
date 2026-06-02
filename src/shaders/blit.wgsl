struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vertexMain(@builtin(vertex_index) vid: u32) -> VSOut {
    // Oversize triangle
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0),
    );

    let p = positions[vid];
    var out: VSOut;
    out.position = vec4f(p, 0.0, 1.0);
    
    out.uv = vec2f((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return out;
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;

@fragment
fn fragmentMain(@location(0) uv: vec2f) -> @location(0) vec4f {
    return textureSample(src, srcSampler, uv);
}
