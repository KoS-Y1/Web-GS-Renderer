@group(0) @binding(0)

var outputImage: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8, 1)
fn computeMain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(outputImage);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }

    let coord = vec2i(gid.xy);
    let color = vec4f(f32(gid.x) / f32(dims.x), f32(gid.y) / f32(dims.y), 0.0f, 1.0f);

    textureStore(outputImage, coord, color);
}