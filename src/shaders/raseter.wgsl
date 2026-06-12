struct GlobalUniforms {
    textureSize: vec2f,
};
@group(0) @binding(0) var<uniform> uniforms: GlobalUniforms;
@group(0) @binding(1) var<storage, read> gsParams: array<f32>;
@group(0) @binding(2) var<storage, read> physicalIndices: array<u32>;
@group(0) @binding(3) var<storage, read> tileRanges: array<u32, 32u>;
@group(0) @binding(4) var output: texture_storage_2d<rgba8unorm, write>;

const TILE_SIZE_X = 16u;
const TILE_SIZE_Y = 16u;

var<workgroup> localPhysicalIndices: array<u32, 256>;

@compute @workgroup_size(TILE_PER_ROW, TILE_PER_COLOMN )
fn computeMain(@builtin(local_invocation_id) lid: vec3u, @builtin(local_invocation_index) lindex: u32, @builtin(workgroup_id) wid: vec3u) {
    let tileSizePx = vec2u(ceil(uniforms.textureSize / vec2f(f32(TILE_PER_ROW), f32(TILE_PER_COLOMN))));
    let px = wid.xy * tileSizePx + lid.xy;
    let isInside = any(px >= vec2u(0)) && any(px < vec2u(uniforms.textureSize));

}

// Helpers to get property 
fn getPropertyF(offset: u32, propIndex: u32) -> f32 {
    return gsParams[offset + propIndex];
}

fn getPropertyVec3f(offset: u32, propIndex: u32) -> vec3f {
    return vec3f(
        gsParams[offset + propIndex * 3],
        gsParams[offset + propIndex * 3 + 1],
        gsParams[offset + propIndex * 3 + 2]
    );
}

fn getPropertyVec4f(offset: u32, propIndex: u32) -> vec4f {
    return vec4f(
        gsParams[offset + propIndex * 4],
        gsParams[offset + propIndex * 4 + 1],
        gsParams[offset + propIndex * 4 + 2],
        gsParams[offset + propIndex * 4 + 3]
    );
}