struct GlobalUniforms {
    view: mat4x4f,
    viewProj: mat4x4f,
    focal: vec2f,
    tanFov: vec2f,
    textureSize: vec2f,
    count: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
};

@group(0) @binding(0) var<uniform> uniforms: GlobalUniforms;
@group(0) @binding(1) var<storage, read> gsParams: array<f32>;
@group(0) @binding(2) var<storage, read> keys: array<u32>;
@group(0) @binding(3) var<storage, read> physicalIndices: array<u32>;
@group(0) @binding(4) var<storage, read> instanceCount: u32;

// TODO: test ouptut here
@group(0) @binding(5) var outputImage: texture_storage_2d<rgba8unorm, write>;

const TILE_PER_ROW = 4u;
const TILE_PER_COLOMN = 4u;
const SMALL_VALUE = 0.0000001f;

@compute @workgroup_size(32)
fn computeMain(@builtin(global_invocation_id) gid: vec3u) {
    let gindex = gid.x;
    let count = uniforms.count;
    if gindex >= instanceCount {
        return;
    }

    let physicalIndex = physicalIndices[gindex];
    if physicalIndex >= count {
        return;
    }

    let positionOffset = u32(0);
    let scaleOffset = positionOffset + count * 3;
    let quaternionOffset = scaleOffset + count * 3;
    let colorOffset = quaternionOffset + count * 4;
    let opacityOffset = colorOffset + count * 3;
    let shOffest = opacityOffset + count;

    let position = getPropertyVec3f(positionOffset, physicalIndex);

    let positionView = uniforms.view * vec4f(position, 1.0f);

    let positionClip = uniforms.viewProj * vec4f(position, 1.0f);
    let positionNdc = positionClip.xyz / (positionClip.w + SMALL_VALUE);
    let positionPixel = vec2f(
        ndcToPixel(positionNdc.x, uniforms.textureSize.x),
        ndcToPixel(positionNdc.y, uniforms.textureSize.y)
    );

    let tileId = keys[gindex] >> 16u;
    let px = vec2u(positionPixel);
    let tx = tileId % TILE_PER_ROW;
    let ty = tileId / TILE_PER_ROW;
    let col = vec3f(f32(tx) / f32(TILE_PER_ROW - 1u),
        f32(ty) / f32(TILE_PER_COLOMN - 1u),
        0.5f);
    textureStore(outputImage, px, vec4f(col, 1.0f));
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
fn ndcToPixel(v: f32, s: f32) -> f32 {
    return ((v + 1.0f) * s - 1.0f) * 0.5f;
}