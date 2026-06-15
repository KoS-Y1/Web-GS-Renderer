struct GlobalUniforms {
    view: mat4x4f,
    viewProj: mat4x4f,
    cameraPos: vec3f,
    count: u32,
    focal: vec2f,
    tanFov: vec2f,
    textureSize: vec2f,
    padding0: u32,
    padding1: u32,
};

struct IndirectArgUniform {
    maxInstanceCount: u32,
};

@group(0) @binding(0) var<uniform> uniforms: GlobalUniforms;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read> splatMeta: array<vec4u>;
@group(0) @binding(3) var<uniform> arg: IndirectArgUniform;

@group(1) @binding(0) var<storage, read_write> outKeys: array<u32>;
@group(1) @binding(1) var<storage, read_write> outPhysicaslIndices: array<u32>;

const TILE_SIZE_X = 16u;
const EMIT_WORKGROUP_SIZE = 256u;

@compute @workgroup_size(EMIT_WORKGROUP_SIZE)
fn computeMain(
    @builtin(local_invocation_index) lindex: u32,
    @builtin(workgroup_id) wid: vec3u,
    @builtin(num_workgroups) numWg: vec3u
) {
    let gindex = (wid.y * numWg.x + wid.x) * EMIT_WORKGROUP_SIZE + lindex;
    if gindex >= uniforms.count {
        return;
    }

    let base = offsets[gindex];
    let n = offsets[gindex + 1] - base;
    if n == 0u {
        return;
    }

    let metaData = splatMeta[gindex];
    let minTileX = metaData.x & 0xFFFFu;
    let minTileY = metaData.x >> 16u;
    let maxTileX = metaData.y & 0xFFFFu;
    let maxTileY = metaData.y >> 16u;
    let depth16U = metaData.z & 0xFFFFu;

    let tilesPerRow = (u32(uniforms.textureSize.x) + TILE_SIZE_X - 1u) / TILE_SIZE_X;

    var slot = base;
    for (var ty = minTileY; ty <= maxTileY; ty++) {
        for (var tx = minTileX; tx <= maxTileX; tx++) {
            if slot >= arg.maxInstanceCount {
                return;
            }

            let tileId = ty * tilesPerRow + tx;
            outKeys[slot] = (tileId << 16u) | depth16U;
            outPhysicaslIndices[slot] = gindex;
            slot++;
        }
    }
}