// Based on https://github1s.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu
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
@group(0) @binding(0) var<uniform> uniforms: GlobalUniforms;
@group(0) @binding(1) var<storage, read> conicOpacities: array<vec4f>;
@group(0) @binding(2) var<storage, read> pixelPositons: array<vec2f>;
@group(0) @binding(3) var<storage, read> colors: array<vec3f>;
@group(0) @binding(4) var<storage, read> physicalIndices: array<u32>;
@group(0) @binding(5) var<storage, read> tileRanges: array<u32>;
@group(0) @binding(6) var output: texture_storage_2d<rgba8unorm, write>;

const TILE_SIZE_X = 16u;
const TILE_SIZE_Y = 16u;
const TIlE_SIZE = TILE_SIZE_X * TILE_SIZE_Y;

const ALPHA_CLAMP = 0.99f;
const EPSILON = 1.0f / 255.0f;
const BLEND_THRESHOLD = 0.0001f;

var<workgroup> localPhysicalIndices: array<u32, 256>;
var<workgroup> localConicOpacities: array<vec4f, 256>;
var<workgroup> localPixelPositions: array<vec2f, 256>;
var<workgroup> doneCount: atomic<u32>;
var<workgroup> allDone: u32;

@compute @workgroup_size(TILE_SIZE_X, TILE_SIZE_Y )
fn computeMain(@builtin(local_invocation_index) lindex: u32, @builtin(workgroup_id) wid: vec3u, @builtin(global_invocation_id) gid: vec3u) {
    let px = gid.xy;
    let isInside = all(px < vec2u(uniforms.textureSize));
    var done = !isInside;

    let tilesPerRow = (u32(uniforms.textureSize.x) + TILE_SIZE_X - 1u) / TILE_SIZE_X;
    let tileIndex = wid.y * tilesPerRow + wid.x;
    let start = tileRanges[tileIndex * 2];
    let end = tileRanges[tileIndex * 2 + 1];
    var toDo = end - start;

    var contributor = 0u;
    var lastContributor = 0u;
    var T = 1.0f;
    var color = vec3f(0.0f);

    for (var i = start; i < end; i += TIlE_SIZE) {
        // Break once every pixel in the tile is done (saturated or outside).
        workgroupBarrier();
        if lindex == 0u {
            atomicStore(&doneCount, 0u);
        }
        workgroupBarrier();
        if done {
            atomicAdd(&doneCount, 1u);
        }
        workgroupBarrier();
        if lindex == 0u {
            allDone = select(0u, 1u, atomicLoad(&doneCount) == TIlE_SIZE);
        }
        if workgroupUniformLoad(&allDone) == 1u {
            break;
        }

        // Cooperatively load this batch into shared mem (guard against reading past the tile range).
        let loadIndex = i + lindex;
        if loadIndex < end {
            let physicalIndex = physicalIndices[loadIndex];
            localPhysicalIndices[lindex] = physicalIndex;
            localConicOpacities[lindex] = conicOpacities[physicalIndex];
            localPixelPositions[lindex] = pixelPositons[physicalIndex];
        }
        workgroupBarrier();

        for (var j = 0u; !done && j < min(TIlE_SIZE, toDo); j++) {
            contributor++;

            let conicOpacity = localConicOpacities[j];
            let power = calculatePower(j, conicOpacity, vec2f(px));
            if power > 0.0f {
                continue;
            }

            let alpha = min(ALPHA_CLAMP, conicOpacity.w * exp(power));
            if alpha < EPSILON {
                continue;
            }
            let testT = T * (1.0f - alpha);
            if testT < BLEND_THRESHOLD {
                done = true;
                continue;
            }

            color += colors[localPhysicalIndices[j]] * alpha * T;

            T = testT;
            lastContributor = contributor;
        }

        toDo -= TIlE_SIZE;
    }

    if isInside {
        textureStore(output, px, vec4f(color, 1.0f));
    }
}

fn calculatePower(localIndex: u32, conicOpacity: vec4f, px: vec2f) -> f32 {
    let positionPixel = localPixelPositions[localIndex];
    let d = positionPixel - px;
    return -0.5f * (conicOpacity.x * d.x * d.x + conicOpacity.z * d.y * d.y) - conicOpacity.y * d.x * d.y;
}