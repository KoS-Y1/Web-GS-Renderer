// Mostly based on https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu
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

// gsParams layout:
// positions: array<f32, count * 3>
// scales: array<f32, count * 3>
// quaternions: array<f32, count * 4>
// colors: array<f32, count * 3>
// opacites: array<f32, count>
// shs: array<f32, count * 45>
// All data is stored in a flat array<f32>
@group(0) @binding(1) var<storage, read> gsParams: array<f32>;

// Key for radix sort [tileId | depth]
@group(1) @binding(0) var<storage, read_write> outputKeys: array<u32>;
@group(1) @binding(1) var<storage, read_write> outputPhysicalIndices: array<u32>;
@group(1) @binding(2) var<storage, read_write> outputInstnaceCount: atomic<u32>;
@group(1) @binding(3) var<storage, read_write> outputConicOpacity: array<vec4f>;
@group(1) @binding(4) var<storage, read_write> outputPixelPositons: array<vec2f>;
@group(1) @binding(5) var<storage, read_write> outputColors: array<vec3f>;

const SMALL_VALUE = 0.0000001f;

const MAX_INSTANCE_FACTOR = 64u;
const Z_NEAR_VIEW = 0.2f;
const FRUSTUM_EXTENTED = 1.3f;
const TILE_SIZE_X = 16u;
const TILE_SIZE_Y = 16u;
const DEGREE = 3u;
const MAX_COEFFICIENTS = 16u;

const SH_C0 = 0.28209479177387814f;
const SH_C1 = 0.4886025119029199f;
const SH_C2_0 = 1.0925484305920792f;
const SH_C2_1 = -1.0925484305920792f;
const SH_C2_2 = 0.31539156525252005f;
const SH_C2_3 = -1.0925484305920792f;
const SH_C2_4 = 0.5462742152960396f;
const SH_C3_0 = -0.5900435899266435f;
const SH_C3_1 = 2.890611442640554f;
const SH_C3_2 = -0.4570457994644658f;
const SH_C3_3 = 0.3731763325901154f;
const SH_C3_4 = -0.4570457994644658f;
const SH_C3_5 = 1.445305721320277f;
const SH_C3_6 = -0.5900435899266435f;

const PREPROCESS_WORKGROUP_SIZE = 32u;

@compute @workgroup_size(PREPROCESS_WORKGROUP_SIZE)
fn computeMain(@builtin(global_invocation_id) gid: vec3u) {
    let gindex = gid.x;
    let count = uniforms.count;
    if gindex >= count {
        return;
    }

    let positionOffset = u32(0);
    let scaleOffset = positionOffset + count * 3;
    let quaternionOffset = scaleOffset + count * 3;
    let colorOffset = quaternionOffset + count * 4;
    let opacityOffset = colorOffset + count * 3;
    let shOffest = opacityOffset + count;

    let position = getPropertyVec3f(positionOffset, gindex);

    let positionView = uniforms.view * vec4f(position, 1.0f);

    // Near culling
    if positionView.z <= Z_NEAR_VIEW {
        return;
    }

    let positionClip = uniforms.viewProj * vec4f(position, 1.0f);
    let positionNdc = positionClip.xyz / (positionClip.w + SMALL_VALUE);
    let positionPixel = vec2f(
        ndcToPixel(positionNdc.x, uniforms.textureSize.x),
        ndcToPixel(positionNdc.y, uniforms.textureSize.y)
    );
    outputPixelPositons[gindex] = positionPixel;

    let scale = getPropertyVec3f(scaleOffset, gindex);
    let quaternion = getPropertyVec4f(quaternionOffset, gindex);

    let scaleMat3 = scaleToMat3(exp(scale));    // .PLY stores scale as log(scale)
    let rotationMat3 = quantToMat3(normalize(vec4f(quaternion.y, quaternion.z, quaternion.w, quaternion.x)));
    let covariance = rotationMat3 * scaleMat3 * transpose(scaleMat3) * transpose(rotationMat3);

    let covariance2D = calculateCovariance2D(positionView, covariance);
    let det = covariance2D.x * covariance2D.z - covariance2D.y * covariance2D.y;
    let radius = calculateRadius(covariance2D, det);

    let conic = vec3f(
        covariance2D.z / det,
        -covariance2D.y / det,
        covariance2D.x / det
    );
    let opacity = getPropertyF(opacityOffset, gindex);
    outputConicOpacity[gindex] = vec4f(conic, opacity);

    let dir = normalize(position - uniforms.cameraPos);
    outputColors[gindex] = calculateColor(gindex, colorOffset, shOffest, dir);

    // Frustum culling
    if !isInFrustum(radius, positionPixel) {
        return;
    }

    let boundingMinPx = positionPixel - vec2f(radius);
    let boundingMaxPx = positionPixel + vec2f(radius);

    let tilesPerRow = (u32(uniforms.textureSize.x) + TILE_SIZE_X - 1u) / TILE_SIZE_X;
    let tilesPerColumn = (u32(uniforms.textureSize.y) + TILE_SIZE_Y - 1u) / TILE_SIZE_Y;

    // Tile AABB
    let minTileX = u32(clamp(floor(boundingMinPx.x / f32(TILE_SIZE_X)), 0.0f, f32(tilesPerRow - 1u)));
    let minTileY = u32(clamp(floor(boundingMinPx.y / f32(TILE_SIZE_Y)), 0.0f, f32(tilesPerColumn - 1u)));
    let maxTileX = u32(clamp(floor(boundingMaxPx.x / f32(TILE_SIZE_X)), 0.0f, f32(tilesPerRow - 1u)));
    let maxTileY = u32(clamp(floor(boundingMaxPx.y / f32(TILE_SIZE_Y)), 0.0f, f32(tilesPerColumn - 1u)));

    let depthU16 = u32(saturate(positionNdc.z) * 65535.0f);
    let maxCount = count * MAX_INSTANCE_FACTOR;

    for (var ty = minTileY; ty <= maxTileY; ty++) {
        for (var tx = minTileX; tx <= maxTileX; tx++) {
            let tileId = ty * tilesPerRow + tx;
            let key = (tileId << 16u) | (depthU16 & 0xFFFFu);

            let slot = atomicAdd(&outputInstnaceCount, 1u);
            if slot >= maxCount {
                return;
            }
            outputKeys[slot] = key;
            outputPhysicalIndices[slot] = gindex;
        }
    }

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

fn getShRest(offset: u32, propIndex: u32, c: u32) -> vec3f {
    let base = offset + propIndex * 45u + c;
    return vec3f(gsParams[base], gsParams[base + 15u], gsParams[base + 30u]);
}

fn getPropertyVec4f(offset: u32, propIndex: u32) -> vec4f {
    return vec4f(
        gsParams[offset + propIndex * 4],
        gsParams[offset + propIndex * 4 + 1],
        gsParams[offset + propIndex * 4 + 2],
        gsParams[offset + propIndex * 4 + 3]
    );
}

fn scaleToMat3(s: vec3f) -> mat3x3f {
    return mat3x3f(
        vec3f(s.x, 0.0f, 0.0f),
        vec3f(0.0f, s.y, 0.0f),
        vec3f(0.0f, 0.0f, s.z)
    );
}

fn quantToMat3(q: vec4f) -> mat3x3f {
    let x2 = q.x + q.x;
    let y2 = q.y + q.y;
    let z2 = q.z + q.z;

    let xx = q.x * x2;
    let xy = q.x * y2;
    let xz = q.x * z2;
    let yy = q.y * y2;
    let yz = q.y * z2;
    let zz = q.z * z2;
    let wx = q.w * x2;
    let wy = q.w * y2;
    let wz = q.w * z2;

    return mat3x3f(
        vec3f(1.0f - (yy + zz), xy + wz, xz - wy),
        vec3f(xy - wz, 1.0f - (xx + zz), yz + wx),
        vec3f(xz + wy, yz - wx, 1.0f - (xx + yy))
    );
}

fn ndcToPixel(v: f32, s: f32) -> f32 {
    return ((v + 1.0f) * s - 1.0f) * 0.5f;
}

fn calculateJocabian(u: vec4f, f: vec2f) -> mat3x2f {
    return mat3x2f(
        vec2f(f.x / u.z, 0.0),
        vec2f(0.0, f.y / u.z),
        vec2f(-f.x * u.x / (u.z * u.z), -f.y * u.y / (u.z * u.z))
    );
}

fn calculateColor(propIndex: u32, colorOffset: u32, shOffset: u32, dir: vec3f) -> vec3f {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let yz = y * z;
    let xz = x * z;

    var result = SH_C0 * getPropertyVec3f(colorOffset, propIndex)
        - SH_C1 * y * getShRest(shOffset, propIndex, 0u)
        + SH_C1 * z * getShRest(shOffset, propIndex, 1u)
        - SH_C1 * x * getShRest(shOffset, propIndex, 2u)
        + SH_C2_0 * xy * getShRest(shOffset, propIndex, 3u)
        + SH_C2_1 * yz * getShRest(shOffset, propIndex, 4u)
        + SH_C2_2 * (2.0f * zz - xx - yy) * getShRest(shOffset, propIndex, 5u)
        + SH_C2_3 * xz * getShRest(shOffset, propIndex, 6u)
        + SH_C2_4 * (xx - yy) * getShRest(shOffset, propIndex, 7u)
        + SH_C3_0 * y * (3.0f * xx - yy) * getShRest(shOffset, propIndex, 8u)
        + SH_C3_1 * xy * z * getShRest(shOffset, propIndex, 9u)
        + SH_C3_2 * y * (4.0f * zz - xx - yy) * getShRest(shOffset, propIndex, 10u)
        + SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * getShRest(shOffset, propIndex, 11u)
        + SH_C3_4 * x * (4.0f * zz - xx - yy) * getShRest(shOffset, propIndex, 12u)
        + SH_C3_5 * z * (xx - yy) * getShRest(shOffset, propIndex, 13u)
        + SH_C3_6 * x * (xx - 3.0f * yy) * getShRest(shOffset, propIndex, 14u);

    result += 0.5f;
    return max(result, vec3f(0.0f));
}

fn calculateCovariance2D(positionView: vec4f, covariance: mat3x3f) -> vec3f {
    let lim = FRUSTUM_EXTENTED * uniforms.tanFov;
    var t = positionView.xyz;
    t.x = min(lim.x, max(-lim.x, t.x / t.z)) * t.z;
    t.y = min(lim.y, max(-lim.y, t.y / t.z)) * t.z;

    let J = calculateJocabian(vec4f(t, 1.0f), uniforms.focal);
    let W = mat3x3f(
        uniforms.view[0].xyz,
        uniforms.view[1].xyz,
        uniforms.view[2].xyz
    );

    let covariance2D = J * W * covariance * transpose(W) * transpose(J);

    // Apply low-pass filter: every Gaussian should be at least
    // one pixel wide/high. Discard 3rd row and column.   
    return vec3f(covariance2D[0][0] + 0.3f, covariance2D[0][1], covariance2D[1][1] + 0.3f);
}

fn calculateRadius(covariance: vec3f, det: f32) -> f32 {
    let mid = 0.5f * (covariance.x + covariance.z);
    let lambda = mid + sqrt(max(0.1f, mid * mid - det));
    return ceil(3.0f * sqrt(lambda));
}

fn isInFrustum(radius: f32, positionPixel: vec2f) -> bool {
    if positionPixel.x + radius < 0.0f || positionPixel.x - radius > uniforms.textureSize.x ||
        positionPixel.y + radius < 0.0f || positionPixel.y - radius > uniforms.textureSize.y {
        return false;
    }

    return true;
}
