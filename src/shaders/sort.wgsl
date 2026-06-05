struct GlobalUniforms {
    view: mat4x4f,
    focal: vec2f,
    near: f32,
    tileWidth: f32,
    tileHeight: f32,
    textureWidth: f32,
    textureHeight: f32,
    count: u32,
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
@group(0) @binding(1) var<storage, read> gsParms: array<f32>;

@compute @workgroup_size(32)
fn computeMain(@builtin(global_invocation_index) gindex: u32) {
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

    let positon = getPropertyVec3f(positionOffset, gindex);
    let scale = getPropertyVec3f(scaleOffset, gindex);
    let quaternion = getPropertyVec4f(quaternionOffset, gindex);

    let scaleMat3 = scaleToMat3(exp(scale));    // .PLY stores scale as log(scale)
    // TODO: rot may be stored as (w, x, y, z)
    let rotationMat3 = quantToMat3(normalize(quaternion));
    let covariance = rotationMat3 * scaleMat3 * transpose(scaleMat3) * transpose(rotationMat3);

    let positionView = uniforms.view * vec4f(positon, 1.0f);
    // Near plane culling
    if positionView.z < uniforms.near {
        return;
    }

    // Pinhole camera projection
    let positionPixel = vec2f(
        uniforms.focal.x * positionView.x / positionView.z + uniforms.textureWidth * 0.5f,
        uniforms.focal.y * positionView.y / positionView.z + uniforms.textureHeight * 0.5f
    );

    let J = calculateJocabian(positionView, uniforms.focal);
    let W = mat3x3f(
        uniforms.view[0].xyz,
        uniforms.view[1].xyz,
        uniforms.view[2].xyz
    );
    let covarianceClip = J * W * covariance * transpose(W) * transpose(J);
    let c00 = covarianceClip[0][0];
    let c01 = covarianceClip[0][1]; // [1][0] symmetric
    let c11 = covarianceClip[1][1];
    let mid = 0.5f * (c00 + c11);
    let det = c00 * c11 - c01 * c01;
    let lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    let lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
    let radius = ceil(3.0f * sqrt(max(lambda1, lambda2)));

    // Frustum culling
    if positionPixel.x + radius < 0.0f || positionPixel.x - radius > uniforms.textureWidth ||
        positionPixel.y + radius < 0.0f || positionPixel.y - radius > uniforms.textureHeight {
        return;
    }
}

// Helpers to get property 
fn getPropertyF(offset: u32, propIndex: u32) -> f32 {
    return gsParms[offset + propIndex];
}

fn getPropertyVec3f(offset: u32, propIndex: u32) -> vec3f {
    return vec3f(
        gsParms[offset + propIndex * 3],
        gsParms[offset + propIndex * 3 + 1],
        gsParms[offset + propIndex * 3 + 2]
    );
}

fn getPropertyVec4f(offset: u32, propIndex: u32) -> vec4f {
    return vec4f(
        gsParms[offset + propIndex * 4],
        gsParms[offset + propIndex * 4 + 1],
        gsParms[offset + propIndex * 4 + 2],
        gsParms[offset + propIndex * 4 + 3]
    );
}

// Helper to build scale matrix
fn scaleToMat3(s: vec3f) -> mat3x3f {
    return mat3x3f(
        vec3f(s.x, 0.0f, 0.0f),
        vec3f(0.0f, s.y, 0.0f),
        vec3f(0.0f, 0.0f, s.z)
    );
}

// Helper to build rotation matrix
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

// Helper to calculate Jocabian matrix
fn calculateJocabian(u: vec4f, f: vec2f) -> mat3x2f {
    return mat3x2f(
        vec2f(f.x / u.z, 0.0),
        vec2f(0.0, f.y / u.z),
        vec2f(-f.x * u.x / (u.z * u.z), -f.y * u.y / (u.z * u.z))
    );
}

// Based on https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu
fn frustumCulling(covarianceClip: mat2x2f) -> bool {
    
}