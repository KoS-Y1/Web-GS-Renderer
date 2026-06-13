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
@group(0) @binding(1) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> outInstanceCount: u32;

const OFFSET_SCAN_WORKGROUP_SIZE = 256u;

var<workgroup> temp: array<u32, OFFSET_SCAN_WORKGROUP_SIZE>;
var<workgroup> carry: u32;

@compute @workgroup_size(OFFSET_SCAN_WORKGROUP_SIZE)
fn computeMain(@builtin(local_invocation_index) lindex: u32) {
    let n = uniforms.count + 1u;
    if lindex == 0u {
        carry = 0u;
    }
    workgroupBarrier();

    // Similar to Radix sort
    let blockCounts = (n + OFFSET_SCAN_WORKGROUP_SIZE - 1u) / OFFSET_SCAN_WORKGROUP_SIZE;
    for (var b = 0u; b < blockCounts; b++) {
        let i = b * OFFSET_SCAN_WORKGROUP_SIZE + lindex;
        var c = 0u;
        if i < n {
            c = offsets[i];
        }
        temp[lindex] = c;
        workgroupBarrier();

        for (var shift = 1u; shift < OFFSET_SCAN_WORKGROUP_SIZE; shift = shift << 1u) {
            var prefix = 0u;
            if shift <= lindex {
                prefix = temp[lindex - shift];
            }
            workgroupBarrier();
            temp[lindex] = temp[lindex] + prefix;
            workgroupBarrier();
        }

        let inLanePrefix = temp[lindex] - c;
        let blockTotal = temp[OFFSET_SCAN_WORKGROUP_SIZE - 1u];

        if i < n {
            offsets[i] = carry + inLanePrefix;
        }
        workgroupBarrier();

        if lindex == 0u {
            carry += blockTotal;
        }
        workgroupBarrier();
    }

    if lindex == 0u {
        outInstanceCount = carry;
    }
}