// Based on 39.2.4 Arrays of Arbitrary Size
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
struct RadixUniforms {
    instanceCount: u32,
    blockCountMax: u32,
    shiftAmount: u32,
    countsBufferLength: u32,
};

@group(0) @binding(0) var<uniform> uniforms: RadixUniforms;
@group(0) @binding(1) var<storage, read_write> counts: array<u32>;
@group(0) @binding(2) var<storage, read_write> blockSums: array<u32>;

const SCAN_PARALLEL_WORKGROUP_SIZE = 256u;

var<workgroup> temp: array<u32, 256u>;

@compute @workgroup_size(SCAN_PARALLEL_WORKGROUP_SIZE)
fn scanLocalMain(@builtin(local_invocation_index) lindex: u32, @builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) numWg: vec3u) {
    let windex = wid.y * numWg.x + wid.x;
    let numBlocks = (uniforms.countsBufferLength + SCAN_PARALLEL_WORKGROUP_SIZE - 1u) / SCAN_PARALLEL_WORKGROUP_SIZE;

    let n = uniforms.countsBufferLength;
    let i = windex * SCAN_PARALLEL_WORKGROUP_SIZE + lindex;

    var c = 0u;
    if i < n {
        c = counts[i];
    }
    temp[lindex] = c;
    workgroupBarrier();

    scanBlockInclusive(lindex);

    let blockTotal = temp[SCAN_PARALLEL_WORKGROUP_SIZE - 1u];
    if i < n {
        counts[i] = temp[lindex] - c;   // block local prefix
    }
    if lindex == 0u && windex < numBlocks {
        blockSums[windex] = blockTotal;     // block sum
    }
}

@compute @workgroup_size(SCAN_PARALLEL_WORKGROUP_SIZE)
fn scanBlockSumsMain(@builtin(local_invocation_index) lindex: u32) {
    let numBlocks = (uniforms.countsBufferLength + SCAN_PARALLEL_WORKGROUP_SIZE - 1) / SCAN_PARALLEL_WORKGROUP_SIZE;

    var carry = 0u;
    let chunks = (numBlocks + SCAN_PARALLEL_WORKGROUP_SIZE - 1u) / SCAN_PARALLEL_WORKGROUP_SIZE;
    for (var b = 0u; b < chunks; b++) {
        let i = b * SCAN_PARALLEL_WORKGROUP_SIZE + lindex;

        var c = 0u;
        if i < numBlocks {
            c = blockSums[i];
        }
        temp[lindex] = c;
        workgroupBarrier();

        scanBlockInclusive(lindex);

        let inLanePrefix = temp[lindex] - c; // exclusive within chunk
        let chunkTotal = temp[SCAN_PARALLEL_WORKGROUP_SIZE - 1u];

        if i < numBlocks {
            {
                blockSums[i] = carry + inLanePrefix; // global exclusive offset
            }
        }
        workgroupBarrier();

        carry += chunkTotal;
        workgroupBarrier();
    }
}

@compute  @workgroup_size(SCAN_PARALLEL_WORKGROUP_SIZE)
fn scanAddOffsetMain(@builtin(local_invocation_index) lindex: u32, @builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) numWg: vec3u) {
    let windex = wid.y * numWg.x + wid.x;
    let n = uniforms.countsBufferLength;
    let i = windex * SCAN_PARALLEL_WORKGROUP_SIZE + lindex;
    if i < n {
        counts[i] += blockSums[windex];
    }
}

// Inclusive Hillis-Steele over temp[],  caller converts to exclusive
fn scanBlockInclusive(lindex: u32) {
    for (var shift = 1u; shift < SCAN_PARALLEL_WORKGROUP_SIZE; shift = shift << 1u) {
        var prefix = 0u;

        if shift <= lindex {
            prefix = temp[lindex - shift];
        }

        workgroupBarrier();
        temp[lindex] = temp[lindex] + prefix;
        workgroupBarrier();
    }
}