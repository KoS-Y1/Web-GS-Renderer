// Radix sort (parallel scan variant): count -> 3-phase prefix sum -> reorder.
// Scan based on 39.2.4 Arrays of Arbitrary Size
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
struct RadixUniforms {
    instanceCount: u32,
    blockCountMax: u32,
    shiftAmount: u32,
    countsBufferLength: u32,
};

@group(0) @binding(0) var<uniform> uniforms: RadixUniforms;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> blockSums: array<u32>;
@group(0) @binding(4) var<storage, read> physicalIndex: array<u32>;

@group(1) @binding(0) var<storage, read_write> outKeys: array<u32>;
@group(1) @binding(1) var<storage, read_write> outPhysicalIndices: array<u32>;

const RADIX_BLOCK_SIZE = 256u;
const RADIX_DIGITS = 256u;
const SCAN_PARALLEL_WORKGROUP_SIZE = 256u;

var<workgroup> localCounts: array<atomic<u32>, RADIX_DIGITS>;

@compute @workgroup_size(RADIX_BLOCK_SIZE)
fn countMain(@builtin(local_invocation_index) lindex: u32, @builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) numWg: vec3u) {
    atomicStore(&localCounts[lindex], 0u);
    workgroupBarrier();

    let windex = wid.y * numWg.x + wid.x;
    let i = windex * RADIX_BLOCK_SIZE + lindex;
    if i < uniforms.instanceCount {
        let d = (keys[i] >> uniforms.shiftAmount) & 0xFFu;
        atomicAdd(&localCounts[d], 1u);
    }
    workgroupBarrier();

    // 2D over-dispatch can yield windex beyond the real block count; guard the write.
    if windex < uniforms.blockCountMax {
        counts[lindex * uniforms.blockCountMax + windex] = atomicLoad(&localCounts[lindex]);
    }
}

var<workgroup> temp: array<u32, SCAN_PARALLEL_WORKGROUP_SIZE>;

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
            blockSums[i] = carry + inLanePrefix; // global exclusive offset
        }
        workgroupBarrier();

        carry += chunkTotal;
        workgroupBarrier();
    }
}

@compute @workgroup_size(SCAN_PARALLEL_WORKGROUP_SIZE)
fn scanAddOffsetMain(@builtin(local_invocation_index) lindex: u32, @builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) numWg: vec3u) {
    let windex = wid.y * numWg.x + wid.x;
    let n = uniforms.countsBufferLength;
    let i = windex * SCAN_PARALLEL_WORKGROUP_SIZE + lindex;
    if i < n {
        counts[i] += blockSums[windex];
    }
}

// Inclusive Hillis-Steele over temp[], caller converts to exclusive
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

var<workgroup> localDigit: array<u32, RADIX_BLOCK_SIZE>;

@compute @workgroup_size(RADIX_BLOCK_SIZE)
fn reorderMain(@builtin(local_invocation_index) lindex: u32, @builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) numWg: vec3u) {
    let windex = wid.y * numWg.x + wid.x;
    let i = windex * RADIX_BLOCK_SIZE + lindex; // global index

    var d = 0u;
    var valid = false;
    if i < uniforms.instanceCount {
        valid = true;
        d = (keys[i] >> uniforms.shiftAmount) & 0xFFu;
    }
    localDigit[lindex] = d;
    workgroupBarrier();

    if valid {
        var rank = 0u;
        // Count same digits before current thread in this workgroup
        for (var j = 0u; j < lindex; j++) {
            if localDigit[j] == d {
                rank++;
            }
        }

        let base = counts[d * uniforms.blockCountMax + windex];
        let pos = base + rank;

        outKeys[pos] = keys[i];
        outPhysicalIndices[pos] = physicalIndex[i];
    }
}
