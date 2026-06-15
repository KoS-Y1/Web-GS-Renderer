struct RadixUniforms {
    instanceCount: u32,
    blockCountMax: u32,
    shiftAmount: u32,
    countsBufferLength: u32,
};

@group(0) @binding(0) var<uniform> uniforms: RadixUniforms;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> counts: array<u32>;
@group(0) @binding(4) var<storage, read> physicalIndex: array<u32>;

@group(1) @binding(0) var<storage, read_write> outKeys: array<u32>;
@group(1) @binding(1) var<storage, read_write> outPhysicalIndices: array<u32>;

const RADIX_BLOCK_SIZE = 256u;
const RADIX_DIGITS = 256u;
const SCAN_WORKGROUP_SIZE = 256u;

var<workgroup> localCounts: array<atomic<u32>, RADIX_DIGITS>;

@compute @workgroup_size(RADIX_BLOCK_SIZE)
fn countMain(@builtin(local_invocation_index) lindex: u32, @builtin(workgroup_id) wid: vec3u) {
    atomicStore(&localCounts[lindex], 0u);
    workgroupBarrier();

    let windex = wid.x;
    let i = windex * RADIX_BLOCK_SIZE + lindex;
    if i < uniforms.instanceCount {
        let d = (keys[i] >> uniforms.shiftAmount) & 0xFFu;
        atomicAdd(&localCounts[d], 1u);
    }
    workgroupBarrier();

    counts[lindex * uniforms.blockCountMax + windex] = atomicLoad(&localCounts[lindex]);
}

var<workgroup> temp: array<u32, SCAN_WORKGROUP_SIZE>;
var<workgroup> prevLocalPrefixSum: u32;

@compute @workgroup_size(SCAN_WORKGROUP_SIZE)
fn scanMain(@builtin(local_invocation_index) lidx: u32) {
    if lidx == 0u {
        prevLocalPrefixSum = 0u;
    }
    workgroupBarrier();

    let blockCounts = (uniforms.countsBufferLength + SCAN_WORKGROUP_SIZE - 1u) / SCAN_WORKGROUP_SIZE;
    for (var b = 0u; b < blockCounts; b++) {
        let n = uniforms.countsBufferLength;
        let i = b * SCAN_WORKGROUP_SIZE + lidx; // index at the global count buffer

        var c = 0u;
        if i < n {
            c = counts[i];
        }
        temp[lidx] = c;
        workgroupBarrier();

        // Hillis-Steele scan within the current block
        for (var shift = 1u; shift < SCAN_WORKGROUP_SIZE; shift = shift << 1u) {
            var prefix = 0u;

            if shift <= lidx {
                prefix = temp[lidx - shift];
            }

            workgroupBarrier(); // Wait for all reads
            temp[lidx] = temp[lidx] + prefix;
            workgroupBarrier(); // Wait for all writes
        }
        let inLanePrefix = temp[lidx] - c;
        let inLaneLastPrefix = temp[SCAN_WORKGROUP_SIZE - 1u];

        if i < n {
            counts[i] = prevLocalPrefixSum + inLanePrefix;
        }
        workgroupBarrier();

        if lidx == 0u {
            prevLocalPrefixSum = prevLocalPrefixSum + inLaneLastPrefix;
        }
        workgroupBarrier();
    }
}

var<workgroup> localDigit: array<u32, RADIX_BLOCK_SIZE>;

@compute @workgroup_size(RADIX_BLOCK_SIZE)
fn reorderMain(@builtin(local_invocation_index) lindex: u32, @builtin(workgroup_id) wid: vec3u) {
    let windex = wid.x;
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
