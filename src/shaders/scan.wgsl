struct RadixUniforms {
    instanceCount: u32,
    blockCountMax: u32,
    shiftAmount: u32,
    countsBufferLength: u32,
};
@group(0) @binding(0) var<uniform> uniforms: RadixUniforms;
@group(0) @binding(1) var<storage, read_write> counts: array<u32>;

const SCAN_WORKGROUP_SIZE = 256u;

var<workgroup> temp: array<u32, SCAN_WORKGROUP_SIZE>;
var<workgroup> prevLocalPrefixSum: u32;

@compute @workgroup_size(SCAN_WORKGROUP_SIZE)
fn computeMain(@builtin(local_invocation_index)lindex: u32) {
    if lindex == 0u {
        prevLocalPrefixSum = 0u;
    }
    workgroupBarrier();

    let blockCounts = (uniforms.countsBufferLength + SCAN_WORKGROUP_SIZE - 1u) / SCAN_WORKGROUP_SIZE;
    for (var b = 0u; b < blockCounts; b++) {
        let n = uniforms.countsBufferLength;
        let i = b * SCAN_WORKGROUP_SIZE + lindex; // index at the global count buffer

        var c = 0u;
        if i < n {
            c = counts[i];
        }
        temp[lindex] = c;
        workgroupBarrier();

        // Hillis-Steele scan
        // Calculate prefix sum in current block
        for (var shift = 1u; shift < SCAN_WORKGROUP_SIZE; shift = shift << 1u) {
            var prefix = 0u;

            if shift <= lindex {
                prefix = temp[lindex - shift];
            }

            workgroupBarrier(); // Wait for all reads
            temp[lindex] = temp[lindex] + prefix;
            workgroupBarrier(); // Wait for all writes
        }
        let inLanePrefix = temp[lindex] - c;
        let inLaneLastPrefix = temp[SCAN_WORKGROUP_SIZE - 1u];

        if i < n {
            counts[i] = prevLocalPrefixSum + inLanePrefix;
        }
        workgroupBarrier();

        if lindex == 0u {
            prevLocalPrefixSum = prevLocalPrefixSum + inLaneLastPrefix;
        }
        workgroupBarrier();
    }
}
