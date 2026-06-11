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
fn computeMain(@builtin(local_invocation_index)lidx: u32) {
    if lidx == 0u {
        prevLocalPrefixSum = 0u;
    }
    workgroupBarrier();

    let blockCounts = (uniforms.countsBufferLength + SCAN_WORKGROUP_SIZE - 1u) / SCAN_WORKGROUP_SIZE;
    for (var b = 0u; b < blockCounts; b++) {
        let i = b * SCAN_WORKGROUP_SIZE + lidx; // index at the global count buffer

        var c = 0u;
        if i < uniforms.countsBufferLength {
            c = counts[i];
        }
        temp[lidx] = c;
        workgroupBarrier();

        // Hillis-Steele scan
        // Calculate prefix sum in current block
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

        if i < uniforms.countsBufferLength {
            counts[i] = prevLocalPrefixSum + inLanePrefix;
        }
        workgroupBarrier();

        if lidx == 0u {
            prevLocalPrefixSum = prevLocalPrefixSum + inLaneLastPrefix;
        }
        workgroupBarrier();
    }
}
