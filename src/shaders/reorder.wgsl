struct RadixUniforms {
    instasnceCount: u32,
    blockCountMax: u32,
    shiftAmount: u32,
    countsBufferLength: u32,
};

@group(0) @binding(0) var<uniform> uniforms: RadixUniforms;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read> physicalIndex: array<u32>;
@group(0) @binding(3) var<storage, read> prefixSums: array<u32>;

@group(1) @binding(0) var<storage, read_write> outKeys: array<u32>;
@group(1) @binding(1) var<storage,read_write> outPhysicalIndecies: array<u32>;

const RADIX_BLOCK_SIZE = 256u;

var<workgroup> localDigit: array<u32, RADIX_BLOCK_SIZE>;

@compute @workgroup_size(RADIX_BLOCK_SIZE)
fn computeMain(@builtin(local_invocation_index) lindex: u32, @builtin(workgroup_id) wid: vec3u) {
    let windex = wid.x;
    let i = windex * RADIX_BLOCK_SIZE + lindex; // global index

    // Load d to localDigit 
    var d = 0u;
    var valid = false;
    if i < uniforms.instasnceCount {
        valid = true;
        d = (keys[i] >> uniforms.shiftAmount) & 0xFFu;
    }
    localDigit[lindex] = d;
    workgroupBarrier();

    if valid {
        var rank = 0u;
        // Find how many digits before current thread are the same as current digit in this workgroup
        for (var j = 0u; j < lindex; j++) {
            if localDigit[j] == d {
                rank++;
            }
        }

        let base = prefixSums[d * uniforms.blockCountMax + windex];
        let pos = base + rank;

        outKeys[pos] = keys[i];
        outPhysicalIndecies[pos] = physicalIndex[i];
    }
}