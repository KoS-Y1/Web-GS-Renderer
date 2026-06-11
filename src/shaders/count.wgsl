struct RadixUniforms {
    instanceCount: u32,
    blockCountMax: u32,
    shiftAmount: u32,
    countsBufferLength: u32,
};

@group(0) @binding(0) var<uniform> uniforms: RadixUniforms;
@group(0) @binding(1) var<storage, read> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> outCounts: array<u32>;

const RADIX_BLOCK_SIZE = 256u;
const RADIX_DIGITS = 256u;

var<workgroup> localCounts: array<atomic<u32>, RADIX_DIGITS>;

@compute @workgroup_size(RADIX_BLOCK_SIZE)
fn computeMain(@builtin(local_invocation_index) lindex: u32, @builtin(workgroup_id) wid: vec3u) {
    atomicStore(&localCounts[lindex], 0u);
    workgroupBarrier();

    let windex = wid.x;
    let i = windex * RADIX_BLOCK_SIZE + lindex;
    if i < uniforms.instanceCount {
        let d = (keys[i] >> uniforms.shiftAmount) & 0xFFu;
        atomicAdd(&localCounts[d], 1u);
    }
    workgroupBarrier();

    outCounts[lindex * uniforms.blockCountMax + windex] = atomicLoad(&localCounts[lindex]);
}