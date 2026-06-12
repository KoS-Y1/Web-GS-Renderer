struct IndirectArgUniform {
    maxInstanceCount: u32,
};
@group(0) @binding(0) var<uniform> uniforms: IndirectArgUniform;
@group(0) @binding(1) var<storage, read> instanceCount: u32;
@group(0) @binding(2) var<storage, read_write> outArgBuffer: array<u32>;
@group(0) @binding(3) var<storage, read_write> outRadixUniform: array<u32>;

const RADIX_BLOCK_SIZE = 256u;    
const RADIX_PASS_COUNT = 4u;
const RADIX_UNIFORM_STRIDE_U32 = 64u;  

@compute @workgroup_size(1)
fn computeMain() {
    let n = min(instanceCount, uniforms.maxInstanceCount);

    outArgBuffer[0] = (n + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE; 
    outArgBuffer[1] = 1u;
    outArgBuffer[2] = 1u;

    for (var p = 0u; p < RADIX_PASS_COUNT; p++) {
        outRadixUniform[p * RADIX_UNIFORM_STRIDE_U32] = n;
    }
}