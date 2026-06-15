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
const MAX_DISPATCH_DIM = 65535u;

@compute @workgroup_size(1)
fn computeMain() {
    let n = min(instanceCount, uniforms.maxInstanceCount);

    // Split the block count into a 2D grid so neither dimension exceeds the dispatch limit.
    let blocks = (n + RADIX_BLOCK_SIZE - 1u) / RADIX_BLOCK_SIZE;
    let gridX = max(1u, min(blocks, MAX_DISPATCH_DIM));
    outArgBuffer[0] = gridX;
    outArgBuffer[1] = (blocks + gridX - 1u) / gridX;
    outArgBuffer[2] = 1u;

    for (var p = 0u; p < RADIX_PASS_COUNT; p++) {
        outRadixUniform[p * RADIX_UNIFORM_STRIDE_U32] = n;
    }
}