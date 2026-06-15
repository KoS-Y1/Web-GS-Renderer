@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> instanceCount: u32;
@group(0) @binding(2) var<storage, read_write> outRanges: array<u32>; 

const TILE_RANGE_WORKGROUP_SIZE = 256u;

@compute @workgroup_size(TILE_RANGE_WORKGROUP_SIZE)
fn computeMain(
    @builtin(local_invocation_index) lindex: u32,
    @builtin(workgroup_id) wid: vec3u,
    @builtin(num_workgroups) numWg: vec3u
) {
    let gindex = (wid.y * numWg.x + wid.x) * TILE_RANGE_WORKGROUP_SIZE + lindex;
    if gindex >= instanceCount {
        return;
    }

    let tileIndex = keys[gindex] >> 16u;

    if gindex == 0u {
        outRanges[tileIndex * 2] = 0u;
    }
    else {
        let prevTileIndex = keys[gindex - 1] >> 16u;

        if tileIndex > prevTileIndex {
            outRanges[prevTileIndex * 2 + 1] = gindex;
            outRanges[tileIndex * 2] = gindex;
        }
    }

    if gindex == instanceCount - 1u {
        outRanges[tileIndex * 2 + 1] = instanceCount;
    }
}