## Models

- [Strawberry](https://superspl.at/scene/84df8849)
- [Cochem Imperial Castle, Germany](https://superspl.at/scene/9b18007e)

## Optimization

### Single Work Group Scan

| Step         | ms           |
|--------------|--------------|
| preprocess   | 0.25         |
| offset scan  | 3.888        |
| emit         | 0.053        |
| indirect arg | 0.002        |
| count        | 0.16         |
| scan         | 1414.27      |
| reorder      | 0.39         |
| tile ranges  | 0.027        |
| raster       | 0.774        |
| blit         | 0.008        |
| **TOTAL**    | **1419.821** |

### Parallel Scan

| Step            | Time (ms) |
|-----------------|---|
| preprocess      | 0.727 |
| offset scan     | 4.782 |
| emit            | 0.309 |
| indirect arg    | 0.003 |
| count           | 0.689 |
| scan local      | 4.389 |
| scan block sums | 4.745 |
| scan add offset | 3.989 |
| reorder         | 3.372 |
| tile ranges     | 0.061 |
| raster          | 1.202 |
| blit            | 0.007 |
| **TOTAL**       | **24.274** |

## Reference

- [WebGPU Fundamentals ](https://webgpufundamentals.org/)
- [WebGPU Samples](https://github.com/webgpu/webgpu-samples)
- [The PLY Format](https://developer.playcanvas.com/user-manual/gaussian-splatting/formats/ply/)
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/pdf/2308.04079)
- [Differential Gaussian Rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [Introduction to GPU Radix Sort](https://gpuopen.com/download/Introduction_to_GPU_Radix_Sort.pdf)
- [GPU Gems 3, Chapter 39. Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)
