import {mat4, vec3} from "wgpu-matrix";

import {createShaderModule} from "../gpu/device.js";
import {fail} from "../utils/utils.js";

import blitWGSL from "../shaders/blit.wgsl?raw"
import preprocessWGSL from "../shaders/preprocess.wgsl?raw"
import indirectArgWGSL from "../shaders/indirect_arg.wgsl?raw"
import countWGSL from "../shaders/count.wgsl?raw"
import scanWGSL from "../shaders/scan.wgsl?raw"
import reorderWGSL from "../shaders/reorder.wgsl?raw"

const SCREEN_VERTEX_COUNT = 4;
const MAX_INSTANCE_FACTOR = 8;
const PREPROCESS_WORKGROUP_SIZE = 32;
const RADIX_BLOCK_SIZE = 256; // count/reorder radix block size (elements per workgroup)

const RADIX_PING_PONG_COUNT = 2;


/*
struct GlobalUniforms {
    view: mat4x4f,
    viewProj: mat4x4f,
    focal: vec2f,
    tanFov: vec2f,
    textureSize: vec2f,
    count: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
};
 */
// 2 mat4 (128) + 3 vec2 (24) + count + 3 paddings (16) = 168, rounded up to 16-byte multiple.
const GLOBAL_UNIFORM_SIZE = 176;

/*
struct IndirectArgUniform {
    maxInstanceCount: u32,
};
 */
const INDIRECT_ARG_UNIFORM_SIZE = 4;

/*
struct RadixUniforms {
    instasnceCount: u32,
    blockCountMax: u32,
    shiftAmount: u32,
    countsBufferLength: u32,
};
 */
const RADIX_UNIFORM_SIZE = 16;
const RADIX_DIGITS = 256;
const RADIX_PASS_COUNT = 4;
const RADIX_UNIFORM_STRIDE = 256;


export class Renderer {
    #device;
    #context;
    #format;
    #width;
    #height;
    #frameCount;

    #gsBuffers;
    #currentGs;
    #previousGs;

    #finalImage;
    #finalImageView;

    #keysBuffers;
    #physicalIndicesBuffers;
    #countPrefixSumBuffer;
    #instanceCountBuffer;
    #radixIndirecArgBuffer;

    #globalUniformBuffer;
    #radixUniformBuffer;
    #indirectArgUniformBuffer;

    #linearSampler;

    #preprocessPipeline;
    #preprocessBindGroup0;
    #preprocessBindGroup1;

    #indirectArgPipeline;
    #indirectArgBindGroup;

    #countPipeline;
    #countBindGroups;

    #scanPipeline;
    #scanBindGroup;

    #reorderPipeline;
    #reorderBindGroup0s;
    #reorderBindGroup1s;

    #blitPipeline;
    #blitBindGroup;

    #finalImageDirty;

    constructor(device, context, format) {
        this.#device = device;
        this.#context = context;
        this.#format = format;

        this.#globalUniformBuffer = this.#device.createBuffer({
            label: "global uniform buffer",
            size: GLOBAL_UNIFORM_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.#radixUniformBuffer = this.#device.createBuffer({
            label: "radix uniform buffer",
            size: RADIX_PASS_COUNT * RADIX_UNIFORM_STRIDE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.#indirectArgUniformBuffer = this.#device.createBuffer({
            label: "indirect arg uniform buffer",
            size: INDIRECT_ARG_UNIFORM_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.#instanceCountBuffer = this.#device.createBuffer({
            label: "instance count buffer",
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.#radixIndirecArgBuffer = this.#device.createBuffer({
            label: "indirect args buffer",
            size: 3 * 4,
            usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const createComputePipeline = (name, shaderModule, groupLayouts) => {
            return this.#device.createComputePipeline({
                label: `${name} compute pipeline`,
                layout: device.createPipelineLayout({
                    bindGroupLayouts: groupLayouts,
                }),
                compute: {
                    module: shaderModule,
                    entryPoint: "computeMain"
                }
            })
        }

        const createRenderPipeline = (name, shaderModule, groupLayouts, primitive) => {
            return this.#device.createRenderPipeline({
                label: `${name} render pipeline`,
                layout: device.createPipelineLayout({
                    bindGroupLayouts: groupLayouts,
                }),
                vertex: {
                    module: shaderModule,
                    entryPoint: "vertexMain",
                },
                fragment: {
                    module: shaderModule,
                    entryPoint: "fragmentMain",
                    targets: [{format: this.#format}]
                },
                primitive: primitive
            })
        }

        const createPipeline = (name, code, create, groupEntries) => {
            const shaderModule = createShaderModule(device, `${name} shader`, code);

            const groupLayouts = groupEntries.map(
                (entry) => this.#device.createBindGroupLayout({entries: entry})
            );

            return create(name, shaderModule, groupLayouts);
        }

        this.#preprocessPipeline = createPipeline(
            "preprocess",
            preprocessWGSL,
            createComputePipeline,
            [
                [
                    {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
                    {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}}
                ],
                [
                    {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
                    {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
                    {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
                ]
            ],
        );

        this.#indirectArgPipeline = createPipeline(
            "indirect arg",
            indirectArgWGSL,
            createComputePipeline,
            [
                [
                    {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
                    {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
                    {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
                    {binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
                ]
            ]
        );

        this.#countPipeline = createPipeline(
            "count",
            countWGSL,
            createComputePipeline,
            [
                [
                    {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform", hasDynamicOffset: true}},
                    {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}},
                    {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
                ]
            ],
        );

        this.#scanPipeline = createPipeline(
            "scan",
            scanWGSL,
            createComputePipeline,
            [
                [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {type: "uniform", hasDynamicOffset: true},
                    },
                    {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
                ]
            ],
        )

        this.#reorderPipeline = createPipeline(
            "reorder",
            reorderWGSL,
            createComputePipeline,
            [
                [
                    {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform", hasDynamicOffset: true}},
                    {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}},
                    {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}},
                    {binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}},
                ],
                [
                    {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
                    {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
                ]
            ],
        );

        this.#blitPipeline = createPipeline(
            "blit",
            blitWGSL,
            (name, shaderModule, groupLayouts) =>
                createRenderPipeline(name,
                    shaderModule,
                    groupLayouts,
                    {
                        topology: "triangle-strip",
                        cullMode: "none"
                    }),
            [
                [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.FRAGMENT,
                        texture: {format: this.#format, viewDimension: "2d"}
                    },
                    {binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {type: "filtering"}}
                ]
            ],
        );

        this.#linearSampler = this.#device.createSampler({
            label: "linear sampler",
            magFilter: "linear",
            minFilter: "linear",
        });

        this.#width = 0;
        this.#height = 0;

        this.#gsBuffers = new Map();
        this.#currentGs = "";
        this.#previousGs = "";

        this.#keysBuffers = [];
        this.#physicalIndicesBuffers = [];
        this.#countPrefixSumBuffer = null;

        this.#preprocessBindGroup0 = null;
        this.#preprocessBindGroup1 = null;
        this.#indirectArgBindGroup = null;
        this.#countBindGroups = [];
        this.#scanBindGroup = null;
        this.#reorderBindGroup0s = [];
        this.#reorderBindGroup1s = [];
        this.#blitBindGroup = null;

        this.#finalImageDirty = true;
    }

    setGs(name) {
        if (!this.#gsBuffers.has(name)) {
            fail(`Invalid Gs buffer for name "${name}"`);
        }
        this.#currentGs = name;
    }

    uploadGsData(data, name) {
        const buffer = this.#device.createBuffer({
            label: `${name} gs buffer`,
            size: data.packed.byteLength,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
        });
        new Float32Array(buffer.getMappedRange()).set(data.packed);
        buffer.unmap();

        this.#gsBuffers.set(name, {buffer, count: data.count});
    }

    execute() {
        this.#resize();
        this.#update();

        const encoder = this.#device.createCommandEncoder({
            label: "frame encoder",
        });
        this.#render(encoder);
        this.#device.queue.submit([encoder.finish()]);

        ++this.#frameCount;
    }

    #resize() {
        const {width, height} = this.#context.canvas;
        if (width === this.#width && height === this.#height) {
            return;
        }
        this.#width = width;
        this.#height = height;

        this.#finalImage?.destroy();
        this.#finalImage = this.#device.createTexture({
            label: "compute output",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.#finalImageView = this.#finalImage.createView();

        this.#finalImageDirty = true;
    }

    #update() {
        if (!this.#gsBuffers.has(this.#currentGs)) {
            fail(`GS ${name} does not exist`);
        }
        const gs = this.#gsBuffers.get(this.#currentGs);
        const maxInstance = gs.count * MAX_INSTANCE_FACTOR;
        let gsDirty = false;

        if (this.#currentGs !== this.#previousGs) {
            this.#previousGs = this.#currentGs;

            const createStorageBuffer = (name, size) => {
                return this.#device.createBuffer({
                    label: name,
                    size: size,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
                });
            };

            this.#keysBuffers = Array.from({length: RADIX_PING_PONG_COUNT}, (_, i) => {
                this.#keysBuffers[i]?.destroy();
                return createStorageBuffer(`key buffer ${i}`, maxInstance * 4);
            });
            this.#physicalIndicesBuffers = Array.from({length: RADIX_PING_PONG_COUNT}, (_, i) => {
                this.#physicalIndicesBuffers[i]?.destroy();
                return createStorageBuffer(`physical indices buffer ${i}`, maxInstance * 4);
            })
            this.#countPrefixSumBuffer?.destroy();
            this.#countPrefixSumBuffer = createStorageBuffer("count prefix sum buffer", RADIX_DIGITS * Math.ceil(maxInstance / RADIX_BLOCK_SIZE) * 4);

            gsDirty = true;
        }

        const self = this;
        updateGlobalUniformBuffer();
        updateInddirectArgUniformBuffer();
        updateRadixUniformBuffer();

        if (this.#finalImageDirty) {
            this.#blitBindGroup = this.#device.createBindGroup({
                label: "blit bindGroup",
                layout: this.#blitPipeline.getBindGroupLayout(0),
                entries: [
                    {binding: 0, resource: this.#finalImageView},
                    {binding: 1, resource: this.#linearSampler},
                ],
            });
            this.#finalImageDirty = false;
        }

        if (gsDirty) {
            this.#preprocessBindGroup0 = this.#device.createBindGroup({
                label: "preprocess bind group 0",
                layout: this.#preprocessPipeline.getBindGroupLayout(0),
                entries: [
                    {binding: 0, resource: {buffer: this.#globalUniformBuffer}},
                    {binding: 1, resource: {buffer: gs.buffer}},
                ],
            });

            this.#preprocessBindGroup1 = this.#device.createBindGroup({
                label: "preprocess bind group 1",
                layout: this.#preprocessPipeline.getBindGroupLayout(1),
                entries: [
                    {binding: 0, resource: {buffer: this.#keysBuffers[0]}},
                    {binding: 1, resource: {buffer: this.#physicalIndicesBuffers[0]}},
                    {binding: 2, resource: {buffer: this.#instanceCountBuffer}},
                ],
            });

            this.#indirectArgBindGroup = this.#device.createBindGroup({
                label: "indirect arg bind group",
                layout: this.#indirectArgPipeline.getBindGroupLayout(0),
                entries: [
                    {binding: 0, resource: {buffer: this.#indirectArgUniformBuffer}},
                    {binding: 1, resource: {buffer: this.#instanceCountBuffer}},
                    {binding: 2, resource: {buffer: this.#radixIndirecArgBuffer}},
                    {binding: 3, resource: {buffer: this.#radixUniformBuffer}},
                ],
            });

            this.#countBindGroups[0] = this.#device.createBindGroup({
                label: "count bind group (ping, read keys[0])",
                layout: this.#countPipeline.getBindGroupLayout(0),
                entries: [
                    {binding: 0, resource: {buffer: this.#radixUniformBuffer, size: RADIX_UNIFORM_SIZE}},
                    {binding: 1, resource: {buffer: this.#keysBuffers[0]}},
                    {binding: 2, resource: {buffer: this.#countPrefixSumBuffer}},
                ],
            });
            this.#countBindGroups[1] = this.#device.createBindGroup({
                label: "count bind group (pong, read keys[1])",
                layout: this.#countPipeline.getBindGroupLayout(0),
                entries: [
                    {binding: 0, resource: {buffer: this.#radixUniformBuffer, size: RADIX_UNIFORM_SIZE}},
                    {binding: 1, resource: {buffer: this.#keysBuffers[1]}},
                    {binding: 2, resource: {buffer: this.#countPrefixSumBuffer}},
                ],
            });

            this.#scanBindGroup = this.#device.createBindGroup({
                label: "scan bind group",
                layout: this.#scanPipeline.getBindGroupLayout(0),
                entries: [
                    {binding: 0, resource: {buffer: this.#radixUniformBuffer, size: RADIX_UNIFORM_SIZE}},
                    {binding: 1, resource: {buffer: this.#countPrefixSumBuffer}},
                ],
            });

            this.#reorderBindGroup0s[0] = this.#device.createBindGroup({
                label: "reorder bind group 0 (ping, read keys[0])",
                layout: this.#reorderPipeline.getBindGroupLayout(0),
                entries: [
                    {binding: 0, resource: {buffer: this.#radixUniformBuffer, size: RADIX_UNIFORM_SIZE}},
                    {binding: 1, resource: {buffer: this.#keysBuffers[0]}},
                    {binding: 2, resource: {buffer: this.#physicalIndicesBuffers[0]}},
                    {binding: 3, resource: {buffer: this.#countPrefixSumBuffer}},
                ],
            });
            this.#reorderBindGroup1s[0] = this.#device.createBindGroup({
                label: "reorder bind group 1 (ping, write keys[1])",
                layout: this.#reorderPipeline.getBindGroupLayout(1),
                entries: [
                    {binding: 0, resource: {buffer: this.#keysBuffers[1]}},
                    {binding: 1, resource: {buffer: this.#physicalIndicesBuffers[1]}},
                ],
            });

            this.#reorderBindGroup0s[1] = this.#device.createBindGroup({
                label: "reorder bind group 0 (pong, read keys[1])",
                layout: this.#reorderPipeline.getBindGroupLayout(0),
                entries: [
                    {binding: 0, resource: {buffer: this.#radixUniformBuffer, size: RADIX_UNIFORM_SIZE}},
                    {binding: 1, resource: {buffer: this.#keysBuffers[1]}},
                    {binding: 2, resource: {buffer: this.#physicalIndicesBuffers[1]}},
                    {binding: 3, resource: {buffer: this.#countPrefixSumBuffer}},
                ],
            });
            this.#reorderBindGroup1s[1] = this.#device.createBindGroup({
                label: "reorder bind group 1 (pong, write keys[0])",
                layout: this.#reorderPipeline.getBindGroupLayout(1),
                entries: [
                    {binding: 0, resource: {buffer: this.#keysBuffers[0]}},
                    {binding: 1, resource: {buffer: this.#physicalIndicesBuffers[0]}},
                ],
            });
        }

        function updateGlobalUniformBuffer() {
            // TODO: fixed view, proj for now
            const uniformBytes = new ArrayBuffer(GLOBAL_UNIFORM_SIZE);
            const floatData = new Float32Array(uniformBytes);
            const uintData = new Uint32Array(uniformBytes);

            const viewOffset = 0;
            const viewProjOffset = viewOffset + 4 * 4;
            const focalOffset = viewProjOffset + 4 * 4;
            const tanFovOffset = focalOffset + 2;
            const textureSizeOffset = tanFovOffset + 2;
            const countOffset = textureSizeOffset + 2;

            const aspect = self.#width / self.#height;
            const fovY = 60 * Math.PI / 180;
            const view = mat4.lookAt(vec3.create(0, 0, 0), vec3.create(0, 0, 1), vec3.create(0, 1, 0));
            const proj = mat4.perspective(fovY, aspect, 0.2, 1000);
            const viewProj = mat4.multiply(proj, view);
            const tanFovY = Math.tan(fovY / 2);
            const tanFovX = tanFovY * aspect;
            const focalX = self.#width / (2 * tanFovX);
            const focalY = self.#height / (2 * tanFovY);

            floatData.set(view, viewOffset);
            floatData.set(viewProj, viewProjOffset);
            floatData.set([focalX, focalY], focalOffset);
            floatData.set([tanFovX, tanFovY], tanFovOffset);
            floatData.set([self.#width, self.#height], textureSizeOffset);
            uintData[countOffset] = gs.count;

            self.#device.queue.writeBuffer(self.#globalUniformBuffer, 0, uniformBytes);
        }

        function updateRadixUniformBuffer() {
            const blockCountMax = Math.ceil(maxInstance / RADIX_BLOCK_SIZE);
            const countsBufferLength = RADIX_DIGITS * blockCountMax;
            const radixBytes = new ArrayBuffer(RADIX_PASS_COUNT * RADIX_UNIFORM_STRIDE);
            for (let pass = 0; pass < RADIX_PASS_COUNT; ++pass) {
                const slot = new Uint32Array(radixBytes, pass * RADIX_UNIFORM_STRIDE, 4);
                slot[0] = maxInstance;        // instanceCount
                slot[1] = blockCountMax;      // blockCountMax
                slot[2] = pass * 8;           // shiftAmount
                slot[3] = countsBufferLength; // countsBufferLength
            }
            self.#device.queue.writeBuffer(self.#radixUniformBuffer, 0, radixBytes);
        }

        function updateInddirectArgUniformBuffer() {
            const uniformBytes = new ArrayBuffer(INDIRECT_ARG_UNIFORM_SIZE);
            const uintData = new Uint32Array(uniformBytes);
            uintData[0] = maxInstance;
            self.#device.queue.writeBuffer(self.#indirectArgUniformBuffer, 0, uniformBytes);
        }

    }

    #render(encoder) {
        const count = this.#gsBuffers.get(this.#currentGs).count;

        this.#keysBuffers.forEach(buffer => encoder.clearBuffer(buffer));
        this.#physicalIndicesBuffers.forEach(buffer => encoder.clearBuffer(buffer));
        encoder.clearBuffer(this.#instanceCountBuffer);

        const executePass = (passEncoder, pipeline, bindGroups, execute) => {
            passEncoder.setPipeline(pipeline);
            bindGroups.forEach((entry, i) => {
                const [bindGroup, dynamicOffsets] = Array.isArray(entry) ? entry : [entry];
                passEncoder.setBindGroup(i, bindGroup, dynamicOffsets);
            });
            execute(passEncoder);
            passEncoder.end();
        };

        executePass(
            encoder.beginComputePass({label: "preprocess pass"}),
            this.#preprocessPipeline,
            [this.#preprocessBindGroup0, this.#preprocessBindGroup1],
            (passEncoder) => passEncoder.dispatchWorkgroups(Math.ceil(count / PREPROCESS_WORKGROUP_SIZE)),
        );

        executePass(
            encoder.beginComputePass({label: "indirect arg pass"}),
            this.#indirectArgPipeline,
            [this.#indirectArgBindGroup],
            (passEncoder) => passEncoder.dispatchWorkgroups(1),
        );

        // Radix sort
        for (let i = 0; i < RADIX_PASS_COUNT; ++i) {
            const parity = i & 1;
            const dynamicOffset = [i * RADIX_UNIFORM_STRIDE];

            encoder.clearBuffer(this.#countPrefixSumBuffer);

            executePass(
                encoder.beginComputePass({label: "count pass"}),
                this.#countPipeline,
                [[this.#countBindGroups[parity], dynamicOffset]],
                (passEncoder) => passEncoder.dispatchWorkgroupsIndirect(this.#radixIndirecArgBuffer, 0),
            );

            executePass(
                encoder.beginComputePass({label: "scan pass"}),
                this.#scanPipeline,
                [[this.#scanBindGroup, dynamicOffset]],
                (passEncoder) => passEncoder.dispatchWorkgroups(1),
            );

            executePass(
                encoder.beginComputePass({label: "reorder pass"}),
                this.#reorderPipeline,
                [
                    [this.#reorderBindGroup0s[parity], dynamicOffset],
                    this.#reorderBindGroup1s[parity],
                ],
                (passEncoder) => passEncoder.dispatchWorkgroupsIndirect(this.#radixIndirecArgBuffer, 0),
            );
        }

        executePass(
            encoder.beginRenderPass({
                label: "blit pass",
                colorAttachments: [{
                    view: this.#context.getCurrentTexture().createView(),
                    clearValue: {r: 0, g: 0, b: 0, a: 0},
                    loadOp: "clear",
                    storeOp: "store",
                }],
            }),
            this.#blitPipeline,
            [this.#blitBindGroup],
            (passEncoder) => passEncoder.draw(SCREEN_VERTEX_COUNT),
        );

    }


}
