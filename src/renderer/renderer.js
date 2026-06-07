import {mat4, vec3} from "wgpu-matrix";

import {createShaderModule} from "../gpu/device.js";

import blitWGSL from "../shaders/blit.wgsl?raw"
import preprocessWGSL from "../shaders/preprocess.wgsl?raw"
import sortWGSL from "../shaders/sort.wgsl?raw"

const SCREE_VERTEX_COUNT = 4;

// Must match MAX_COUNT_FACTOR in preprocess.wgsl: output buffers hold count * factor instances.
const MAX_INSTANCE_FACTOR = 8;

const WORKGROUP_SIZE = 32;

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
// 2 mat4 (128) + 3 vec2 (24) + count + 3 padding (16) = 168, rounded up to 16-byte multiple.
const GLOBAL_UNIFORM_SIZE = 176;

export class Renderer {
    #device;
    #context;
    #format;
    #width;
    #height;

    #gsBuffers;
    #currentGs;

    #uniformBuffer;

    #linearSampler;

    #preprocessPipeline;
    #preprocessBindGroup0;
    #preprocessBindGroup1;

    #sortPipeline;
    #sortBindGroup;

    #blitPipeline;
    #blitBindGroup;

    #computeOutput;
    #computeOutputView;

    #keysBuffer;
    #physicalIndexBuffer;
    #instanceCountBuffer;
    #outputBufferCount;

    #bindGroupsDirty;

    constructor(device, context, format) {
        this.#device = device;
        this.#context = context;
        this.#format = format;

        this.#gsBuffers = new Map();
        this.#currentGs = "strawberry";

        this.#uniformBuffer = this.#device.createBuffer({
            label: "uniform buffer",
            size: GLOBAL_UNIFORM_SIZE,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // --- Preprocess pipeline ---------------------------------------------
        const preprocessShaderModule = createShaderModule(this.#device, "preprocess shader", preprocessWGSL);
        // group(0): uniforms + gs params (read-only)
        const preprocessGroup0Layout = this.#device.createBindGroupLayout({
            entries: [
                {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
                {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}},
            ]
        });
        // group(1): keys + physical indices + instance counter (all read_write)
        const preprocessGroup1Layout = this.#device.createBindGroupLayout({
            entries: [
                {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
                {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
                {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
            ]
        });
        this.#preprocessPipeline = this.#device.createComputePipeline({
            label: "preprocess pipeline",
            layout: this.#device.createPipelineLayout({
                bindGroupLayouts: [preprocessGroup0Layout, preprocessGroup1Layout]
            }),
            compute: {module: preprocessShaderModule, entryPoint: "computeMain"}
        });

        // --- Sort pipeline (debug visualizer) --------------------------------
        const sortShaderModule = createShaderModule(this.#device, "sort shader", sortWGSL);
        const sortGroup0Layout = this.#device.createBindGroupLayout({
            entries: [
                {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
                {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}},
                {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}},
                {binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}},
                {binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}},
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {format: "rgba8unorm", viewDimension: "2d"}
                },
            ]
        });
        this.#sortPipeline = this.#device.createComputePipeline({
            label: "sort pipeline",
            layout: this.#device.createPipelineLayout({bindGroupLayouts: [sortGroup0Layout]}),
            compute: {module: sortShaderModule, entryPoint: "computeMain"}
        });

        // --- Blit pipeline ---------------------------------------------------
        const blitShaderModule = createShaderModule(this.#device, "blit shader", blitWGSL);
        this.#blitPipeline = this.#device.createRenderPipeline({
            label: "blit pipeline",
            layout: "auto",
            vertex: {module: blitShaderModule, entryPoint: "vertexMain"},
            fragment: {
                module: blitShaderModule,
                entryPoint: "fragmentMain",
                targets: [{format: this.#format}],
            },
            primitive: {topology: "triangle-strip", cullMode: "none"},
        });

        this.#linearSampler = this.#device.createSampler({
            label: "linear sampler",
            magFilter: "linear",
            minFilter: "linear",
        });

        this.#width = 0;
        this.#height = 0;
        this.#computeOutput = null;
        this.#computeOutputView = null;
        this.#keysBuffer = null;
        this.#physicalIndexBuffer = null;
        this.#instanceCountBuffer = null;
        this.#outputBufferCount = 0;
        this.#preprocessBindGroup0 = null;
        this.#preprocessBindGroup1 = null;
        this.#sortBindGroup = null;
        this.#blitBindGroup = null;
        this.#bindGroupsDirty = true;
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
        const encoder = this.#device.createCommandEncoder({
            label: "frame encoder",
        });

        this.#resize();
        this.#update();
        this.#ensureBindGroups();
        this.#render(encoder);

        this.#device.queue.submit([encoder.finish()]);
    }

    #resize() {
        const {width, height} = this.#context.canvas;
        if (width === this.#width && height === this.#height) {
            return;
        }
        this.#width = width;
        this.#height = height;

        this.#computeOutput?.destroy();
        this.#computeOutput = this.#device.createTexture({
            label: "compute output",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.#computeOutputView = this.#computeOutput.createView();

        this.#bindGroupsDirty = true;
    }

    #update() {
        const gs = this.#gsBuffers.get(this.#currentGs);

        // Output buffers depend on gs count, not canvas size — (re)create when the count changes.
        if (this.#outputBufferCount !== gs.count) {
            const maxInstances = gs.count * MAX_INSTANCE_FACTOR;

            this.#keysBuffer?.destroy();
            this.#keysBuffer = this.#device.createBuffer({
                label: "keys buffer",
                size: maxInstances * 4,
                usage: GPUBufferUsage.STORAGE,
            });
            this.#physicalIndexBuffer?.destroy();
            this.#physicalIndexBuffer = this.#device.createBuffer({
                label: "physical index buffer",
                size: maxInstances * 4,
                usage: GPUBufferUsage.STORAGE,
            });
            this.#instanceCountBuffer?.destroy();
            this.#instanceCountBuffer = this.#device.createBuffer({
                label: "instance count buffer",
                size: 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });

            this.#outputBufferCount = gs.count;
            this.#bindGroupsDirty = true;
        }

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

        const aspect = this.#width / this.#height;
        const fovY = 60 * Math.PI / 180;
        const view = mat4.lookAt(vec3.create(0, 0, 0), vec3.create(0, 0, 1), vec3.create(0, 1, 0));
        const proj = mat4.perspective(fovY, aspect, 0.2, 1000);
        const viewProj = mat4.multiply(proj, view);
        const tanFovY = Math.tan(fovY / 2);
        const tanFovX = tanFovY * aspect;
        const focalX = this.#width / (2 * tanFovX);
        const focalY = this.#height / (2 * tanFovY);

        floatData.set(view, viewOffset);
        floatData.set(viewProj, viewProjOffset);
        floatData.set([focalX, focalY], focalOffset);
        floatData.set([tanFovX, tanFovY], tanFovOffset);
        floatData.set([this.#width, this.#height], textureSizeOffset);
        uintData[countOffset] = gs.count;

        this.#device.queue.writeBuffer(this.#uniformBuffer, 0, uniformBytes);
    }

    #ensureBindGroups() {
        if (!this.#bindGroupsDirty) {
            return;
        }
        const gs = this.#gsBuffers.get(this.#currentGs);

        this.#preprocessBindGroup0 = this.#device.createBindGroup({
            label: "preprocess bind group 0",
            layout: this.#preprocessPipeline.getBindGroupLayout(0),
            entries: [
                {binding: 0, resource: {buffer: this.#uniformBuffer}},
                {binding: 1, resource: {buffer: gs.buffer}},
            ],
        });
        this.#preprocessBindGroup1 = this.#device.createBindGroup({
            label: "preprocess bind group 1",
            layout: this.#preprocessPipeline.getBindGroupLayout(1),
            entries: [
                {binding: 0, resource: {buffer: this.#keysBuffer}},
                {binding: 1, resource: {buffer: this.#physicalIndexBuffer}},
                {binding: 2, resource: {buffer: this.#instanceCountBuffer}},
            ],
        });

        this.#sortBindGroup = this.#device.createBindGroup({
            label: "sort bind group",
            layout: this.#sortPipeline.getBindGroupLayout(0),
            entries: [
                {binding: 0, resource: {buffer: this.#uniformBuffer}},
                {binding: 1, resource: {buffer: gs.buffer}},
                {binding: 2, resource: {buffer: this.#keysBuffer}},
                {binding: 3, resource: {buffer: this.#physicalIndexBuffer}},
                {binding: 4, resource: {buffer: this.#instanceCountBuffer}},
                {binding: 5, resource: this.#computeOutputView},
            ],
        });

        this.#blitBindGroup = this.#device.createBindGroup({
            label: "blit bindGroup",
            layout: this.#blitPipeline.getBindGroupLayout(0),
            entries: [
                {binding: 0, resource: this.#computeOutputView},
                {binding: 1, resource: this.#linearSampler},
            ],
        });

        this.#bindGroupsDirty = false;
    }

    #render(encoder) {
        const count = this.#gsBuffers.get(this.#currentGs).count;
        const maxInstances = count * MAX_INSTANCE_FACTOR;

        // Reset the atomic instance counter before preprocess accumulates into it.
        encoder.clearBuffer(this.#instanceCountBuffer);

        const preprocessPass = encoder.beginComputePass({label: "preprocess pass"});
        preprocessPass.setPipeline(this.#preprocessPipeline);
        preprocessPass.setBindGroup(0, this.#preprocessBindGroup0);
        preprocessPass.setBindGroup(1, this.#preprocessBindGroup1);
        preprocessPass.dispatchWorkgroups(Math.ceil(count / WORKGROUP_SIZE));
        preprocessPass.end();

        // Debug visualization: one thread per stored instance (the shader early-outs past instanceCount).
        // We don't know the real instance count on the CPU, so dispatch for the worst case.
        const maxWorkgroups = this.#device.limits.maxComputeWorkgroupsPerDimension;
        let sortWorkgroups = Math.ceil(maxInstances / WORKGROUP_SIZE);
        if (sortWorkgroups > maxWorkgroups) {
            console.warn(
                `sort dispatch ${sortWorkgroups} exceeds device limit ${maxWorkgroups}; ` +
                `some instances will not be visualized. Use indirect dispatch from instanceCount instead.`
            );
            sortWorkgroups = maxWorkgroups;
        }

        const sortPass = encoder.beginComputePass({label: "sort pass"});
        sortPass.setPipeline(this.#sortPipeline);
        sortPass.setBindGroup(0, this.#sortBindGroup);
        sortPass.dispatchWorkgroups(sortWorkgroups);
        sortPass.end();

        const renderPass = encoder.beginRenderPass({
            label: "blit pass",
            colorAttachments: [{
                view: this.#context.getCurrentTexture().createView(),
                clearValue: {r: 0, g: 0, b: 0, a: 0},
                loadOp: "clear",
                storeOp: "store",
            }],
        });
        renderPass.setPipeline(this.#blitPipeline);
        renderPass.setBindGroup(0, this.#blitBindGroup);
        renderPass.draw(SCREE_VERTEX_COUNT);
        renderPass.end();
    }
}
