import {mat4, vec3} from "wgpu-matrix";

import {createShaderModule} from "../gpu/device.js";

import blitWGSL from "../shaders/blit.wgsl?raw"
import sortWGSL from "../shaders/sort.wgsl?raw"

const SCREE_VERTEX_COUNT = 4;
const TILE_COUNT_X = 4;
const TILE_COUNT_Y = 4;

/*
struct GlobalUniforms {
    view: mat4x4f,
    viewProj: mat4x4f,
    focal: vec2f,
    tanFov: vec2f,
    textureSize: vec2f,
    tileSize: vec2f,
    count: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
};
 */
const GLOBAL_UNIFORM_SIZE = (4 * 4 * 2 + 2 * 4 + 4) * 4;

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

    #sortPipeline;
    #sortBindGroup;

    #blitPipeline;
    #blitBindGroup;

    #computeOutput;

    constructor(device, context, format) {
        this.#device = device;
        this.#context = context;
        this.#format = format;

        this.#gsBuffers = new Map();
        this.#currentGs = "strawberry";

        this.#uniformBuffer = this.#device.createBuffer({
            label: "uniform buffer",
            size: GLOBAL_UNIFORM_SIZE * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });


        // TODO: refactor
        const sortShaderModule = createShaderModule(this.#device, "sort shader", sortWGSL);
        const sortBindGroupLayout = this.#device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "uniform",
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage",
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        format: "rgba8unorm",
                        viewDimension: "2d",
                    }
                }
            ]
        });
        const sortLayout = this.#device.createPipelineLayout({
            bindGroupLayouts: [sortBindGroupLayout]
        });
        this.#sortPipeline = this.#device.createComputePipeline({
            label: "sort pipeline",
            layout: sortLayout,
            compute: {
                module: sortShaderModule,
                entryPoint: "computeMain",
            }
        });

        const blitShaderModule = createShaderModule(this.#device, "blit shader", blitWGSL);
        this.#blitPipeline = this.#device.createRenderPipeline({
            label: "blit pipeline",
            layout: "auto",
            vertex: {
                module: blitShaderModule,
                entryPoint: "vertexMain",
            },
            fragment: {
                module: blitShaderModule,
                entryPoint: "fragmentMain",
                targets: [{format: this.#format}],
            },
            primitive: {
                topology: "triangle-strip",
                cullMode: "none",
            },
        });

        this.#linearSampler = this.#device.createSampler({
            label: "linear sampler",
            magFilter: "linear",
            minFilter: "linear",
        });

        this.#width = 0;
        this.#height = 0;
        this.#computeOutput = null;
        this.#sortBindGroup = null;
        this.#blitBindGroup = null;
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
        const computeOutputView = this.#computeOutput.createView();

        this.#sortBindGroup = this.#device.createBindGroup({
            label: "sort bind group",
            layout: this.#sortPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {buffer: this.#uniformBuffer}
                    // {
                    //         buffer: this.#uniformBuffer,
                    //         size: GLOBAL_UNIFORM_SIZE * 4
                    //     }
                },
                {
                    binding: 1,
                    resource: {buffer: this.#gsBuffers.get(this.#currentGs).buffer}
                    // {
                    // buffer: this.#gsBuffers.get(this.#currentGs).buffer,
                    // size: this.#gsBuffers.get(this.#currentGs).buffer.byteLength
                    // }
                },
                {
                    binding: 2,
                    resource: computeOutputView

                }
            ],
        });

        this.#blitBindGroup = this.#device.createBindGroup({
            label: "blit bindGroup",
            layout: this.#blitPipeline.getBindGroupLayout(0),
            entries: [
                {binding: 0, resource: computeOutputView},
                {binding: 1, resource: this.#linearSampler},
            ],
        });
    }

    #update() {
        // TODO: fixed view, proj for now
        const uniformBytes = new ArrayBuffer(GLOBAL_UNIFORM_SIZE);
        const floatData = new Float32Array(uniformBytes);
        const uintData = new Uint32Array(uniformBytes);

        const viewOffset = 0;
        const viewProjOffset = viewOffset + 4 * 4;
        const focalOffset = viewProjOffset + 4 * 4;
        const tanFovOffset = focalOffset + 2;
        const textureSizeOffset = tanFovOffset + 2;
        const tileSizeOffset = textureSizeOffset + 2;
        const countOffset = tileSizeOffset + 2;

        const aspect = this.#width / this.#height;
        const fovY = 60 * Math.PI / 180;
        const view = mat4.lookAt(vec3.create(0, 0, 0), vec3.create(0, 0, 1), vec3.create(0, 1, 0));
        const proj = mat4.perspective(fovY, aspect, 0.2, 1000);
        const viewProj = mat4.multiply(proj, view);
        const tanFovY = Math.tan(fovY / 2);
        const tanFovX = tanFovY * aspect;
        const focalX = this.#width / (2 * tanFovX);
        const focalY = this.#height / (2 * tanFovY);

        const tileWidth = this.#width / TILE_COUNT_X;
        const tileHeight = this.#height / TILE_COUNT_Y;

        floatData.set(view, viewOffset);
        floatData.set(viewProj, viewProjOffset);
        floatData.set([focalX, focalY], focalOffset);
        floatData.set([tanFovX, tanFovY], tanFovOffset);
        floatData.set([this.#width, this.#height], textureSizeOffset);
        floatData.set([tileWidth, tileHeight], tileSizeOffset);
        uintData[countOffset] = this.#gsBuffers.get(this.#currentGs).count;

        this.#device.queue.writeBuffer(this.#uniformBuffer, 0, uniformBytes);
    }


    #render(encoder) {
        const computePass = encoder.beginComputePass({label: "sort pass"});
        computePass.setPipeline(this.#sortPipeline);
        computePass.setBindGroup(0, this.#sortBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.#gsBuffers.get(this.#currentGs).count / 32));
        computePass.end();

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

    #endFrame() {

    }
}