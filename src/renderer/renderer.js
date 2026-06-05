import computeWGSL from "../shaders/compute.wgsl?raw"
import blitWGSL from "../shaders/blit.wgsl?raw"

import {createShaderModule} from "../gpu/device.js";

const SCREE_VERTEX_COUNT = 4;

export class Renderer {
    constructor(device, context, format) {
        this.device = device;
        this.context = context;
        this.format = format;

        this.gsBuffers = new Map();

        const computeShaderModule = createShaderModule(device, "compute shader", computeWGSL);
        this.computePipeline = device.createComputePipeline({
            label: "compute pipeline",
            layout: "auto",
            compute: {
                module: computeShaderModule,
                entryPoint: "computeMain",
            },
        });

        const blitShaderModule = createShaderModule(device, "blit shader", blitWGSL);
        this.blitPipeline = device.createRenderPipeline({
            label: "blit pipeline",
            layout: "auto",
            vertex: {
                module: blitShaderModule,
                entryPoint: "vertexMain",
            },
            fragment: {
                module: blitShaderModule,
                entryPoint: "fragmentMain",
                targets: [{format: this.format}],
            },
            primitive: {
                topology: "triangle-strip",
                cullMode: "none",
            },
        });

        this.linearSampler = device.createSampler({
            label: "linear sampler",
            magFilter: "linear",
            minFilter: "linear",
        });

        this.width = 0;
        this.height = 0;
        this.computeOutput = null;
        this.computeBindGroup = null;
        this.blitBindGroup = null;
    }

    resize() {
        const {width, height} = this.context.canvas;
        if (width === this.width && height === this.height) {
            return;
        }
        this.width = width;
        this.height = height;

        this.computeOutput?.destroy();
        this.computeOutput = this.device.createTexture({
            label: "compute output",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        const computeOutputView = this.computeOutput.createView();

        this.computeBindGroup = this.device.createBindGroup({
            label: "compute bindGroup",
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [{binding: 0, resource: computeOutputView}],
        });

        this.blitBindGroup = this.device.createBindGroup({
            label: "blit bindGroup",
            layout: this.blitPipeline.getBindGroupLayout(0),
            entries: [
                {binding: 0, resource: computeOutputView},
                {binding: 1, resource: this.linearSampler},
            ],
        });
    }

    render() {
        const encoder = this.device.createCommandEncoder({
            label: "frame encoder",
        });

        const computePass = encoder.beginComputePass({label: "compute pass"});
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
        computePass.end();

        const renderPass = encoder.beginRenderPass({
            label: "blit pass",
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: {r: 0, g: 0, b: 0, a: 1},
                loadOp: "clear",
                storeOp: "store",
            }],
        });
        renderPass.setPipeline(this.blitPipeline);
        renderPass.setBindGroup(0, this.blitBindGroup);
        renderPass.draw(SCREE_VERTEX_COUNT);
        renderPass.end();

        this.device.queue.submit([encoder.finish()]);
    }

    uploadGsData(data, name) {
        const buffer = this.device.createBuffer({
            label: `${name} gs buffer`,
            size: data.packed.byteLength,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
        });
        new Float32Array(buffer.getMappedRange()).set(data.packed.byteLength);
        buffer.unmap();

        this.gsBuffers.set(name, {buffer, count: data.count});
    }
}