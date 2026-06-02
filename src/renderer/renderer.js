import computeWGSL from "../shaders/compute.wgsl?raw"
import blitWGSL from "../shaders/blit.wgsl?raw"

const WIDTH = 512;
const HEIGHT = 512;

export class Renderer {
    constructor(device, context, format) {
        this.device = device;
        this.context = context;
        this.format = format;

        const computeShaderModule = device.createShaderModule({
            label: "compute shader",
            code: computeWGSL,
        });
        this.computePipeline = device.createComputePipeline({
            label: "compute pipeline",
            layout: "auto",
            compute: {
                module: computeShaderModule,
                entryPoint: "computeMain",
            },
        });

        this.computeOutput = device.createTexture({
            label: "compute output",
            size: [WIDTH, HEIGHT],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        const computeOutputView = this.computeOutput.createView();

        this.computeBindGroup = device.createBindGroup({
            label: "compute bindGroup",
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [{binding: 0, resource: computeOutputView}],
        });

        const blitShaderModule = device.createShaderModule({
            label: "blit shader",
            code: blitWGSL,
        });
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
            primitive: {topology: "triangle-list"},
        });

        const sampler = device.createSampler({
            label: "blit sampler",
            magFilter: "linear",
            minFilter: "linear",
        });
        this.blitBindGroup = device.createBindGroup({
            label: "blit bindGroup",
            layout: this.blitPipeline.getBindGroupLayout(0),
            entries: [
                {binding: 0, resource: computeOutputView},
                {binding: 1, resource: sampler},
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
        computePass.dispatchWorkgroups(Math.ceil(WIDTH / 8), Math.ceil(HEIGHT / 8));
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
        renderPass.draw(3);
        renderPass.end();

        this.device.queue.submit([encoder.finish()]);
    }
}
