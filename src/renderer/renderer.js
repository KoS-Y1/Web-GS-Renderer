import testWGSL from "../shaders/test.wgsl?raw"

export class Renderer {
    constructor(device, context, format) {
        this.device = device;
        this.context = context;
        this.format = format;

        const testShaderModule = device.createShaderModule({
            label: "test shader",
            code: testWGSL
        });
        this.testPipeline = device.createRenderPipeline({
            label: "test pipeline",
            layout: "auto",
            vertex: {
                module: testShaderModule,
            },
            fragment: {
                module: testShaderModule,
                targets: [{format: format}],
            },
        });
        this.renderPassDesc = {
            label: "test canvas renderPass",
            colorAttachments: [{
                clearValue: [0.3, 0.3, 0.3, 1.0],
                loadOp: "clear",
                storeOp: "store",
            }]
        }
    }

    render() {
        this.renderPassDesc.colorAttachments[0].view = this.context.getCurrentTexture().createView();

        const encoder = this.device.createCommandEncoder({
            label: "test encoder",
        })

        const pass = encoder.beginRenderPass(this.renderPassDesc);
        pass.setPipeline(this.testPipeline);
        pass.draw(3);
        pass.end();

        const commandBuffer = encoder.finish();
        this.device.queue.submit([commandBuffer]);
    }
}
