import {requestDevice} from "./gpu/device.js";
import {configureCanvas, resizeCanvas} from "./gpu/context.js";
import {Renderer} from "./renderer/renderer.js";

const GPU_CANVAS = "gpu-canvas";

async function main() {
    const device = await requestDevice();

    const canvas = document.getElementById(GPU_CANVAS);
    const {context, format} = configureCanvas(device, canvas);

    const renderer = new Renderer(device, context, format);
    resizeCanvas(device, canvas, () => {
        renderer.resize();
        renderer.render();
    });
}

try {
    await main();
} catch (e) {
    console.error(`Fatal error in main: ${e}`);
}