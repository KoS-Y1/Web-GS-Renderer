import {requestDevice} from "./gpu/device.js";
import {configureCanvas} from "./gpu/context.js";
import {Renderer} from "./renderer/renderer.js";

async function main() {
    const device = await requestDevice();

    const canvas = document.getElementById("gpu-canvas");
    const {context, format} = configureCanvas(device, canvas);

    const renderer = new Renderer(device, context, format);
    renderer.render();
}

main();