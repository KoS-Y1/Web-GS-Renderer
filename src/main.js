import {requestDevice} from "./gpu/device.js";
import {configureCanvas, resizeCanvas} from "./gpu/context.js";
import {Renderer} from "./renderer/renderer.js";
import {loadProjectPly} from "./loader/loader.js";

const GPU_CANVAS = "gpu-canvas";

const STRAWBERRY_PLY = "../assets/strawberry.ply";
const CASTLE_PLY = "../assets/castle.ply"

async function main() {
    const device = await requestDevice();

    const canvas = document.getElementById(GPU_CANVAS);
    const {context, format} = configureCanvas(device, canvas);

    const loadAllPly = Promise.all([
        loadProjectPly(STRAWBERRY_PLY),
        loadProjectPly(CASTLE_PLY),
    ]);

    const renderer = new Renderer(device, context, format);

    const [strawberryData, castleData] = await loadAllPly;
    renderer.uploadGsData(strawberryData, "strawberry");
    renderer.uploadGsData(castleData, "castle");

    resizeCanvas(device, canvas, () => {
        renderer.execute()
    });

}

try {
    await main();
} catch (e) {
    console.error(`Fatal error in main: ${e}`);
}