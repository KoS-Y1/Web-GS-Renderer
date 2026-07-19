import {requestDevice} from "./gpu/device.js";
import {configureCanvas, resizeCanvas} from "./gpu/context.js";
import {Renderer} from "./renderer/renderer.js";
// import {loadProjectPly} from "./loader/loader.js";
import {loadUserPly} from "./loader/loader.js";
import {UI} from "./ui/ui.js";
import {showLoading, hideLoading, setLoadingError} from "./ui/loading.js";

const GPU_CANVAS = "gpu-canvas";

// const MODELS = [
//     {name: "strawberry", url: "../assets/strawberry.ply"},
//     {name: "castle", url: "../assets/castle.ply"},
// ];

async function main() {
    const device = await requestDevice();

    const canvas = document.getElementById(GPU_CANVAS);
    const {context, format} = configureCanvas(device, canvas);

    // const loadAllPly = Promise.all(MODELS.map((model) => loadProjectPly(model.url)));

    const ui = new UI();
    const renderer = new Renderer(device, context, format, (data) => ui.updateDebug(data));

    // const loaded = await loadAllPly;
    // MODELS.forEach((model, i) => renderer.uploadGsData(loaded[i], model.name));

    // renderer.setGs(MODELS[0].name);

    // const loadedNames = new Set(MODELS.map((model) => model.name));
    const loadedNames = new Set();
    const uniqueName = (base) => {
        let name = base;
        for (let n = 2; loadedNames.has(name); ++n) {
            name = `${base} (${n})`;
        }
        loadedNames.add(name);
        return name;
    };

    ui.setModels(
        // MODELS.map((model, i) => ({name: model.name, count: loaded[i].count})),
        // MODELS[0].name,
        [],
        "",
        {
            onSelect: (name) => renderer.setGs(name),
            onImport: async (file) => {
                showLoading(`Loading ${file.name}…`);
                // Yield a frame so the overlay paints before the synchronous parse blocks the thread.
                await new Promise((resolve) => requestAnimationFrame(resolve));
                try {
                    const data = await loadUserPly(file);
                    const name = uniqueName(file.name.replace(/\.ply$/i, ""));
                    renderer.uploadGsData(data, name);
                    renderer.setGs(name);
                    ui.addModel(name, data.count);
                } catch (e) {
                    console.error(`Failed to import ${file.name}:`, e);
                    if (loadedNames.size === 0) {
                        ui.showDropzone();
                    }
                } finally {
                    hideLoading();
                }
            },
        },
    );

    resizeCanvas(device, canvas, () => {
        renderer.execute()
    });

    let lastFrameTime = performance.now();
    let dtSmooth = 0;
    function frame() {
        const now = performance.now();
        const dt = now - lastFrameTime;
        lastFrameTime = now;
        if (dt > 0) {
            dtSmooth = dtSmooth === 0 ? dt : dtSmooth * 0.9 + dt * 0.1;
            ui.updateFps(1000 / dtSmooth);
        }
        renderer.execute();
        requestAnimationFrame(frame);
    }

    hideLoading();
    ui.showDropzone();
    requestAnimationFrame(frame);
}

try {
    await main();
} catch (e) {
    console.error(`Fatal error in main: ${e}`);
    setLoadingError("Failed to load. See console for details.");
}