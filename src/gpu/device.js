import {fail} from "../utils/utils.js";

export async function requestDevice() {
    if (!navigator.gpu) {
        fail("navigator.gpu is not defined - WebGPU is not available in this browser");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        fail("requestAdapter returned null - this renderer can't run on this system");
    }

    const device = await adapter.requestDevice({
        requiredLimits: {
            maxBufferSize: adapter.limits.maxBufferSize,
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        }
    });
    if (!device) {
        fail("Unable to get a device for an unknown readson ");
    }

    device.lost.then((reason) => {
        fail(`Device lost ("${reason.reason}"):\n${reason.message}`);
    });
    device.addEventListener("uncapturederror", (ev) => {
        fail(`Uncaptured error:\n${ev.error.message}`);
    });

    return device;
}

export function createShaderModule(device, label, code) {
    const module = device.createShaderModule({label, code});

    module.getCompilationInfo().then((info) => {
        for (const m of info.messages) {
            const where = `${label}:${m.lineNum}:${m.linePos}`;
            const text = `[${m.type} ${where} - ${m.message}]`;

            if (m.type == "error") {
                console.error(text);
            } else if (m.type == "warning") {
                console.warn(text);
            } else {
                console.info(text);
            }
        }
    })

    return module;
}