export function configureCanvas(device, canvas) {
    const context = canvas.getContext("webgpu");
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({device: device, format: format, alphaMode: "premultiplied"});
    console.log(`presentation format: ${format}`);

    return {context, format};
}

export function resizeCanvas(device, canvas, onResize) {
    const observer = new ResizeObserver(entries => {
        entries.forEach((entry) => {
            const width = entry.devicePixelContentBoxSize?.[0].inlineSize ?? entry.contentBoxSize[0].inlineSize * devicePixelRatio;
            const height = entry.devicePixelContentBoxSize?.[0].blockSize ?? entry.contentBoxSize[0].blockSize * devicePixelRatio;
            const maxSize = device.limits.maxTextureDimension2D;
            const canvas = entry.target;
            canvas.width = Math.max(1, Math.min(Math.floor(width), maxSize));
            canvas.height = Math.max(1, Math.min(Math.floor(height), maxSize));

            onResize?.();
        });
    })
    observer.observe(canvas);
}