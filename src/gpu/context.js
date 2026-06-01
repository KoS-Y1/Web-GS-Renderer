export function configureCanvas(device, canvas) {
    const context = canvas.getContext("webgpu");
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({device: device, format: format, alphaMode: "premultiplied"});
    console.log(`presentation format: ${format}`);

    return {context, format};
}
