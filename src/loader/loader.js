import {fail} from "../utils/utils.js";

// Required attribute for Gaussian splatting file
const REQUIRED_ATTRIBUTES = [
    "x", "y", "z",                                                                          // position
    "scale_0", "scale_1", "scale_2",                                                        // scale
    "rot_0", "rot_1", "rot_2", "rot_3",                                                     // rotation
    "opacity",                                                                              // opacity
    "f_dc_0", "f_dc_1", "f_dc_2",                                                           // direct color component
    ...Array.from({length: 45}, (_, i) => `f_rest_${i}`)      // spherical harmonic coefficients
];

const TYPE_SIZE = {
    char: 1, uchar: 1, int8: 1, uint8: 1,
    short: 2, ushort: 2, int16: 2, uint16: 2,
    int: 4, uint: 4, int32: 4, uint32: 4,
    float: 4, float32: 4,
    double: 8, float64: 8,
};

const READERS = {
    char: (v, o) => v.getInt8(o),
    uchar: (v, o) => v.getUint8(o),
    int8: (v, o) => v.getInt8(o),
    uint8: (v, o) => v.getUint8(o),
    short: (v, o) => v.getInt16(o, true),
    ushort: (v, o) => v.getUint16(o, true),
    int16: (v, o) => v.getInt16(o, true),
    uint16: (v, o) => v.getUint16(o, true),
    int: (v, o) => v.getInt32(o, true),
    uint: (v, o) => v.getUint32(o, true),
    int32: (v, o) => v.getInt32(o, true),
    uint32: (v, o) => v.getUint32(o, true),
    float: (v, o) => v.getFloat32(o, true),
    float32: (v, o) => v.getFloat32(o, true),
    double: (v, o) => v.getFloat64(o, true),
    float64: (v, o) => v.getFloat64(o, true),
}

// Load project provided PLY file
export async function loadProjectPly(url) {
    const response = await fetch(url);
    if (!response.ok) {
        fail(`Failed to load PLY (${url}): ${response.status} ${response.statusText}`);
    }

    const buffer = await response.arrayBuffer();
    return loadPly(url, buffer);
}

export async function loadUserPly(file) {
    const buffer = await file.arrayBuffer();
    return loadPly(file.name, buffer);
}

function loadPly(fileName, buffer) {
    let now = new Date();
    console.log(`[${now.getHours()}:${now.getMinutes()}:${now.getSeconds()}:${now.getMilliseconds()}]Started loading ${fileName}`);

    const bytes = new Uint8Array(buffer);
    const bodyStart = findHeaderEnd();
    const headerText = new TextDecoder("ascii").decode(bytes.subarray(0, bodyStart));
    const header = parseHeader(fileName, headerText);

    const attributes = new Set(header.properties.map((p) => p.name));
    for (const attribute of REQUIRED_ATTRIBUTES) {
        if (!attributes.has(attribute)) {
            fail(`Invalid PLY (${fileName}): missing required property "${attribute}"`);
        }
    }

    const {count, properties, stride} = header;
    const view = new DataView(buffer);
    const propertiesByName = new Map(properties.map((p) => [p.name, p]));
    const out = {
        count: count,
        packed: new Float32Array(count * (3 /*position*/ + 3 /*scale*/ + 4 /*rotation*/ + 3 /*color*/ + 1 /*opacity*/ + 45 /*sh*/)),
    };

    const positionOffst = 0;
    const scaleOffset = positionOffst + count * 3;
    const rotationOffset = scaleOffset + count * 3;
    const colorOffset = rotationOffset + count * 4;
    const opacityOffset = colorOffset + count * 3;
    const shsOffset = opacityOffset + count;

    let base = 0;
    for (let i = 0; i < count; ++i) {
        base = bodyStart + i * stride

        out.packed[positionOffst + i * 3] = get("x");
        out.packed[positionOffst + i * 3 + 1] = get("y");
        out.packed[positionOffst + i * 3 + 2] = get("z");

        out.packed[scaleOffset + i * 3] = get("scale_0");
        out.packed[scaleOffset + i * 3 + 1] = get("scale_1");
        out.packed[scaleOffset + i * 3 + 2] = get("scale_2");

        out.packed[rotationOffset + i * 4] = get("rot_0");
        out.packed[rotationOffset + i * 4 + 1] = get("rot_1");
        out.packed[rotationOffset + i * 4 + 2] = get("rot_2");
        out.packed[rotationOffset + i * 4 + 3] = get("rot_3");

        out.packed[colorOffset + i * 3] = get("f_dc_0");
        out.packed[colorOffset + i * 3 + 1] = get("f_dc_1");
        out.packed[colorOffset + i * 3 + 2] = get("f_dc_2");

        out.packed[opacityOffset + i] = get("opacity");

        getSphericalHarmonic(i);
    }

    now = new Date();
    console.log(`[${now.getHours()}:${now.getMinutes()}:${now.getSeconds()}:${now.getMilliseconds()}]Finish loading ${fileName}`);

    return out;

    function findHeaderEnd() {
        const text = new TextDecoder("ascii").decode(bytes);
        const idx = text.indexOf("end_header");

        if (idx === -1) {
            fail(`Invalid PLY (${fileName}): "${"end_header"}" not found`);
        }

        let end = idx + "end_header".length;

        if (bytes[end] === "\r".charCodeAt(0)) {
            ++end;
        }
        if (bytes[end] === "\n".charCodeAt(0)) {
            ++end;
        }

        return end;
    }

    function get(name) {
        const p = propertiesByName.get(name);
        return READERS[p.type](view, base + p.offset);
    }

    function getSphericalHarmonic(i) {
        for (let j = 0; j < 45; ++j) {
            out.packed[shsOffset + i * 45 + j] = get(`f_rest_${j}`);
        }
    }

}

function parseHeader(fileName, headerText) {
    let count = 0;
    let littleEndian = false;
    let currentElement = null;
    let hasFormat = false;

    const properties = [];
    let stride = 0;

    for (const line of headerText.split("\n")) {
        const words = line.trim().split(/\s+/);
        if (words[0] === "format") {
            parseFormat(words);
        }
        if (words[0] === "element") {
            parseElement(words);
        }
        // Only need vertex property
        if (words[0] === "property" && currentElement === "vertex") {
            parseProperty(words);
        }
    }

    if (!hasFormat) {
        fail(`Invalid PLY (${fileName}): missing "format" line`);
    }
    if (!littleEndian) {
        fail(`Invalid PLY (${fileName}): unsupported format`);
    }
    if (count <= 0) {
        fail(`Invalid PLY (${fileName}): vertex count missing or zero`);
    }

    return {count, properties, stride};

    function parseFormat(words) {
        hasFormat = true;
        littleEndian = words[1] === "binary_little_endian";
    }

    function parseElement(words) {
        currentElement = words[1];

        if (currentElement === "vertex") {
            count = parseInt(words[2], 10);
        }
    }

    function parseProperty(words) {
        if (words[1] === "list") {
            fail(`Invalid PLY (${fileName}): list properties are not supported"`);
        }

        const type = words[1];
        const name = words[2]

        if (!(type in TYPE_SIZE)) {
            fail(`Invalid PLY (${fileName}): unknown property type "${type}"`);
        }

        properties.push({name, type, offset: stride});
        stride += TYPE_SIZE[type];
    }
}