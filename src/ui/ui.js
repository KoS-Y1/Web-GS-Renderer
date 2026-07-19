import {Pane} from "tweakpane";

const DROPZONE_ID = "dropzone";
const DROPZONE_BROWSE_ID = "dropzone-browse";

export class UI {
    #pane;

    #dropzone;

    #sceneFolder;
    #modelState = {model: "", splats: 0};
    #modelCounts = new Map();
    #modelSelector;
    #splatBinding;
    #onSelect;

    #passFolder;
    #metrics = {};
    #performance = {fps: 0};
    #passes = new Map();
    #fpsBindings = [];

    constructor() {
        this.#pane = new Pane({title: "Debug"});

        this.#sceneFolder = this.#pane.addFolder({title: "Scene", expanded: true});

        const performanceFolder = this.#pane.addFolder({title: "Performance", expanded: true});
        this.#fpsBindings.push(
            performanceFolder.addBinding(this.#performance, "fps", {
                readonly: true,
                format: (v) => v.toFixed(1),
            }),
            performanceFolder.addBinding(this.#performance, "fps", {
                readonly: true,
                view: "graph",
                min: 0,
                max: 144,
                label: "fps graph",
            }),
        );

        this.#passFolder = this.#pane.addFolder({title: "GPU pass (ms)", expanded: true});
    }

    setModels(models, current, {onSelect, onImport}) {
        this.#onSelect = onSelect;
        models.forEach((model) => this.#modelCounts.set(model.name, model.count));
        this.#modelState.model = current;
        this.#modelState.splats = this.#modelCounts.get(current) ?? 0;

        this.#sceneFolder.addButton({title: "Import PLY…"}).on("click", () => this.#openFilePicker(onImport));
        this.#splatBinding = this.#sceneFolder.addBinding(this.#modelState, "splats", {
            readonly: true,
            format: (v) => Math.round(v).toLocaleString(),
            label: "splats",
        });
        this.#rebuildSelector();
        this.#enableDrop(onImport);
        this.#enableDropzone(onImport);
    }

    addModel(name, count) {
        this.#modelCounts.set(name, count);
        this.#modelState.model = name;
        this.#modelState.splats = count;
        this.#rebuildSelector();
        this.#splatBinding.refresh();
        this.hideDropzone();
    }

    showDropzone() {
        this.#dropzone?.classList.remove("hidden");
    }

    hideDropzone() {
        this.#dropzone?.classList.remove("dragging");
        this.#dropzone?.classList.add("hidden");
    }

    #enableDropzone(onImport) {
        this.#dropzone = document.getElementById(DROPZONE_ID);
        document
            .getElementById(DROPZONE_BROWSE_ID)
            ?.addEventListener("click", () => this.#openFilePicker(onImport));

        // Nested elements fire dragenter/dragleave too, so track depth instead of toggling per event.
        let depth = 0;
        window.addEventListener("dragenter", () => {
            if (++depth === 1) {
                this.#dropzone?.classList.add("dragging");
            }
        });
        window.addEventListener("dragleave", () => {
            if (--depth <= 0) {
                depth = 0;
                this.#dropzone?.classList.remove("dragging");
            }
        });
        window.addEventListener("drop", () => {
            depth = 0;
            this.#dropzone?.classList.remove("dragging");
        });
    }

    #rebuildSelector() {
        // Nothing to pick from until the user imports a file.
        if (this.#modelCounts.size === 0) {
            return;
        }

        this.#modelSelector?.dispose();

        const options = {};
        for (const name of this.#modelCounts.keys()) {
            options[name] = name;
        }

        this.#modelSelector = this.#sceneFolder.addBinding(this.#modelState, "model", {options, label: "model"});
        this.#modelSelector.on("change", (ev) => {
            this.#modelState.splats = this.#modelCounts.get(ev.value) ?? 0;
            this.#splatBinding.refresh();
            this.#onSelect(ev.value);
        });
    }

    #openFilePicker(onImport) {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".ply";
        input.multiple = true;
        input.addEventListener("change", () => {
            [...input.files].forEach((file) => onImport(file));
        });
        input.click();
    }

    #enableDrop(onImport) {
        window.addEventListener("dragover", (e) => e.preventDefault());
        window.addEventListener("drop", (e) => {
            e.preventDefault();
            [...(e.dataTransfer?.files ?? [])]
                .filter((file) => /\.ply$/i.test(file.name))
                .forEach((file) => onImport(file));
        });
    }

    updateFps(fps) {
        this.#performance.fps = fps;
        this.#fpsBindings.forEach((binding) => binding.refresh());
    }

    updateDebug(profilerData) {
        for (const [label, {ms}] of Object.entries(profilerData)) {
            this.#metrics[label] = ms;
            if (!this.#passes.has(label)) {
                const folder = this.#passFolder.addFolder({title: label, expanded: false});
                const bindings = [
                    folder.addBinding(this.#metrics, label, {
                        readonly: true,
                        format: (v) => v.toFixed(3),
                        label: "ms",
                    }),
                    folder.addBinding(this.#metrics, label, {
                        readonly: true,
                        view: "graph",
                        min: 0,
                        label: "graph",
                    }),
                ];
                this.#passes.set(label, bindings);
            }
            this.#passes.get(label).forEach((binding) => binding.refresh());
        }
    }
}
