export class GpuProfiler {
    #enabled;
    #device;
    #capacity;
    #printEvery;

    #querySet;
    #resolveBuffer;
    #readBuffer;

    #labels = [];
    #index = 0;

    #pending = false;
    #captured = null;

    #sums = new Map();
    #order = [];
    #frames = 0;

    constructor(device, {capacity = 32, printEvery = 8} = {}) {
        this.#enabled = device.features.has("timestamp-query");
        this.#device = device;
        this.#capacity = capacity;
        this.#printEvery = printEvery;

        if (!this.#enabled) {
            console.warn("GpuProfiler disabled: 'timestamp-query' feature unavailable");
            return;
        }
        console.log(`GpuProfiler enabled — averaging every ${printEvery} rendered frames`);

        this.#querySet = device.createQuerySet({type: "timestamp", count: capacity * 2});
        this.#resolveBuffer = device.createBuffer({
            label: "profiler resolve buffer",
            size: capacity * 2 * 8,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        this.#readBuffer = device.createBuffer({
            label: "profiler read buffer",
            size: capacity * 2 * 8,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
    }

    begin() {
        this.#index = 0;
        this.#labels = [];
    }

    write(label) {
        if (!this.#enabled || this.#index >= this.#capacity) {
            return undefined;
        }
        const i = this.#index++;
        this.#labels.push(label);
        return {
            querySet: this.#querySet,
            beginningOfPassWriteIndex: i * 2,
            endOfPassWriteIndex: i * 2 + 1,
        };
    }

    resolve(encoder) {
        if (!this.#enabled || this.#labels.length === 0) {
            return;
        }
        const used = this.#labels.length;
        encoder.resolveQuerySet(this.#querySet, 0, used * 2, this.#resolveBuffer, 0);

        if (!this.#pending) {
            encoder.copyBufferToBuffer(this.#resolveBuffer, 0, this.#readBuffer, 0, used * 2 * 8);
            this.#captured = this.#labels.slice();
        }
    }

    afterSubmit() {
        if (!this.#enabled || this.#pending || !this.#captured) {
            return;
        }
        const labels = this.#captured;
        this.#captured = null;
        this.#pending = true;

        this.#readBuffer.mapAsync(GPUMapMode.READ).then(() => {
            const stamps = new BigInt64Array(this.#readBuffer.getMappedRange().slice(0));
            this.#readBuffer.unmap();
            this.#pending = false;
            accumulate(labels, stamps);
        }).catch((e) => {
            this.#pending = false;
            console.error("GpuProfiler readback failed:", e);
        });

        const accumulate = (labels, stamps) => {
            for (let i = 0; i < labels.length; ++i) {
                const ns = Number(stamps[i * 2 + 1] - stamps[i * 2]);
                const label = labels[i];
                if (!this.#sums.has(label)) {
                    this.#order.push(label);
                }
                // Same label used by multiple passes (e.g. radix) sums per frame.
                this.#sums.set(label, (this.#sums.get(label) ?? 0) + ns / 1e6);
            }
            ++this.#frames;

            const rows = {};
            let total = 0;
            for (const label of this.#order) {
                const avg = this.#sums.get(label) / this.#frames;
                total += avg;
                rows[label] = {ms: +avg.toFixed(3)};
            }
            rows["TOTAL"] = {ms: +total.toFixed(3)};
            console.clear();
            console.log(`GPU pass ms (avg of ${this.#frames} frames):`);
            console.table(rows);

            if (this.#frames >= this.#printEvery) {
                this.#sums.clear();
                this.#order = [];
                this.#frames = 0;
            }
        }
    }


}
