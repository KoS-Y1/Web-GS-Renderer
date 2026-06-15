import {vec3, mat4, utils} from "wgpu-matrix";

const {degToRad} = utils;

const DEFAULT_YAW = -90.0;
const DEFUALT_PITCH = 0.0;
const DEFAULT_FOV = 60.0;

const DEFAULT_SPEED = 1.0;
const DEFAULT_SENSITIVITY = 1.0;

const WORLD_UP = vec3.create(0.0, 1.0, 0.0);
const Z_NEAR = 0.2;
const Z_FAR = 1000.0;

export class Camera {
    #view;
    #projection;

    #eye;

    #yaw;
    #pitch;

    #fov;
    #ratio;

    #width;
    #height;

    #keys = new Set();

    #dirty = true;

    constructor(width, height) {
        this.reset(width, height);

        window.addEventListener("keydown", (e) => this.#keys.add(e.code));
        window.addEventListener("keyup", (e) => this.#keys.delete(e.code));

        window.addEventListener("mousemove", (e) => {
            if (!this.#keys.has("ShiftLeft")) {
                return;
            }
            this.#yaw += e.movementX * DEFAULT_SENSITIVITY;
            this.#pitch -= e.movementY * DEFAULT_SENSITIVITY;
            this.#pitch = Math.max(-89.0, Math.min(89.0, this.#pitch));
            this.#updateViewMatrix();
            this.#dirty = true;
        });
    }

    reset(width, height) {
        this.#width = width;
        this.#height = height;

        this.#eye = vec3.create();

        this.#yaw = DEFAULT_YAW;
        this.#pitch = DEFUALT_PITCH;

        this.#fov = DEFAULT_FOV;
        this.#ratio = width / height;

        this.#updateViewMatrix();
        this.#updateProjectionMatrix();

        this.#dirty = true;
    }

    resize(width, height) {
        this.#width = width;
        this.#height = height;
        this.#ratio = width / height;

        this.#updateProjectionMatrix();

        this.#dirty = true;
    }

    pollDirty() {
        const dirty = this.#dirty;
        this.#dirty = false;
        return dirty;
    }

    getUniform() {
        const tanFovY = Math.tan(degToRad(this.#fov) / 2);
        const tanFovX = tanFovY * this.#ratio;

        return {
            view: this.#view,
            projView: mat4.multiply(this.#projection, this.#view),
            eye: this.#eye,
            focal: [this.#width / (2 * tanFovX), this.#height / (2 * tanFovY)],
            tanFov: [tanFovX, tanFovY]
        };
    }

    update(deltaTime) {
        const {forward, right} = this.#calculateCameraFrame();
        const moveForward = vec3.negate(forward);
        const moveRight = vec3.negate(right);
        const dist = DEFAULT_SPEED * deltaTime;

        let moved = false;
        const step = (dir, sign) => {
            this.#eye = vec3.addScaled(this.#eye, dir, sign * dist);
            moved = true;
        };

        if (this.#keys.has("KeyW")) {
            step(moveForward, 1);
        }
        if (this.#keys.has("KeyS")) {
            step(moveForward, -1);
        }
        if (this.#keys.has("KeyD")) {
            step(moveRight, 1);
        }
        if (this.#keys.has("KeyA")) {
            step(moveRight, -1);
        }
        if (this.#keys.has("KeyQ")) {
            step(WORLD_UP, 1);
        }
        if (this.#keys.has("KeyE")) {
            step(WORLD_UP, -1);
        }

        if (moved) {
            this.#updateViewMatrix();
            this.#dirty = true;
        }
    }

    #updateViewMatrix() {
        const {forward, right, up} = this.#calculateCameraFrame();
        this.#view = mat4.lookAt(this.#eye, vec3.add(this.#eye, forward), up);
    }

    #updateProjectionMatrix() {
        this.#projection = mat4.perspective(degToRad(this.#fov), this.#ratio, Z_NEAR, Z_FAR);
    }

    #calculateCameraFrame() {
        const yawRadian = degToRad(this.#yaw);
        const pitchRadian = degToRad(this.#pitch);

        const forward = vec3.normalize(
            vec3.create(
                Math.cos(yawRadian) * Math.cos(pitchRadian),
                Math.sin(pitchRadian),
                Math.sin(yawRadian) * Math.cos(pitchRadian)
            ));
        const right = vec3.normalize(vec3.cross(forward, WORLD_UP));
        const up = vec3.normalize(vec3.cross(right, forward));

        return {forward, right, up};
    }
}
