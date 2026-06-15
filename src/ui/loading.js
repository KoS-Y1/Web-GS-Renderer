const ID = "loading";

function element() {
    return document.getElementById(ID);
}

export function showLoading(message = "Loading…") {
    const el = element();
    if (!el) {
        return;
    }
    el.classList.remove("hidden", "error");
    el.querySelector(".loading-text").textContent = message;
}

export function hideLoading() {
    element()?.classList.add("hidden");
}

export function setLoadingError(message) {
    const el = element();
    if (!el) {
        return;
    }
    el.classList.remove("hidden");
    el.classList.add("error");
    el.querySelector(".loading-text").textContent = message;
}
