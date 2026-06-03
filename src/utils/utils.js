// Fail by showing a console error, and a dialog box if possible
// Based on https://github.com/webgpu/webgpu-samples/blob/904fe349a66cae556eaff2bebf4ff00e59d184b5/sample/util.ts
export const fail = (() => {
    function createErrorOutput() {
        if (typeof document === "undefined") {
            return {
                show(msg) {
                    console.error(msg);
                },
            };
        }

        const dialogBox = document.createElement("dialog");
        dialogBox.close();
        document.body.append(dialogBox);

        const dialogText = document.createElement("pre");
        dialogText.style.whiteSpace = "pre-wrap";
        dialogBox.append(dialogText);

        const closeButton = document.createElement("button");
        closeButton.textContent = "OK";
        closeButton.onclick = () => dialogBox.close();
        dialogBox.append(closeButton);

        return {
            show(msg) {
                dialogText.textContent = msg;
                if (!dialogBox.open) {
                    dialogBox.showModal();
                }
                console.error(msg);
            },
        };
    }

    let output;

    return (message) => {
        if (!output) {
            output = createErrorOutput();
        }

        output.show(message);
        throw new Error(message);
    };
})();
