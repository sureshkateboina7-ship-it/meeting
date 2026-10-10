document.getElementById("image-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    const output = document.getElementById("image-output");
    const result = document.getElementById("image-result");
    result.textContent = "Processing...";
    result.classList.add("loading");
    output.classList.add("hidden");

    try {
        const response = await fetch("/upload-image", { method: "POST", body: formData });
        const data = await response.json();
        result.textContent = JSON.stringify(data, null, 2);
        if (data.processed_image) {
            output.src = `/${data.processed_image}`;
            output.classList.remove("hidden");
        }
    } catch (error) {
        result.textContent = `Error: ${error.message}`;
    } finally {
        result.classList.remove("loading");
    }
});
