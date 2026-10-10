document.getElementById("video-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    const video = document.getElementById("video-output");
    const result = document.getElementById("video-result");
    result.textContent = "Processing video. This may take some time...";
    result.classList.add("loading");
    video.classList.add("hidden");

    try {
        const response = await fetch("/upload-video", { method: "POST", body: formData });
        const data = await response.json();
        result.textContent = JSON.stringify(data, null, 2);
        const outputPath = data?.stats?.output_path;
        if (outputPath) {
            video.src = `/${outputPath}`;
            video.classList.remove("hidden");
        }
    } catch (error) {
        result.textContent = `Error: ${error.message}`;
    } finally {
        result.classList.remove("loading");
    }
});
