const startButton = document.getElementById("start-live");
const stopButton = document.getElementById("stop-live");
const streamImage = document.getElementById("live-stream");
const liveState = document.getElementById("live-state");

startButton?.addEventListener("click", () => {
    const source = document.getElementById("live-source")?.value || "0";
    const email = document.getElementById("live-email")?.value || "";
    const query = new URLSearchParams({ source, email });
    streamImage.src = `/live-detection?${query.toString()}`;
    streamImage.classList.remove("hidden");
    liveState.textContent = "Live stream running...";
    liveState.classList.add("loading");
});

stopButton?.addEventListener("click", () => {
    streamImage.src = "";
    streamImage.classList.add("hidden");
    liveState.textContent = "Stream stopped.";
    liveState.classList.remove("loading");
});
