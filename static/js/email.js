document.getElementById("email-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.target;
    const payload = {
        violation_id: form.violation_id.value ? Number(form.violation_id.value) : null,
        plate_number: form.plate_number.value,
        image_path: form.image_path.value,
        timestamp: form.timestamp.value,
        email: form.email.value,
    };
    const result = document.getElementById("email-result");
    result.textContent = "Sending...";
    result.classList.add("loading");

    try {
        const response = await fetch("/send-email", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        result.textContent = JSON.stringify(data, null, 2);
    } catch (error) {
        result.textContent = `Error: ${error.message}`;
    } finally {
        result.classList.remove("loading");
    }
});
