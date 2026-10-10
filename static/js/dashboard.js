async function updateViolation(id) {
    const plateInput = document.querySelector(`.plate-input[data-id="${id}"]`);
    const emailInput = document.querySelector(`.email-input[data-id="${id}"]`);
    const response = await fetch(`/update-violation/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            plate_number: plateInput.value.trim(),
            email: emailInput.value.trim(),
        }),
    });
    const data = await response.json();
    window.alert(data.message || data.error || "Unknown response");
    if (response.ok) {
        await loadViolations();
    }
}

async function rerunOcr(id) {
    const response = await fetch(`/rerun-ocr/${id}`, { method: "POST" });
    const data = await response.json();
    window.alert(data.message || data.error || "Unknown response");
    if (response.ok) {
        await loadViolations();
    }
}

async function sendViolationEmail(id) {
    const emailInput = document.querySelector(`.email-input[data-id="${id}"]`);
    const email = emailInput?.value?.trim() || window.prompt("Enter recipient email:");
    if (!email) return;

    const response = await fetch("/send-email", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ violation_id: id, email }),
    });
    const data = await response.json();
    window.alert(data.message || data.error || "Unknown response");
}

async function deleteViolation(id) {
    const confirmed = window.confirm(`Delete violation #${id}?`);
    if (!confirmed) return;

    const response = await fetch(`/delete-violation/${id}`, { method: "DELETE" });
    const data = await response.json();
    if (!response.ok) {
        window.alert(data.error || "Delete failed");
        return;
    }
    await loadViolations();
}

async function loadViolations() {
    const refreshButton = document.getElementById("refresh-violations");
    refreshButton?.classList.add("loading");
    const response = await fetch("/get-violations");
    const data = await response.json();
    const rows = data.violations || [];
    const tbody = document.querySelector("#violations-table tbody");
    tbody.innerHTML = "";

    let readable = 0;
    let emailCount = 0;

    rows.forEach((row) => {
        if (
            row.plate_number &&
            row.plate_number !== "NOTFOUND" &&
            row.plate_number !== "UNREADABLE" &&
            row.plate_number !== "NUMBER PLATE NOT FOUND"
        ) {
            readable += 1;
        }
        if (row.email) {
            emailCount += 1;
        }

        const imageCell = row.image_exists
            ? `<img src="/${row.image_path}" alt="violation" />`
            : `<span>Image missing</span>`;

        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${row.id}</td>
            <td>
                <input class="plate-input" data-id="${row.id}" value="${row.plate_number || ""}" />
                <div><small>Extracted Plate</small></div>
            </td>
            <td>${imageCell}</td>
            <td>
                <input class="email-input" data-id="${row.id}" value="${row.email || ""}" placeholder="Enter email" />
                <div><small>Editable Email</small></div>
            </td>
            <td>${row.timestamp}</td>
            <td class="action-cell">
                <button class="btn save-row" data-id="${row.id}">Save</button>
                <button class="btn secondary rerun-ocr" data-id="${row.id}">Re-run OCR</button>
                <button class="btn secondary send-mail" data-id="${row.id}">Send Email</button>
                <button class="btn danger delete-row" data-id="${row.id}">Delete</button>
            </td>
        `;
        tbody.appendChild(tr);
    });

    document.querySelectorAll(".save-row").forEach((button) => {
        button.addEventListener("click", () => updateViolation(button.dataset.id));
    });
    document.querySelectorAll(".rerun-ocr").forEach((button) => {
        button.addEventListener("click", () => rerunOcr(button.dataset.id));
    });
    document.querySelectorAll(".send-mail").forEach((button) => {
        button.addEventListener("click", () => sendViolationEmail(button.dataset.id));
    });
    document.querySelectorAll(".delete-row").forEach((button) => {
        button.addEventListener("click", () => deleteViolation(button.dataset.id));
    });

    document.getElementById("metric-total").textContent = String(rows.length);
    document.getElementById("metric-readable").textContent = String(readable);
    document.getElementById("metric-email").textContent = String(emailCount);
    refreshButton?.classList.remove("loading");
}

document.getElementById("refresh-violations")?.addEventListener("click", loadViolations);
loadViolations().catch(console.error);
