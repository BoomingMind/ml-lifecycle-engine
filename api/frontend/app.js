const fileInput = document.getElementById("fileInput");
const apiKeyInput = document.getElementById("apiKeyInput");
const predictBtn = document.getElementById("predictBtn");
const statusEl = document.getElementById("status");
const previewImg = document.getElementById("previewImg");
const overlay = document.getElementById("overlay");
const jsonOut = document.getElementById("jsonOut");

// Browser localStorage key used to remember the API key between page refreshes.
const API_KEY_STORAGE = "defectguard_api_key";

function setStatus(text, kind = "info") {
  // Update small status text in the UI so users know what is happening.
  statusEl.textContent = text;
  statusEl.style.color = kind === "error" ? "var(--danger)" : "var(--muted)";
}

function pretty(obj) {
  // Convert a JS object into indented JSON text for the response panel.
  return JSON.stringify(obj, null, 2);
}

function clearOverlay() {
  // The canvas sits above the image; clearing it removes any old boxes.
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function resizeCanvasToImage() {
  // Keep the drawing canvas the same visible size as the image element.
  // This is important so detection boxes line up with the photo.
  const rect = previewImg.getBoundingClientRect();
  overlay.width = Math.max(1, Math.floor(rect.width));
  overlay.height = Math.max(1, Math.floor(rect.height));
}

function drawBoxes(payload) {
  // Start from a clean canvas each time new predictions arrive.
  clearOverlay();
  if (!payload || !Array.isArray(payload.boxes)) return;

  const ctx = overlay.getContext("2d");
  const rect = previewImg.getBoundingClientRect();
  // YOLO gives coordinates relative to the image's real pixel size.
  // The browser may show the image scaled, so we compute scale factors.
  const sx = rect.width / previewImg.naturalWidth;
  const sy = rect.height / previewImg.naturalHeight;

  // Set drawing style once before looping over all boxes.
  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(94, 234, 212, 0.95)";
  ctx.fillStyle = "rgba(94, 234, 212, 0.16)";
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace";

  const scores = Array.isArray(payload.scores) ? payload.scores : [];
  const classNames = Array.isArray(payload.class_names) ? payload.class_names : [];

  payload.boxes.forEach((b, i) => {
    if (!Array.isArray(b) || b.length !== 4) return;
    const [x1, y1, x2, y2] = b;

    // Convert model coordinates into on-screen coordinates.
    const rx = x1 * sx;
    const ry = y1 * sy;
    const rw = (x2 - x1) * sx;
    const rh = (y2 - y1) * sy;

    // Draw the box fill + border.
    ctx.fillRect(rx, ry, rw, rh);
    ctx.strokeRect(rx, ry, rw, rh);

    // Read matching score/class name for the current detection if available.
    const score = typeof scores[i] === "number" ? scores[i] : null;
    const label = typeof classNames[i] === "string" ? classNames[i] : "defect";
    const tag = score === null ? label : `${label} ${(score * 100).toFixed(1)}%`;

    // Measure the text so we can draw a background label behind it.
    const pad = 4;
    const tw = ctx.measureText(tag).width;
    const th = 16;
    // Clamp the label so it stays inside the visible canvas.
    const tx = Math.max(0, Math.min(overlay.width - tw - pad * 2, rx));
    const ty = Math.max(th, ry);

    // Draw label background and text.
    ctx.fillStyle = "rgba(10, 17, 31, 0.85)";
    ctx.fillRect(tx, ty - th, tw + pad * 2, th);
    ctx.fillStyle = "rgba(94, 234, 212, 0.95)";
    ctx.fillText(tag, tx + pad, ty - 4);

    // Reset fill style so the next box fill uses the translucent color again.
    ctx.fillStyle = "rgba(94, 234, 212, 0.16)";
  });
}

function loadApiKey() {
  // Restore saved API key so users do not have to paste it every refresh.
  const v = localStorage.getItem(API_KEY_STORAGE);
  apiKeyInput.value = v ?? "";
}

function saveApiKey() {
  // Save latest input value into browser local storage.
  localStorage.setItem(API_KEY_STORAGE, apiKeyInput.value ?? "");
}

fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  if (!f) return;
  // Reset previous output when a new file is selected.
  setStatus("Image selected");
  jsonOut.textContent = "";
  clearOverlay();
  // Create a temporary browser URL so the image can be previewed instantly.
  previewImg.src = URL.createObjectURL(f);
});

// Save API key while the user types or when they leave the field.
apiKeyInput.addEventListener("change", saveApiKey);
apiKeyInput.addEventListener("keyup", saveApiKey);

previewImg.addEventListener("load", () => {
  // Once the browser knows the final image size, resize overlay canvas to match.
  resizeCanvasToImage();
  clearOverlay();
});

window.addEventListener("resize", () => {
  if (!previewImg.src) return;
  // If window size changes, image layout may change too, so resize overlay again.
  resizeCanvasToImage();
});

predictBtn.addEventListener("click", async () => {
  const f = fileInput.files?.[0];
  if (!f) {
    setStatus("Select an image first", "error");
    return;
  }

  predictBtn.disabled = true;
  setStatus("Predicting...");

  try {
    // FormData lets us send a file upload exactly like a normal browser form.
    const form = new FormData();
    form.append("file", f);

    // Build optional auth headers.
    const headers = {};
    const apiKey = apiKeyInput.value?.trim();
    if (apiKey) headers["X-API-Key"] = apiKey;

    // Call the backend prediction API.
    const resp = await fetch("/predict", { method: "POST", body: form, headers });
    // Try to parse JSON response. If server returns non-JSON, body becomes null.
    const body = await resp.json().catch(() => null);

    if (!resp.ok) {
      // Show error information in the response panel.
      setStatus(`Error ${resp.status}`, "error");
      jsonOut.textContent = pretty(body ?? { error: "Request failed" });
      clearOverlay();
      return;
    }

    // Success: show raw JSON and draw returned boxes on the image.
    jsonOut.textContent = pretty(body);
    setStatus("Done");
    drawBoxes(body);
  } catch (e) {
    // Network errors happen before the server gives a normal response.
    setStatus("Network error", "error");
    jsonOut.textContent = String(e);
    clearOverlay();
  } finally {
    // Re-enable the button no matter what happened.
    predictBtn.disabled = false;
  }
});

// Run once at page load.
loadApiKey();
