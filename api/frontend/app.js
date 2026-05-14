const fileInput = document.getElementById("fileInput");
const predictBtn = document.getElementById("predictBtn");
const statusEl = document.getElementById("status");
const previewImg = document.getElementById("previewImg");
const overlay = document.getElementById("overlay");
const jsonOut = document.getElementById("jsonOut");

function setStatus(text, kind = "info") {
  statusEl.textContent = text;
  statusEl.style.color = kind === "error" ? "var(--danger)" : "var(--muted)";
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

function clearOverlay() {
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function resizeCanvasToImage() {
  const rect = previewImg.getBoundingClientRect();
  overlay.width = Math.max(1, Math.floor(rect.width));
  overlay.height = Math.max(1, Math.floor(rect.height));
}

function drawBoxes(payload) {
  clearOverlay();
  if (!payload || !Array.isArray(payload.boxes)) return;

  const ctx = overlay.getContext("2d");
  const rect = previewImg.getBoundingClientRect();
  const sx = rect.width / previewImg.naturalWidth;
  const sy = rect.height / previewImg.naturalHeight;

  ctx.lineWidth = 2;
  ctx.strokeStyle = "rgba(94, 234, 212, 0.95)";
  ctx.fillStyle = "rgba(94, 234, 212, 0.16)";
  ctx.font = "12px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, Courier New, monospace";

  const scores = Array.isArray(payload.scores) ? payload.scores : [];
  const classNames = Array.isArray(payload.class_names) ? payload.class_names : [];

  payload.boxes.forEach((b, i) => {
    if (!Array.isArray(b) || b.length !== 4) return;
    const [x1, y1, x2, y2] = b;

    const rx = x1 * sx;
    const ry = y1 * sy;
    const rw = (x2 - x1) * sx;
    const rh = (y2 - y1) * sy;

    ctx.fillRect(rx, ry, rw, rh);
    ctx.strokeRect(rx, ry, rw, rh);

    const score = typeof scores[i] === "number" ? scores[i] : null;
    const label = typeof classNames[i] === "string" ? classNames[i] : "defect";
    const tag = score === null ? label : `${label} ${(score * 100).toFixed(1)}%`;

    const pad = 4;
    const tw = ctx.measureText(tag).width;
    const th = 16;
    const tx = Math.max(0, Math.min(overlay.width - tw - pad * 2, rx));
    const ty = Math.max(th, ry);

    ctx.fillStyle = "rgba(10, 17, 31, 0.85)";
    ctx.fillRect(tx, ty - th, tw + pad * 2, th);
    ctx.fillStyle = "rgba(94, 234, 212, 0.95)";
    ctx.fillText(tag, tx + pad, ty - 4);

    ctx.fillStyle = "rgba(94, 234, 212, 0.16)";
  });
}

fileInput.addEventListener("change", () => {
  const f = fileInput.files?.[0];
  if (!f) return;
  setStatus("Image selected");
  jsonOut.textContent = "";
  clearOverlay();
  previewImg.src = URL.createObjectURL(f);
});

previewImg.addEventListener("load", () => {
  resizeCanvasToImage();
  clearOverlay();
});

window.addEventListener("resize", () => {
  if (!previewImg.src) return;
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
    const form = new FormData();
    form.append("file", f);

    const resp = await fetch("/predict", { method: "POST", body: form });
    const body = await resp.json().catch(() => null);

    if (!resp.ok) {
      setStatus(`Error ${resp.status}`, "error");
      jsonOut.textContent = pretty(body ?? { error: "Request failed" });
      clearOverlay();
      return;
    }

    jsonOut.textContent = pretty(body);
    setStatus("Done");
    drawBoxes(body);
  } catch (e) {
    setStatus("Network error", "error");
    jsonOut.textContent = String(e);
    clearOverlay();
  } finally {
    predictBtn.disabled = false;
  }
});

