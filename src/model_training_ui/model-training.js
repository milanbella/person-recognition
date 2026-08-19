"use strict";

const app = { token: localStorage.getItem("modelTrainingToken") || "", session: null, frames: [], frame: null, image: null, box: null, dragStart: null };
const el = Object.fromEntries([...document.querySelectorAll("[id]")].map(node => [node.id, node]));

function authHeaders(json = false) {
  const headers = { Authorization: `Bearer ${app.token}` };
  if (json) headers["Content-Type"] = "application/json";
  return headers;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

function status(message, failed = false) {
  el.status.textContent = message;
  el.status.style.color = failed ? "#aa3d24" : "";
}

async function connect() {
  app.token = el.token.value.trim();
  localStorage.setItem("modelTrainingToken", app.token);
  try {
    const [products, cameras] = await Promise.all([
      api("/model-training/api/products"), api("/model-training/api/cameras")
    ]);
    renderOptions(el.product, products.products, item => item.code, item => `${item.name} (${item.code})`);
    renderOptions(el.camera, cameras.cameras, item => item.cameraIndex, item => `Camera ${item.cameraNumber} - ${item.status}`);
    el.camera.dataset.items = JSON.stringify(cameras.cameras);
    updateCamera();
    try { app.session = await api("/model-training/api/sessions/active"); } catch (_) { app.session = null; }
    renderSession();
    await loadQueue();
    status(products.products.length ? "Connected." : "Connected. Refreshing product catalog...");
    if (!products.products.length) {
      const refreshed = await api("/model-training/api/products/refresh", { method: "POST", headers: authHeaders() });
      renderOptions(el.product, refreshed.products, item => item.code, item => `${item.name} (${item.code})`);
      status(`Loaded ${refreshed.count} products.`);
    }
  } catch (error) { status(error.message, true); }
}

function renderOptions(select, items, value, label) {
  select.replaceChildren(...items.map(item => {
    const option = document.createElement("option"); option.value = value(item); option.textContent = label(item); return option;
  }));
}

function updateCamera() {
  const cameras = JSON.parse(el.camera.dataset.items || "[]");
  const camera = cameras.find(item => String(item.cameraIndex) === el.camera.value);
  if (!camera) return;
  el.stream.src = `${camera.streamUrl}?t=${Date.now()}`;
  el["camera-state"].textContent = camera.status;
}

async function startSession() {
  try {
    app.session = await api("/model-training/api/sessions", {
      method: "POST", headers: authHeaders(true), body: JSON.stringify({
        productCode: el.product.value, cameraIndex: Number(el.camera.value), scenario: el.scenario.value
      })
    });
    renderSession(); status(`Session started for ${app.session.productName}.`);
  } catch (error) { status(error.message, true); }
}

function renderSession() {
  const active = app.session && app.session.status === "active";
  el.capture.disabled = !active;
  el["session-summary"].textContent = active
    ? `${app.session.productName} / camera ${app.session.cameraNumber} / ${app.session.scenario} / ${app.session.frameCount} captures`
    : "No active session";
}

async function captureOne() {
  if (!app.session) return;
  el.capture.disabled = true; status("Requesting native 4K still...");
  try {
    const frame = await api(`/model-training/api/sessions/${app.session.sessionId}/captures`, { method: "POST", headers: authHeaders() });
    app.session.frameCount += 1; renderSession();
    app.frames.push(frame); await showFrame(frame); status("Capture registered. Draw one tight box.");
  } catch (error) { status(error.message, true); }
  finally { el.capture.disabled = false; }
}

async function loadQueue() {
  const result = await api("/model-training/api/frames?status=needs_review&limit=200");
  app.frames = result.frames; el["review-count"].textContent = `${app.frames.length} pending`;
  if (app.frames.length) await showFrame(app.frames[0]); else clearFrame();
}

async function showFrame(frame) {
  app.frame = frame; app.box = frame.annotations[0] || null;
  const image = new Image();
  image.onload = () => { app.image = image; resizeCanvas(); renderCanvas(); };
  image.src = `${frame.imageUrls.review}&t=${Date.now()}`;
  el["frame-caption"].textContent = `${frame.productName} / camera ${frame.cameraNumber} / ${frame.scenario}`;
  el["canvas-wrap"].classList.remove("empty"); el["empty-review"].hidden = true; el["review-canvas"].style.display = "block";
  setReviewEnabled(true);
}

function clearFrame() {
  app.frame = null; app.image = null; app.box = null;
  el["canvas-wrap"].classList.add("empty"); el["empty-review"].hidden = false; el["review-canvas"].style.display = "none";
  setReviewEnabled(false); el["review-count"].textContent = "0 pending";
}

function setReviewEnabled(enabled) { ["accept", "redraw", "not-visible", "uncertain", "reject"].forEach(id => el[id].disabled = !enabled); }
function resizeCanvas() { if (!app.image) return; el["review-canvas"].width = app.image.naturalWidth; el["review-canvas"].height = app.image.naturalHeight; }
function renderCanvas() {
  if (!app.image) return;
  const canvas = el["review-canvas"], ctx = canvas.getContext("2d");
  ctx.drawImage(app.image, 0, 0, canvas.width, canvas.height);
  if (!app.box) return;
  const x = app.box.x1 * canvas.width, y = app.box.y1 * canvas.height;
  const w = (app.box.x2 - app.box.x1) * canvas.width, h = (app.box.y2 - app.box.y1) * canvas.height;
  ctx.strokeStyle = "#d8f26a"; ctx.lineWidth = Math.max(3, canvas.width / 400); ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = "#d8f26a"; ctx.fillRect(x, Math.max(0, y - 30), Math.min(w, 260), 30);
  ctx.fillStyle = "#17211c"; ctx.font = "bold 20px sans-serif"; ctx.fillText(app.frame.productCode, x + 7, Math.max(21, y - 8));
}

function point(event) {
  const rect = el["review-canvas"].getBoundingClientRect();
  return { x: Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)), y: Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height)) };
}

el["review-canvas"].addEventListener("pointerdown", event => { event.preventDefault(); el["review-canvas"].setPointerCapture(event.pointerId); app.dragStart = point(event); app.box = null; renderCanvas(); });
el["review-canvas"].addEventListener("pointermove", event => {
  if (!app.dragStart) return; event.preventDefault(); const current = point(event);
  app.box = { productCode: app.frame.productCode, x1: Math.min(app.dragStart.x, current.x), y1: Math.min(app.dragStart.y, current.y), x2: Math.max(app.dragStart.x, current.x), y2: Math.max(app.dragStart.y, current.y) }; renderCanvas();
});
el["review-canvas"].addEventListener("pointerup", event => { event.preventDefault(); app.dragStart = null; });
el["review-canvas"].addEventListener("contextmenu", event => event.preventDefault());

async function accept() {
  if (!app.frame || !app.box || app.box.x2 - app.box.x1 < .003 || app.box.y2 - app.box.y1 < .003) { status("Draw a non-empty product box first.", true); return; }
  try {
    await api(`/model-training/api/frames/${app.frame.frameId}/annotations`, { method: "PUT", headers: authHeaders(true), body: JSON.stringify({ boxes: [app.box] }) });
    await api(`/model-training/api/frames/${app.frame.frameId}/accept`, { method: "POST", headers: authHeaders() });
    await advance("Frame accepted.");
  } catch (error) { status(error.message, true); }
}

async function finalize(action, message) {
  if (!app.frame) return;
  try { await api(`/model-training/api/frames/${app.frame.frameId}/${action}`, { method: "POST", headers: authHeaders() }); await advance(message); }
  catch (error) { status(error.message, true); }
}

async function advance(message) { app.frames = app.frames.filter(item => item.frameId !== app.frame.frameId); el["review-count"].textContent = `${app.frames.length} pending`; if (app.frames.length) await showFrame(app.frames[0]); else clearFrame(); status(message); }
async function exportDataset() { try { const result = await api("/model-training/api/datasets", { method: "POST", headers: authHeaders() }); el["export-result"].textContent = JSON.stringify({ datasetVersion: result.datasetVersion, counts: result.counts, warning: result.splitWarning, path: result.path }, null, 2); status(`${result.datasetVersion} exported.`); } catch (error) { status(error.message, true); } }

el.token.value = app.token;
el.connect.addEventListener("click", connect); el["start-session"].addEventListener("click", startSession); el.capture.addEventListener("click", captureOne); el.camera.addEventListener("change", updateCamera);
el.accept.addEventListener("click", accept); el.redraw.addEventListener("click", () => { app.box = null; renderCanvas(); });
el["not-visible"].addEventListener("click", () => finalize("not-visible", "Valid negative recorded."));
el.uncertain.addEventListener("click", () => finalize("uncertain", "Frame marked uncertain.")); el.reject.addEventListener("click", () => finalize("reject", "Frame rejected.")); el.export.addEventListener("click", exportDataset);
window.addEventListener("resize", renderCanvas);
if (app.token) connect();
