"use strict";

const app = { token: localStorage.getItem("modelTrainingToken") || "", connected: false, captureInFlight: false, queueRefreshInFlight: false, products: [], session: null, workingSessions: [], exportedSessions: [], reviewSessionId: null, reviewMode: "working", readOnly: false, frames: [], frame: null, frameIndex: -1, image: null, box: null, dragStart: null };
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
  app.connected = false;
  app.token = el.token.value.trim();
  localStorage.setItem("modelTrainingToken", app.token);
  try {
    const [products, cameras] = await Promise.all([
      api("/model-training/api/products"), api("/model-training/api/cameras")
    ]);
    app.products = products.products;
    renderProducts();
    renderOptions(el.camera, cameras.cameras, item => item.cameraIndex, item => `Camera ${item.cameraNumber} - ${item.status}`);
    el.camera.dataset.items = JSON.stringify(cameras.cameras);
    restoreCameraSelection();
    updateCamera();
    try { app.session = await api("/model-training/api/sessions/active"); } catch (_) { app.session = null; }
    renderSession();
    await refreshSessionLists(false);
    await loadQueue();
    app.connected = true;
    status(products.products.length ? "Connected." : "Connected. Refreshing product catalog...");
    if (!products.products.length) {
      const refreshed = await api("/model-training/api/products/refresh", { method: "POST", headers: authHeaders() });
      app.products = refreshed.products;
      renderProducts();
      status(`Loaded ${refreshed.count} products.`);
    }
  } catch (error) { status(error.message, true); }
}

function renderOptions(select, items, value, label) {
  select.replaceChildren(...items.map(item => {
    const option = document.createElement("option"); option.value = value(item); option.textContent = label(item); return option;
  }));
}

function sessionLabel(session, exported = false) {
  const base = `${session.productName} · Camera ${session.cameraNumber} · ${session.scenario}`;
  if (exported) return `${base} · ${session.exportedCount} frames · ${session.datasetVersions.join(", ")}`;
  return `${base} · ${session.pendingCount} pending · ${session.unexportedCount} ready`;
}

function renderSessionSelect(select, sessions, placeholder, exported = false) {
  const selected = select.value;
  const options = [document.createElement("option")];
  options[0].value = ""; options[0].textContent = sessions.length ? placeholder : `No ${placeholder.toLowerCase()}`;
  for (const session of sessions) {
    const option = document.createElement("option");
    option.value = session.sessionId;
    option.textContent = sessionLabel(session, exported);
    options.push(option);
  }
  select.replaceChildren(...options);
  if (sessions.some(session => session.sessionId === selected)) select.value = selected;
}

async function refreshSessionLists(preserveSelection = true) {
  const previousId = preserveSelection ? app.reviewSessionId : null;
  const [working, exported] = await Promise.all([
    api("/model-training/api/sessions?group=working"),
    api("/model-training/api/sessions?group=exported")
  ]);
  app.workingSessions = working.sessions;
  app.exportedSessions = exported.sessions;
  renderSessionSelect(el["working-session"], app.workingSessions, "Choose working session");
  renderSessionSelect(el["exported-session"], app.exportedSessions, "Choose exported session", true);

  const preserved = previousId && (
    (app.reviewMode === "working" && app.workingSessions.some(item => item.sessionId === previousId))
    || (app.reviewMode === "exported" && app.exportedSessions.some(item => item.sessionId === previousId))
  );
  if (preserved) {
    el[`${app.reviewMode}-session`].value = previousId;
    return;
  }
  const preferred = app.session && app.workingSessions.find(item => item.sessionId === app.session.sessionId);
  const first = preferred || app.workingSessions[0];
  app.reviewMode = "working";
  app.readOnly = false;
  app.reviewSessionId = first ? first.sessionId : null;
  el["working-session"].value = app.reviewSessionId || "";
  el["exported-session"].value = "";
}

async function selectReviewSession(mode) {
  const select = el[`${mode}-session`];
  app.reviewMode = mode;
  app.readOnly = mode === "exported";
  app.reviewSessionId = select.value || null;
  el[mode === "working" ? "exported-session" : "working-session"].value = "";
  await loadQueue();
}

function renderProducts() {
  const selected = el.product.value;
  const query = el["product-search"].value.trim().toLocaleLowerCase();
  const matches = app.products.filter(item => {
    if (!query) return true;
    return [item.name, item.code, item.barCode]
      .filter(value => value !== null && value !== undefined)
      .some(value => String(value).toLocaleLowerCase().includes(query));
  });
  renderOptions(el.product, matches, item => item.code, item => `${item.name} (${item.code})`);
  if (matches.some(item => item.code === selected)) el.product.value = selected;
  el["product-match-count"].textContent = `${matches.length} of ${app.products.length} products`;
  el["start-session"].disabled = matches.length === 0;
}

function updateCamera() {
  const cameras = JSON.parse(el.camera.dataset.items || "[]");
  const camera = cameras.find(item => String(item.cameraIndex) === el.camera.value);
  if (!camera) return;
  localStorage.setItem("modelTrainingCameraIndex", String(camera.cameraIndex));
  el.stream.src = `${camera.streamUrl}?t=${Date.now()}`;
  el["camera-state"].textContent = camera.status;
}

function restoreCameraSelection() {
  const storedCameraIndex = localStorage.getItem("modelTrainingCameraIndex");
  if (storedCameraIndex === null) return;
  const available = [...el.camera.options].some(option => option.value === storedCameraIndex);
  if (available) el.camera.value = storedCameraIndex;
}

function selectedSessionConfig() {
  return {
    productCode: el.product.value,
    cameraIndex: Number(el.camera.value),
    scenario: el.scenario.value
  };
}

function sessionMatchesSelection() {
  if (!app.session || app.session.status !== "active") return false;
  const selected = selectedSessionConfig();
  return app.session.productCode === selected.productCode
    && Number(app.session.cameraIndex) === selected.cameraIndex
    && app.session.scenario === selected.scenario;
}

async function createSessionFromSelection() {
  app.session = await api("/model-training/api/sessions", {
    method: "POST",
    headers: authHeaders(true),
    body: JSON.stringify(selectedSessionConfig())
  });
  renderSession();
  app.reviewMode = "working";
  app.readOnly = false;
  app.reviewSessionId = app.session.sessionId;
  await refreshSessionLists();
  return app.session;
}

async function startSession() {
  try {
    await createSessionFromSelection();
    await loadQueue();
    status(`Session started for ${app.session.productName}.`);
  } catch (error) { status(error.message, true); }
}

function renderSession() {
  const active = app.session && app.session.status === "active";
  if (active) {
    el["product-search"].value = "";
    renderProducts();
    el.product.value = app.session.productCode;
    el.camera.value = String(app.session.cameraIndex);
    localStorage.setItem("modelTrainingCameraIndex", String(app.session.cameraIndex));
    el.scenario.value = app.session.scenario;
    updateCamera();
  }
  el.capture.disabled = !active || app.captureInFlight;
  el["clear-session"].disabled = !active;
  el["session-summary"].textContent = active
    ? `${app.session.productName} / camera ${app.session.cameraNumber} / ${app.session.scenario} / ${app.session.frameCount} captures`
    : "No active session";
}

async function captureOne() {
  if (!app.session) return;
  app.captureInFlight = true;
  renderSession();
  try {
    if (!sessionMatchesSelection()) {
      const selectedScenario = el.scenario.value;
      status(`Starting a new ${selectedScenario} session...`);
      await createSessionFromSelection();
    }
    status("Requesting native 4K still...");
    const frame = await api(`/model-training/api/sessions/${app.session.sessionId}/captures`, { method: "POST", headers: authHeaders() });
    app.session.frameCount += 1; renderSession();
    app.reviewMode = "working"; app.readOnly = false; app.reviewSessionId = app.session.sessionId;
    await refreshSessionLists();
    await loadQueue();
    await showFrame(frame); status("Capture registered. Draw one tight box.");
  } catch (error) { status(error.message, true); }
  finally { app.captureInFlight = false; renderSession(); }
}

async function loadQueue(preserveCurrent = false, pendingCount = null) {
  const currentFrameId = app.frame && app.frame.frameId;
  if (!app.reviewSessionId) { app.frames = []; clearFrame(); return; }
  const session = encodeURIComponent(app.reviewSessionId);
  const path = app.readOnly
    ? `/model-training/api/frames?sessionId=${session}&exportedOnly=true&limit=500`
    : `/model-training/api/frames?status=needs_review&sessionId=${session}&limit=500`;
  const result = await api(path);
  app.frames = result.frames;
  updatePendingCount(pendingCount === null ? app.frames.length : pendingCount);
  const currentFrame = currentFrameId && app.frames.find(item => item.frameId === currentFrameId);
  if (preserveCurrent && currentFrame) return;
  if (app.frames.length) await showFrame(app.frames[0]); else clearFrame();
}

function updatePendingCount(count) {
  el["review-count"].textContent = app.readOnly ? `${count} exported` : `${count} pending`;
}

async function refreshQueueState() {
  if (!app.connected || app.queueRefreshInFlight || document.visibilityState === "hidden") return;
  app.queueRefreshInFlight = true;
  try {
    const previousMode = app.reviewMode;
    const previousSessionId = app.reviewSessionId;
    await refreshSessionLists();
    if (previousMode !== app.reviewMode || previousSessionId !== app.reviewSessionId) {
      await loadQueue();
      return;
    }
    if (app.readOnly || !app.reviewSessionId) return;
    const state = await api(`/model-training/api/frames/review-state?sessionId=${encodeURIComponent(app.reviewSessionId)}&limit=500`);
    updatePendingCount(state.count);
    const localIds = app.frames.map(item => item.frameId);
    const changed = localIds.length !== state.frameIds.length
      || localIds.some((frameId, index) => frameId !== state.frameIds[index]);
    if (changed) await loadQueue(true, state.count);
  } catch (error) {
    console.warn("Could not refresh model-training review queue:", error);
  } finally {
    app.queueRefreshInFlight = false;
  }
}

async function showFrame(frame) {
  app.frame = frame; app.box = frame.annotations[0] || null;
  app.frameIndex = app.frames.findIndex(item => item.frameId === frame.frameId);
  const image = new Image();
  image.onload = () => { app.image = image; resizeCanvas(); renderCanvas(); };
  image.src = `${frame.imageUrls.review}&t=${Date.now()}`;
  const position = app.frameIndex >= 0 ? ` / frame ${app.frameIndex + 1} of ${app.frames.length}` : "";
  const mode = app.readOnly ? " / exported read-only" : "";
  el["frame-caption"].textContent = `${frame.productName} / camera ${frame.cameraNumber} / ${frame.scenario}${position}${mode}`;
  el["canvas-wrap"].classList.remove("empty"); el["empty-review"].hidden = true; el["review-canvas"].style.display = "block";
  setReviewEnabled(true);
}

function clearFrame() {
  app.frame = null; app.frameIndex = -1; app.image = null; app.box = null;
  el["canvas-wrap"].classList.add("empty"); el["empty-review"].hidden = false; el["review-canvas"].style.display = "none";
  setReviewEnabled(false); updatePendingCount(0);
}

function setReviewEnabled(enabled) {
  ["accept", "redraw", "not-visible", "uncertain", "reject"].forEach(id => el[id].disabled = !enabled || app.readOnly);
  el["previous-frame"].disabled = !enabled || app.frameIndex <= 0;
  el["next-frame"].disabled = !enabled || app.frameIndex < 0 || app.frameIndex >= app.frames.length - 1;
}

async function moveFrame(offset) {
  const next = app.frameIndex + offset;
  if (next >= 0 && next < app.frames.length) await showFrame(app.frames[next]);
}
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

async function advance(message) {
  const removedIndex = app.frameIndex;
  app.frames = app.frames.filter(item => item.frameId !== app.frame.frameId);
  updatePendingCount(app.frames.length);
  if (app.frames.length) await showFrame(app.frames[Math.min(removedIndex, app.frames.length - 1)]); else clearFrame();
  await refreshSessionLists();
  status(message);
}
async function exportDataset() {
  try {
    const result = await api("/model-training/api/datasets", { method: "POST", headers: authHeaders() });
    el["export-result"].textContent = JSON.stringify({ datasetVersion: result.datasetVersion, counts: result.counts, warning: result.splitWarning, path: result.path }, null, 2);
    await refreshSessionLists();
    status(`${result.datasetVersion} exported.`);
  } catch (error) { status(error.message, true); }
}

async function clearCurrentSession() {
  if (!confirm("Delete the active session and all of its captured training files? Exported datasets remain unchanged.")) return;
  try {
    const result = await api("/model-training/api/sessions/active", {
      method: "DELETE", headers: authHeaders(true),
      body: JSON.stringify({ confirmation: "CLEAR CURRENT SESSION" })
    });
    app.session = null;
    await refreshSessionLists(false);
    await loadQueue();
    renderSession();
    status(`Cleared current session and ${result.frameCount} frames.`);
  } catch (error) { status(error.message, true); }
}

async function clearAllTrainingData() {
  if (!confirm("Permanently delete ALL sessions, captures, annotations, and exported datasets?")) return;
  const phrase = prompt("Type CLEAR ALL TRAINING DATA to confirm:");
  if (phrase !== "CLEAR ALL TRAINING DATA") { status("Clear-all cancelled: confirmation did not match.", true); return; }
  try {
    const result = await api("/model-training/api/training-data", {
      method: "DELETE", headers: authHeaders(true),
      body: JSON.stringify({ confirmation: phrase })
    });
    app.session = null; app.frames = []; app.workingSessions = []; app.exportedSessions = []; app.reviewSessionId = null; clearFrame(); renderSession();
    renderSessionSelect(el["working-session"], [], "Working sessions");
    renderSessionSelect(el["exported-session"], [], "Exported sessions", true);
    el["export-result"].textContent = "";
    status(`Cleared ${result.sessions} sessions, ${result.frames} frames, and ${result.datasets} datasets.`);
  } catch (error) { status(error.message, true); }
}

el.token.value = app.token;
el.connect.addEventListener("click", connect); el["start-session"].addEventListener("click", startSession); el.capture.addEventListener("click", captureOne); el.camera.addEventListener("change", updateCamera);
el["working-session"].addEventListener("change", () => selectReviewSession("working"));
el["exported-session"].addEventListener("change", () => selectReviewSession("exported"));
el["previous-frame"].addEventListener("click", () => moveFrame(-1));
el["next-frame"].addEventListener("click", () => moveFrame(1));
el["product-search"].addEventListener("input", renderProducts);
el.accept.addEventListener("click", accept); el.redraw.addEventListener("click", () => { app.box = null; renderCanvas(); });
el["not-visible"].addEventListener("click", () => finalize("not-visible", "Valid negative recorded."));
el.uncertain.addEventListener("click", () => finalize("uncertain", "Frame marked uncertain.")); el.reject.addEventListener("click", () => finalize("reject", "Frame rejected.")); el.export.addEventListener("click", exportDataset);
el["clear-session"].addEventListener("click", clearCurrentSession);
el["clear-all-data"].addEventListener("click", clearAllTrainingData);
window.addEventListener("resize", renderCanvas);
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") refreshQueueState(); });
window.setInterval(refreshQueueState, 2000);
if (app.token) connect();
