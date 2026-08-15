(() => {
  "use strict";

  const app = {
    state: null,
    selectedCamera: null,
    selectedObservation: null,
    observationPayload: null,
    recentObservations: new Map(),
    eventSource: null,
    timeline: [],
    eventFeedback: new Map(),
    report: null,
    worldState: null,
    shelfSync: null,
    productCameraEvidence: null,
    frozenProductCameras: new Map(),
    productFreezePending: new Set(),
    productSnapshotVisitId: null,
    openShopPending: false,
    visitProposals: new Map(),
    automaticVisitMappings: new Set(),
    lastWorldPollAt: 0,
    pollTimer: null,
    runTimer: null,
    token: localStorage.getItem("shopOperatorToken") || "",
  };

  const el = (id) => document.getElementById(id);
  const annotationButtons = [...document.querySelectorAll("[data-annotation]")];
  const physicalButtons = [...document.querySelectorAll("[data-physical]")];
  const OBSERVATION_GRACE_MILLISECONDS = 2500;
  const MONITORED_EVENT_TYPES = new Set([
    "entry_accepted",
    "leave_accepted",
    "shop_entry_bound",
    "shop_entry_bind_skipped",
    "shop_entry_bind_failed",
    "shop_leave_persisted",
    "shop_leave_persist_failed",
  ]);
  const VERIFIABLE_EVENT_TYPES = new Set(["entry_accepted", "leave_accepted"]);

  async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    if (options.body && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }
    if (app.token) {
      headers.set("Authorization", `Bearer ${app.token}`);
    }
    const response = await fetch(path, {...options, headers});
    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      try {
        const payload = await response.json();
        message = payload.detail || message;
      } catch (_) {
        // Keep HTTP status when the body is not JSON.
      }
      throw new Error(message);
    }
    return response.json();
  }

  async function bootstrap() {
    try {
      app.state = await api("/operator/api/state");
      setConnected(true);
      renderState();
      connectEvents();
      startPolling();
    } catch (error) {
      setConnected(false);
      showToast(error.message);
      setTimeout(bootstrap, 2000);
    }
  }

  function setConnected(online) {
    el("connection-dot").classList.toggle("online", online);
    el("connection-label").textContent = online ? "Live" : "Disconnected";
  }

  function renderState() {
    renderRun();
    renderCameras();
    renderSubjects();
    mergeTimeline(app.state?.recentEvents || []);
    setActionAvailability();
    void pollWorldState();
  }

  function renderRun() {
    const run = app.state?.activeRun;
    el("run-name").textContent = run ? `${run.scenario} · ${run.runId}` : "No active test";
    el("start-run").disabled = Boolean(run);
    el("open-shop").disabled = !run || app.openShopPending;
    el("open-shop").textContent = app.openShopPending ? "Opening…" : "Open shop";
    el("stop-run").disabled = !run;
    if (app.runTimer) clearInterval(app.runTimer);
    if (!run) {
      el("run-clock").textContent = "Start a run before recording physical facts.";
      return;
    }
    const updateClock = () => {
      const elapsed = Math.max(0, Date.now() - run.startedAtUnixMilliseconds);
      const minutes = Math.floor(elapsed / 60000);
      const seconds = Math.floor((elapsed % 60000) / 1000);
      el("run-clock").textContent = `${minutes}:${String(seconds).padStart(2, "0")} elapsed · verifier ${run.verifier}`;
    };
    updateClock();
    app.runTimer = setInterval(updateClock, 1000);
  }

  function renderCameras() {
    const container = el("camera-tabs");
    container.replaceChildren();
    const cameras = app.state?.cameras || [];
    if (app.selectedCamera === null && cameras.length) {
      app.selectedCamera = cameras.find((camera) => camera.status === "active")?.id ?? cameras[0].id;
    }
    cameras.forEach((camera) => {
      const button = document.createElement("button");
      button.className = `camera-tab ${camera.id === app.selectedCamera ? "active" : ""} ${camera.status}`;
      const title = document.createElement("strong");
      title.textContent = `Camera ${camera.id + 1}`;
      const detail = document.createElement("span");
      detail.textContent = `${camera.role} · ${camera.visiblePersonCount} visible`;
      button.append(title, detail);
      button.addEventListener("click", () => selectCamera(camera.id));
      container.append(button);
    });
    updateStream();
  }

  function selectCamera(cameraId) {
    app.selectedCamera = cameraId;
    app.selectedObservation = null;
    app.observationPayload = null;
    app.recentObservations.clear();
    renderCameras();
    renderSelected();
    pollObservation();
  }

  function updateStream() {
    const image = el("camera-stream");
    const empty = el("camera-empty");
    if (app.selectedCamera === null) {
      image.removeAttribute("src");
      empty.hidden = false;
      return;
    }
    const expected = `/stream/${app.selectedCamera}`;
    if (!image.src.endsWith(expected)) image.src = expected;
    empty.hidden = true;
  }

  function renderSubjects() {
    const select = el("subject-select");
    const current = select.value;
    select.replaceChildren();
    const subjects = app.state?.activeRun?.subjects || [];
    subjects.forEach((subject) => {
      const option = document.createElement("option");
      option.value = subject.subjectId;
      option.textContent = subject.displayName;
      select.append(option);
    });
    if (subjects.some((subject) => subject.subjectId === current)) {
      select.value = current;
    }
  }

  function selectedSubjectId() {
    return el("subject-select").value || null;
  }

  function subjectProposalKey() {
    const runId = app.state?.activeRun?.runId;
    const subjectId = selectedSubjectId();
    return runId && subjectId ? `${runId}:${subjectId}` : null;
  }

  function visitMappingState(payload = app.worldState) {
    const key = subjectProposalKey();
    const resolution = payload?.resolution || {};
    const visitId = resolution.visitId;
    const selectedVisitId = app.selectedObservation?.person?.visitId;
    if (resolution.status === "confirmed" && Number.isInteger(visitId)) {
      if (Number.isInteger(selectedVisitId) && selectedVisitId !== visitId) {
        return {visitId: selectedVisitId, status: "manual_override", current: true};
      }
      return {visitId, status: "confirmed", current: true};
    }
    if (resolution.status === "single_candidate" && Number.isInteger(visitId)) {
      return {visitId, status: "automatic", current: true};
    }
    if (resolution.status === "single_observer_candidate" && Number.isInteger(visitId)) {
      const proposal = {visitId, status: resolution.status, current: true};
      if (key) app.visitProposals.set(key, proposal);
      return proposal;
    }
    if (
      ["ambiguous", "ambiguous_observer_candidates"].includes(resolution.status)
      && Number.isInteger(selectedVisitId)
    ) {
      return {visitId: selectedVisitId, status: "manual_candidate", current: true};
    }
    const remembered = key ? app.visitProposals.get(key) : null;
    return remembered ? {...remembered, current: false} : null;
  }

  async function autoMapSingleEntranceCandidate(payload) {
    const run = app.state?.activeRun;
    const subjectId = selectedSubjectId();
    const resolution = payload?.resolution || {};
    const visitId = resolution.visitId;
    if (
      !run
      || !subjectId
      || resolution.status !== "single_candidate"
      || !Number.isInteger(visitId)
    ) {
      return false;
    }
    const mappingKey = `${run.runId}:${subjectId}:${visitId}`;
    if (app.automaticVisitMappings.has(mappingKey)) return false;
    app.automaticVisitMappings.add(mappingKey);
    try {
      await api(`/operator/api/test-runs/${encodeURIComponent(run.runId)}/annotations`, {
        method: "POST",
        body: JSON.stringify({
          annotationType: "subject_visit_mapping",
          subjectId,
          visitId,
          mappingSource: "automatic_single_entrance_candidate",
          clientRecordedAtUnixMilliseconds: Date.now(),
        }),
      });
      return true;
    } catch (error) {
      app.automaticVisitMappings.delete(mappingKey);
      showToast(`Automatic visit selection failed: ${error.message}`);
      return false;
    }
  }

  function startPolling() {
    if (app.pollTimer) clearInterval(app.pollTimer);
    pollObservation();
    app.pollTimer = setInterval(pollObservation, 400);
  }

  async function pollObservation() {
    if (app.selectedCamera === null) return;
    try {
      const payload = await api(`/observer-cameras/${app.selectedCamera}/observations`);
      app.observationPayload = payload;
      updateRecentObservations(payload);
      el("frame-age").textContent = payload.frame
        ? `Frame ${payload.frame.rgbSequenceNumber} · ${payload.frame.ageMilliseconds} ms`
        : "Camera starting";
      if (app.selectedObservation) {
        const refreshed = recentObservation(
          app.selectedObservation.person.trackId
        );
        app.selectedObservation = refreshed || null;
      }
      drawOverlay();
      renderSelected();
      updateSystemAnswer();
      if (Date.now() - app.lastWorldPollAt >= 1000) void pollWorldState();
    } catch (error) {
      el("frame-age").textContent = "Observation unavailable";
      drawOverlay();
    }
  }

  function drawOverlay() {
    const canvas = el("observation-overlay");
    const stage = el("camera-stage");
    const payload = app.observationPayload;
    const observations = recentObservationEntries();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(stage.clientWidth * ratio);
    canvas.height = Math.round(stage.clientHeight * ratio);
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);
    if (!payload?.frame) {
      renderObservationHitboxes(null, []);
      return;
    }

    const frameWidth = payload.frame.width;
    const frameHeight = payload.frame.height;
    const scale = Math.min(stage.clientWidth / frameWidth, stage.clientHeight / frameHeight);
    const renderedWidth = frameWidth * scale;
    const renderedHeight = frameHeight * scale;
    const offsetX = (stage.clientWidth - renderedWidth) / 2;
    const offsetY = (stage.clientHeight - renderedHeight) / 2;
    const geometry = {scale, offsetX, offsetY};
    renderObservationHitboxes(geometry, observations);

    observations.forEach((entry, index) => {
      const person = entry.person;
      const box = person.boundingBox;
      const x = offsetX + box.x1 * scale;
      const y = offsetY + box.y1 * scale;
      const width = (box.x2 - box.x1) * scale;
      const height = (box.y2 - box.y1) * scale;
      const selected = app.selectedObservation?.person.trackId === person.trackId;
      context.strokeStyle = selected ? "#ff6a3d" : "#a8ffcf";
      context.lineWidth = selected ? 4 : 2;
      context.setLineDash(entry.recent ? [7, 5] : []);
      context.strokeRect(x, y, width, height);
      context.setLineDash([]);
      const label = `${index + 1} · T${person.trackId} · V${person.visitId ?? "?"} · C${person.customerId ?? "?"}${entry.recent ? " · recent" : ""}`;
      context.font = "700 14px Bahnschrift, sans-serif";
      const labelWidth = context.measureText(label).width + 14;
      context.fillStyle = selected ? "#ff6a3d" : "rgba(8, 38, 32, .9)";
      context.fillRect(x, Math.max(0, y - 25), labelWidth, 25);
      context.fillStyle = "white";
      context.fillText(label, x + 7, Math.max(17, y - 7));
    });
  }

  function renderObservationHitboxes(geometry, observations) {
    const layer = el("observation-hitboxes");
    const activeTrackIds = new Set();
    if (!geometry) {
      layer.replaceChildren();
      return;
    }

    observations.forEach((entry) => {
      const person = entry.person;
      const trackId = String(person.trackId);
      activeTrackIds.add(trackId);
      let button = [...layer.children].find(
        (candidate) => candidate.dataset.trackId === trackId
      );
      if (!button) {
        button = document.createElement("button");
        button.type = "button";
        button.className = "person-hitbox";
        button.dataset.trackId = trackId;
        bindObservationSelectionControl(button);
        button.addEventListener("contextmenu", (event) => event.preventDefault());
        layer.append(button);
      }
      const box = person.boundingBox;
      button.style.left = `${geometry.offsetX + box.x1 * geometry.scale}px`;
      button.style.top = `${geometry.offsetY + box.y1 * geometry.scale}px`;
      button.style.width = `${(box.x2 - box.x1) * geometry.scale}px`;
      button.style.height = `${(box.y2 - box.y1) * geometry.scale}px`;
      button.classList.toggle(
        "selected",
        app.selectedObservation?.person.trackId === person.trackId
      );
      button.classList.toggle("recent", entry.recent);
      button.setAttribute(
        "aria-label",
        `Select track ${person.trackId}, visit ${person.visitId ?? "unassigned"}`
      );
    });

    [...layer.children].forEach((button) => {
      if (!activeTrackIds.has(button.dataset.trackId)) button.remove();
    });
  }

  function selectObservationFromControl(event) {
    event.preventDefault();
    const trackId = Number.parseInt(event.currentTarget.dataset.trackId, 10);
    selectObservationByTrackId(trackId);
  }

  function bindObservationSelectionControl(button) {
    button.addEventListener("touchend", (event) => {
      event.preventDefault();
      event.stopPropagation();
      button._lastTouchSelectionAt = Date.now();
      selectObservationFromControl(event);
    }, {passive: false});
    button.addEventListener("click", (event) => {
      if (
        button._lastTouchSelectionAt &&
        Date.now() - button._lastTouchSelectionAt < 750
      ) {
        event.preventDefault();
        return;
      }
      selectObservationFromControl(event);
    });
  }

  function selectObservationByTrackId(trackId) {
    const observation = recentObservation(trackId);
    if (!observation) return;
    app.selectedObservation = observation;
    renderSelected();
    drawOverlay();
    showToast(`Selected track ${app.selectedObservation.person.trackId}`);
  }

  function renderObservationChoices() {
    const container = el("observation-choices");
    const observations = recentObservationEntries();
    if (!observations.length) {
      if (container.dataset.state !== "empty") {
        const message = document.createElement("p");
        message.textContent = "Wait until the selected camera reports a visible person.";
        container.replaceChildren(message);
        container.dataset.state = "empty";
      }
      return;
    }

    if (container.dataset.state !== "tracks") {
      container.replaceChildren();
      container.dataset.state = "tracks";
    }
    const activeTrackIds = new Set();
    observations.forEach((entry) => {
      const person = entry.person;
      const trackId = String(person.trackId);
      activeTrackIds.add(trackId);
      let button = [...container.querySelectorAll("button")].find(
        (candidate) => candidate.dataset.trackId === trackId
      );
      if (!button) {
        button = document.createElement("button");
        button.type = "button";
        button.className = "button compact observation-choice";
        button.dataset.trackId = trackId;
        bindObservationSelectionControl(button);
        button.addEventListener("contextmenu", (event) => event.preventDefault());
        container.append(button);
      }
      button.textContent = `Select track ${person.trackId} · visit ${person.visitId ?? "unassigned"}${entry.recent ? " · recently visible" : ""}`;
      button.classList.toggle(
        "selected",
        app.selectedObservation?.person.trackId === person.trackId
      );
      button.classList.toggle("recent", entry.recent);
    });
    [...container.querySelectorAll("button")].forEach((button) => {
      if (!activeTrackIds.has(button.dataset.trackId)) button.remove();
    });
  }

  function renderSelected() {
    const selection = app.selectedObservation;
    const person = selection?.person;
    const details = el("selected-details");
    details.replaceChildren();
    renderObservationChoices();
    if (!person || !selection?.frame) {
      const visibleCount = recentObservationEntries().length;
      el("selected-title").textContent = visibleCount
        ? "Select a detected person"
        : "No selectable person visible";
      el("selected-freshness").textContent = "None";
      setActionAvailability();
      return;
    }
    el("selected-title").textContent = `Track ${person.trackId} · Visit ${person.visitId ?? "unassigned"}`;
    el("selected-freshness").textContent = `${observationAgeMilliseconds(selection)} ms`;
    const rows = [
      ["Track state", person.trackStatus],
      ["Visit origin", person.visitOrigin || "none"],
      ["Customer", person.customerId || person.customerBindingStatus],
      ["Match", `${person.visitMatch.state}${person.visitMatch.matchedScore == null ? "" : ` · ${person.visitMatch.matchedScore.toFixed(3)}`}`],
      ["Association", person.visitMatch.reason || person.visitMatch.decision || "existing mapping"],
      ["Depth", person.depth ? `${Math.round(person.depth.depthMm)} mm` : "unavailable"],
      ["Box", `${person.boundingBox.x1},${person.boundingBox.y1} → ${person.boundingBox.x2},${person.boundingBox.y2}`],
    ];
    rows.forEach(([key, value]) => {
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = key;
      dd.textContent = value;
      details.append(dt, dd);
    });
    setActionAvailability();
  }

  function observationReference() {
    const selection = app.selectedObservation;
    const person = selection?.person;
    if (!person || !selection.frame || !selection.camera) return null;
    return {
      cameraIndex: selection.camera.id,
      deviceId: selection.camera.deviceId,
      rgbSequenceNumber: selection.frame.rgbSequenceNumber,
      hostSyncedSeconds: selection.frame.hostSyncedSeconds,
      trackId: person.trackId,
      observedVisitId: person.visitId,
      observedCustomerId: person.customerId,
    };
  }

  function updateRecentObservations(payload) {
    const now = Date.now();
    (payload.observations || []).forEach((person) => {
      app.recentObservations.set(String(person.trackId), {
        person,
        camera: payload.camera,
        frame: payload.frame,
        lastSeenAt: now,
        recent: false,
      });
    });
    pruneRecentObservations(now);
  }

  function pruneRecentObservations(now = Date.now()) {
    for (const [trackId, entry] of app.recentObservations) {
      if (now - entry.lastSeenAt > OBSERVATION_GRACE_MILLISECONDS) {
        app.recentObservations.delete(trackId);
      }
    }
  }

  function recentObservationEntries() {
    const now = Date.now();
    pruneRecentObservations(now);
    return [...app.recentObservations.values()].map((entry) => ({
      ...entry,
      recent: now - entry.lastSeenAt > 500,
    }));
  }

  function recentObservation(trackId) {
    return recentObservationEntries().find(
      (entry) => entry.person.trackId === trackId
    ) || null;
  }

  function observationAgeMilliseconds(selection) {
    return Math.max(
      0,
      Date.now() - selection.frame.publishedAtUnixMilliseconds
    );
  }

  function setActionAvailability() {
    const active = Boolean(app.state?.activeRun);
    const selected = active && Boolean(observationReference());
    annotationButtons.forEach((button) => { button.disabled = !selected; });
    physicalButtons.forEach((button) => { button.disabled = !active; });
    el("add-note").disabled = !active;
    const mapButton = el("map-subject-visit");
    if (mapButton) {
      const mapping = visitMappingState();
      const canMap = active && mapping
        && !["confirmed", "automatic"].includes(mapping.status);
      mapButton.disabled = !canMap;
    }
  }

  async function pollWorldState() {
    app.lastWorldPollAt = Date.now();
    const run = app.state?.activeRun;
    const subjectId = selectedSubjectId();
    if (!run || !subjectId) {
      app.worldState = null;
      app.shelfSync = null;
      app.productCameraEvidence = null;
      resetProductSnapshots();
      renderWorldState();
      return;
    }
    try {
      let worldState = await api(
        `/operator/api/test-runs/${encodeURIComponent(run.runId)}/subjects/${encodeURIComponent(subjectId)}/world-state?captureQuery=true&persistQuery=false`
      );
      if (await autoMapSingleEntranceCandidate(worldState)) {
        worldState = await api(
          `/operator/api/test-runs/${encodeURIComponent(run.runId)}/subjects/${encodeURIComponent(subjectId)}/world-state?captureQuery=true&persistQuery=false`
        );
      }
      app.worldState = worldState;
      const visitId = app.worldState?.claims?.visitId;
      if (Number.isInteger(visitId)) {
        try {
          app.shelfSync = await api(`/operator/api/shop-shelf-sync/${visitId}`);
        } catch (error) {
          app.shelfSync = {
            enabled: true,
            visitId,
            status: "unavailable",
            local: null,
            cloud: null,
            pendingCount: 0,
            lastError: error.message,
          };
        }
        if (
          Number.isInteger(app.productSnapshotVisitId)
          && app.productSnapshotVisitId !== visitId
        ) {
          resetProductSnapshots();
        }
        app.productSnapshotVisitId = visitId;
        app.productCameraEvidence = await api(
          `/world-state/visits/${visitId}/product-observations`
        );
      } else {
        app.shelfSync = null;
        app.productCameraEvidence = null;
        resetProductSnapshots();
      }
      renderWorldState();
    } catch (error) {
      el("world-state-summary").textContent = error.message;
    }
  }

  function renderWorldState() {
    const payload = app.worldState;
    const details = el("world-state-details");
    details.replaceChildren();
    if (!payload) {
      el("world-state-title").textContent = "No subject state";
      el("world-state-freshness").textContent = "Unknown";
      el("world-state-summary").textContent = "Start a test run to query subject state.";
      el("map-subject-visit").hidden = true;
      renderShelfPositionEvidence(null);
      renderShelfSync(null);
      renderProductRecognition(null);
      setActionAvailability();
      return;
    }
    const resolution = payload.resolution || {};
    const claims = payload.claims || {};
    const visitText = claims.visitId == null ? "unresolved" : `visit ${claims.visitId}`;
    el("world-state-title").textContent = `${visitText} · ${resolution.status || "unknown"}`;
    el("world-state-freshness").textContent = payload.freshness || "unknown";
    const ambiguous = ["ambiguous", "ambiguous_observer_candidates"].includes(resolution.status);
    const observerProposal = resolution.status === "single_observer_candidate";
    el("world-state-summary").textContent = ambiguous
      ? `Multiple candidate visits: ${(resolution.candidateVisitIds || []).join(", ")}. Select your person track before trusting subject answers.`
      : observerProposal
      ? `Likely observer-only visit ${resolution.visitId}. The system sees this person but did not observe an entrance. Confirm if this is you.`
      : `Revision ${payload.revision} · ${claims.visibility || "unknown visibility"}`;
    const mapButton = el("map-subject-visit");
    const mapping = visitMappingState(payload);
    const mappingActionable = mapping
      && !["confirmed", "automatic"].includes(mapping.status);
    mapButton.hidden = !mappingActionable;
    mapButton.textContent = !mapping
      ? "Use proposed visit"
      : mapping.status === "manual_override"
      ? `Use selected visit ${mapping.visitId} instead`
      : mapping.status === "manual_candidate"
      ? `Use selected visit ${mapping.visitId} for me`
      : mapping.current
      ? `Use visit ${mapping.visitId} for me`
      : `Use last proposed visit ${mapping.visitId} for me`;
    const rows = [
      ["Inside", displayWorldValue(claims.inside)],
      ["Entrance", displayWorldValue(claims.entranceConfirmed)],
      ["Customer", displayWorldValue(claims.customerId)],
      ["Visible cameras", (claims.visibleOnCameraIndexes || []).map((index) => index + 1).join(", ") || "none"],
      ["Shelf position", `${displayWorldValue(claims.shelfPositionId)} · ${claims.shelfPositionFreshness || "unknown"}`],
      ["Shelf distance", claims.shelfPositionDistanceMm == null ? "unknown" : `${Math.round(claims.shelfPositionDistanceMm)} mm`],
      ["Product", claims.productLabel == null ? "unknown" : `${claims.productLabel} (${claims.productId})`],
      ["Product freshness", claims.productRecognitionFreshness || "unknown"],
    ];
    rows.forEach(([key, value]) => {
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = key;
      dd.textContent = value;
      details.append(dt, dd);
    });
    renderShelfPositionEvidence(payload);
    renderShelfSync(app.shelfSync);
    renderProductRecognition(payload);
    setActionAvailability();
  }

  function renderShelfSync(sync) {
    const status = sync?.status || "unknown";
    const badge = el("cloud-shelf-sync-status");
    badge.textContent = status.replaceAll("_", " ");
    badge.className = `pill sync-status ${status}`;
    el("local-shelf-position").textContent = formatShelfSyncPosition(sync?.local);
    el("cloud-shelf-position").textContent = formatShelfSyncPosition(sync?.cloud);
    if (!sync) {
      el("cloud-shelf-sync-detail").textContent = "No synchronization state.";
      return;
    }
    const details = [];
    if (sync.cloud?.sourceRevision != null) details.push(`revision ${sync.cloud.sourceRevision}`);
    if (sync.pendingCount) details.push(`${sync.pendingCount} queued`);
    if (sync.attempts) details.push(`attempt ${sync.attempts}`);
    if (sync.debounceRemainingMilliseconds) {
      details.push(`${sync.debounceRemainingMilliseconds} ms stability wait`);
    }
    if (sync.lastSuccessfulSyncUnixMilliseconds) {
      details.push(`${Math.max(0, Date.now() - sync.lastSuccessfulSyncUnixMilliseconds)} ms since push`);
    }
    if (sync.lastError) details.push(sync.lastError);
    el("cloud-shelf-sync-detail").textContent = details.join(" · ") || "Waiting for shelf evidence.";
  }

  function formatShelfSyncPosition(position) {
    if (!position) return "Unknown";
    const shelf = position.shelfId == null ? "Cleared" : `Shelf ${position.shelfId}`;
    const distance = position.distanceMm == null ? "" : ` · ${Math.round(position.distanceMm)} mm`;
    return `${shelf}${distance}`;
  }

  function renderProductRecognition(payload) {
    const recognition = payload?.visit?.productRecognition;
    const summary = el("product-recognition-summary");
    const evidence = el("product-camera-evidence");
    evidence.replaceChildren();
    const cameraEvidence = app.productCameraEvidence;
    if (!payload || !cameraEvidence) {
      summary.textContent = "No product evidence for this visit.";
      updateProductFreezeAllControl([]);
      return;
    }
    const best = recognition?.bestCandidate;
    const cameras = cameraEvidence.cameras || [];
    const fullFrameDebug = cameras.some(
      (camera) => camera.scope === "full_frame"
    );
    const frozenCount = cameras.filter(
      (camera) => app.frozenProductCameras.has(camera.cameraIndex)
    ).length;
    const beliefSummary = fullFrameDebug
      ? "Full-frame debug mode: camera detections are shown but are not treated as products held by this visit."
      : best
      ? `Visit belief: ${best.label} · ${(100 * (best.bestScore ?? best.score ?? 0)).toFixed(0)}% · ${best.confirmations ?? 1} confirmation(s)`
      : "Visit belief: no product detected.";
    summary.textContent = frozenCount
      ? `${beliefSummary} ${frozenCount} camera snapshot(s) frozen.`
      : beliefSummary;
    updateProductFreezeAllControl(cameras);
    cameras.forEach((camera) => {
      const snapshot = app.frozenProductCameras.get(camera.cameraIndex);
      const displayedCamera = snapshot?.camera || camera;
      const frozen = Boolean(snapshot);
      const card = document.createElement("article");
      card.className = `product-camera-card ${displayedCamera.freshness || "unknown"} ${frozen ? "frozen" : ""}`;

      const heading = document.createElement("div");
      heading.className = "product-camera-heading";
      const title = document.createElement("strong");
      title.textContent = `Camera ${camera.cameraIndex + 1}`;
      const actions = document.createElement("div");
      actions.className = "product-camera-heading-actions";
      const state = document.createElement("span");
      state.className = frozen ? "product-camera-frozen-label" : "";
      state.textContent = frozen ? "FROZEN" : displayedCamera.freshness || "unknown";
      const freeze = document.createElement("button");
      freeze.type = "button";
      freeze.className = "button product-camera-freeze";
      freeze.textContent = frozen ? "Resume" : "Freeze";
      freeze.disabled = (
        !displayedCamera.cropAvailable
        || app.productFreezePending.has(camera.cameraIndex)
      );
      freeze.addEventListener("click", () => {
        if (frozen) {
          app.frozenProductCameras.delete(camera.cameraIndex);
          renderProductRecognition(app.worldState);
        } else {
          void freezeProductCamera(camera.cameraIndex);
        }
      });
      actions.append(state, freeze);
      heading.append(title, actions);
      card.append(heading);

      const metadata = document.createElement("p");
      metadata.className = "product-camera-metadata";
      const captured = snapshot
        ? `frozen ${new Date(snapshot.capturedAtUnixMilliseconds).toLocaleTimeString()}`
        : `${displayedCamera.ageMilliseconds} ms old`;
      metadata.textContent = displayedCamera.cropAvailable
        ? `${displayedCamera.scope === "full_frame" ? "full frame" : `track ${displayedCamera.trackId}`} · frame ${displayedCamera.rgbSequenceNumber} · ${captured} · ${displayedCamera.inferenceMilliseconds} ms inference`
        : "No person crop has been processed for this visit.";
      card.append(metadata);

      if (displayedCamera.cropAvailable) {
        const image = document.createElement("img");
        image.loading = "lazy";
        image.alt = `Camera ${camera.cameraIndex + 1} product detections for visit ${cameraEvidence.visitId}`;
        image.src = snapshot?.imageUrl
          || `/world-state/visits/${cameraEvidence.visitId}/product-observations/${camera.cameraIndex}/crop.jpg?observed=${displayedCamera.observedAtUnixMilliseconds}`;
        card.append(image);
      }

      const list = document.createElement("ul");
      list.className = "product-candidate-list";
      (displayedCamera.candidates || []).forEach((candidate) => {
        const item = document.createElement("li");
        item.textContent = `${candidate.productId} · ${candidate.label} · ${(100 * (candidate.score ?? 0)).toFixed(0)}%`;
        list.append(item);
      });
      if (!list.childElementCount && displayedCamera.cropAvailable) {
        const item = document.createElement("li");
        item.textContent = displayedCamera.scope === "full_frame"
          ? "No products detected in this frame."
          : "No products detected in this crop.";
        list.append(item);
      }
      card.append(list);
      evidence.append(card);
    });
  }

  function resetProductSnapshots() {
    app.frozenProductCameras.clear();
    app.productFreezePending.clear();
    app.productSnapshotVisitId = null;
  }

  function updateProductFreezeAllControl(cameras) {
    const button = el("freeze-all-products");
    const available = cameras.filter((camera) => camera.cropAvailable);
    const allFrozen = (
      available.length > 0
      && available.every((camera) => app.frozenProductCameras.has(camera.cameraIndex))
    );
    button.disabled = !available.length || app.productFreezePending.size > 0;
    button.textContent = allFrozen ? "Resume all" : "Freeze all";
  }

  async function freezeProductCamera(cameraIndex, render = true) {
    const visitId = app.productCameraEvidence?.visitId;
    if (!Number.isInteger(visitId) || app.productFreezePending.has(cameraIndex)) return;
    app.productFreezePending.add(cameraIndex);
    if (render) renderProductRecognition(app.worldState);
    try {
      const snapshot = await api(
        `/world-state/visits/${visitId}/product-observations/${cameraIndex}/snapshot`,
        {method: "POST"}
      );
      if (app.productCameraEvidence?.visitId === visitId) {
        app.frozenProductCameras.set(cameraIndex, snapshot);
      }
    } catch (error) {
      showToast(error.message);
    } finally {
      app.productFreezePending.delete(cameraIndex);
      if (render) renderProductRecognition(app.worldState);
    }
  }

  el("freeze-all-products").addEventListener("click", async () => {
    const cameras = (app.productCameraEvidence?.cameras || []).filter(
      (camera) => camera.cropAvailable
    );
    if (!cameras.length) return;
    if (cameras.every((camera) => app.frozenProductCameras.has(camera.cameraIndex))) {
      app.frozenProductCameras.clear();
      renderProductRecognition(app.worldState);
      return;
    }
    await Promise.all(
      cameras
        .filter((camera) => !app.frozenProductCameras.has(camera.cameraIndex))
        .map((camera) => freezeProductCamera(camera.cameraIndex, false))
    );
    renderProductRecognition(app.worldState);
  });

  function renderShelfPositionEvidence(payload) {
    const container = el("shelf-position-evidence");
    container.replaceChildren();
    const measurements = payload?.visit?.shelfMeasurements || [];
    if (!measurements.length) {
      const empty = document.createElement("p");
      empty.className = "shelf-evidence-empty";
      empty.textContent = "No current shelf distance measurements for this visit.";
      container.append(empty);
      return;
    }
    const position = payload.visit?.shelfPosition;
    const scroll = document.createElement("div");
    scroll.className = "shelf-evidence-scroll";
    const table = document.createElement("table");
    table.className = "shelf-evidence-table";
    const head = document.createElement("thead");
    const heading = document.createElement("tr");
    ["Shelf", "Camera", "Marker", "Distance", "Freshness"].forEach((label) => {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = label;
      heading.append(cell);
    });
    head.append(heading);
    const body = document.createElement("tbody");
    measurements.forEach((measurement) => {
      const row = document.createElement("tr");
      const selected = position
        && measurement.shelfId === position.shelfId
        && measurement.cameraIndex === position.cameraIndex
        && measurement.markerId === position.markerId;
      if (selected) row.className = "selected-position";
      const values = [
        `Shelf ${measurement.shelfId}${selected ? " · selected" : ""}`,
        Number.isInteger(measurement.cameraIndex) ? `Camera ${measurement.cameraIndex + 1}` : "unknown",
        measurement.markerId ?? "unknown",
        measurement.distanceMm == null ? "unknown" : `${Math.round(measurement.distanceMm)} mm`,
        measurement.freshness || "unknown",
      ];
      values.forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = String(value);
        row.append(cell);
      });
      body.append(row);
    });
    table.append(head, body);
    scroll.append(table);
    container.append(scroll);
  }

  function displayWorldValue(value) {
    if (value === null || value === undefined || value === "") return "unknown";
    if (typeof value === "boolean") return value ? "yes" : "no";
    return String(value);
  }

  el("map-subject-visit").addEventListener("click", async () => {
    const visitId = visitMappingState()?.visitId;
    if (!Number.isInteger(visitId)) return showToast("No single proposed visit is available.");
    const response = await annotate("subject_visit_mapping", {visitId});
    if (response) await pollWorldState();
  });
  el("subject-select").addEventListener("change", () => {
    resetProductSnapshots();
    void pollWorldState();
  });

  async function annotate(annotationType, extra = {}, withObservation = false) {
    const run = app.state?.activeRun;
    const subjectId = selectedSubjectId();
    if (!run) {
      showToast("Start a feedback recording first.");
      return null;
    }
    const payload = {
      annotationType,
      subjectId,
      clientRecordedAtUnixMilliseconds: Date.now(),
      ...extra,
    };
    if (withObservation) {
      const reference = observationReference();
      if (!reference) return showToast("Select a fresh person rectangle.");
      payload.observationRef = reference;
    }
    try {
      const response = await api(`/operator/api/test-runs/${run.runId}/annotations`, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      app.report = response.report;
      renderReport();
      showToast(`Recorded: ${annotationType.replaceAll("_", " ")}`);
      return response;
    } catch (error) {
      showToast(error.message);
      return null;
    }
  }

  annotationButtons.forEach((button) => {
    button.addEventListener("click", () => annotate(button.dataset.annotation, {}, true));
  });
  physicalButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const type = button.dataset.physical;
      const extra = type === "subject_visible_but_not_detected"
        ? {cameraIndex: app.selectedCamera}
        : {};
      annotate(type, extra);
    });
  });

  el("add-note").addEventListener("click", () => {
    const input = el("note-text");
    const note = input.value.trim();
    if (!note) return;
    annotate("note", {note});
    input.value = "";
  });

  el("start-run").addEventListener("click", () => {
    el("start-form").elements.token.value = app.token;
    el("start-dialog").showModal();
  });
  el("open-shop").addEventListener("click", async () => {
    if (!app.state?.activeRun) return showToast("Start a test run first.");
    if (!confirm("Physically unlock the shop and create a test shopping customer?")) return;
    app.openShopPending = true;
    renderRun();
    try {
      const response = await api("/operator/api/shop/open", {method: "POST"});
      showToast(`Shop opened · customer ${response.customerId}`);
      await refreshState();
    } catch (error) {
      showToast(error.message);
    } finally {
      app.openShopPending = false;
      renderRun();
    }
  });
  el("cancel-start").addEventListener("click", () => el("start-dialog").close());
  el("start-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    app.token = String(form.get("token") || app.token).trim();
    if (app.token) localStorage.setItem("shopOperatorToken", app.token);
    const payload = {
      scenario: "shop-walk",
      verifier: "Milan",
      subjects: [{
        subjectId: "milan",
        displayName: "Milan",
        expectedCustomerId: null,
      }],
      notes: "",
    };
    try {
      await api("/operator/api/test-runs", {method: "POST", body: JSON.stringify(payload)});
      el("start-dialog").close();
      await refreshState();
      showToast("Test run started.");
    } catch (error) {
      showToast(error.message);
    }
  });

  el("stop-run").addEventListener("click", async () => {
    const run = app.state?.activeRun;
    if (!run || !confirm("Stop and analyze this test run?")) return;
    try {
      const response = await api(`/operator/api/test-runs/${run.runId}/stop`, {method: "POST"});
      app.report = response.report;
      renderReport();
      await refreshState();
      showToast("Test stopped and exported.");
    } catch (error) {
      showToast(error.message);
    }
  });

  async function refreshState() {
    app.state = await api("/operator/api/state");
    renderState();
  }

  function connectEvents() {
    if (app.eventSource) app.eventSource.close();
    const after = app.state?.lastEventId || 0;
    app.eventSource = new EventSource(`/operator/api/events?afterEventId=${after}`);
    app.eventSource.onopen = () => setConnected(true);
    app.eventSource.onerror = () => setConnected(false);
    app.eventSource.onmessage = receiveEvent;
    const types = [
      "track_appeared", "track_disappeared", "visit_assignment_changed",
      "customer_binding_changed", "entry_accepted", "leave_accepted",
      "shop_entry_bound", "shop_entry_bind_skipped", "shop_entry_bind_failed",
      "shop_leave_persisted", "shop_leave_persist_failed",
      "human_annotation_created",
      "test_run_started", "test_run_stopped", "resync_required",
    ];
    types.forEach((type) => app.eventSource.addEventListener(type, receiveEvent));
  }

  function receiveEvent(message) {
    if (message.type === "resync_required") {
      refreshState();
      return;
    }
    let event;
    try { event = JSON.parse(message.data); } catch (_) { return; }
    appendTimeline(event);
    if ([
      "test_run_started", "test_run_stopped", "customer_binding_changed",
      "entry_accepted", "leave_accepted", "track_appeared",
      "track_disappeared", "shop_entry_bound", "shop_entry_bind_skipped",
      "shop_entry_bind_failed", "shop_leave_persisted", "shop_leave_persist_failed",
    ].includes(event.eventType)) {
      refreshState();
    }
  }

  function appendTimeline(event) {
    mergeTimeline([event]);
  }

  function mergeTimeline(events) {
    const byId = new Map(
      app.timeline.map((event) => [event.eventId, event])
    );
    events.forEach((event) => {
      const annotationType = event.payload?.annotationType;
      if (
        event.eventType === "human_annotation_created"
        && ["system_event_correct", "system_event_incorrect"].includes(annotationType)
      ) {
        const systemEventId = event.payload?.payload?.systemEventId;
        if (Number.isInteger(systemEventId)) {
          app.eventFeedback.set(
            systemEventId,
            annotationType === "system_event_correct" ? "correct" : "wrong"
          );
        }
      }
      byId.set(event.eventId, event);
    });
    app.timeline = [...byId.values()]
      .sort((left, right) => right.eventId - left.eventId)
      .slice(0, 250);
    renderTimeline();
  }

  function renderTimeline() {
    const container = el("timeline");
    container.replaceChildren();
    const monitored = app.timeline.filter(
      (event) => MONITORED_EVENT_TYPES.has(event.eventType)
    );
    if (!monitored.length) {
      const empty = document.createElement("p");
      empty.className = "timeline-empty";
      empty.textContent = "Waiting for the next shop event.";
      container.append(empty);
      return;
    }
    monitored.forEach((item) => {
      const row = document.createElement("div");
      row.className = `event-card ${eventCardClass(item.eventType)}`;
      const heading = document.createElement("div");
      heading.className = "event-card-heading";
      const kind = document.createElement("strong");
      kind.textContent = eventTitle(item);
      const time = document.createElement("time");
      time.textContent = new Date(item.occurredAtUnixMilliseconds).toLocaleTimeString();
      const summary = document.createElement("span");
      summary.className = "event-card-summary";
      summary.textContent = eventSummary(item);
      heading.append(kind, time);
      const actions = document.createElement("div");
      actions.className = "event-verdict-actions";
      const feedback = app.eventFeedback.get(item.eventId);
      if (!VERIFIABLE_EVENT_TYPES.has(item.eventType)) {
        const result = document.createElement("strong");
        const successful = ["shop_entry_bound", "shop_leave_persisted"].includes(item.eventType);
        const skipped = item.eventType === "shop_entry_bind_skipped";
        result.className = `event-verdict ${successful ? "correct" : "wrong"}`;
        result.textContent = successful
          ? "Database confirmed"
          : skipped ? "Customer not bound" : "Database update failed";
        actions.append(result);
      } else if (feedback) {
        const result = document.createElement("strong");
        result.className = `event-verdict ${feedback}`;
        result.textContent = feedback === "correct" ? "Confirmed correct" : "Marked wrong";
        actions.append(result);
      } else {
        const correct = document.createElement("button");
        correct.type = "button";
        correct.className = "button event-correct";
        correct.textContent = "Correct";
        correct.disabled = !app.state?.activeRun;
        correct.addEventListener("click", () => annotateEventFeedback(item, true));
        const wrong = document.createElement("button");
        wrong.type = "button";
        wrong.className = "button danger-soft event-wrong";
        wrong.textContent = "Wrong";
        wrong.disabled = !app.state?.activeRun;
        wrong.addEventListener("click", () => annotateEventFeedback(item, false));
        actions.append(correct, wrong);
      }
      row.append(heading, summary, actions);
      container.append(row);
    });
  }

  async function annotateEventFeedback(event, correct) {
    const response = await annotate(
      correct ? "system_event_correct" : "system_event_incorrect",
      {
        systemEventId: event.eventId,
        systemEventType: event.eventType,
        systemEventVisitId: event.visitId,
        systemEventPayload: event.payload || {},
      }
    );
    if (!response) return;
    app.eventFeedback.set(event.eventId, correct ? "correct" : "wrong");
    renderTimeline();
  }

  function eventCardClass(eventType) {
    if (eventType === "entry_accepted") return "entry";
    if (["leave_accepted", "shop_leave_persist_failed", "shop_entry_bind_failed"].includes(eventType)) return "leave";
    if (["shop_entry_bound", "shop_leave_persisted"].includes(eventType)) return "entry";
    return "transition";
  }

  function eventTitle(event) {
    if (event.eventType === "entry_accepted") return "ENTRY";
    if (event.eventType === "leave_accepted") return "LEAVE";
    if (event.eventType === "shop_entry_bound") return "SERVER ENTRY BOUND";
    if (event.eventType === "shop_entry_bind_skipped") return "SERVER ENTRY NOT BOUND";
    if (event.eventType === "shop_entry_bind_failed") return "SERVER ENTRY FAILED";
    if (event.eventType === "shop_leave_persisted") return "SERVER DEPARTURE SAVED";
    if (event.eventType === "shop_leave_persist_failed") return "SERVER DEPARTURE FAILED";
    return "TRANSITION";
  }

  function eventSummary(event) {
    const payload = event.payload || {};
    switch (event.eventType) {
      case "entry_accepted": return `Visit ${event.visitId} entered on camera ${event.cameraIndex + 1}.`;
      case "leave_accepted": return `Visit ${event.visitId} left on camera ${event.cameraIndex + 1}.`;
      case "shop_entry_bound": return `Visit ${event.visitId} was bound to customer ${payload.customerId}.`;
      case "shop_entry_bind_skipped": return `Visit ${event.visitId} was not bound: ${formatReason(payload.reason)}.`;
      case "shop_entry_bind_failed": return `Visit ${event.visitId} binding failed: ${payload.error ?? formatReason(payload.reason)}.`;
      case "shop_leave_persisted": return `Visit ${event.visitId} · customer ${payload.customerId ?? "unknown"} · shopLeftAt ${formatServerTime(payload.shopLeftAt)}.`;
      case "shop_leave_persist_failed": return `Visit ${event.visitId} was not marked left: ${payload.error ?? "unknown server error"}`;
      case "track_appeared": return `Track ${event.trackId} appeared · visit ${event.visitId ?? "unassigned"}`;
      case "track_disappeared": return `Track ${event.trackId} disappeared`;
      case "visit_assignment_changed": return `Track ${event.trackId} moved to visit ${event.visitId ?? "unassigned"}`;
      case "customer_binding_changed": return `Visit ${event.visitId} customer ${payload.customerId ?? payload.previousCustomerId ?? "changed"}`;
      case "human_annotation_created": return String(payload.annotationType || "Human annotation").replaceAll("_", " ");
      default: return event.eventType.replaceAll("_", " ");
    }
  }

  function formatServerTime(value) {
    if (!value) return "missing";
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString();
  }

  function formatReason(value) {
    return String(value || "unknown reason").replaceAll("_", " ");
  }

  function updateSystemAnswer() {
    const observations = app.observationPayload?.observations || [];
    if (!app.observationPayload?.frame) {
      el("system-answer").textContent = "The selected camera has not published a processed frame.";
      return;
    }
    if (!observations.length) {
      el("system-answer").textContent = `Camera ${app.selectedCamera + 1} currently sees no people.`;
      return;
    }
    const descriptions = observations.map(
      (person) => `track ${person.trackId}, visit ${person.visitId ?? "unassigned"}, customer ${person.customerId ?? person.customerBindingStatus}`
    );
    el("system-answer").textContent = `Camera ${app.selectedCamera + 1} sees ${observations.length} ${observations.length === 1 ? "person" : "people"}: ${descriptions.join("; ")}.`;
  }

  function renderReport() {
    if (!app.report) return;
    const summary = app.report.summary;
    el("result-summary").textContent = `${app.report.status} · ${summary.pass} pass / ${summary.fail} fail / ${summary.pending} pending`;
  }

  let toastTimer;
  function showToast(message) {
    const toast = el("toast");
    toast.textContent = message;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("show"), 3500);
  }

  window.addEventListener("resize", drawOverlay);
  bootstrap();
})();
