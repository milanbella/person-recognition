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
    pollTimer: null,
    runTimer: null,
    token: localStorage.getItem("shopOperatorToken") || "",
    voice: {
      peerConnection: null,
      localStream: null,
      dataChannel: null,
      sessionId: null,
      statusTimer: null,
      muted: false,
      starting: false,
    },
  };

  const el = (id) => document.getElementById(id);
  const annotationButtons = [...document.querySelectorAll("[data-annotation]")];
  const physicalButtons = [...document.querySelectorAll("[data-physical]")];
  const OBSERVATION_GRACE_MILLISECONDS = 2500;
  const MONITORED_EVENT_TYPES = new Set([
    "entry_accepted",
    "leave_accepted",
    "shelf_approach",
    "shelf_departure",
  ]);

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
    renderShelves();
    mergeTimeline(app.state?.recentEvents || []);
    setActionAvailability();
  }

  function renderRun() {
    const run = app.state?.activeRun;
    el("run-name").textContent = run ? `${run.scenario} · ${run.runId}` : "No active test";
    el("start-run").disabled = Boolean(run);
    el("analyze-run").disabled = !run;
    el("stop-run").disabled = !run;
    el("start-voice").disabled = !run || app.voice.starting || Boolean(app.voice.peerConnection);
    if (!run && app.voice.peerConnection) void disconnectVoice(true, "Test run ended");
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

  function renderShelves() {
    const select = el("missing-shelf");
    const current = select.value;
    const shelves = app.state?.shelves || [];
    const shelfIds = [...new Set(shelves.map((shelf) => shelf.shelfId))]
      .sort((left, right) => left - right);
    select.replaceChildren();
    shelfIds.forEach((shelfId) => {
      const option = document.createElement("option");
      option.value = String(shelfId);
      option.textContent = `Shelf ${shelfId}`;
      select.append(option);
    });
    if (shelfIds.some((shelfId) => String(shelfId) === current)) {
      select.value = current;
    }
    select.disabled = !shelfIds.length;
  }

  function selectedSubjectId() {
    return el("subject-select").value || null;
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
    el("shelf-approach").disabled = !active;
    el("shelf-departure").disabled = !active;
    el("add-note").disabled = !active;
  }

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
      if (annotationType === "physical_entry" || annotationType === "physical_leave") {
        await refreshState();
      }
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

  el("shelf-approach").addEventListener("click", () => markShelf("shelf_approach"));
  el("shelf-departure").addEventListener("click", () => markShelf("shelf_departure"));
  function markShelf(type) {
    const shelfId = Number.parseInt(el("missing-shelf").value, 10);
    if (!Number.isInteger(shelfId)) return showToast("Shelf ID must be an integer.");
    annotate(type, {shelfId});
  }

  el("add-note").addEventListener("click", () => {
    const input = el("note-text");
    const note = input.value.trim();
    if (!note) return;
    annotate("note", {note});
    input.value = "";
  });

  el("start-run").addEventListener("click", () => el("start-dialog").showModal());
  el("cancel-start").addEventListener("click", () => el("start-dialog").close());
  el("start-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    app.token = String(form.get("token") || "").trim();
    if (app.token) localStorage.setItem("shopOperatorToken", app.token);
    const expected = String(form.get("expectedCustomerId") || "").trim();
    const payload = {
      scenario: String(form.get("scenario")),
      verifier: String(form.get("verifier")),
      subjects: [{
        subjectId: String(form.get("subjectId")),
        displayName: String(form.get("displayName")),
        expectedCustomerId: expected || null,
      }],
      notes: String(form.get("notes") || ""),
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
      await disconnectVoice(true, "Test run stopped");
      const response = await api(`/operator/api/test-runs/${run.runId}/stop`, {method: "POST"});
      app.report = response.report;
      renderReport();
      await refreshState();
      showToast("Test stopped and exported.");
    } catch (error) {
      showToast(error.message);
    }
  });

  el("analyze-run").addEventListener("click", async () => {
    const run = app.state?.activeRun;
    if (!run) return;
    try {
      app.report = await api(`/operator/api/test-runs/${run.runId}/analyze`, {method: "POST"});
      renderReport();
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
      "shelf_approach", "shelf_departure", "human_annotation_created",
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
    if (shouldSpeakEvent(event)) speak(eventSummary(event));
    if ([
      "test_run_started", "test_run_stopped", "customer_binding_changed",
      "entry_accepted", "leave_accepted", "track_appeared",
      "track_disappeared", "shelf_approach", "shelf_departure",
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
      if (feedback) {
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
    if (eventType === "leave_accepted") return "leave";
    return "shelf";
  }

  function eventTitle(event) {
    if (event.eventType === "entry_accepted") return "ENTRY";
    if (event.eventType === "leave_accepted") return "LEAVE";
    if (event.eventType === "shelf_approach") return "SHELF APPROACH";
    return "SHELF LEAVE";
  }

  function eventSummary(event) {
    const payload = event.payload || {};
    switch (event.eventType) {
      case "entry_accepted": return `Visit ${event.visitId} entered on camera ${event.cameraIndex + 1}.`;
      case "leave_accepted": return `Visit ${event.visitId} left on camera ${event.cameraIndex + 1}.`;
      case "track_appeared": return `Track ${event.trackId} appeared · visit ${event.visitId ?? "unassigned"}`;
      case "track_disappeared": return `Track ${event.trackId} disappeared`;
      case "visit_assignment_changed": return `Track ${event.trackId} moved to visit ${event.visitId ?? "unassigned"}`;
      case "customer_binding_changed": return `Visit ${event.visitId} customer ${payload.customerId ?? payload.previousCustomerId ?? "changed"}`;
      case "shelf_approach": return `Visit ${event.visitId} approached shelf ${payload.shelfId}.`;
      case "shelf_departure": return `Visit ${event.visitId} left shelf ${payload.shelfId}.`;
      case "human_annotation_created": return String(payload.annotationType || "Human annotation").replaceAll("_", " ");
      default: return event.eventType.replaceAll("_", " ");
    }
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

  el("speak-state").addEventListener("click", () => speak(el("system-answer").textContent));
  function shouldSpeakEvent(event) {
    return !app.voice.peerConnection && el("speak-events").checked && [
      "visit_assignment_changed", "customer_binding_changed", "entry_accepted",
      "leave_accepted", "shelf_approach", "shelf_departure",
    ].includes(event.eventType);
  }
  function speak(text) {
    if (!("speechSynthesis" in window)) return showToast("Speech output is not supported.");
    speechSynthesis.cancel();
    speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  }

  el("start-voice").addEventListener("click", startRealtimeVoice);
  el("mute-voice").addEventListener("click", toggleVoiceMute);
  el("disconnect-voice").addEventListener("click", () => disconnectVoice(true));

  async function startRealtimeVoice() {
    if (!app.state?.activeRun) return showToast("Start a test run first.");
    if (!app.token) return showToast("Enter the operator token when starting the test run.");
    if (!window.isSecureContext) {
      return showToast("Microphone access requires HTTPS.");
    }
    if (!navigator.mediaDevices?.getUserMedia || !window.RTCPeerConnection) {
      return showToast("This browser does not support WebRTC microphone sessions.");
    }
    app.voice.starting = true;
    setVoiceStatus("starting", "Requesting microphone access");
    renderRun();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {echoCancellation: true, noiseSuppression: true, autoGainControl: true},
      });
      const peerConnection = new RTCPeerConnection();
      app.voice.localStream = stream;
      app.voice.peerConnection = peerConnection;
      stream.getAudioTracks().forEach((track) => peerConnection.addTrack(track, stream));
      peerConnection.ontrack = (event) => {
        const audio = el("voice-audio");
        audio.srcObject = event.streams[0];
        void audio.play().catch(() => {
          setVoiceStatus("connected", "Tap Start voice test again to enable audio playback");
        });
      };
      peerConnection.onconnectionstatechange = () => {
        const state = peerConnection.connectionState;
        if (state === "connected") setVoiceStatus("connected", "Microphone active · listening for shop events");
        if (["failed", "closed", "disconnected"].includes(state) && app.voice.peerConnection) {
          void disconnectVoice(false, `WebRTC ${state}`);
        }
      };
      const dataChannel = peerConnection.createDataChannel("oai-events");
      app.voice.dataChannel = dataChannel;
      dataChannel.addEventListener("message", receiveRealtimeBrowserEvent);
      dataChannel.addEventListener("open", () => {
        setVoiceStatus("connected", "Microphone active · listening for shop events");
      });
      dataChannel.addEventListener("close", () => {
        if (app.voice.peerConnection) setVoiceStatus("error", "Realtime control channel closed");
      });

      const offer = await peerConnection.createOffer();
      await peerConnection.setLocalDescription(offer);
      const response = await fetch("/operator/voice/sessions", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${app.token}`,
          "Content-Type": "application/sdp",
          "X-Operator-Id": app.state.activeRun.verifier || "mobile-operator",
        },
        body: offer.sdp,
      });
      if (!response.ok) throw new Error(await voiceHttpError(response));
      app.voice.sessionId = response.headers.get("X-Voice-Session-Id");
      if (!app.voice.sessionId) throw new Error("Voice service returned no session ID.");
      await peerConnection.setRemoteDescription({type: "answer", sdp: await response.text()});
      app.voice.statusTimer = setInterval(pollVoiceStatus, 2000);
      setVoiceStatus("connected", "Connecting audio and server controls");
    } catch (error) {
      await disconnectVoice(false, error.message);
      showToast(error.message);
    } finally {
      app.voice.starting = false;
      renderRun();
    }
  }

  async function voiceHttpError(response) {
    try {
      const payload = await response.json();
      return payload.detail || `${response.status} ${response.statusText}`;
    } catch (_) {
      return `${response.status} ${response.statusText}`;
    }
  }

  function receiveRealtimeBrowserEvent(message) {
    let event;
    try { event = JSON.parse(message.data); } catch (_) { return; }
    if (event.type === "input_audio_buffer.speech_started") {
      setVoiceStatus("listening", "Hearing you…");
    } else if (event.type === "input_audio_buffer.speech_stopped") {
      setVoiceStatus("thinking", "Checking your feedback…");
    } else if (event.type === "response.output_audio_transcript.done" && event.transcript) {
      setVoiceStatus("connected", event.transcript);
    } else if (event.type === "error") {
      setVoiceStatus("error", event.error?.message || "Realtime API error");
    }
  }

  async function pollVoiceStatus() {
    if (!app.voice.sessionId) return;
    try {
      const status = await api(`/operator/voice/sessions/${app.voice.sessionId}`);
      if (status.status === "ended") {
        await disconnectVoice(false, status.disconnectReason || "Voice session ended");
      }
    } catch (error) {
      setVoiceStatus("error", error.message);
    }
  }

  function toggleVoiceMute() {
    const track = app.voice.localStream?.getAudioTracks()[0];
    if (!track) return;
    app.voice.muted = !app.voice.muted;
    track.enabled = !app.voice.muted;
    el("mute-voice").textContent = app.voice.muted ? "Unmute" : "Mute";
    setVoiceStatus(
      app.voice.muted ? "muted" : "connected",
      app.voice.muted ? "Microphone muted" : "Microphone active · listening for shop events"
    );
  }

  async function disconnectVoice(notifyServer = true, detail = "Voice test is off") {
    const sessionId = app.voice.sessionId;
    app.voice.sessionId = null;
    if (app.voice.statusTimer) clearInterval(app.voice.statusTimer);
    app.voice.statusTimer = null;
    app.voice.dataChannel?.close();
    app.voice.peerConnection?.close();
    app.voice.localStream?.getTracks().forEach((track) => track.stop());
    el("voice-audio").srcObject = null;
    app.voice.peerConnection = null;
    app.voice.localStream = null;
    app.voice.dataChannel = null;
    app.voice.muted = false;
    el("mute-voice").textContent = "Mute";
    setVoiceStatus("disconnected", detail);
    renderRun();
    if (notifyServer && sessionId) {
      try {
        await api(`/operator/voice/sessions/${sessionId}`, {method: "DELETE"});
      } catch (_) {
        // Closing the peer connection still terminates browser audio immediately.
      }
    }
  }

  function setVoiceStatus(state, detail) {
    el("voice-bar").className = `voice-bar ${state}`;
    const labels = {
      disconnected: "Voice test is off",
      starting: "Starting voice test",
      connected: "Voice test connected",
      listening: "Listening",
      thinking: "Processing feedback",
      muted: "Voice test muted",
      error: "Voice unavailable",
    };
    el("voice-status").textContent = labels[state] || "Voice test";
    el("voice-detail").textContent = detail;
    const active = Boolean(app.voice.peerConnection);
    el("mute-voice").disabled = !active;
    el("disconnect-voice").disabled = !active;
    el("start-voice").disabled = !app.state?.activeRun || active || app.voice.starting;
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
  window.addEventListener("beforeunload", () => {
    app.voice.localStream?.getTracks().forEach((track) => track.stop());
    app.voice.peerConnection?.close();
  });
  bootstrap();
})();
