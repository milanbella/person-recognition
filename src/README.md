# Source Reset

This source folder has been reset for a new architecture:

- OAK camera as USB image source
- host PC GPU for detection, tracking, entrance logic, and recognition

The old on-device `RVC2` experiment scripts were intentionally removed.

## Current Baseline

- `pipeline/`
  - shared runtime modules for the host-side pipeline
  - phase scripts and the future final pipeline should import these modules instead of owning duplicate logic
  - current extracted modules include:
    - `pipeline.config`
      - shared default paths and parameter values
      - harness scripts should reference these defaults so tuning changes have one home
    - `pipeline.camera`
      - shared OAK device discovery and explicit device selection helpers
      - provides `--device-id` / `--list-devices` support for live scripts
    - `pipeline.detection`
      - generic person detector protocol/factory, ONNX YOLO detector wrapper, detection arg helpers, and drawing helpers
      - YOLO is the person detector backend and `../models/yolo26n.onnx` is the default model
      - comparison models are `../models/yolo11n.onnx` and `../models/yolo11s.onnx`; COCO class id `0` is `person`
    - `pipeline.detectors`
      - detector adapter API re-exports for future detector backends
    - `pipeline.tracking`
      - generic person tracker protocol/factory, current IoU tracking logic, tracking arg helpers, and drawing helpers
      - current tracker backend is `iou`; use `--tracker-backend iou`
    - `pipeline.entrance`
      - entrance-line state, crossing logic, and debug drawing helpers
    - `pipeline.evidence`
      - evidence-crop buffering and event capture helpers
    - `pipeline.embedding`
      - offline face-embedding extraction from saved evidence event folders
      - shared by replay entrypoints and the legacy Phase 5 harness
    - `pipeline.review`
      - embedding-similarity review math and HTML generation
      - shared by replay entrypoints and the legacy Phase 6 harnesses
    - `pipeline.identity`
      - local identity grouping and HTML review logic
      - shared by replay entrypoints and the legacy Phase 7 harnesses
    - `pipeline.face_identity`
      - generic face recognizer protocol/factory and current InsightFace/ArcFace replay-local identity assignment
      - current face backend is `insightface`; use `--face-backend insightface`
      - attaches observed `face_person_###` labels to tracked faces during RGBD replay
      - face recognition is limited to `NEW` / `TRACKED` person tracks above the minimum face-track size thresholds
    - `pipeline.body_evidence`
      - generic per-track body evidence extractor protocol/factory
      - current body backend is `hsv`; use `--body-backend hsv`
      - wraps the existing upper/lower clothing HSV histogram evidence used by visit matching
    - `pipeline.aruco_markers`
      - OpenCV ArUco marker detection and drawing helpers
      - current door-marker prototype defaults to `DICT_4X4_50` and door marker IDs `0`, `1`, `2`, `3`
    - `pipeline.visit_identity`
      - within-visit physical-person identity layer above temporary `track_id`
      - reattaches new track ids to existing `visit_id` values using clothing/body appearance, depth, and recent timing
      - default visit match threshold is `0.45`
      - treats fragmented `face_person_###` labels as evidence attached to a visit, not as the only visit identity source
      - frame overlays show `V#_E` for entrance-confirmed visits and `V#_O` for observer-only visits
    - `pipeline.visit_registry`
      - shop-wide active visit registry for synchronized multi-camera replay
      - merges entrance-camera plane events into one `entrance_confirmed` visit by timestamp window
      - lets observer cameras attach to entrance-confirmed visits or create `observer_only` visits when no match is found
      - supports temporal entrance-to-observer handoff for new observer tracks that appear shortly after an entrance event
      - builds visit observations from normalized evidence, not raw RGB frames
      - uses `FrameEvidence` for one camera frame and `TrackVisitEvidence` for one track's visit-matching evidence
- `pipeline.entry_session`
  - typed entry-event/session building, offline correlation, and HTML review helpers
  - shared by replay entrypoints and the legacy Phase 8 harnesses
    - `pipeline.depth`
      - shared helpers for sampling aligned stereo depth inside tracked person boxes
      - includes depth-threshold and calibrated-plane entrance trigger modes
      - calibrated plane mode is the default; threshold mode remains available for fallback/debug
      - depth trigger functions return `DepthEntranceResult` with `entered_track_ids`, `exited_track_ids`, `depth_samples`, and `signed_distances_mm`

- `main.py`
  - host-side camera capture baseline
  - connects to the OAK camera with `depthai`
  - receives RGB frames on the PC
  - shows a live preview

- `detect_door_aruco.py`
  - live 4K RGB prototype for detecting OpenCV ArUco markers around the entrance door
  - defaults to `DICT_4X4_50` and highlights door marker IDs `0`, `1`, `2`, `3`
  - supports explicit OAK selection with `--device-id`
  - intended for marker visibility/prototyping only; it does not fit or save entrance planes yet

- `record_rgbd_stream.py`
  - host-side RGB plus aligned depth recorder for one OAK camera
  - writes an `oak_<device-id>.rgbd\` folder with `rgb.avi`, `frames.jsonl`, and 16-bit depth PNG frames
  - intended for later depth-based replay and tuning

- `replay_synced_rgbd_streams.py`
  - replays multiple recorded RGBD streams in sync using recorded RGB frame timestamps
  - shows synchronized tiled RGB views; depth view is hidden by default and can be enabled with `--show-depth-window`
  - accepts one or more `--device-id` values and derives the matching RGBD recording folders
  - defaults to calibrated plane-trigger mode
  - runs replay-local face identity assignment by default; disable it with `--disable-face-recognition`
  - assigns shared registry-owned `visit_id` labels across the synchronized replay
  - defaults every stream to `--camera-role entrance`
  - supports `--camera-role entrance_observer` for entrance cameras that should also contribute observer evidence
  - supports `--camera-role observer` for in-shop observer streams
  - observer-only streams do not require plane calibration and never emit entrance events
  - supports temporal entrance-to-observer handoff via `--observer-handoff-*` tuning flags
  - when exactly one eligible entrance-confirmed visit exists, observer matching uses a `0.25` fallback threshold; otherwise the normal threshold remains unchanged
  - unmatched visible observer tracks remain provisional for 3 seconds and retry matching before an `observer_only` visit is created
  - supports `--log-plane-trace` for per-frame plane signed-distance debugging on entrance-capable cameras
  - enables plane track-split recovery by default to recover entry/leave events when a tracker split happens exactly at the entrance plane; disable it with `--disable-plane-track-split-recovery`
  - supports `--output-dir` for replay artifacts: visit decisions, track visit evidence, entrance/leave plane-crossing events, and final visit summaries

- `live_synced_rgbd_streams.py`
  - live multi-OAK RGBD pipeline equivalent to synchronized replay
  - opens all `--device-id` cameras at once and pairs current raw and processing RGB outputs by exact sequence number
  - `--frame-width` and `--frame-height` configure current raw RGB capture, MJPEG streaming, GUI output, and observer API coordinates; defaults are `1920x1080`
  - `--processing-width` and `--processing-height` configure a separate current RGB output plus aligned stereo depth; defaults are `1280x720`
  - enables StereoDepth's `7x7` on-device median filter by default; use `--depth-median-filter off`, `3x3`, or `5x5` to change spatial smoothing, while host track-ROI depth sampling continues to reject spatial outliers
  - runs YOLO/tracking on current `1280x720` RGB, maps tracks to the matching full-resolution frame, and runs InsightFace only on the full-resolution person crops
  - never puts full-resolution frames in the depth-delay buffer; it retains only the latest frame for raw GUI/MJPEG output, while `--processing-buffer-seconds` retains processing frames plus track, face, and body evidence snapshots
  - attaches delayed depth to cached evidence by exact sequence when counters are shared, otherwise by nearest host-synchronized capture timestamp; differences above `--max-rgb-depth-delta-ms` (default `250`) are rejected
  - GUI preview shows the delayed synchronized tracking overlay by default; use `--show-raw-preview` for the current raw RGB view
  - YOLO inference remains independently configured at `640x384`, maps detections to the `1280x720` processing frame, and then scales tracks to the configured RGB resolution
  - uses one shared `VisitRegistry` across all live streams
  - supports the same `--camera-role entrance`, `entrance_observer`, and `observer` behavior as synced replay
  - observer-only streams do not require plane calibration and never emit entrance/leave events
  - runs face recognition on every eligible processed track frame and accumulates recognized face identity IDs as visit evidence; disable it with `--disable-face-recognition`
  - supports `--observer-single-active-fallback-threshold` and `--observer-provisional-seconds` for conservative delayed observer-only visit creation
  - supports optional shop API integration:
    - entry binds `visit_id` to the latest recent unbound `ShoppingCustomer`
    - leave marks the matching `ShoppingCustomer` as left by `visitId`
  - persists operational live visit state to `state/shop_state.sqlite` by default; override with `--state-db`
  - supports opt-in aggregated profiling with `--log-performance`; `LIVE_PERF` reports camera polling, RGB conversion, YOLO, tracking, depth, face, body, registry/I/O, overlays, depth colorization, MJPEG publication, GUI, and total-cycle timings
  - supports `--log-plane-trace` for per-frame plane signed-distance debugging on entrance-capable cameras
  - enables plane track-split recovery by default to recover entry/leave events when a tracker split happens exactly at the entrance plane; disable it with `--disable-plane-track-split-recovery`
  - supports `--output-dir` for live artifacts: visit decisions, track visit evidence, entrance/leave plane-crossing events, live config, and final visit summaries
  - optionally serves a mobile shop-test console at `/operator/` on the configured streaming port
  - operator runs, physical ground-truth annotations, system transition events, and analysis results persist in the same `--state-db`
  - exported run artifacts are written below `--operator-runs-root` (default `test-runs`)
  - enable it explicitly with `--enable-operator-console --operator-api-token <secret>`; it is disabled by default
  - the operator console reuses raw MJPEG plus observer JSON and draws selectable person boxes in the browser; it does not run another detector
  - accepted entry/leave, visit assignment, customer binding, track, and camera transitions are available through `/operator/api/events`; current shelf position is exposed through world state
  - continuously materializes system-believed shop state in the same SQLite DB and exposes it through `/world-state`; this read API is available whenever MJPEG streaming is enabled, even when the operator console is disabled
  - optionally runs `../models/best.onnx` asynchronously on expanded full-resolution person crops with `--enable-product-recognition`; fixed batch-three inference never blocks person tracking and stale crop jobs are replaced by newer ones

## System-Believed World State

The live service maintains a revisioned projection of cameras, tracks, visits,
customers, and shelf position. It deliberately represents what the system
believes; human physical ground truth remains in operator annotations.

For every visit with fresh shelf observations, `shelfPosition` is the shelf
marker with the smallest measured 3D distance across all cameras. Shelf
approach/departure thresholds and events are not used for this position.

Read the complete current projection:

```bash
curl -s http://127.0.0.1:8002/world-state | jq
```

Entity endpoints:

```text
GET /world-state/visits/{visit_id}
GET /world-state/visits/{visit_id}/shelf-position
GET /world-state/visits/{visit_id}/products
GET /world-state/visits/{visit_id}/product-crop.jpg
GET /world-state/visits/{visit_id}/product-observations
GET /world-state/visits/{visit_id}/product-observations/{camera_index}/crop.jpg
GET /world-state/shelves/{shelf_id}
GET /world-state/cameras/{camera_index}
GET /world-state/revisions/{revision}
```

Query only the current shelf position for a visit:

```bash
curl -s http://127.0.0.1:8002/world-state/visits/1/shelf-position | jq
```

The response includes the selected `position` plus `measurements` for every
camera/marker distance that participated in the decision. Camera indexes in the
API are zero-based; the operator console displays them as Camera 1, Camera 2,
and so on.

Enable product recognition and query the latest result associated with a
visit:

```bash
python ./live_synced_rgbd_streams.py ... \
  --enable-product-recognition \
  --log-product-recognition

curl -s http://127.0.0.1:8002/world-state/visits/1/products | jq
```

Product inference uses the synchronized raw RGB image, expands each active
person rectangle by 30%, letterboxes that crop to the model's fixed
`1280x1280` input, and batches up to three crops. The result means a product
was recognized near the person; it does not by itself prove the product is
physically held.

For detector debugging, disable person cropping and scan every complete raw
camera frame:

```bash
python ./live_synced_rgbd_streams.py ... \
  --enable-product-recognition \
  --product-full-frame \
  --log-product-recognition
```

Full-frame results appear in the operator console's per-camera product cards,
but are deliberately not promoted to a visit's held-product belief.

Current rows are checkpointed asynchronously in `state/shop_state.sqlite`.
Observation writes are coalesced so camera processing does not wait for SQLite.
After restart, visit lifecycle is restored while camera, track, and shelf
measurements remain explicitly stale until fresh observations arrive.

With an active operator run, query one physical test subject:

```bash
curl -s \
  http://127.0.0.1:8002/operator/api/test-runs/RUN_ID/subjects/milan/world-state \
  | jq
```

The response resolves the subject to a confirmed visit, a single proposed
candidate, an ambiguous candidate set, or unknown. Use
`captureQuery=true` only when creating a confirmation/correction; it stores an
opaque snapshot reference so feedback is validated against the exact answer.

## Shop Test Operator Console

The console is disabled by default. Enable it on the MJPEG API port with an
explicit bearer token:

```bash
python ./live_synced_rgbd_streams.py \
  ... \
  --enable-operator-console \
  --operator-api-token 'replace-with-a-long-random-token'
```

The console is then available at:

```text
http://<shop-server>:8002/operator/
```

The browser asks for the same token when starting a run. All mutation endpoints
require `Authorization: Bearer <token>`. When `--enable-operator-console` is
omitted, the operator page and API routes are not registered.

## Realtime Voice Shop Testing

`shop_voice_agent.py` is a separate companion process for hands-free physical
verification. Voice testing is query-driven by default: walk normally and ask
questions such as “what is my visit ID?” or “what shelf am I nearest?”. The
agent reads revisioned world state and records subsequent confirmations or
corrections against the exact queried snapshot. It does not announce every
generated event. Add `--announce-major-events` if ENTRY and LEAVE announcements
are also wanted.

The phone sends microphone audio directly to the OpenAI Realtime
API over WebRTC. The permanent OpenAI key remains on the shop server, while a
server-side sideband connection validates and executes fixed feedback tools
through the existing operator annotation API.

Prerequisites:

- enable the operator console and start a test run before voice mode
- serve `/operator/` over HTTPS because mobile browsers require a secure origin
  for microphone access
- install `requirements.txt`, including the `websockets` sideband client
- use the same operator token for the recognition and voice services
- store the OpenAI API key only on the server

Development startup:

```bash
export OPENAI_API_KEY='replace-with-project-api-key'
python ./shop_voice_agent.py \
  --operator-api-base-url http://127.0.0.1:8002 \
  --operator-api-token 'same-token-as-person-recognition-live' \
  --state-db state/shop_state.sqlite
```

Production should use credential files instead of command-line secrets. An
example unit using systemd credentials is provided at
`deploy/person-recognition-voice.service.example`.

Route the browser-facing voice namespace to the companion service on port
`8003`; leave the rest of the operator console routed to port `8002`:

```nginx
location /operator/voice/ {
    proxy_pass http://127.0.0.1:8003/operator/voice/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

Health check:

```bash
curl http://127.0.0.1:8003/operator/voice/health
```

The voice service is opt-in. If it stops or OpenAI is unavailable, camera
processing, streaming, event persistence, and manual feedback on port `8002`
continue normally. Voice audit sessions and tool calls are stored in the same
SQLite database; raw audio is never stored. Short transcripts can be disabled
with `--disable-transcript-retention`, which also disables the separate input
transcription model and its associated usage.

- `replay_depth_tuner.py`
  - replays one recorded RGBD stream through detection, tracking, and depth-based entrance logic
  - writes replayed depth entrance/leave event timing logs from recorded timestamps and aligned recorded depth
  - defaults to calibrated plane-trigger mode; `--depth-trigger-mode threshold` remains available for fallback/debug
  - accepts `--camera-role` as a single-stream debug label; full entrance/observer role resolution lives in `replay_synced_rgbd_streams.py`
  - supports `--log-visit-decisions` for single-stream `visit_id` creation/matching debug output
  - uses `observer_only` for unconfirmed local visit hypotheses and promotes the visit to `entrance_confirmed` when a plane entry event fires
  - runs replay-local face identity assignment by default; disable it with `--disable-face-recognition`
  - skips face assignment for `LOST` / `REMOVED` tracks and tracks below `--face-min-track-width-px` / `--face-min-track-height-px`
  - writes `visit_id` and attached face identity ids into depth event logs

- `fit_plane_from_tags.py`
  - interactive plane-calibration utility for recorded RGBD streams
  - lets you click 3 tagged door-corner points, fits a 3D plane from recorded depth, and prints the CLI args for plane-based entrance detection

- `fit_plane_from_aruco.py`
  - automatic recorded-RGBD plane-calibration utility for entrance door ArUco markers
  - defaults to `DICT_4X4_50` marker IDs `0`, `1`, `2`, `3`
  - samples aligned depth at detected marker centers, fits a plane from at least 3 valid marker points, and saves `plane_fit_<device-id>.json`
  - keeps `fit_plane_from_tags.py` available as the manual fallback

- `final_pipeline.py`
  - first unified live pipeline entrypoint built on shared modules
  - runs host-side detection, tracking, entrance logic, and optional evidence capture
  - demonstrates how the eventual live pipeline should depend on shared modules instead of phase harness imports
  - supports explicit OAK selection with `--device-id`

- `depth_entrance_live.py`
  - first live prototype for depth-based entrance triggering
  - uses CAM_A RGB plus CAM_B/C stereo depth aligned to RGB
  - samples depth near the lower body and emits `DEPTH_ENTRY_EVENT` / `DEPTH_LEAVE_EVENT` when a tracked person crosses the configured depth/plane trigger

- `phase1_host_detection_scrfd.py`
  - legacy-named host-side Step 2 detector harness
  - uses the shared YOLO ONNX detector
  - reads OAK USB frames and draws person detections on the host
  - prefers CUDA when available and falls back to CPU when the local GPU runtime is incomplete

## Model Provenance

- default detector model:
  - `C:\wi\luxonis\person-recognition\models\yolo26n.onnx`
- source:
  - Ultralytics YOLO26n checkpoint: `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt`
  - exported with Ultralytics `8.4.92`, ONNX opset 20, `end2end=False`, and no embedded NMS
- local SHA256:
  - `354cdf0ec4144d797f4ef77ecdb8227303d0c3122e43a391de6d44e6de37155f`
- ONNX metadata:
  - input `[1, 3, 384, 640]`
  - output `[1, 84, 5040]`
  - live RGB camera, streaming, and evidence crops default to `1920x1080`; detector inference is independently resized to `640x384`
- production caveat:
  - verify Ultralytics licensing for commercial production deployment

## Model Adapter Boundary

- person detection, face recognition, and body evidence are now selected through small backend factories
- current defaults:
  - `--detector-backend yolo`
  - `--tracker-backend iou`
  - `--face-backend insightface`
  - `--body-backend hsv`
- future detector replacement should add a new adapter and factory case, then keep downstream `Detection` output unchanged
- future tracker replacement should add a new adapter and factory case, then keep downstream `Track` output unchanged
- future face replacement should keep returning `RecognizedFace`
- future body ReID replacement should keep returning per-track `BodyEvidence`
- tracking, depth plane logic, visit identity, visit registry, and event logging should not import model-specific classes directly

- `phase2_host_tracking_scrfd.py`
  - legacy-named host-side tracking baseline on top of person detections
  - uses a small local IoU-based tracker first, so tracking can be validated before adding a heavier tracker dependency
  - draws track IDs, track states, and short centroid histories on the host

- `phase3_host_entrance_line_scrfd.py`
  - legacy-named host-side entrance-line logic on top of YOLO tracking
  - adds one configurable line, side classification, short centroid history, and one-shot entry events
  - logs `ENTRY_EVENT track_id=...` when a track crosses from outside to inside
  - confirmed to emit entrance events in the running system

- `phase4_host_recognition_evidence_scrfd.py`
  - legacy-named host-side recognition evidence collection on top of YOLO entrance events
  - saves pre-entry and post-entry crops for the entering track
  - keeps recognition out of scope and focuses only on evidence capture quality

- `phase5_host_embedding_arcface.py`
  - legacy Phase 5 harness around the shared `pipeline.embedding` module
  - scans saved evidence event folders offline
  - runs face detection plus ArcFace embeddings from the local InsightFace `buffalo_l` pack
  - writes per-event embedding outputs and summaries without doing identity matching yet

- `01_replay_embeddings.py`
  - replay-oriented offline entrypoint for embedding generation
  - uses the same shared `pipeline.embedding` module without phase-specific framing

- `phase6_embedding_similarity_review.py`
  - legacy Phase 6 harness around the shared `pipeline.review` module
  - computes pairwise cosine similarities and writes review artifacts

- `phase6_embedding_similarity_html.py`
  - legacy Phase 6 HTML harness around the shared `pipeline.review` module

- `phase7_local_identity_matcher.py`
  - legacy Phase 7 harness around the shared `pipeline.identity` module
  - applies cosine-threshold local identity grouping and writes assignment outputs

- `phase7_local_identity_html.py`
  - legacy Phase 7 HTML harness around the shared `pipeline.identity` module

- `02_replay_similarity_review.py`
  - replay-oriented offline entrypoint for embedding similarity analysis

- `03_replay_similarity_html.py`
  - replay-oriented offline entrypoint for rendering similarity review HTML

- `04_replay_identity.py`
  - replay-oriented offline entrypoint for local identity assignment

- `05_replay_identity_html.py`
  - replay-oriented offline entrypoint for rendering local identity review HTML

- `contracts.py`
  - shared Python contract module for the next system layer
  - defines typed records for:
    - per-camera `EntryEvent`
    - merged `EntrySessionPacket`
    - backend `shopping_customer_id` candidates
    - in-shop `ObserverObservation`
    - observer association results
  - includes stable JSON-friendly `to_dict()` / `from_dict()` helpers

- `phase8_entry_session_builder.py`
  - legacy Phase 8 harness around the shared `pipeline.entry_session` module
  - builds typed `EntryEvent` / `EntrySessionPacket` artifacts through shared logic

- `phase8_entry_session_html.py`
  - legacy Phase 8 HTML harness around the shared `pipeline.entry_session` module

- `06_replay_entry_sessions.py`
  - replay-oriented offline entrypoint for building typed entry-session artifacts

- `07_replay_entry_sessions_html.py`
  - replay-oriented offline entrypoint for rendering entry-session review HTML

## What Comes Next

After that, the next steps should add:

- entry-event quality scoring
- multi-camera entry-session assembly
- backend `shopping_customer_id` association
- observer-camera re-association inside the shop

## Evaluation

- a concrete 2-camera evaluation procedure is documented in [two-camera-evaluation-workflow.md](/abs/path/C:/wi/luxonis/llm/person-recognition/doc/two-camera-evaluation-workflow.md)
- a 3-entrance plus observer-camera replay workflow is documented in [shop-rgbd-replay-testing-workflow.md](/abs/path/C:/wi/luxonis/llm/person-recognition/doc/shop-rgbd-replay-testing-workflow.md)
- reusable label templates live in:
  - [camera_map.example.json](/abs/path/C:/wi/luxonis/person-recognition/src/eval_templates/camera_map.example.json)
  - [single_camera_event_review.example.csv](/abs/path/C:/wi/luxonis/person-recognition/src/eval_templates/single_camera_event_review.example.csv)
  - [entry_ground_truth.example.csv](/abs/path/C:/wi/luxonis/person-recognition/src/eval_templates/entry_ground_truth.example.csv)
  - [shop_visit_ground_truth.example.csv](/abs/path/C:/wi/luxonis/person-recognition/src/eval_templates/shop_visit_ground_truth.example.csv)
  - [shop_visit_review.example.csv](/abs/path/C:/wi/luxonis/person-recognition/src/eval_templates/shop_visit_review.example.csv)

## Two Cameras

- live OAK scripts now support:
  - `--list-devices`
  - `--device-id <mxid>`
- the intended two-entrance-camera operating model is one process per camera with explicit `--device-id`
- use the offline replay workflow afterward to correlate and review the resulting artifacts

## Design Rule

- shared modules define the real pipeline behavior
- shared config defines the default parameter values and canonical data paths
- phase scripts are thin harnesses for:
  - live testing
  - replay
  - artifact writing
  - visual review
- artifacts are debug/review outputs, not the architecture itself
- the future final live pipeline should call the same shared modules and may choose whether to write artifacts
- replay entrypoints should do the same for offline processing
