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
      - generic person detector protocol/factory, current SCRFD detector wrapper, detection arg helpers, and drawing helpers
      - current detector backend is `scrfd`; use `--detector-backend scrfd` and `--model <onnx-path>`
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
      - treats fragmented `face_person_###` labels as evidence attached to a visit, not as the only visit identity source
    - `pipeline.visit_registry`
      - shop-wide active visit registry for synchronized multi-camera replay
      - merges entrance-camera plane events into one `entrance_confirmed` visit by timestamp window
      - lets observer cameras attach to entrance-confirmed visits or create `observer_only` visits when no match is found
      - builds visit observations from normalized evidence, not raw RGB frames
      - uses `FrameEvidence` for one camera frame and `TrackVisitEvidence` for one track's visit-matching evidence
- `pipeline.entry_session`
  - typed entry-event/session building, offline correlation, and HTML review helpers
  - shared by replay entrypoints and the legacy Phase 8 harnesses
    - `pipeline.depth`
      - shared helpers for sampling aligned stereo depth inside tracked person boxes
      - includes a first depth-threshold entrance prototype
      - depth trigger functions return `DepthEntranceResult` with `entered_track_ids`, `depth_samples`, and `signed_distances_mm`

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
  - shows synchronized tiled RGB views and optional synchronized tiled depth views
  - accepts one or more `--device-id` values and derives the matching RGBD recording folders
  - can optionally run replay-local face identity assignment with `--enable-face-recognition`
  - assigns shared registry-owned `visit_id` labels across the synchronized replay
  - defaults every stream to `--camera-role entrance`
  - supports `--camera-role entrance_observer` for entrance cameras that should also contribute observer evidence
  - supports `--camera-role observer` for in-shop observer streams
  - supports `--output-dir` for replay artifacts: visit decisions, track visit evidence, entrance events, and final visit summaries

- `replay_depth_tuner.py`
  - replays one recorded RGBD stream through detection, tracking, and depth-based entrance logic
  - writes replayed depth entrance-event timing logs from recorded timestamps and aligned recorded depth
  - can optionally run replay-local face identity assignment with `--enable-face-recognition`
  - writes `visit_id` and attached face identity ids into depth event logs

- `fit_plane_from_tags.py`
  - interactive plane-calibration utility for recorded RGBD streams
  - lets you click 3 tagged door-corner points, fits a 3D plane from recorded depth, and prints the CLI args for plane-based entrance detection

- `final_pipeline.py`
  - first unified live pipeline entrypoint built on shared modules
  - runs host-side detection, tracking, entrance logic, and optional evidence capture
  - demonstrates how the eventual live pipeline should depend on shared modules instead of phase harness imports
  - supports explicit OAK selection with `--device-id`

- `depth_entrance_live.py`
  - first live prototype for depth-based entrance triggering
  - uses CAM_A RGB plus CAM_B/C stereo depth aligned to RGB
  - samples depth near the lower body and emits `DEPTH_ENTRY_EVENT` when a tracked person crosses a depth threshold

- `phase1_host_detection_scrfd.py`
  - host-side Step 2 detector harness
  - host-side SCRFD detection using the InsightFace ONNX wrapper
  - reads OAK USB frames and draws person detections on the host
  - prefers CUDA when available and falls back to CPU when the local GPU runtime is incomplete

## Model Provenance

- default detector model:
  - `C:\wi\luxonis\person-recognition\models\scrfd_person_2.5g.onnx`
- source:
  - InsightFace v0.7 release asset: `https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_person_2.5g.onnx`
  - SourceForge InsightFace mirror listing: `https://sourceforge.net/projects/insightface.mirror/files/v0.7/`
- local SHA256:
  - `76522ba15eecb0712780509e912884aba066e9834be0c85761918cdcf76de5b5`
- ONNX metadata:
  - `producer_name=pytorch`
  - `producer_version=1.7`
  - `graph_name=torch-jit-export`
  - `opset=11`
- production caveat:
  - InsightFace model downloads are documented as non-commercial research assets; verify licensing before production deployment or replace this detector with a production-safe model.

## Model Adapter Boundary

- person detection, face recognition, and body evidence are now selected through small backend factories
- current defaults preserve existing behavior:
  - `--detector-backend scrfd`
  - `--tracker-backend iou`
  - `--face-backend insightface`
  - `--body-backend hsv`
- future detector replacement should add a new adapter and factory case, then keep downstream `Detection` output unchanged
- future tracker replacement should add a new adapter and factory case, then keep downstream `Track` output unchanged
- future face replacement should keep returning `RecognizedFace`
- future body ReID replacement should keep returning per-track `BodyEvidence`
- tracking, depth plane logic, visit identity, visit registry, and event logging should not import model-specific classes directly

- `phase2_host_tracking_scrfd.py`
  - host-side tracking baseline on top of SCRFD detections
  - uses a small local IoU-based tracker first, so tracking can be validated before adding a heavier tracker dependency
  - draws track IDs, track states, and short centroid histories on the host

- `phase3_host_entrance_line_scrfd.py`
  - host-side entrance-line logic on top of SCRFD tracking
  - adds one configurable line, side classification, short centroid history, and one-shot entry events
  - logs `ENTRY_EVENT track_id=...` when a track crosses from outside to inside
  - confirmed to emit entrance events in the running system

- `phase4_host_recognition_evidence_scrfd.py`
  - host-side recognition evidence collection on top of SCRFD entrance events
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
