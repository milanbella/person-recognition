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
      - SCRFD detector wrapper, detection arg helpers, and drawing helpers
    - `pipeline.tracking`
      - track state, IoU tracking logic, tracking arg helpers, and drawing helpers
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
    - `pipeline.entry_session`
      - typed entry-event/session building, offline correlation, and HTML review helpers
      - shared by replay entrypoints and the legacy Phase 8 harnesses

- `main.py`
  - host-side camera capture baseline
  - connects to the OAK camera with `depthai`
  - receives RGB frames on the PC
  - shows a live preview

- `final_pipeline.py`
  - first unified live pipeline entrypoint built on shared modules
  - runs host-side detection, tracking, entrance logic, and optional evidence capture
  - demonstrates how the eventual live pipeline should depend on shared modules instead of phase harness imports
  - supports explicit OAK selection with `--device-id`

- `phase1_host_detection_scrfd.py`
  - host-side Step 2 detector harness
  - host-side SCRFD detection using the InsightFace ONNX wrapper
  - reads OAK USB frames and draws person detections on the host
  - prefers CUDA when available and falls back to CPU when the local GPU runtime is incomplete

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
- reusable label templates live in:
  - [camera_map.example.json](/abs/path/C:/wi/luxonis/person-recognition/src/eval_templates/camera_map.example.json)
  - [single_camera_event_review.example.csv](/abs/path/C:/wi/luxonis/person-recognition/src/eval_templates/single_camera_event_review.example.csv)
  - [entry_ground_truth.example.csv](/abs/path/C:/wi/luxonis/person-recognition/src/eval_templates/entry_ground_truth.example.csv)

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
