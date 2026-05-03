# Source Reset

This source folder has been reset for a new architecture:

- OAK camera as USB image source
- host PC GPU for detection, tracking, entrance logic, and recognition

The old on-device `RVC2` experiment scripts were intentionally removed.

## Current Baseline

- `main.py`
  - host-side camera capture baseline
  - connects to the OAK camera with `depthai`
  - receives RGB frames on the PC
  - shows a live preview

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
  - host-side Step 8 embedding harness
  - scans saved evidence event folders offline
  - runs face detection plus ArcFace embeddings from the local InsightFace `buffalo_l` pack
  - writes per-event embedding outputs and summaries without doing identity matching yet

- `phase6_embedding_similarity_review.py`
  - offline embedding review helper
  - reads per-event mean embeddings
  - computes pairwise cosine similarities
  - writes a matrix and neighbor report so embedding usefulness can be judged before identity logic

- `phase6_embedding_similarity_html.py`
  - offline visual reviewer
  - turns the similarity review into a local HTML page with side-by-side evidence thumbnails

- `phase7_local_identity_matcher.py`
  - offline local identity matcher
  - processes event mean embeddings in time order
  - applies cosine-threshold matching with `unknown`/new-identity behavior
  - writes assignment and gallery summaries before any live integration

- `phase7_local_identity_html.py`
  - offline visual reviewer for identity assignments
  - renders each `person_###` gallery group with the member event crops side by side

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
  - first contract-driven builder for the next identity layer
  - converts saved evidence/embedding artifacts into typed `EntryEvent` JSON files
  - merges nearby events into `EntrySessionPacket` JSON files by time window
  - gives the project a concrete offline artifact flow for entry-session assembly

- `phase8_entry_session_html.py`
  - visual reviewer for the entry-session layer
  - shows merged sessions, member event crops, and non-merge / ambiguity decisions

## What Comes Next

After that, the next steps should add:

- entry-event quality scoring
- multi-camera entry-session assembly
- backend `shopping_customer_id` association
- observer-camera re-association inside the shop
