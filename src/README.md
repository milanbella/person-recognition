# Source Directory

Current system documentation is maintained in:

```text
C:\wi\luxonis\llm\person-recognition\doc\README.md
```

This directory contains runtime entrypoints, shared pipeline modules, web
assets, deployment examples, and tests. Use script `--help` output for exact
CLI options.

Primary entrypoints:

- `live_synced_rgbd_streams.py`: production multi-camera live service
- `replay_synced_rgbd_streams.py`: synchronized RGB-D replay
- `model_training_app.py`: product dataset collection/review/export service
- `record_rgbd_stream.py`: one-camera RGB-D recorder
- `fit_plane_from_aruco.py`: recorded entrance-plane calibration
- `calibrate_shelf_anchors.py`: live or recorded shelf calibration
- `calibrate_shelf_regions.py`: normalized shelf polygons plus synchronized
  multi-frame 3D depth-grid calibration
- `preview_product_detection.py`: one-camera full-frame product preview
- `preview_body_crop_product_detection.py`: multi-camera body-crop product preview
- `preview_body_crop_pose.py`: multi-camera tracked-person YOLO pose preview

Shared production behavior belongs in `pipeline/`; phase-numbered scripts are
legacy experiment/review harnesses and must not become the only implementation
of runtime logic.

Run tests from this directory:

```bash
python -m unittest discover -s tests
```
