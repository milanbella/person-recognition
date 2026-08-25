# Person Recognition

Multi-camera OAK RGB-D processing for shop visits, entrance/leave detection,
cross-camera visit identity, shelf position, product recognition, operator
testing, and product-model dataset collection.

## Documentation

Current documentation is maintained under:

```text
C:\wi\luxonis\llm\person-recognition\doc\README.md
```

Start with:

- `architecture.md`
- `live-service-operations.md`
- `http-api.md`
- `calibration.md`
- `model-training.md`

Historical implementation plans are stored in the documentation archive and
must not be treated as current operational instructions.

## Primary Entrypoints

- `src/live_synced_rgbd_streams.py`: production multi-camera live service.
- `src/replay_synced_rgbd_streams.py`: synchronized recorded RGB-D replay.
- `src/model_training_app.py`: product dataset capture/review/export service.
- `src/calibrate_shelf_anchors.py`: per-camera shelf calibration.
- `src/fit_plane_from_aruco.py`: recorded entrance-plane calibration.
- `src/preview_product_detection.py`: one-camera full-frame product preview.
- `src/preview_body_crop_product_detection.py`: multi-camera body-crop product preview.

Use each entrypoint's `--help` output for the exact current CLI contract.
