from __future__ import annotations

import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from model_training.store import ModelTrainingStore


class YoloDatasetExporter:
    def __init__(self, root: Path, store: ModelTrainingStore) -> None:
        self.root = Path(root).resolve()
        self.datasets_root = self.root / "datasets"
        self.datasets_root.mkdir(parents=True, exist_ok=True)
        self.store = store

    def export(self) -> dict[str, Any]:
        rows = self.store.export_rows()
        if not rows:
            raise ValueError("No accepted frames are available for export.")
        version = self.store.next_dataset_version()
        final_path = self.datasets_root / version
        temporary = self.datasets_root / f".{version}.tmp"
        if final_path.exists() or temporary.exists():
            raise RuntimeError(f"Dataset destination already exists: {final_path}")

        classes = sorted({str(row["productCode"]) for row in rows})
        class_ids = {code: index for index, code in enumerate(classes)}
        model_labels = _model_labels(self.store, classes)
        assignments, warning = _session_assignments(rows)
        manifest_items: list[dict[str, Any]] = []
        try:
            for split in {"train", "val", "test"}:
                (temporary / "images" / split).mkdir(parents=True, exist_ok=True)
                (temporary / "labels" / split).mkdir(parents=True, exist_ok=True)
            for row in rows:
                split = assignments[str(row["sessionId"])]
                frame_id = str(row["frameId"])
                source = self.store.frame_image_path(frame_id, "original")
                image_target = temporary / "images" / split / f"{frame_id}.jpg"
                _link_or_copy(source, image_target)
                labels = []
                for box in row["annotations"]:
                    x_center = (float(box["x1"]) + float(box["x2"])) / 2
                    y_center = (float(box["y1"]) + float(box["y2"])) / 2
                    width = float(box["x2"]) - float(box["x1"])
                    height = float(box["y2"]) - float(box["y1"])
                    labels.append(
                        f"{class_ids[str(box['productCode'])]} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"
                    )
                label_target = temporary / "labels" / split / f"{frame_id}.txt"
                label_target.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="ascii")
                manifest_items.append(
                    {
                        "frameId": frame_id,
                        "sessionId": row["sessionId"],
                        "split": split,
                        "productCode": row["productCode"],
                        "reviewOutcome": row["reviewOutcome"],
                        "sha256": row["sha256"],
                        "annotationCount": len(row["annotations"]),
                    }
                )
            data_yaml = [f"path: {final_path}", "train: images/train", "val: images/val", "test: images/test", "names:"]
            data_yaml.extend(
                f"  {index}: {json.dumps(model_labels[code])}"
                for code, index in class_ids.items()
            )
            (temporary / "data.yaml").write_text("\n".join(data_yaml) + "\n", encoding="utf-8")
            (temporary / "class-map.json").write_text(
                json.dumps(
                    {
                        "classes": classes,
                        "classIds": class_ids,
                        "modelLabels": model_labels,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            manifest = {
                "datasetVersion": version,
                "taskType": "sku_detection",
                "createdAtUnixMilliseconds": int(time.time() * 1000),
                "splitWarning": warning,
                "counts": dict(Counter(item["split"] for item in manifest_items)),
                "classes": classes,
                "modelLabels": model_labels,
                "items": manifest_items,
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            temporary.replace(final_path)
            self.store.save_dataset(version, final_path, manifest)
            return {**manifest, "path": str(final_path)}
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


def _model_labels(
    store: ModelTrainingStore,
    product_codes: list[str],
) -> dict[str, str]:
    products = {
        str(product["code"]): product
        for product in store.list_products(active_only=False)
    }
    labels: dict[str, str] = {}
    assigned_ids: dict[int, str] = {}
    for code in product_codes:
        product = products.get(code)
        if product is None or product.get("id") is None:
            raise ValueError(
                f"Product {code!r} has no stable shop product ID for model export."
            )
        source_product_id = int(product["id"])
        if source_product_id <= 0:
            raise ValueError(
                f"Product {code!r} has invalid shop product ID {source_product_id}."
            )
        model_product_id = source_product_id - 1
        previous_code = assigned_ids.get(model_product_id)
        if previous_code is not None and previous_code != code:
            raise ValueError(
                "Products have duplicate model product ID "
                f"{model_product_id:03d}: {previous_code!r} and {code!r}."
            )
        assigned_ids[model_product_id] = code
        labels[code] = f"{model_product_id:03d}_{code}"
    return labels


def _session_assignments(rows: list[Mapping[str, Any]]) -> tuple[dict[str, str], str | None]:
    sessions_by_product: dict[str, set[str]] = defaultdict(set)
    gold_sessions: set[str] = set()
    for row in rows:
        session_id = str(row["sessionId"])
        sessions_by_product[str(row["productCode"])].add(session_id)
        if row["datasetIntent"] == "gold_test":
            gold_sessions.add(session_id)
    development = {code: sorted(items - gold_sessions) for code, items in sessions_by_product.items()}
    if any(len(items) < 3 for items in development.values()):
        return ({str(row["sessionId"]): "test" if str(row["sessionId"]) in gold_sessions else "train" for row in rows},
                "Insufficient independent development sessions for validation/test; exported development data as train-only.")
    assignments: dict[str, str] = {session_id: "test" for session_id in gold_sessions}
    for sessions in development.values():
        for session_id in sessions[:-2]:
            assignments[session_id] = "train"
        assignments[sessions[-2]] = "val"
        assignments[sessions[-1]] = "test"
    return assignments, None


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
