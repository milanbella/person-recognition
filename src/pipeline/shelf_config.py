from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipeline.aruco_markers import DEFAULT_ARUCO_DICTIONARY, DEFAULT_DOOR_MARKER_IDS


DEFAULT_SHELF_CONFIG_PATH = Path("config") / "shelves.json"
DEFAULT_SHELF_CALIBRATIONS_DIR = Path("shelf_calibrations")


@dataclass(frozen=True)
class ShelfPersonDepthConfig:
    center_x_fraction: float = 0.50
    center_y_fraction: float = 0.50
    width_fraction: float = 0.30
    height_fraction: float = 0.25
    min_valid_pixels: int = 25
    fallback_center_y_fractions: tuple[float, ...] = (0.42, 0.58)


@dataclass(frozen=True)
class ShelfDefaults:
    approach_distance_mm: float = 900.0
    departure_distance_mm: float = 1150.0
    approach_dwell_milliseconds: int = 500
    departure_dwell_milliseconds: int = 800
    lost_visit_grace_milliseconds: int = 1500
    owner_switch_margin_mm: float = 200.0
    owner_switch_dwell_milliseconds: int = 500


@dataclass(frozen=True)
class ShelfDefinition:
    shelf_id: int
    label: str
    marker_id: int
    approach_distance_mm: float
    departure_distance_mm: float
    approach_dwell_milliseconds: int
    departure_dwell_milliseconds: int
    lost_visit_grace_milliseconds: int
    owner_switch_margin_mm: float
    owner_switch_dwell_milliseconds: int
    marker_ids: tuple[int, ...] = ()

    @property
    def all_marker_ids(self) -> tuple[int, ...]:
        return self.marker_ids or (self.marker_id,)


@dataclass(frozen=True)
class ShelfWatchingConfig:
    schema_version: int
    aruco_dictionary: str
    marker_size_mm: float
    person_depth: ShelfPersonDepthConfig
    shelves: tuple[ShelfDefinition, ...]

    def shelf_by_id(self) -> dict[int, ShelfDefinition]:
        return {shelf.shelf_id: shelf for shelf in self.shelves}


_ROOT_FIELDS = {
    "schemaVersion",
    "arucoDictionary",
    "markerSizeMm",
    "defaults",
    "shelves",
}
_DEFAULT_FIELDS = {
    "approachDistanceMm",
    "departureDistanceMm",
    "approachDwellMilliseconds",
    "departureDwellMilliseconds",
    "lostVisitGraceMilliseconds",
    "ownerSwitchMarginMm",
    "ownerSwitchDwellMilliseconds",
    "personDepthRoiCenterXFraction",
    "personDepthRoiCenterYFraction",
    "personDepthRoiWidthFraction",
    "personDepthRoiHeightFraction",
    "personDepthMinValidPixels",
    "personDepthFallbackCenterYFractions",
}
_SHELF_FIELDS = {
    "shelfId",
    "label",
    "markerId",
    "markerIds",
    "approachDistanceMm",
    "departureDistanceMm",
    "approachDwellMilliseconds",
    "departureDwellMilliseconds",
    "lostVisitGraceMilliseconds",
    "ownerSwitchMarginMm",
    "ownerSwitchDwellMilliseconds",
}
_LEGACY_SHELF_FIELDS = _SHELF_FIELDS | {"cameraDeviceIds"}


def _require_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return value


def _reject_unknown_fields(
    payload: Mapping[str, Any],
    *,
    allowed: set[str],
    field_name: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            f"{field_name} contains unknown fields: {', '.join(unknown)}"
        )


def _positive_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number.") from exc
    if result <= 0.0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return result


def _non_negative_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    if value < 0:
        raise ValueError(f"{field_name} must not be negative.")
    return value


def _fraction(value: Any, *, field_name: str, allow_zero: bool) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number.") from exc
    lower_valid = result >= 0.0 if allow_zero else result > 0.0
    if not lower_valid or result > 1.0:
        comparison = "between 0.0 and 1.0" if allow_zero else "greater than 0.0 and no greater than 1.0"
        raise ValueError(f"{field_name} must be {comparison}.")
    return result


def _defaults_from_payload(payload: Mapping[str, Any]) -> tuple[ShelfDefaults, ShelfPersonDepthConfig]:
    _reject_unknown_fields(payload, allowed=_DEFAULT_FIELDS, field_name="defaults")
    defaults = ShelfDefaults(
        approach_distance_mm=_positive_float(
            payload.get("approachDistanceMm", 900.0),
            field_name="defaults.approachDistanceMm",
        ),
        departure_distance_mm=_positive_float(
            payload.get("departureDistanceMm", 1150.0),
            field_name="defaults.departureDistanceMm",
        ),
        approach_dwell_milliseconds=_non_negative_int(
            payload.get("approachDwellMilliseconds", 500),
            field_name="defaults.approachDwellMilliseconds",
        ),
        departure_dwell_milliseconds=_non_negative_int(
            payload.get("departureDwellMilliseconds", 800),
            field_name="defaults.departureDwellMilliseconds",
        ),
        lost_visit_grace_milliseconds=_non_negative_int(
            payload.get("lostVisitGraceMilliseconds", 1500),
            field_name="defaults.lostVisitGraceMilliseconds",
        ),
        owner_switch_margin_mm=_positive_float(
            payload.get("ownerSwitchMarginMm", 200.0),
            field_name="defaults.ownerSwitchMarginMm",
        ),
        owner_switch_dwell_milliseconds=_non_negative_int(
            payload.get("ownerSwitchDwellMilliseconds", 500),
            field_name="defaults.ownerSwitchDwellMilliseconds",
        ),
    )
    if defaults.departure_distance_mm <= defaults.approach_distance_mm:
        raise ValueError(
            "defaults.departureDistanceMm must be greater than "
            "defaults.approachDistanceMm."
        )

    fallback_values = payload.get(
        "personDepthFallbackCenterYFractions",
        [0.42, 0.58],
    )
    if not isinstance(fallback_values, list):
        raise ValueError(
            "defaults.personDepthFallbackCenterYFractions must be an array."
        )
    person_depth = ShelfPersonDepthConfig(
        center_x_fraction=_fraction(
            payload.get("personDepthRoiCenterXFraction", 0.50),
            field_name="defaults.personDepthRoiCenterXFraction",
            allow_zero=True,
        ),
        center_y_fraction=_fraction(
            payload.get("personDepthRoiCenterYFraction", 0.50),
            field_name="defaults.personDepthRoiCenterYFraction",
            allow_zero=True,
        ),
        width_fraction=_fraction(
            payload.get("personDepthRoiWidthFraction", 0.30),
            field_name="defaults.personDepthRoiWidthFraction",
            allow_zero=False,
        ),
        height_fraction=_fraction(
            payload.get("personDepthRoiHeightFraction", 0.25),
            field_name="defaults.personDepthRoiHeightFraction",
            allow_zero=False,
        ),
        min_valid_pixels=_non_negative_int(
            payload.get("personDepthMinValidPixels", 25),
            field_name="defaults.personDepthMinValidPixels",
        ),
        fallback_center_y_fractions=tuple(
            _fraction(
                value,
                field_name=f"defaults.personDepthFallbackCenterYFractions[{index}]",
                allow_zero=True,
            )
            for index, value in enumerate(fallback_values)
        ),
    )
    if person_depth.min_valid_pixels <= 0:
        raise ValueError("defaults.personDepthMinValidPixels must be positive.")
    return defaults, person_depth


def _resolved_shelf(
    payload: Mapping[str, Any],
    *,
    index: int,
    defaults: ShelfDefaults,
    schema_version: int,
) -> ShelfDefinition:
    field_name = f"shelves[{index}]"
    _reject_unknown_fields(
        payload,
        allowed=_LEGACY_SHELF_FIELDS if schema_version == 1 else _SHELF_FIELDS,
        field_name=field_name,
    )
    if "shelfId" not in payload:
        raise ValueError(f"{field_name}.shelfId is required.")
    marker_fields = [field for field in ("markerId", "markerIds") if field in payload]
    if not marker_fields:
        raise ValueError(
            f"{field_name} requires either markerId or markerIds."
        )
    if len(marker_fields) > 1:
        raise ValueError(
            f"{field_name} must not specify both markerId and markerIds."
        )

    shelf_id = payload["shelfId"]
    if isinstance(shelf_id, bool) or not isinstance(shelf_id, int) or shelf_id < 0:
        raise ValueError(f"{field_name}.shelfId must be a non-negative integer.")
    if "markerId" in payload:
        raw_marker_ids = [payload["markerId"]]
    else:
        raw_marker_ids = payload["markerIds"]
        if not isinstance(raw_marker_ids, list) or not raw_marker_ids:
            raise ValueError(f"{field_name}.markerIds must be a non-empty array.")
    marker_ids: list[int] = []
    for marker_index, marker_id in enumerate(raw_marker_ids):
        if (
            isinstance(marker_id, bool)
            or not isinstance(marker_id, int)
            or marker_id < 0
        ):
            raise ValueError(
                f"{field_name}.markerIds[{marker_index}] must be a non-negative integer."
            )
        marker_ids.append(marker_id)
    if len(marker_ids) != len(set(marker_ids)):
        raise ValueError(f"{field_name}.markerIds contains duplicates.")
    label = payload.get("label", f"Shelf {shelf_id}")
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"{field_name}.label must be a non-empty string.")
    if schema_version == 1:
        camera_ids = payload.get("cameraDeviceIds")
        if (
            not isinstance(camera_ids, list)
            or not camera_ids
            or any(
                not isinstance(value, str) or not value.strip()
                for value in camera_ids
            )
        ):
            raise ValueError(
                f"{field_name}.cameraDeviceIds must be a non-empty string array."
            )
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError(f"{field_name}.cameraDeviceIds contains duplicates.")

    approach_distance_mm = _positive_float(
        payload.get("approachDistanceMm", defaults.approach_distance_mm),
        field_name=f"{field_name}.approachDistanceMm",
    )
    departure_distance_mm = _positive_float(
        payload.get("departureDistanceMm", defaults.departure_distance_mm),
        field_name=f"{field_name}.departureDistanceMm",
    )
    if departure_distance_mm <= approach_distance_mm:
        raise ValueError(
            f"{field_name}.departureDistanceMm must be greater than "
            f"{field_name}.approachDistanceMm."
        )

    return ShelfDefinition(
        shelf_id=shelf_id,
        label=label.strip(),
        marker_id=marker_ids[0],
        approach_distance_mm=approach_distance_mm,
        departure_distance_mm=departure_distance_mm,
        approach_dwell_milliseconds=_non_negative_int(
            payload.get(
                "approachDwellMilliseconds",
                defaults.approach_dwell_milliseconds,
            ),
            field_name=f"{field_name}.approachDwellMilliseconds",
        ),
        departure_dwell_milliseconds=_non_negative_int(
            payload.get(
                "departureDwellMilliseconds",
                defaults.departure_dwell_milliseconds,
            ),
            field_name=f"{field_name}.departureDwellMilliseconds",
        ),
        lost_visit_grace_milliseconds=_non_negative_int(
            payload.get(
                "lostVisitGraceMilliseconds",
                defaults.lost_visit_grace_milliseconds,
            ),
            field_name=f"{field_name}.lostVisitGraceMilliseconds",
        ),
        owner_switch_margin_mm=_positive_float(
            payload.get("ownerSwitchMarginMm", defaults.owner_switch_margin_mm),
            field_name=f"{field_name}.ownerSwitchMarginMm",
        ),
        owner_switch_dwell_milliseconds=_non_negative_int(
            payload.get(
                "ownerSwitchDwellMilliseconds",
                defaults.owner_switch_dwell_milliseconds,
            ),
            field_name=f"{field_name}.ownerSwitchDwellMilliseconds",
        ),
        marker_ids=tuple(marker_ids),
    )


def load_shelf_config(path: Path) -> ShelfWatchingConfig:
    try:
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid shelf configuration JSON {path}: {exc}") from exc

    payload = _require_mapping(raw_payload, field_name="root")
    _reject_unknown_fields(payload, allowed=_ROOT_FIELDS, field_name="root")
    schema_version = payload.get("schemaVersion", 1)
    if schema_version not in {1, 2}:
        raise ValueError(
            f"Unsupported shelf configuration schemaVersion {schema_version!r}; "
            "expected 1 or 2."
        )
    dictionary = payload.get("arucoDictionary", DEFAULT_ARUCO_DICTIONARY)
    if not isinstance(dictionary, str) or not dictionary:
        raise ValueError("arucoDictionary must be a non-empty string.")
    marker_size_mm = _positive_float(
        payload.get("markerSizeMm", 80.0),
        field_name="markerSizeMm",
    )
    defaults_payload = _require_mapping(
        payload.get("defaults", {}),
        field_name="defaults",
    )
    defaults, person_depth = _defaults_from_payload(defaults_payload)
    shelf_payloads = payload.get("shelves")
    if not isinstance(shelf_payloads, list) or not shelf_payloads:
        raise ValueError("shelves must be a non-empty array.")
    shelves = tuple(
        _resolved_shelf(
            _require_mapping(value, field_name=f"shelves[{index}]"),
            index=index,
            defaults=defaults,
            schema_version=schema_version,
        )
        for index, value in enumerate(shelf_payloads)
    )

    shelf_ids = [shelf.shelf_id for shelf in shelves]
    marker_ids = [
        marker_id
        for shelf in shelves
        for marker_id in shelf.all_marker_ids
    ]
    duplicate_shelf_ids = sorted(
        shelf_id for shelf_id in set(shelf_ids) if shelf_ids.count(shelf_id) > 1
    )
    duplicate_marker_ids = sorted(
        marker_id for marker_id in set(marker_ids) if marker_ids.count(marker_id) > 1
    )
    if duplicate_shelf_ids:
        raise ValueError(f"Duplicate shelfId values: {duplicate_shelf_ids}")
    if duplicate_marker_ids:
        raise ValueError(f"Duplicate markerId values: {duplicate_marker_ids}")
    reserved = sorted(set(marker_ids) & set(DEFAULT_DOOR_MARKER_IDS))
    if reserved:
        raise ValueError(
            f"Shelf marker IDs overlap reserved door marker IDs: {reserved}"
        )

    return ShelfWatchingConfig(
        schema_version=schema_version,
        aruco_dictionary=dictionary,
        marker_size_mm=marker_size_mm,
        person_depth=person_depth,
        shelves=shelves,
    )


def validate_shelf_config_for_live_cameras(
    config: ShelfWatchingConfig,
    *,
    camera_device_ids: Sequence[str],
    observer_capable_device_ids: set[str],
) -> None:
    configured = set(camera_device_ids)
    errors = [
        f"observer-capable camera {device_id} is absent from --device-id"
        for device_id in sorted(observer_capable_device_ids - configured)
    ]
    if not observer_capable_device_ids:
        errors.append("shelf watching requires at least one observer-capable camera")
    if errors:
        raise ValueError("Invalid shelf camera configuration: " + "; ".join(errors))
