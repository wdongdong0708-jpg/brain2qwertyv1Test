import hashlib
import json


def timeline_cache_key(timeline) -> str:
    payload = json.dumps(timeline, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def neuro_cache_key(event) -> str:
    return hashlib.sha256(event._splittable_event_uid().encode()).hexdigest()


def configure_cache(data, *, read_only=False) -> None:
    """Use identical, filesystem-safe item keys on Windows and Linux."""
    data.study.infra_timelines.item_uid = timeline_cache_key
    data.neuro.infra.item_uid = neuro_cache_key
    mode = "read-only" if read_only else "cached"
    data.study.infra_timelines.mode = mode
    data.neuro.infra.mode = mode


def cache_signature(data) -> dict:
    """Return exca's actual UIDs; execution mode and cache location are excluded."""
    return {
        "timelines": data.study.infra_timelines.uid(),
        "neuro": data.neuro.infra.uid(),
        "timeline_item_key": "sha256-sorted-json-v1",
        "neuro_item_key": "sha256-splittable-event-uid-v1",
    }
