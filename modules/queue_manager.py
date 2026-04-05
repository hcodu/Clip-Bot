"""
queue_manager.py

Stage 6: Save finished clips to the queue directory with companion metadata JSON.
Also provides queue inspection and status update utilities.

Usage (CLI):
    python -m modules.queue_manager --list ./queue
    python -m modules.queue_manager --status clip_a3f2b1 uploaded ./queue
"""

import argparse
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

from utils.file_utils import make_clip_id, ensure_dirs
from utils.logger import get_logger

log = get_logger("queue_manager")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enqueue_clip(
    final_video_path: str | Path,
    metadata: dict,
    queue_dir: str | Path,
) -> dict:
    """
    Copy final_video_path into queue_dir as <clip_id>.mp4 and write
    a companion <clip_id>.json. Returns the finalized metadata dict.
    """
    queue_dir = Path(queue_dir)
    ensure_dirs(queue_dir)
    final_video_path = Path(final_video_path)

    clip_id = metadata.get("clip_id") or make_clip_id()
    dest_video = queue_dir / f"{clip_id}.mp4"
    dest_json = queue_dir / f"{clip_id}.json"

    shutil.copy2(str(final_video_path), str(dest_video))

    meta = metadata.copy()
    meta["clip_id"] = clip_id
    meta["status"] = meta.get("status", "pending")
    meta["queued_at"] = datetime.now(timezone.utc).isoformat()
    if "files" not in meta:
        meta["files"] = {}
    meta["files"]["final_video"] = str(dest_video)

    _write_json(dest_json, meta)
    log.info("Enqueued %s → %s", final_video_path.name, dest_video)
    return meta


def get_queue(
    queue_dir: str | Path,
    status_filter: str | None = None,
) -> list[dict]:
    """
    Return all clips in queue_dir, optionally filtered by status.
    Sorted by 'queued_at' ascending (oldest first).
    status_filter: None | "pending" | "uploaded" | "failed"
    """
    queue_dir = Path(queue_dir)
    if not queue_dir.exists():
        return []

    clips = []
    for json_file in sorted(queue_dir.glob("*.json")):
        try:
            meta = _read_json(json_file)
            if status_filter is None or meta.get("status") == status_filter:
                clips.append(meta)
        except Exception as e:
            log.warning("Could not read queue file %s: %s", json_file, e)

    clips.sort(key=lambda c: c.get("queued_at", ""))
    return clips


def mark_clip_status(
    clip_id: str,
    queue_dir: str | Path,
    status: str,
    extra_fields: dict | None = None,
) -> dict:
    """
    Update the status of a queued clip. Optionally merge extra_fields into
    the 'upload' sub-dict. Returns the updated metadata.
    """
    queue_dir = Path(queue_dir)
    json_path = queue_dir / f"{clip_id}.json"

    if not json_path.exists():
        raise FileNotFoundError(f"Queue entry not found: {json_path}")

    meta = _read_json(json_path)
    meta["status"] = status

    if extra_fields:
        if "upload" not in meta:
            meta["upload"] = {}
        meta["upload"].update(extra_fields)

    _write_json(json_path, meta)
    log.info("Clip %s status → %s", clip_id, status)
    return meta


def build_metadata(
    clip_dict: dict,
    source_video: str | Path,
) -> dict:
    """
    Assemble a full metadata dict from the accumulated clip processing state.
    clip_dict is built up progressively by each pipeline stage.
    """
    clip_id = clip_dict.get("clip_id", make_clip_id())

    return {
        "clip_id": clip_id,
        "status": "pending",
        "queued_at": None,  # set by enqueue_clip
        "source": {
            "video_file": str(source_video),
            "start_sec": clip_dict.get("start_sec"),
            "end_sec": clip_dict.get("end_sec"),
            "duration_sec": clip_dict.get("duration"),
            "scene_index": clip_dict.get("scene_index"),
        },
        "files": {
            "final_video": None,
            "raw_clip": str(clip_dict.get("path", "")),
            "cropped_clip": str(clip_dict.get("cropped_path", "")) if clip_dict.get("cropped_path") else None,
            "tts_audio": str(clip_dict.get("tts_audio_path", "")) if clip_dict.get("tts_audio_path") else None,
        },
        "video": {
            "width": clip_dict.get("video_width"),
            "height": clip_dict.get("video_height"),
            "fps": clip_dict.get("fps"),
            "duration_sec": clip_dict.get("duration"),
            "codec": "h264",
            "audio_codec": "aac",
        },
        "transcription": clip_dict.get("transcription", {}),
        "tts": {
            "enabled": clip_dict.get("tts_enabled", True),
            "model": clip_dict.get("tts_model"),
            "text_used": clip_dict.get("transcription", {}).get("full_text"),
            "audio_duration_sec": clip_dict.get("tts_duration"),
            "volume": clip_dict.get("tts_volume"),
        },
        "upload": {
            "attempted_at": None,
            "uploaded_at": None,
            "tiktok_url": None,
            "tiktok_post_id": None,
            "error": None,
            "retry_count": 0,
        },
        "processing_host": platform.node(),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect or update the clip queue.")
    subparsers = parser.add_subparsers(dest="command")

    list_p = subparsers.add_parser("list", help="List queued clips.")
    list_p.add_argument("queue_dir", help="Queue directory path.")
    list_p.add_argument("--status", default=None, help="Filter by status.")

    status_p = subparsers.add_parser("status", help="Update a clip's status.")
    status_p.add_argument("clip_id", help="Clip ID to update.")
    status_p.add_argument("new_status", choices=["pending", "uploaded", "failed"])
    status_p.add_argument("queue_dir", help="Queue directory path.")

    args = parser.parse_args()

    from utils.logger import setup_logger
    setup_logger()

    if args.command == "list":
        clips = get_queue(args.queue_dir, status_filter=args.status)
        if not clips:
            print("Queue is empty.")
        for c in clips:
            print(f"  [{c['status']:8s}] {c['clip_id']}  queued: {c.get('queued_at', 'n/a')}")

    elif args.command == "status":
        updated = mark_clip_status(args.clip_id, args.queue_dir, args.new_status)
        print(f"Updated {args.clip_id} → {updated['status']}")
