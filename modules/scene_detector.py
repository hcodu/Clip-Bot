"""
scene_detector.py

Stage 1: Detect scenes in a source video and export 30–60 second clips.

Usage (CLI):
    python -m modules.scene_detector --input episode.mp4 --output-dir temp/raw_clips
"""

import argparse
import subprocess
from pathlib import Path

from scenedetect import ContentDetector, SceneManager, open_video
from utils.file_utils import make_clip_id, ensure_dirs
from utils.logger import get_logger

log = get_logger("scene_detector")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_scenes(
    video_path: str | Path,
    output_dir: str | Path,
    threshold: float = 27.0,
    min_clip_duration: float = 30.0,
    max_clip_duration: float = 60.0,
    detector: str = "content",
) -> list[dict]:
    """
    Detect scenes in video_path, merge/split to fit [min, max] duration,
    and export each clip to output_dir.

    Returns a list of clip metadata dicts:
      [{"clip_id", "path", "start_sec", "end_sec", "duration", "source_video"}, ...]
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    ensure_dirs(output_dir)

    log.info("Detecting scenes in: %s", video_path)

    raw_scenes = _run_scene_detection(video_path, threshold=threshold, detector=detector)
    log.info("Raw scene count: %d", len(raw_scenes))

    scenes = _merge_short_scenes(raw_scenes, min_clip_duration)
    scenes = _split_long_scenes(scenes, max_clip_duration)
    log.info("After merge/split: %d clips", len(scenes))

    clips = []
    for i, (start_sec, end_sec) in enumerate(scenes):
        clip_id = make_clip_id("clip")
        output_path = output_dir / f"{clip_id}_raw.mp4"
        export_clip(video_path, output_path, start_sec, end_sec)
        clip = {
            "clip_id": clip_id,
            "path": output_path,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "duration": round(end_sec - start_sec, 3),
            "source_video": str(video_path),
            "scene_index": i,
        }
        clips.append(clip)
        log.debug("Exported %s (%.1fs–%.1fs)", clip_id, start_sec, end_sec)

    log.info("Exported %d clips to %s", len(clips), output_dir)
    return clips


def export_clip(
    video_path: str | Path,
    output_path: str | Path,
    start_sec: float,
    end_sec: float,
) -> Path:
    """
    Export a single clip using FFmpeg. Re-encodes with libx264/aac for
    broad compatibility. Returns output_path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end_sec - start_sec

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg export failed for {output_path}:\n{result.stderr}")

    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_scene_detection(
    video_path: Path,
    threshold: float = 27.0,
    detector: str = "content",
) -> list[tuple[float, float]]:
    """
    Run PySceneDetect and return a list of (start_sec, end_sec) tuples.
    """
    video = open_video(str(video_path))
    manager = SceneManager()

    if detector == "content":
        manager.add_detector(ContentDetector(threshold=threshold))
    else:
        # Fallback to ContentDetector for unsupported types
        log.warning("Detector '%s' not implemented, falling back to 'content'.", detector)
        manager.add_detector(ContentDetector(threshold=threshold))

    manager.detect_scenes(video)
    scene_list = manager.get_scene_list()

    scenes = []
    for start_tc, end_tc in scene_list:
        start_sec = start_tc.get_seconds()
        end_sec = end_tc.get_seconds()
        if end_sec > start_sec:
            scenes.append((start_sec, end_sec))

    return scenes


def _merge_short_scenes(
    scenes: list[tuple[float, float]],
    min_duration: float,
) -> list[tuple[float, float]]:
    """
    Merge adjacent scenes until each is at least min_duration seconds.
    The last group may be shorter if there's nothing left to merge into.
    """
    if not scenes:
        return scenes

    merged = []
    group_start, group_end = scenes[0]

    for start, end in scenes[1:]:
        current_duration = group_end - group_start
        if current_duration < min_duration:
            # Extend the current group
            group_end = end
        else:
            merged.append((group_start, group_end))
            group_start, group_end = start, end

    merged.append((group_start, group_end))
    return merged


def _split_long_scenes(
    scenes: list[tuple[float, float]],
    max_duration: float,
) -> list[tuple[float, float]]:
    """
    Split scenes longer than max_duration into equal-length segments.
    """
    result = []
    for start, end in scenes:
        duration = end - start
        if duration <= max_duration:
            result.append((start, end))
        else:
            # Calculate how many equal segments we need
            n_segments = int(duration / max_duration) + (1 if duration % max_duration > 0 else 0)
            seg_duration = duration / n_segments
            for i in range(n_segments):
                seg_start = start + i * seg_duration
                seg_end = min(start + (i + 1) * seg_duration, end)
                result.append((seg_start, seg_end))
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect scenes and export clips.")
    parser.add_argument("--input", required=True, help="Path to source video file.")
    parser.add_argument("--output-dir", default="./temp/raw_clips", help="Directory for exported clips.")
    parser.add_argument("--threshold", type=float, default=27.0, help="Scene detection sensitivity.")
    parser.add_argument("--min-duration", type=float, default=30.0, help="Minimum clip duration (seconds).")
    parser.add_argument("--max-duration", type=float, default=60.0, help="Maximum clip duration (seconds).")
    args = parser.parse_args()

    from utils.logger import setup_logger
    setup_logger()

    clips = detect_scenes(
        video_path=args.input,
        output_dir=args.output_dir,
        threshold=args.threshold,
        min_clip_duration=args.min_duration,
        max_clip_duration=args.max_duration,
    )
    for c in clips:
        print(f"  {c['clip_id']}  {c['start_sec']:.1f}s–{c['end_sec']:.1f}s  ({c['duration']:.1f}s)")
