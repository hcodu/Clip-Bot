"""
crop_converter.py

Stage 2: Center-crop widescreen clips to 9:16 vertical (TikTok format).

Usage (CLI):
    python -m modules.crop_converter --input clip.mp4 --output clip_vertical.mp4
"""

import argparse
import json
import subprocess
from pathlib import Path

from utils.file_utils import ensure_dirs
from utils.logger import get_logger

log = get_logger("crop_converter")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def crop_to_vertical(
    input_path: str | Path,
    output_path: str | Path,
    target_width: int = 1080,
    target_height: int = 1920,
) -> Path:
    """
    Center-crop input video from its native aspect ratio to target_width x target_height
    (default 1080x1920, i.e. 9:16 vertical).

    Audio is copied without re-encoding. Video is re-encoded with libx264.
    Returns output_path.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    ensure_dirs(output_path.parent)

    # Probe source to get actual dimensions using ffprobe
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_streams",
        "-of", "json",
        str(input_path)
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if probe_result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe_result.stderr}")
    
    probe = json.loads(probe_result.stdout)
    video_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "video"),
        None,
    )
    if video_stream is None:
        raise ValueError(f"No video stream found in {input_path}")

    src_w = int(video_stream["width"])
    src_h = int(video_stream["height"])
    log.debug("Source resolution: %dx%d", src_w, src_h)

    crop_filter = get_crop_filter(src_w, src_h, target_aspect=(target_width, target_height))
    log.debug("FFmpeg filter: %s", crop_filter)

    # Build FFmpeg command via subprocess for full control
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", crop_filter,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg crop failed for {input_path}:\n{result.stderr}")

    log.info("Cropped %s → %s", input_path.name, output_path.name)
    return output_path


def get_crop_filter(
    source_width: int,
    source_height: int,
    target_aspect: tuple[int, int] = (9, 16),
) -> str:
    """
    Return the FFmpeg vf filter string to center-crop source to target_aspect,
    then scale to the target pixel dimensions.

    For a 1920x1080 source → 9:16:
      crop width = 1080 * 9/16 = 607.5 → 607
      center offset = (1920 - 607) / 2 = 656
      filter: crop=607:1080:656:0,scale=1080:1920
    """
    target_w_ratio, target_h_ratio = target_aspect

    # We always crop the full height and compute the matching width
    crop_h = source_height
    crop_w = int(source_height * target_w_ratio / target_h_ratio)

    # Center horizontally
    x_offset = (source_width - crop_w) // 2
    y_offset = 0

    # Scale to exact target pixel size
    scale_w = target_w_ratio * (1920 // target_h_ratio) if target_h_ratio <= 16 else target_w_ratio
    # Resolve actual output pixel dimensions
    # Convention: target_aspect is passed as (width, height) pixels when >100,
    # or as ratio (9, 16) when small. Detect by value.
    if target_w_ratio > 100:
        # Pixel dimensions passed directly
        out_w, out_h = target_w_ratio, target_h_ratio
    else:
        # Ratio only — scale to 1080p vertical by default
        out_h = 1920
        out_w = int(out_h * target_w_ratio / target_h_ratio)

    return f"crop={crop_w}:{crop_h}:{x_offset}:{y_offset},scale={out_w}:{out_h}:flags=lanczos"


def batch_crop(
    clip_metadata_list: list[dict],
    output_dir: str | Path,
    target_width: int = 1080,
    target_height: int = 1920,
) -> list[dict]:
    """
    Crop a list of clip dicts (from scene_detector) to vertical.
    Each dict must have a 'path' key. Adds 'cropped_path' key to each dict.
    Returns the updated list.
    """
    output_dir = Path(output_dir)
    ensure_dirs(output_dir)

    for clip in clip_metadata_list:
        clip_id = clip["clip_id"]
        input_path = Path(clip["path"])
        output_path = output_dir / f"{clip_id}_crop.mp4"

        try:
            crop_to_vertical(input_path, output_path, target_width, target_height)
            clip["cropped_path"] = output_path
        except Exception as e:
            log.error("Crop failed for %s: %s", clip_id, e)
            clip["cropped_path"] = None
            clip["error"] = str(e)

    return clip_metadata_list


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Center-crop video to 9:16 vertical.")
    parser.add_argument("--input", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Output video path.")
    parser.add_argument("--width", type=int, default=1080, help="Target width (default 1080).")
    parser.add_argument("--height", type=int, default=1920, help="Target height (default 1920).")
    args = parser.parse_args()

    from utils.logger import setup_logger
    setup_logger()

    result = crop_to_vertical(args.input, args.output, args.width, args.height)
    print(f"Output: {result}")
