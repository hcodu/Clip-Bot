"""
audio_mixer.py

Stage 5: Mix TTS voiceover at low volume over the original clip audio.

Usage (CLI):
    python -m modules.audio_mixer --video clip.mp4 --tts tts.wav --output final.mp4
"""

import argparse
import subprocess
from pathlib import Path

import ffmpeg

from utils.file_utils import ensure_dirs
from utils.logger import get_logger

log = get_logger("audio_mixer")

# Warn if TTS audio exceeds clip duration by this many seconds
TTS_DURATION_WARN_THRESHOLD = 5.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mix_audio(
    video_path: str | Path,
    tts_audio_path: str | Path,
    output_path: str | Path,
    tts_volume: float = 0.25,
    original_volume: float = 1.0,
    tts_start_offset: float = 0.0,
    output_codec: str = "aac",
    output_bitrate: str = "192k",
) -> Path:
    """
    Mix TTS audio over the original clip audio and produce a final MP4.

    - TTS audio is resampled to 48kHz and mixed at tts_volume (0.0–1.0).
    - Original audio plays at original_volume.
    - amix duration=first ensures output matches video duration exactly.
    - Video stream is copied without re-encoding.
    Returns output_path.
    """
    video_path = Path(video_path)
    tts_audio_path = Path(tts_audio_path)
    output_path = Path(output_path)
    ensure_dirs(output_path.parent)

    # Check duration alignment and warn if TTS is much longer
    video_duration = _get_duration(video_path)
    tts_duration = _get_duration(tts_audio_path)

    if tts_duration and video_duration:
        overage = tts_duration - video_duration
        if overage > TTS_DURATION_WARN_THRESHOLD:
            log.warning(
                "TTS audio (%.1fs) is %.1fs longer than clip (%.1fs). "
                "It will be cut off at clip end.",
                tts_duration, overage, video_duration,
            )

    # Build the FFmpeg filter graph
    # [0:a] = original video audio, [1:a] = TTS wav
    delay_ms = int(tts_start_offset * 1000)
    tts_filter = f"volume={tts_volume},aresample=48000"
    if delay_ms > 0:
        tts_filter += f",adelay={delay_ms}|{delay_ms}"

    filter_complex = (
        f"[0:a]volume={original_volume}[orig];"
        f"[1:a]{tts_filter}[tts];"
        f"[orig][tts]amix=inputs=2:duration=first:dropout_transition=0[aout]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(tts_audio_path),
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", output_codec,
        "-b:a", output_bitrate,
        "-movflags", "+faststart",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio mix failed for {video_path.name}:\n{result.stderr}")

    log.info("Mixed audio: %s → %s", video_path.name, output_path.name)
    return output_path


def mix_audio_batch(
    clip_metadata_list: list[dict],
    output_dir: str | Path,
    tts_volume: float = 0.25,
    original_volume: float = 1.0,
    tts_start_offset: float = 0.5,
    output_codec: str = "aac",
    output_bitrate: str = "192k",
) -> list[dict]:
    """
    Mix TTS audio for all clips. Each dict needs 'cropped_path' and
    'tts_audio_path'. Adds 'mixed_path' key. Returns updated list.
    """
    output_dir = Path(output_dir)
    ensure_dirs(output_dir)

    for clip in clip_metadata_list:
        clip_id = clip.get("clip_id", "unknown")
        video = clip.get("cropped_path") or clip.get("path")
        tts_audio = clip.get("tts_audio_path")

        if not video:
            log.warning("No video path for clip %s, skipping mix.", clip_id)
            clip["mixed_path"] = None
            continue

        if not tts_audio:
            log.info("No TTS audio for clip %s, using original audio only.", clip_id)
            clip["mixed_path"] = Path(video)
            continue

        output_path = output_dir / f"{clip_id}_mixed.mp4"
        try:
            mix_audio(
                video_path=video,
                tts_audio_path=tts_audio,
                output_path=output_path,
                tts_volume=tts_volume,
                original_volume=original_volume,
                tts_start_offset=tts_start_offset,
                output_codec=output_codec,
                output_bitrate=output_bitrate,
            )
            clip["mixed_path"] = output_path
            clip["tts_volume"] = tts_volume
        except Exception as e:
            log.error("Audio mix failed for clip %s: %s", clip_id, e)
            clip["mixed_path"] = None
            clip["mix_error"] = str(e)

    return clip_metadata_list


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_duration(path: Path) -> float | None:
    """Return media duration in seconds using ffprobe, or None on error."""
    try:
        probe = ffmpeg.probe(str(path))
        return float(probe["format"]["duration"])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mix TTS audio over a video clip.")
    parser.add_argument("--video", required=True, help="Input video clip.")
    parser.add_argument("--tts", required=True, help="TTS audio .wav file.")
    parser.add_argument("--output", required=True, help="Output video path.")
    parser.add_argument("--tts-volume", type=float, default=0.25, help="TTS track volume (0.0–1.0).")
    parser.add_argument("--original-volume", type=float, default=1.0, help="Original audio volume.")
    parser.add_argument("--offset", type=float, default=0.5, help="Seconds to delay TTS onset.")
    args = parser.parse_args()

    from utils.logger import setup_logger
    setup_logger()

    result = mix_audio(
        video_path=args.video,
        tts_audio_path=args.tts,
        output_path=args.output,
        tts_volume=args.tts_volume,
        original_volume=args.original_volume,
        tts_start_offset=args.offset,
    )
    print(f"Output: {result}")
