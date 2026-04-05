"""
transcriber.py

Stage 3: Transcribe clip audio using local OpenAI Whisper.

Usage (CLI):
    python -m modules.transcriber --input clip.mp4 --model-size base
"""

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from utils.logger import get_logger

if TYPE_CHECKING:
    import whisper as whisper_module

log = get_logger("transcriber")

# Module-level model cache: {model_size: model_instance}
_MODEL_CACHE: dict[str, "whisper_module.Whisper"] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe_clip(
    clip_path: str | Path,
    model_size: str = "base",
    language: str | None = None,
    word_timestamps: bool = True,
    device: str = "auto",
) -> dict:
    """
    Transcribe a video or audio clip using local Whisper.

    Returns the Whisper result dict, augmented with a 'full_text' key:
    {
      "text": "...",
      "language": "en",
      "segments": [...],
      "full_text": "...",
      "whisper_model": "base",
    }
    """
    import torch

    clip_path = Path(clip_path)
    resolved_device = _resolve_device(device)
    model = load_whisper_model(model_size, resolved_device)

    # Pre-extract audio to a temp WAV for reliability
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)

    try:
        _extract_audio_to_wav(clip_path, wav_path)

        kwargs: dict = {
            "language": language,
            "word_timestamps": word_timestamps,
        }
        if resolved_device == "cpu":
            kwargs["fp16"] = False

        result = model.transcribe(str(wav_path), **kwargs)
    finally:
        wav_path.unlink(missing_ok=True)

    result["full_text"] = result.get("text", "").strip()
    result["whisper_model"] = model_size
    log.info(
        "Transcribed %s: %d chars, language=%s",
        clip_path.name,
        len(result["full_text"]),
        result.get("language"),
    )
    return result


def load_whisper_model(
    model_size: str = "base",
    device: str = "auto",
) -> "whisper_module.Whisper":
    """
    Load and cache a Whisper model. Subsequent calls with the same
    model_size return the cached instance.
    """
    import whisper

    cache_key = f"{model_size}_{device}"
    if cache_key not in _MODEL_CACHE:
        resolved = _resolve_device(device)
        log.info("Loading Whisper model '%s' on %s...", model_size, resolved)
        _MODEL_CACHE[cache_key] = whisper.load_model(model_size, device=resolved)
        log.info("Whisper model loaded.")
    return _MODEL_CACHE[cache_key]


def transcribe_batch(
    clip_metadata_list: list[dict],
    model_size: str = "base",
    language: str | None = None,
    device: str = "auto",
) -> list[dict]:
    """
    Transcribe all clips in clip_metadata_list. Each dict must have a
    'cropped_path' key (falls back to 'path'). Adds 'transcription' key
    to each dict. Returns the updated list.
    """
    # Load model once for all clips
    resolved_device = _resolve_device(device)
    load_whisper_model(model_size, resolved_device)

    for clip in clip_metadata_list:
        source = clip.get("cropped_path") or clip.get("path")
        if not source:
            log.warning("No video path for clip %s, skipping transcription.", clip.get("clip_id"))
            clip["transcription"] = {}
            continue

        try:
            result = transcribe_clip(
                source,
                model_size=model_size,
                language=language,
                device=device,
            )
            clip["transcription"] = {
                "full_text": result["full_text"],
                "language": result.get("language"),
                "whisper_model": model_size,
                "segments": [
                    {
                        "id": s["id"],
                        "start": s["start"],
                        "end": s["end"],
                        "text": s["text"].strip(),
                    }
                    for s in result.get("segments", [])
                ],
            }
        except Exception as e:
            log.error("Transcription failed for %s: %s", clip.get("clip_id"), e)
            clip["transcription"] = {"full_text": "", "error": str(e)}

    return clip_metadata_list


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


def _extract_audio_to_wav(video_path: Path, wav_path: Path) -> None:
    """Extract audio from video to a 16kHz mono WAV using FFmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed:\n{result.stderr}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe a clip with local Whisper.")
    parser.add_argument("--input", required=True, help="Input video or audio file.")
    parser.add_argument("--model-size", default="base",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size.")
    parser.add_argument("--language", default=None, help="Language code (e.g. 'en'), or omit for auto.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    from utils.logger import setup_logger
    setup_logger()

    result = transcribe_clip(args.input, args.model_size, args.language, device=args.device)
    print(result["full_text"])
