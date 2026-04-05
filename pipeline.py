"""
pipeline.py

Orchestrates the full Clip Bot pipeline for a single source video.

Stages:
  1. detect    — scene detection + raw clip export
  2. crop      — 16:9 → 9:16 vertical conversion
  3. transcribe — Whisper transcription
  4. tts       — Coqui TTS voiceover generation
  5. mix       — TTS audio overlay at low volume
  6. queue     — save finished clips to queue

Usage:
    python pipeline.py --input episode.mp4
    python pipeline.py --input episode.mp4 --resume-from tts
    python pipeline.py --input episode.mp4 --config my_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_config, resolve_paths
from utils.file_utils import make_temp_dirs, ensure_dirs, video_file_is_valid
from utils.logger import setup_logger, get_logger

log = get_logger("pipeline")

STAGES = ["detect", "crop", "transcribe", "tts", "mix", "queue"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    video_path: str | Path,
    config: dict,
    temp_dir: str | Path | None = None,
    resume_from: str | None = None,
) -> list[dict]:
    """
    Run the full clip bot pipeline on a single source video.

    Args:
        video_path: Path to the source video file.
        config: Loaded and path-resolved config dict.
        temp_dir: Override temp directory (defaults to config paths).
        resume_from: Skip stages before this one (for crash recovery).
                     One of: "detect"|"crop"|"transcribe"|"tts"|"mix"|"queue"

    Returns:
        List of enqueued clip metadata dicts.
    """
    video_path = Path(video_path)
    if not video_file_is_valid(video_path):
        raise FileNotFoundError(f"Video file not found or unsupported: {video_path}")

    # Resolve directories
    paths = config["paths"]
    actual_temp = Path(temp_dir) if temp_dir else paths["temp_dir"]
    queue_dir = paths["queue_dir"]
    temp_dirs = make_temp_dirs(actual_temp)
    ensure_dirs(queue_dir)

    sd_cfg = config["scene_detection"]
    crop_cfg = config["crop"]
    tr_cfg = config["transcription"]
    tts_cfg = config["tts"]
    mix_cfg = config["audio_mix"]

    start_index = STAGES.index(resume_from) if resume_from in STAGES else 0
    log.info("Starting pipeline from stage: %s", STAGES[start_index])

    # -----------------------------------------------------------------------
    # Stage 1: Scene detection
    # -----------------------------------------------------------------------
    clips: list[dict] = []

    if start_index <= STAGES.index("detect"):
        log.info("=== Stage 1: Scene Detection ===")
        from modules.scene_detector import detect_scenes
        clips = detect_scenes(
            video_path=video_path,
            output_dir=temp_dirs["raw_clips"],
            threshold=sd_cfg["threshold"],
            min_clip_duration=sd_cfg["min_clip_duration"],
            max_clip_duration=sd_cfg["max_clip_duration"],
            detector=sd_cfg["detector"],
        )
        log.info("Scene detection complete: %d clips", len(clips))
    else:
        log.info("Skipping stage: detect")

    if not clips:
        log.warning("No clips to process. Exiting.")
        return []

    # -----------------------------------------------------------------------
    # Stage 2: Crop to vertical
    # -----------------------------------------------------------------------
    if start_index <= STAGES.index("crop"):
        log.info("=== Stage 2: Crop to 9:16 Vertical ===")
        from modules.crop_converter import batch_crop
        clips = batch_crop(
            clip_metadata_list=clips,
            output_dir=temp_dirs["cropped_clips"],
            target_width=crop_cfg["target_width"],
            target_height=crop_cfg["target_height"],
        )
        successful = sum(1 for c in clips if c.get("cropped_path"))
        log.info("Crop complete: %d/%d clips succeeded", successful, len(clips))
    else:
        log.info("Skipping stage: crop")

    # -----------------------------------------------------------------------
    # Stage 3: Transcription
    # -----------------------------------------------------------------------
    if start_index <= STAGES.index("transcribe"):
        log.info("=== Stage 3: Transcription ===")
        from modules.transcriber import transcribe_batch
        clips = transcribe_batch(
            clip_metadata_list=clips,
            model_size=tr_cfg["model_size"],
            language=tr_cfg.get("language"),
            device=tr_cfg["device"],
        )
        transcribed = sum(1 for c in clips if c.get("transcription", {}).get("full_text"))
        log.info("Transcription complete: %d/%d clips", transcribed, len(clips))
    else:
        log.info("Skipping stage: transcribe")

    # -----------------------------------------------------------------------
    # Stage 4: TTS generation
    # -----------------------------------------------------------------------
    if tts_cfg.get("enabled", True) and start_index <= STAGES.index("tts"):
        log.info("=== Stage 4: TTS Generation ===")
        from modules.tts_generator import generate_tts_batch
        clips = generate_tts_batch(
            clip_metadata_list=clips,
            output_dir=temp_dirs["tts_audio"],
            model_name=tts_cfg["model_name"],
            gpu=tts_cfg.get("gpu", False),
            agree_to_tos=tts_cfg.get("agree_to_tos", True),
        )
        tts_ok = sum(1 for c in clips if c.get("tts_audio_path"))
        log.info("TTS complete: %d/%d clips", tts_ok, len(clips))
    else:
        log.info("Skipping stage: tts (disabled or resume skip)")

    # -----------------------------------------------------------------------
    # Stage 5: Audio mix
    # -----------------------------------------------------------------------
    if start_index <= STAGES.index("mix"):
        log.info("=== Stage 5: Audio Mix ===")
        from modules.audio_mixer import mix_audio_batch

        # Mixed clips go directly into the queue dir as the final output
        clips = mix_audio_batch(
            clip_metadata_list=clips,
            output_dir=queue_dir,
            tts_volume=mix_cfg["tts_volume"],
            original_volume=mix_cfg["original_volume"],
            tts_start_offset=mix_cfg["tts_start_offset"],
            output_codec=mix_cfg["output_codec"],
            output_bitrate=mix_cfg["output_bitrate"],
        )
        mixed_ok = sum(1 for c in clips if c.get("mixed_path"))
        log.info("Mix complete: %d/%d clips", mixed_ok, len(clips))
    else:
        log.info("Skipping stage: mix")

    # -----------------------------------------------------------------------
    # Stage 6: Enqueue
    # -----------------------------------------------------------------------
    enqueued = []
    if start_index <= STAGES.index("queue"):
        log.info("=== Stage 6: Enqueue ===")
        from modules.queue_manager import enqueue_clip, build_metadata

        for clip in clips:
            final_video = clip.get("mixed_path") or clip.get("cropped_path") or clip.get("path")
            if not final_video or not Path(final_video).exists():
                log.warning("No final video for clip %s, skipping enqueue.", clip.get("clip_id"))
                continue

            meta = build_metadata(clip, video_path)
            meta["tts_volume"] = mix_cfg["tts_volume"]

            try:
                queued = enqueue_clip(final_video, meta, queue_dir)
                enqueued.append(queued)
            except Exception as e:
                log.error("Failed to enqueue clip %s: %s", clip.get("clip_id"), e)
    else:
        log.info("Skipping stage: queue")

    log.info("Pipeline complete. %d clips enqueued.", len(enqueued))
    return enqueued


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clip Bot — process a TV episode into TikTok-ready clips."
    )
    parser.add_argument("--input", required=True, help="Path to source video file.")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config.yaml (default: config.yaml)."
    )
    parser.add_argument(
        "--resume-from",
        choices=STAGES,
        default=None,
        help="Resume pipeline from this stage (skips earlier stages).",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Override temp directory from config.",
    )
    args = parser.parse_args()

    # Boot logger early
    raw_config = load_config(args.config)
    log_cfg = raw_config.get("logging", {})
    setup_logger(
        log_file=raw_config.get("paths", {}).get("log_file"),
        level=log_cfg.get("level", "INFO"),
        console=log_cfg.get("console", True),
    )

    config = resolve_paths(raw_config)

    results = run_pipeline(
        video_path=args.input,
        config=config,
        temp_dir=args.temp_dir,
        resume_from=args.resume_from,
    )

    print(f"\nDone. {len(results)} clip(s) in queue:")
    for r in results:
        print(f"  [{r['status']}] {r['clip_id']}  {r['source']['duration_sec']:.1f}s")
