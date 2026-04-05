"""
tts_generator.py

Stage 4: Generate a TTS voiceover from transcript text using local Coqui TTS.

Usage (CLI):
    python -m modules.tts_generator --text "Hello world" --output out.wav
"""

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from utils.file_utils import ensure_dirs
from utils.logger import get_logger

if TYPE_CHECKING:
    from TTS.api import TTS as TTSModel

log = get_logger("tts_generator")

# Module-level model cache: {cache_key: TTS instance}
_TTS_CACHE: dict[str, "TTSModel"] = {}

DEFAULT_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_tts(
    text: str,
    output_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    speaker_wav: str | Path | None = None,
    language: str = "en",
    gpu: bool = False,
    agree_to_tos: bool = True,
) -> Path:
    """
    Synthesize speech from text and save to output_path (.wav).
    Returns output_path.
    """
    if not text or not text.strip():
        raise ValueError("TTS text cannot be empty.")

    output_path = Path(output_path)
    ensure_dirs(output_path.parent)

    model = load_tts_model(model_name, gpu=gpu, agree_to_tos=agree_to_tos)

    tts_kwargs: dict = {"text": text, "file_path": str(output_path)}

    # Multi-speaker / multilingual models require speaker_wav or language
    if speaker_wav:
        tts_kwargs["speaker_wav"] = str(speaker_wav)
        tts_kwargs["language"] = language
    elif model.is_multi_lingual:
        tts_kwargs["language"] = language

    model.tts_to_file(**tts_kwargs)
    log.info("TTS generated: %s (%d chars → %s)", output_path.name, len(text), output_path)
    return output_path


def load_tts_model(
    model_name: str = DEFAULT_MODEL,
    gpu: bool = False,
    agree_to_tos: bool = True,
) -> "TTSModel":
    """
    Load and cache a Coqui TTS model. Returns cached instance on repeat calls.
    """
    from TTS.api import TTS

    cache_key = f"{model_name}_{'gpu' if gpu else 'cpu'}"
    if cache_key not in _TTS_CACHE:
        log.info("Loading TTS model '%s'...", model_name)
        model = TTS(model_name=model_name, progress_bar=False, gpu=gpu)
        _TTS_CACHE[cache_key] = model
        log.info("TTS model loaded.")
    return _TTS_CACHE[cache_key]


def generate_tts_batch(
    clip_metadata_list: list[dict],
    output_dir: str | Path,
    model_name: str = DEFAULT_MODEL,
    gpu: bool = False,
    agree_to_tos: bool = True,
) -> list[dict]:
    """
    Generate TTS audio for each clip. Reads transcription.full_text from
    each clip dict. Adds 'tts_audio_path' key. Returns updated list.
    """
    output_dir = Path(output_dir)
    ensure_dirs(output_dir)

    # Pre-load model once
    load_tts_model(model_name, gpu=gpu, agree_to_tos=agree_to_tos)

    for clip in clip_metadata_list:
        clip_id = clip.get("clip_id", "unknown")
        text = clip.get("transcription", {}).get("full_text", "").strip()

        if not text:
            log.warning("No transcription text for clip %s, skipping TTS.", clip_id)
            clip["tts_audio_path"] = None
            continue

        output_path = output_dir / f"{clip_id}.wav"
        try:
            generate_tts(
                text=text,
                output_path=output_path,
                model_name=model_name,
                gpu=gpu,
                agree_to_tos=agree_to_tos,
            )
            clip["tts_audio_path"] = output_path
            clip["tts_model"] = model_name
        except Exception as e:
            log.error("TTS failed for clip %s: %s", clip_id, e)
            clip["tts_audio_path"] = None
            clip["tts_error"] = str(e)

    return clip_metadata_list


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TTS audio from text using Coqui TTS.")
    parser.add_argument("--text", help="Text to synthesize.")
    parser.add_argument("--text-file", help="File containing text to synthesize.")
    parser.add_argument("--output", required=True, help="Output .wav file path.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Coqui TTS model name.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference.")
    args = parser.parse_args()

    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    elif args.text:
        text = args.text
    else:
        parser.error("Provide --text or --text-file.")

    from utils.logger import setup_logger
    setup_logger()

    result = generate_tts(text=text, output_path=args.output, model_name=args.model, gpu=args.gpu)
    print(f"Output: {result}")
