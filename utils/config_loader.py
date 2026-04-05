import yaml
from pathlib import Path
from utils.logger import get_logger

log = get_logger("config_loader")

DEFAULTS = {
    "paths": {
        "temp_dir": "./temp",
        "queue_dir": "./queue",
        "log_file": "./logs/clip_bot.log",
        "cookies_file": "./cookies/tiktok_state.json",
    },
    "scene_detection": {
        "detector": "content",
        "threshold": 27.0,
        "min_clip_duration": 30.0,
        "max_clip_duration": 60.0,
    },
    "crop": {
        "target_width": 1080,
        "target_height": 1920,
    },
    "transcription": {
        "model_size": "base",
        "language": None,
        "device": "auto",
        "extract_audio_first": True,
    },
    "tts": {
        "enabled": True,
        "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
        "speaker_wav": None,
        "language": "en",
        "gpu": False,
        "agree_to_tos": True,
    },
    "audio_mix": {
        "tts_volume": 0.25,
        "original_volume": 1.0,
        "tts_start_offset": 0.5,
        "output_codec": "aac",
        "output_bitrate": "192k",
    },
    "upload": {
        "headless": False,
        "delay_between_uploads": 30.0,
        "max_retries": 2,
        "default_hashtags": ["#fyp", "#tvshow"],
        "title_max_length": 100,
    },
    "logging": {
        "level": "INFO",
        "console": True,
        "file": True,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """
    Load config.yaml and merge with defaults.
    Returns a fully-populated config dict.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        log.warning("Config file not found at %s, using defaults.", config_path)
        return DEFAULTS.copy()

    with open(config_path, "r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    config = _deep_merge(DEFAULTS, user_config)
    log.debug("Config loaded from %s", config_path)
    return config


def resolve_paths(config: dict, base_dir: str | Path | None = None) -> dict:
    """
    Resolve relative paths in config['paths'] against base_dir.
    base_dir defaults to the current working directory.
    Returns config with absolute Path objects in config['paths'].
    """
    base = Path(base_dir) if base_dir else Path.cwd()
    resolved = config.copy()
    resolved["paths"] = {}

    for key, value in config["paths"].items():
        if value:
            p = Path(value)
            resolved["paths"][key] = p if p.is_absolute() else (base / p).resolve()
        else:
            resolved["paths"][key] = None

    return resolved
