import shutil
import uuid
from pathlib import Path
from utils.logger import get_logger

log = get_logger("file_utils")


def ensure_dirs(*dirs: str | Path) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def make_temp_dirs(temp_dir: str | Path) -> dict[str, Path]:
    """
    Create and return the standard temp subdirectories.
    Returns dict with keys: raw_clips, cropped_clips, tts_audio.
    """
    base = Path(temp_dir)
    subdirs = {
        "raw_clips": base / "raw_clips",
        "cropped_clips": base / "cropped_clips",
        "tts_audio": base / "tts_audio",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def short_uuid() -> str:
    """Return a 6-character hex UUID prefix."""
    return uuid.uuid4().hex[:6]


def make_clip_id(prefix: str = "clip") -> str:
    """Generate a unique clip ID like 'clip_a3f2b1'."""
    return f"{prefix}_{short_uuid()}"


def safe_copy(src: str | Path, dst: str | Path) -> Path:
    """Copy src to dst, creating parent directories as needed."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))
    log.debug("Copied %s → %s", src, dst)
    return dst


def clean_temp(temp_dir: str | Path) -> None:
    """Delete all files in temp subdirectories (not the dirs themselves)."""
    base = Path(temp_dir)
    for subdir in ("raw_clips", "cropped_clips", "tts_audio"):
        target = base / subdir
        if target.exists():
            for f in target.iterdir():
                if f.is_file():
                    f.unlink()
    log.info("Temp directory cleaned: %s", temp_dir)


def video_file_is_valid(path: str | Path) -> bool:
    """Basic check: file exists and has a recognized video extension."""
    p = Path(path)
    return p.exists() and p.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".ts"}
