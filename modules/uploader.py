"""
uploader.py

Stage 7: Playwright-based TikTok uploader. Posts clips from the queue
on manual demand.

Usage (CLI):
    # First-time auth: save session cookies
    python -m modules.uploader save-cookies --cookies ./cookies/tiktok_state.json

    # Upload the next pending clip
    python -m modules.uploader upload --queue ./queue --cookies ./cookies/tiktok_state.json

    # Upload up to 3 pending clips
    python -m modules.uploader upload --queue ./queue --max 3
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

from utils.logger import get_logger
from modules.queue_manager import get_queue, mark_clip_status

log = get_logger("uploader")

TIKTOK_UPLOAD_URL = "https://www.tiktok.com/upload"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_cookies(
    cookies_path: str | Path,
) -> None:
    """
    Open a browser window, let the user log in to TikTok manually,
    then save the session state to cookies_path.
    Run this once before using upload_clip.
    """
    from playwright.sync_api import sync_playwright

    cookies_path = Path(cookies_path)
    cookies_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Opening browser — please log in to TikTok, then close the tab.")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            channel="chrome",
        )
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://www.tiktok.com/login")

        print("\n[uploader] Log in to TikTok in the browser window.")
        print("[uploader] Once logged in, press ENTER here to save cookies.")
        input()

        context.storage_state(path=str(cookies_path))
        log.info("Cookies saved to %s", cookies_path)
        browser.close()


def upload_clip(
    clip_id: str,
    queue_dir: str | Path,
    headless: bool = False,
    cookies_path: str | Path | None = None,
    default_hashtags: list[str] | None = None,
    title_max_length: int = 100,
) -> bool:
    """
    Upload a single clip from the queue to TikTok.
    Updates queue JSON to 'uploaded' or 'failed'.
    Returns True on success.
    """
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    queue_dir = Path(queue_dir)
    clips = get_queue(queue_dir)
    meta = next((c for c in clips if c["clip_id"] == clip_id), None)

    if meta is None:
        raise FileNotFoundError(f"Clip {clip_id} not found in queue.")

    video_path = Path(meta["files"]["final_video"])
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Build caption
    transcript = meta.get("transcription", {}).get("full_text", "")
    caption = _build_caption(transcript, default_hashtags or [], title_max_length)

    mark_clip_status(clip_id, queue_dir, "uploading", {
        "attempted_at": datetime.now(timezone.utc).isoformat()
    })

    with sync_playwright() as p:
        launch_kwargs: dict = {"headless": headless}
        try:
            browser = p.chromium.launch(channel="chrome", **launch_kwargs)
        except Exception:
            # Fall back to bundled Chromium if system Chrome not found
            log.warning("System Chrome not found, falling back to bundled Chromium.")
            browser = p.chromium.launch(**launch_kwargs)

        context_kwargs: dict = {}
        if cookies_path and Path(cookies_path).exists():
            context_kwargs["storage_state"] = str(cookies_path)

        context = browser.new_context(**context_kwargs)
        page = context.new_page()

        try:
            success = _run_upload_flow(page, video_path, caption)
        except PWTimeout as e:
            log.error("Upload timed out for %s: %s", clip_id, e)
            success = False
        except Exception as e:
            log.error("Upload failed for %s: %s", clip_id, e)
            success = False
        finally:
            browser.close()

    if success:
        mark_clip_status(clip_id, queue_dir, "uploaded", {
            "uploaded_at": datetime.now(timezone.utc).isoformat()
        })
        log.info("Successfully uploaded clip %s", clip_id)
    else:
        mark_clip_status(clip_id, queue_dir, "failed", {
            "error": "Upload flow did not complete."
        })
        log.error("Upload failed for clip %s", clip_id)

    return success


def upload_batch(
    queue_dir: str | Path,
    max_clips: int = 1,
    delay_between: float = 30.0,
    cookies_path: str | Path | None = None,
    default_hashtags: list[str] | None = None,
    title_max_length: int = 100,
) -> list[dict]:
    """
    Upload up to max_clips pending clips from queue_dir sequentially.
    Returns list of result dicts: [{clip_id, success}, ...]
    """
    queue_dir = Path(queue_dir)
    pending = get_queue(queue_dir, status_filter="pending")

    if not pending:
        log.info("No pending clips in queue.")
        return []

    to_upload = pending[:max_clips]
    results = []

    for i, clip_meta in enumerate(to_upload):
        clip_id = clip_meta["clip_id"]
        log.info("Uploading clip %d/%d: %s", i + 1, len(to_upload), clip_id)

        success = upload_clip(
            clip_id=clip_id,
            queue_dir=queue_dir,
            cookies_path=cookies_path,
            default_hashtags=default_hashtags,
            title_max_length=title_max_length,
        )
        results.append({"clip_id": clip_id, "success": success})

        if i < len(to_upload) - 1 and delay_between > 0:
            log.info("Waiting %.0fs before next upload...", delay_between)
            time.sleep(delay_between)

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_upload_flow(page, video_path: Path, caption: str) -> bool:
    """
    Execute the TikTok upload UI flow. Returns True on success.

    NOTE: TikTok's DOM changes periodically. Selectors here use aria-label
    and text-based matching for resilience. If this breaks, inspect the
    upload page and update the selectors below.
    """
    log.debug("Navigating to TikTok upload page...")
    page.goto(TIKTOK_UPLOAD_URL, wait_until="networkidle", timeout=30000)

    # Find the file input inside the upload iframe
    upload_iframe = _find_upload_iframe(page)
    if upload_iframe is None:
        log.error("Could not find upload iframe.")
        return False

    file_input = upload_iframe.locator('input[type="file"]')
    file_input.set_input_files(str(video_path))
    log.debug("File selected: %s", video_path.name)

    # Wait for video preview (client-side processing)
    log.debug("Waiting for video to process...")
    _wait_for_video_preview(page, timeout_ms=120000)

    # Fill caption
    _set_caption(page, upload_iframe, caption)

    # Click Post
    _click_post(page, upload_iframe)

    # Wait for success confirmation
    success = _wait_for_upload_complete(page, timeout_ms=60000)
    return success


def _find_upload_iframe(page):
    """
    Locate the TikTok upload iframe. Returns a FrameLocator or None.
    TikTok uses an iframe for the upload widget; this searches by URL pattern.
    """
    try:
        # Try common iframe patterns
        for selector in [
            'iframe[src*="tiktok.com"]',
            'iframe[src*="upload"]',
            'iframe',
        ]:
            iframes = page.locator(selector)
            if iframes.count() > 0:
                return page.frame_locator(selector).first
    except Exception as e:
        log.debug("Iframe search error: %s", e)
    return None


def _wait_for_video_preview(page, timeout_ms: int = 120000) -> None:
    """Wait for the upload preview to appear, indicating processing is done."""
    try:
        # Look for a video element or progress indicator disappearing
        page.wait_for_selector('video', timeout=timeout_ms)
    except Exception:
        log.warning("Timed out waiting for video preview. Proceeding anyway.")


def _set_caption(page, iframe, caption: str) -> None:
    """Fill the caption/title field."""
    try:
        # Try within iframe first, then top-level page
        for locator in [
            iframe.locator('[contenteditable="true"]').first,
            page.locator('[contenteditable="true"]').first,
        ]:
            try:
                locator.click()
                locator.fill(caption)
                log.debug("Caption set: %s...", caption[:50])
                return
            except Exception:
                continue
    except Exception as e:
        log.warning("Could not set caption: %s", e)


def _click_post(page, iframe) -> None:
    """Click the Post/Publish button."""
    try:
        for locator in [
            iframe.get_by_role("button", name="Post"),
            page.get_by_role("button", name="Post"),
            page.get_by_text("Post", exact=True),
        ]:
            try:
                locator.click(timeout=5000)
                log.debug("Post button clicked.")
                return
            except Exception:
                continue
    except Exception as e:
        log.warning("Could not click Post button: %s", e)


def _wait_for_upload_complete(page, timeout_ms: int = 60000) -> bool:
    """
    Wait for a success indicator after clicking Post.
    Returns True if success detected, False on timeout.
    """
    try:
        # TikTok shows a success toast or redirects — look for either
        page.wait_for_url("**/profile*", timeout=timeout_ms)
        return True
    except Exception:
        pass

    try:
        page.wait_for_selector(
            '[class*="success"], [class*="uploaded"], [class*="complete"]',
            timeout=timeout_ms,
        )
        return True
    except Exception:
        pass

    log.warning("Could not confirm upload completion.")
    return False


def _build_caption(transcript: str, hashtags: list[str], max_length: int) -> str:
    """Build the TikTok caption from the transcript + hashtags."""
    tags = " ".join(hashtags)
    available = max_length - len(tags) - 1  # -1 for space separator

    if available > 20 and transcript:
        text_part = transcript[:available].rsplit(" ", 1)[0]  # don't cut mid-word
        return f"{text_part} {tags}".strip()

    return tags


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TikTok clip uploader.")
    subparsers = parser.add_subparsers(dest="command")

    # save-cookies subcommand
    sc = subparsers.add_parser("save-cookies", help="Authenticate and save browser cookies.")
    sc.add_argument("--cookies", default="./cookies/tiktok_state.json", help="Path to save cookies.")

    # upload subcommand
    up = subparsers.add_parser("upload", help="Upload pending clips from queue.")
    up.add_argument("--queue", default="./queue", help="Queue directory.")
    up.add_argument("--cookies", default="./cookies/tiktok_state.json", help="Cookies file.")
    up.add_argument("--max", type=int, default=1, help="Max clips to upload.")
    up.add_argument("--clip-id", default=None, help="Upload a specific clip by ID.")
    up.add_argument("--delay", type=float, default=30.0, help="Seconds between uploads.")

    args = parser.parse_args()

    from utils.logger import setup_logger
    setup_logger()

    if args.command == "save-cookies":
        save_cookies(args.cookies)

    elif args.command == "upload":
        if args.clip_id:
            success = upload_clip(args.clip_id, args.queue, cookies_path=args.cookies)
            print("Success" if success else "Failed")
        else:
            results = upload_batch(
                args.queue,
                max_clips=args.max,
                delay_between=args.delay,
                cookies_path=args.cookies,
            )
            for r in results:
                print(f"  {r['clip_id']}: {'OK' if r['success'] else 'FAILED'}")
    else:
        parser.print_help()
