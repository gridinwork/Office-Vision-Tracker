"""Download MediaPipe .task models into models/. Safe to run more than once."""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

from app.settings import MIN_MODEL_BYTES, MODEL_URLS, MODELS_DIR


def _download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "OfficeVisionTracker/1.0"})
    print(f"Downloading {destination.name}")
    print(f"  {url}")
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length", "0") or 0)
        downloaded = 0
        with temporary.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk: break
                handle.write(chunk); downloaded += len(chunk)
                if total:
                    percent = downloaded * 100 // total
                    print(f"\r  {percent:3d}%  {downloaded // 1024} KB", end="", flush=True)
        if total: print()
    if downloaded < MIN_MODEL_BYTES:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is too small: {destination.name}")
    temporary.replace(destination)


def ensure_models() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, urls in MODEL_URLS.items():
        destination = MODELS_DIR / name
        if destination.exists() and destination.stat().st_size >= MIN_MODEL_BYTES:
            print(f"Already present: {name} ({destination.stat().st_size // 1024} KB)")
            continue
        last_error = None
        for url in urls:
            try:
                _download(url, destination); last_error = None; break
            except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
                last_error = exc; print(f"  failed: {exc}")
        if last_error is not None or not destination.exists():
            print(f"Model file not found: {name}", file=sys.stderr); raise SystemExit(1)
        print(f"Saved {name} ({destination.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    try: ensure_models()
    except SystemExit: raise
    except Exception as exc:
        print(f"Model download failed: {exc}", file=sys.stderr); raise SystemExit(1) from exc
