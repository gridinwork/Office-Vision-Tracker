"""Office Vision Tracker entry point."""

from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
import logging, sys, threading
from PySide6.QtWidgets import QApplication
from app.main_window import MainWindow, apply_dark_palette
from app.settings import LOGS_DIR

def configure_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger(); root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    file_handler = logging.FileHandler(LOGS_DIR / "app.log", encoding="utf-8"); file_handler.setFormatter(formatter)
    root.handlers.clear(); root.addHandler(file_handler)
    if sys.stderr is not None:
        stream_handler = logging.StreamHandler(); stream_handler.setFormatter(formatter); root.addHandler(stream_handler)

def _thread_exception(args: threading.ExceptHookArgs) -> None:
    logging.getLogger("office_vision").critical("Unhandled thread exception in %s", getattr(args.thread, "name", "thread"), exc_info=(args.exc_type,args.exc_value,args.exc_traceback))

def main() -> int:
    configure_logging(); logging.getLogger("office_vision").info("Office Vision Tracker started")
    sys.excepthook=lambda exc_type,exc,tb: logging.getLogger("office_vision").critical("Unhandled exception",exc_info=(exc_type,exc,tb))
    threading.excepthook=_thread_exception
    app=QApplication(sys.argv); app.setApplicationName("Office Vision Tracker"); app.setStyle("Fusion"); apply_dark_palette(app)
    window=MainWindow(); window.show(); return app.exec()

if __name__ == "__main__": raise SystemExit(main())
