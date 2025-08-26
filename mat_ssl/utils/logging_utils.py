import logging
import os
from datetime import datetime
from typing import Optional


def setup_logging(script_name: str, base_log_dir: Optional[str] = None, level: int = logging.INFO) -> str:
    """
    Configure Python logging to file under mat_ssl/logs and to console.

    Returns the log file path.
    """
    # Determine base project dir (mat_ssl package directory's parent)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # mat_ssl/
    default_log_dir = os.path.join(here, "logs")
    log_dir = base_log_dir or default_log_dir
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(log_dir, f"{script_name}_{timestamp}.log")

    # Root logger config (avoid duplicate handlers if re-called)
    root = logging.getLogger()
    if root.handlers:
        for h in list(root.handlers):
            root.removeHandler(h)

    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    logging.getLogger(__name__).info(f"Logging initialized: {log_path}")
    return log_path

