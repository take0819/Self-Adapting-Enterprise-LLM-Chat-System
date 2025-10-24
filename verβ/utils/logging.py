# utils/logging.py
"""
ロギング初期化ユーティリティ
"""
from __future__ import annotations
import logging
from typing import Optional

def configure_logging(level: int = logging.INFO, fmt: Optional[str] = None) -> None:
    if fmt is None:
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    root = logging.getLogger()
    if not root.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(fmt))
        root.addHandler(h)
    root.setLevel(level)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name or __name__)
