# utils/io.py
"""
軽量 I/O ユーティリティ: JSON の安全な書き込み／読み込み、atomic write
"""
from __future__ import annotations
import json
import os
import tempfile
from typing import Any, Optional

def atomic_write_json(path: str, data: Any, encoding: str = "utf-8") -> None:
    """atomic に JSON ファイルを書き込む"""
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d or ".", prefix=".tmp", text=True)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

def save_json(path: str, data: Any) -> None:
    atomic_write_json(path, data)

def load_json(path: str, default: Optional[Any] = None) -> Any:
    if not path or not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
