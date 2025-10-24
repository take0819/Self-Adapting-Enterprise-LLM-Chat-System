# components/vdb.py
"""
軽量なベクトルDBの簡易実装（ファイルベース永続化つき）。
- add(id, vector=None, text=None, meta=None)
- query(query_or_vector, top_k=5) -> List[(id, score)]
- save(path), load(path)
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json
import math
import hashlib
import logging
import os

logger = logging.getLogger("components.vdb")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)


def _text_to_vector(text: str, dim: int = 384) -> List[float]:
    """
    単純ハッシュベースの埋め込みスタブ（決定的）。
    実用時は埋め込みモデルに置き換えてください。
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # expand to dim floats in [-1,1]
    vec = []
    for i in range(dim):
        b = h[i % len(h)]
        vec.append(((b / 255.0) * 2.0) - 1.0)
    # normalize
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine_sim(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    la = math.sqrt(sum(x * x for x in a)) or 1.0
    lb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (la * lb)


@dataclass
class _VDBItem:
    id: str
    vector: List[float]
    meta: Dict[str, Any] = field(default_factory=dict)
    text: Optional[str] = None


class VDB:
    def __init__(self, path: Optional[str] = None, dim: int = 384):
        self.path = path
        self.dim = dim
        self._index: Dict[str, _VDBItem] = {}
        if path and os.path.exists(path):
            try:
                self.load(path)
            except Exception:
                logger.exception("Failed to load VDB at init")

    def add(self, id: str, vector: Optional[List[float]] = None, text: Optional[str] = None, meta: Optional[Dict[str, Any]] = None):
        meta = meta or {}
        if vector is None:
            if text is None:
                raise ValueError("Either vector or text must be provided")
            vector = _text_to_vector(text, dim=self.dim)
        # if vector length mismatch, normalize/truncate/pad
        if len(vector) != self.dim:
            # simple adjust: pad with zeros or truncate
            vector = (vector + [0.0] * self.dim)[: self.dim]
        self._index[id] = _VDBItem(id=id, vector=vector, meta=meta, text=text)
        logger.debug("VDB add id=%s", id)

    def query(self, query_or_vector: Any, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        query_or_vector: str (text) or list[float].
        戻り値: list of (id, score) sorted by score desc (cosine similarity).
        """
        if isinstance(query_or_vector, str):
            qv = _text_to_vector(query_or_vector, dim=self.dim)
        else:
            qv = list(query_or_vector)
            if len(qv) != self.dim:
                qv = (qv + [0.0] * self.dim)[: self.dim]

        scores = []
        for id, item in self._index.items():
            try:
                sim = _cosine_sim(qv, item.vector)
            except Exception:
                sim = 0.0
            scores.append((id, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.path
        if not path:
            raise ValueError("No path provided to save VDB")
        serial = {}
        for id, it in self._index.items():
            serial[id] = {"vector": it.vector, "meta": it.meta, "text": it.text}
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"dim": self.dim, "index": serial}, f, ensure_ascii=False)
        logger.info("VDB saved to %s", path)

    def load(self, path: Optional[str] = None) -> None:
        path = path or self.path
        if not path or not os.path.exists(path):
            raise ValueError("No VDB file to load")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.dim = data.get("dim", self.dim)
        idx = data.get("index", {})
        self._index = {}
        for id, v in idx.items():
            self._index[id] = _VDBItem(id=id, vector=v.get("vector", [0.0] * self.dim), meta=v.get("meta", {}), text=v.get("text"))
        logger.info("VDB loaded from %s (%d items)", path, len(self._index))

    def clear(self) -> None:
        self._index.clear()
        logger.info("VDB cleared")

    def stats(self) -> Dict[str, Any]:
        return {"n_items": len(self._index), "dim": self.dim}
