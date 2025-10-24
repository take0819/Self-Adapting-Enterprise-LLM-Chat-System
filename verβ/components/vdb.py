# components/vdb.py
"""
高機能 VDB (Vector DB) 実装 - brute-force + optional numpy acceleration

特徴:
- text -> vector のデフォルト埋め込みはハッシュベースの決定的スタブ (差し替え可能)
- numpy があれば高速なベクトル演算を行う
- add / upsert / remove / query / save / load / export_dataset を提供
- query は top_k を返し、(id, score, meta, text) の辞書リストを返す

永続化:
- numpy 利用時は .npz に保存（ベクトル配列を効率的に保存）
- numpy が無い場合は json で保存（ベクトルは list に変換）
"""
from __future__ import annotations
import os
import json
import math
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

logger = logging.getLogger("components.vdb")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


def _default_text_to_vector(text: str, dim: int = 384) -> List[float]:
    """
    デフォルト埋め込みスタブ（決定的）。実運用ではここを実際の埋め込みモデルに差し替えてください。
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vec = [(b / 255.0) * 2.0 - 1.0 for b in h]
    # expand/truncate to dim
    if len(vec) < dim:
        vec = (vec * ((dim // len(vec)) + 1))[:dim]
    else:
        vec = vec[:dim]
    # normalize
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine_similarity_np(query: "np.ndarray", matrix: "np.ndarray") -> "np.ndarray":
    # query: (D,), matrix: (N,D)
    qn = np.linalg.norm(query) or 1.0
    mn = np.linalg.norm(matrix, axis=1)
    denom = qn * mn
    # avoid divide by zero
    denom[denom == 0] = 1.0
    sims = (matrix @ query) / denom
    return sims


class VDB:
    def __init__(self, dim: int = 384, path: Optional[str] = None, text_to_vector_fn=None, use_numpy: Optional[bool] = None):
        """
        Args:
            dim: 埋め込み次元
            path: デフォルト保存パス（省略可）
            text_to_vector_fn: text->vector の関数（None なら内部スタブ）
            use_numpy: None=自動判定(numpy があれば True)、True/False を明示的に指定可
        """
        self.dim = dim
        self.path = path
        self._text_to_vector = text_to_vector_fn or (lambda t: _default_text_to_vector(t, dim=self.dim))
        self._items: Dict[str, Dict[str, Any]] = {}  # id -> {"vector": list/np.array, "text": str, "meta": dict}
        self._use_numpy = use_numpy if use_numpy is not None else (np is not None)
        # cache matrix for performance (np array shape (N,D))
        self._matrix_cache_valid = False
        self._matrix = None  # numpy array if used

    # ---- persistence / I/O ----
    def save(self, path: Optional[str] = None) -> None:
        path = path or self.path
        if not path:
            raise ValueError("No path specified for VDB.save()")
        if self._use_numpy and np is not None:
            # save vectors as npz and meta as json
            meta = {}
            ids = []
            vecs = []
            for k, v in self._items.items():
                ids.append(k)
                vecs.append(np.asarray(v["vector"], dtype=np.float32))
                meta[k] = {"text": v.get("text"), "meta": v.get("meta", {})}
            vecs_arr = np.stack(vecs) if vecs else np.zeros((0, self.dim), dtype=np.float32)
            np.savez_compressed(path, ids=np.array(ids, dtype=object), vectors=vecs_arr)
            with open(f"{path}.meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            logger.info("VDB saved (numpy) to %s (+ .meta.json)", path)
        else:
            serial = {}
            for k, v in self._items.items():
                serial[k] = {"vector": list(v["vector"]), "text": v.get("text"), "meta": v.get("meta", {})}
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"dim": self.dim, "index": serial}, f, ensure_ascii=False)
            logger.info("VDB saved (json) to %s", path)

    def load(self, path: Optional[str] = None) -> None:
        path = path or self.path
        if not path or not os.path.exists(path):
            raise ValueError(f"No VDB file at {path}")
        # try numpy first
        if path.endswith(".npz") or (self._use_numpy and np is not None):
            try:
                data = np.load(path, allow_pickle=True)
                ids = list(data["ids"].tolist())
                vecs = data["vectors"]
                meta_path = f"{path}.meta.json"
                meta = {}
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                self._items = {}
                for i, id in enumerate(ids):
                    self._items[id] = {"vector": vecs[i].tolist(), "text": meta.get(id, {}).get("text"), "meta": meta.get(id, {}).get("meta", {})}
                self._invalidate_cache()
                logger.info("VDB loaded (numpy) from %s (%d items)", path, len(self._items))
                return
            except Exception:
                logger.debug("numpy load failed, falling back to json", exc_info=True)
        # fallback json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        idx = data.get("index", {})
        self._items = {}
        for k, v in idx.items():
            vec = v.get("vector", [])
            self._items[k] = {"vector": vec, "text": v.get("text"), "meta": v.get("meta", {})}
        self._invalidate_cache()
        logger.info("VDB loaded (json) from %s (%d items)", path, len(self._items))

    # ---- index management ----
    def add(self, id: str, text: Optional[str] = None, vector: Optional[List[float]] = None, meta: Optional[Dict[str, Any]] = None, overwrite: bool = True) -> None:
        """
        add or upsert an item
        """
        if vector is None:
            if text is None:
                raise ValueError("Either text or vector must be provided")
            vector = self._text_to_vector(text)
        if len(vector) != self.dim:
            # pad or truncate
            vector = (list(vector) + [0.0] * self.dim)[: self.dim]
        self._items[id] = {"vector": list(vector), "text": text, "meta": meta or {}}
        self._invalidate_cache()

    def remove(self, id: str) -> None:
        if id in self._items:
            del self._items[id]
            self._invalidate_cache()

    def clear(self) -> None:
        self._items.clear()
        self._invalidate_cache()

    def stats(self) -> Dict[str, Any]:
        return {"n_items": len(self._items), "dim": self.dim, "use_numpy": bool(self._use_numpy and np is not None)}

    # ---- search ----
    def _invalidate_cache(self) -> None:
        self._matrix_cache_valid = False
        self._matrix = None

    def _build_matrix(self) -> None:
        if not (self._use_numpy and np is not None):
            self._matrix = None
            self._matrix_cache_valid = False
            return
        if self._matrix_cache_valid:
            return
        ids = []
        vecs = []
        for k, v in self._items.items():
            ids.append(k)
            vecs.append(v["vector"])
        if vecs:
            self._matrix = np.vstack([np.array(x, dtype=np.float32) for x in vecs])
        else:
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        self._index_ids = list(self._items.keys())
        self._matrix_cache_valid = True

    def query(self, query_or_vector: Any, top_k: int = 5, min_score: float = -1.0) -> List[Dict[str, Any]]:
        """
        query_or_vector: str or list[float] or numpy array
        returns: list of dicts [{"id": id, "score": sim, "meta": meta, "text": text}, ...] sorted by score desc
        """
        if isinstance(query_or_vector, str):
            qv = self._text_to_vector(query_or_vector)
        else:
            qv = list(query_or_vector)
        if len(qv) != self.dim:
            qv = (qv + [0.0] * self.dim)[: self.dim]
        # numpy path
        if self._use_numpy and np is not None:
            try:
                self._build_matrix()
                if self._matrix is None or self._matrix.shape[0] == 0:
                    return []
                qarr = np.array(qv, dtype=np.float32)
                sims = _cosine_similarity_np(qarr, self._matrix)  # shape (N,)
                idxs = np.argsort(-sims)[:top_k]
                results = []
                for i in idxs:
                    score = float(sims[i])
                    if score < min_score:
                        continue
                    id = self._index_ids[int(i)]
                    item = self._items[id]
                    results.append({"id": id, "score": score, "meta": item.get("meta"), "text": item.get("text")})
                return results
            except Exception:
                logger.exception("numpy query failed, falling back to python")
        # python brute force
        res = []
        for id, it in self._items.items():
            vec = it["vector"]
            # cosine
            denom = math.sqrt(sum(x * x for x in qv)) * math.sqrt(sum(x * x for x in vec)) or 1.0
            score = sum(a * b for a, b in zip(qv, vec)) / denom
            if score >= min_score:
                res.append((id, score))
        res.sort(key=lambda x: x[1], reverse=True)
        out = []
        for id, score in res[:top_k]:
            it = self._items[id]
            out.append({"id": id, "score": float(score), "meta": it.get("meta"), "text": it.get("text")})
        return out

    # ---- utility ----
    def export_dataset(self) -> List[Dict[str, Any]]:
        """全インデックスを list(dict) 形式で返す"""
        out = []
        for id, it in self._items.items():
            out.append({"id": id, "vector": list(it["vector"]), "text": it.get("text"), "meta": it.get("meta")})
        return out
