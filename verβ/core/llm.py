# core/llm.py
"""
UltraAdvancedLLM — core モジュール用実装

使い方:
  - このファイルを project/core/llm.py として保存してください。
  - core/dataclasses.py, core/config.py, core/enums.py が存在する前提です。
  - components/*.py が無くても import できるように遅延インポートとフォールバックを行っています。
  - 移行後は必ず smoke tests / 回帰テストを実行してください。

注:
  - 外部 LLM クライアント（openai, groq 等）がある場合はそれらを試行的に初期化します。
  - 実際の SDK の引数/戻り値に合わせて _call_model の該当部分を調整してください。
"""
from __future__ import annotations

import os
import time
import json
import hashlib
import logging
import random
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# core 依存（必須ファイル）
from .config import Cfg, load_cfg_from_env
from .dataclasses import Resp, KnowledgeGraph, KnowledgeNode, ModelStats
from .enums import Intent, Strategy

# components は存在すれば使う（遅延インポート）
try:
    from components.vdb import VDB  # type: ignore
except Exception:
    VDB = None  # type: ignore

try:
    from components.tree_of_thoughts import TreeOfThoughts, ThoughtNode  # type: ignore
except Exception:
    TreeOfThoughts = None  # type: ignore

try:
    from components.debate import DebateSystem  # type: ignore
except Exception:
    DebateSystem = None  # type: ignore

try:
    from components.critic import CriticSystem  # type: ignore
except Exception:
    CriticSystem = None  # type: ignore

try:
    from components.constitutional import ConstitutionalAI  # type: ignore
except Exception:
    ConstitutionalAI = None  # type: ignore

# logger
logger = logging.getLogger("core.llm")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class UltraAdvancedLLM:
    """
    高機能 LLM ラッパー。主な public メソッド:
      - query(prompt, top_k=1, **kwargs)
      - generate(prompt, n=1, **kwargs) -> List[Resp]
      - retrieve(query, top_k=5) -> List[Any]
      - save_state(path), load_state(path)
      - add_knowledge(id, name, type_, properties)
      - evaluate(text) -> float
    """

    def __init__(self, cfg: Optional[Cfg] = None, seed: Optional[int] = None, **kwargs: Any):
        self.cfg = cfg or load_cfg_from_env()
        # シード設定（再現性維持）
        if seed is None:
            seed = int(os.getenv("LLM_SEED", str(random.randint(0, 2 ** 31 - 1))))
        self.seed = seed
        random.seed(self.seed)

        # 内部状態
        self.kg: KnowledgeGraph = KnowledgeGraph()
        self.models: Dict[str, ModelStats] = {}
        self._cache: Dict[str, Resp] = {}
        self.ab_test: Dict[str, Any] = {}
        self.meta: Dict[str, Any] = {}

        # 外部クライアントと名前
        self._client: Optional[Any] = None
        self._client_name: str = "stub"
        self._init_clients()

        # components（存在すればインスタンス化）
        try:
            self.vdb = VDB() if (VDB is not None and getattr(self.cfg, "vec_db", True)) else None
        except Exception:
            logger.exception("VDB initialization failed")
            self.vdb = None

        try:
            self.tree = TreeOfThoughts() if (TreeOfThoughts is not None and getattr(self.cfg, "tree_of_thoughts", True)) else None
        except Exception:
            logger.exception("TreeOfThoughts init failed")
            self.tree = None

        try:
            self.debate_sys = DebateSystem() if (DebateSystem is not None and getattr(self.cfg, "debate", True)) else None
        except Exception:
            logger.exception("DebateSystem init failed")
            self.debate_sys = None

        try:
            self.critic = CriticSystem() if (CriticSystem is not None and getattr(self.cfg, "critic", True)) else None
        except Exception:
            logger.exception("CriticSystem init failed")
            self.critic = None

        try:
            self.constitutional = ConstitutionalAI() if (ConstitutionalAI is not None and getattr(self.cfg, "constitutional_ai", False)) else None
        except Exception:
            logger.exception("ConstitutionalAI init failed")
            self.constitutional = None

        logger.info("UltraAdvancedLLM initialized (client=%s, vdb=%s)", self._client_name, bool(self.vdb))

    # --------------------
    # 外部クライアント初期化
    # --------------------
    def _init_clients(self) -> None:
        """
        可能な外部クライアントを試行的に初期化する。
        実運用ではここを環境に応じてカスタマイズしてください。
        """
        # groq (例)
        try:
            import groq_client as groq  # type: ignore

            apikey = getattr(self.cfg, "api_key", None)
            self._client = groq.Client(api_key=apikey)  # type: ignore
            self._client_name = "groq"
            logger.info("Initialized Groq client")
            return
        except Exception:
            logger.debug("Groq client not available or initialization failed", exc_info=True)

        # openai (例)
        try:
            import openai  # type: ignore

            if getattr(self.cfg, "api_key", None):
                openai.api_key = self.cfg.api_key
            self._client = openai
            self._client_name = "openai"
            logger.info("Initialized OpenAI client")
            return
        except Exception:
            logger.debug("OpenAI client not available or init failed", exc_info=True)

        # フォールバック: stub
        self._client = None
        self._client_name = "stub"
        logger.warning("No external LLM client initialized — using stub behavior")

    # --------------------
    # 高レベル API
    # --------------------
    def query(self, prompt: str, top_k: int = 1, use_rag: bool = True, **kwargs: Any) -> Any:
        """
        top_k == 1 -> Resp を返す
        top_k > 1 -> List[Resp] を返す
        """
        if top_k <= 1:
            return self.generate(prompt, n=1, use_rag=use_rag, **kwargs)[0]
        return self.generate(prompt, n=top_k, use_rag=use_rag, **kwargs)

    def generate(self, prompt: str, n: int = 1, use_rag: bool = True, temperature: Optional[float] = None, **kwargs: Any) -> List[Resp]:
        """
        生成パイプライン:
         - preprocess
         - (optional) RAG retrieve
         - model 呼び出し（外部 or stub）
         - (optional) debate/critic を用いた融合
         - postprocess
        """
        temperature = self.cfg.temperature if temperature is None else temperature

        cache_key = f"gen:{_sha1(prompt)}:n={n}:temp={temperature}"
        if cache_key in self._cache and n == 1:
            logger.debug("Cache hit for prompt")
            return [self._cache[cache_key]]

        ctx = self._preprocess(prompt)

        # RAG (retrieval-augmented generation)
        if use_rag and self.vdb is not None:
            try:
                retrieved = self.retrieve(prompt, top_k=5)
                ctx["retrieved"] = retrieved
                logger.debug("RAG retrieved %d items", len(retrieved))
            except Exception:
                logger.exception("RAG retrieve failed")
                ctx["retrieved"] = []
        else:
            ctx["retrieved"] = []

        candidates: List[Resp] = []
        for i in range(max(1, n)):
            final_prompt = self._compose_prompt(prompt, ctx)
            resp = self._call_model(final_prompt, temperature=temperature, **kwargs)
            candidates.append(resp)

        # debate / critic による融合
        if self.debate_sys is not None and len(candidates) > 1:
            try:
                candidates = self._run_debate_and_fusion(prompt, candidates)
            except Exception:
                logger.exception("Debate/fusion step failed; falling back to raw candidates")

        results = [self._postprocess(r) for r in candidates]

        # 単一生成はキャッシュする
        if n == 1 and results:
            self._cache[cache_key] = results[0]

        return results

    # --------------------
    # パイプライン補助
    # --------------------
    def _preprocess(self, prompt: str) -> Dict[str, Any]:
        """簡易的な前処理: メタデータを付与"""
        return {"prompt": prompt, "ts": datetime.utcnow().isoformat()}

    def _compose_prompt(self, prompt: str, ctx: Dict[str, Any]) -> str:
        """retrieved をプロンプトへ付加する（単純連結）"""
        retrieved = ctx.get("retrieved", [])
        if not retrieved:
            return prompt
        snippets: List[str] = []
        for r in retrieved[:5]:
            # r がタプル (id, score, meta, text) など多様な形式に対応
            if isinstance(r, tuple):
                snippets.append(" ".join(map(str, r)))
            else:
                snippets.append(str(r))
        augmentation = "\n\nRetrieved:\n" + "\n---\n".join(snippets)
        return f"{prompt}\n\n{augmentation}"

    def _call_model(self, prompt: str, temperature: float = 0.0, model: Optional[str] = None, **kwargs: Any) -> Resp:
        """
        外部クライアント経由で生成。なければ stub を使う。
        OpenAI や groq の戻り値は SDK に依存するので適宜調整してください。
        """
        start = time.time()
        model_name = model or getattr(self.cfg, "model", "local-stub")

        # groq client
        if self._client_name == "groq" and self._client is not None:
            try:
                # ここは groq SDK の実際の呼び出しに合わせて書き換えること
                ans = self._client.generate(prompt=prompt, model=model_name, temperature=temperature)  # type: ignore
                text = getattr(ans, "text", str(ans))
                conf = float(getattr(ans, "confidence", 1.0))
            except Exception:
                logger.exception("Groq call failed")
                text, conf = self._stub_model(prompt), 0.0

        # openai client
        elif self._client_name == "openai" and self._client is not None:
            try:
                # ChatCompletion 互換の呼び出し (例)
                completion = self._client.ChatCompletion.create(  # type: ignore
                    model=model_name, messages=[{"role": "user", "content": prompt}], temperature=temperature, max_tokens=getattr(self.cfg, "max_tokens", 1024)
                )
                # 安全にテキストを抽出
                choices = getattr(completion, "choices", None) or completion.get("choices", [])
                text_parts: List[str] = []
                for c in choices:
                    if isinstance(c, dict):
                        text_parts.append(c.get("message", {}).get("content", "") or c.get("text", ""))
                    else:
                        text_parts.append(getattr(c, "text", ""))
                text = "".join(text_parts)
                conf = 1.0
            except Exception:
                logger.exception("OpenAI call failed")
                text, conf = self._stub_model(prompt), 0.0

        # fallback stub
        else:
            text = self._stub_model(prompt)
            conf = 0.5

        latency = time.time() - start
        resp = Resp(text=text, conf=conf, tok=len(text.split()), lat=latency, model=model_name, ts=datetime.utcnow())
        return resp

    def _stub_model(self, prompt: str) -> str:
        """外部クライアントがないときの決まり文句出力（決まった振る舞いで再現性を持たせる）"""
        words = prompt.strip().split()
        head = " ".join(words[:60])
        suffix = "..." if len(words) > 60 else ""
        return f"[stub response] {head}{suffix}"

    def _run_debate_and_fusion(self, prompt: str, candidates: List[Resp]) -> List[Resp]:
        """
        DebateSystem を用いた候補融合。APIは DebateSystem 実装に依存するため、
        ここでは柔軟な wrapper を使い、失敗時は元の候補を返す。
        """
        if self.debate_sys is None:
            return candidates
        try:
            texts = [c.text for c in candidates]
            debate_res = self.debate_sys.debate(prompt, texts)
            # DebateResult の形式に依存するが、まず arguments があれば先頭を winner とする
            if hasattr(debate_res, "arguments") and debate_res.arguments:
                winner_text = debate_res.arguments[0].text
                winner_resp = Resp(text=winner_text, conf=0.9, tok=len(winner_text.split()), model="debate-winner", ts=datetime.utcnow())
                return [winner_resp]
        except Exception:
            logger.exception("Debate system failed")
        return candidates

    def _postprocess(self, resp: Resp) -> Resp:
        """生成後処理: Constitutional AI の適用や簡易正規化など"""
        text = resp.text
        if self.constitutional is not None:
            try:
                text = self.constitutional.apply_rules(text)
            except Exception:
                logger.exception("ConstitutionalAI application failed")
        resp.text = text
        return resp

    # --------------------
    # Retrieval / RAG
    # --------------------
    def retrieve(self, query: str, top_k: int = 5) -> List[Any]:
        """
        VDB があればそれを使って類似ドキュメントを返す。存在しない場合は空リストを返す。
        vdb.query の戻り値形式には実装依存である点に注意。
        """
        if self.vdb is None:
            return []
        try:
            return self.vdb.query(query, top_k=top_k)
        except Exception:
            logger.exception("VDB query failed")
            return []

    # --------------------
    # State I/O
    # --------------------
    def save_state(self, path: str) -> None:
        st: Dict[str, Any] = {
            "cfg": vars(self.cfg) if hasattr(self.cfg, "__dict__") else dict(self.cfg.__dict__),
            "seed": self.seed,
            "kg": {k: getattr(v, "__dict__", {}) for k, v in getattr(self.kg, "nodes", {}).items()},
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(st, f, ensure_ascii=False, indent=2, default=str)
            logger.info("State saved to %s", path)
        except Exception:
            logger.exception("Failed to save state to %s", path)

    def load_state(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                st = json.load(f)
            self.seed = st.get("seed", self.seed)
            self.kg = KnowledgeGraph()
            kgdata = st.get("kg", {})
            for nid, ndata in kgdata.items():
                node = KnowledgeNode(id=nid, name=ndata.get("name", nid), type=ndata.get("type", "fact"), properties=ndata.get("properties", {}))
                self.kg.nodes[nid] = node
            logger.info("State loaded from %s", path)
        except Exception:
            logger.exception("Failed to load state from %s", path)

    # --------------------
    # Utilities / Misc
    # --------------------
    def add_knowledge(self, id: str, name: str, type_: str = "fact", properties: Optional[Dict[str, Any]] = None) -> None:
        properties = properties or {}
        node = KnowledgeNode(id=id, name=name, type=type_, properties=properties)
        self.kg.nodes[id] = node

    def register_model(self, name: str) -> None:
        if name not in self.models:
            self.models[name] = ModelStats(name=name)

    def evaluate(self, text: str) -> float:
        """
        出力品質の簡易評価。CriticSystem があればそれを利用する。
        返り値は 0.0..1.0 のスコア。
        """
        if self.critic is not None:
            try:
                return float(self.critic.assess(text))
            except Exception:
                logger.exception("Critic assessment failed")
        # 単純ヒューリスティック: 長さベースの正規化
        words = max(1, len(str(text).split()))
        return min(1.0, words / 100.0)
