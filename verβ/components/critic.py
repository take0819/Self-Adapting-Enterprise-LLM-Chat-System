# components/critic.py
"""
CriticSystem: 出力の品質を数値化（0.0..1.0）し、理由のリストを返す explain メソッドを提供。

評価指標:
 - length (適度な長さが高評価)
 - uniqueness (ユニークワード比)
 - coherence (前文との語彙オーバーラップ)
 - readability (簡易版: 平均文長)
 - safety / taboo (禁止語の簡易チェック; カスタム語リストを渡せる)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import math
import re
import logging

logger = logging.getLogger("components.critic")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


@dataclass
class CriticResult:
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    explanations: List[str] = field(default_factory=list)


class CriticSystem:
    def __init__(self, taboo_words: Optional[List[str]] = None):
        self.taboo = set(w.lower() for w in (taboo_words or []))

    def _unique_ratio(self, text: str) -> float:
        words = [w for w in re.findall(r"\\w+", text.lower())]
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def _avg_sentence_length(self, text: str) -> float:
        sentences = re.split(r"[.!?]+", text)
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0.0
        return sum(lengths) / len(lengths)

    def _safety_score(self, text: str) -> float:
        low = 1.0
        lw = text.lower()
        for w in self.taboo:
            if w in lw:
                low = 0.0
                break
        return low

    def assess(self, text: str, context: Optional[str] = None) -> float:
        """単純に 0..1 のスカラーを返す（内部で正規化）"""
        res = self.evaluate(text, context)
        return res.score

    def evaluate(self, text: str, context: Optional[str] = None) -> CriticResult:
        if not text:
            return CriticResult(score=0.0, details={}, explanations=["empty text"])

        uniq = self._unique_ratio(text)
        avg_sent = self._avg_sentence_length(text)
        # length heuristic: 20..200 words を好む
        n_words = len(re.findall(r"\\w+", text))
        length_score = min(1.0, max(0.0, (n_words - 10) / 190.0))
        uniq_score = uniq
        sent_score = 1.0 - min(1.0, abs(avg_sent - 20) / 20.0)  # 20語前後が好ましい
        safety = self._safety_score(text)

        # coherence: if context provided check overlap
        coherence = 1.0
        if context:
            ctx_words = set(re.findall(r"\\w+", context.lower()))
            txt_words = set(re.findall(r"\\w+", text.lower()))
            if ctx_words:
                overlap = len(ctx_words & txt_words) / len(ctx_words)
                coherence = min(1.0, overlap + 0.2)  # overlap があるほど良い (0..1)

        # weighted aggregation
        score = 0.3 * length_score + 0.25 * uniq_score + 0.2 * sent_score + 0.15 * coherence + 0.1 * safety
        score = max(0.0, min(1.0, score))

        details = {
            "n_words": n_words,
            "unique_ratio": uniq,
            "avg_sentence_length": avg_sent,
            "length_score": length_score,
            "uniq_score": uniq_score,
            "sent_score": sent_score,
            "coherence": coherence,
            "safety": safety,
        }
        explanations = []
        if score < 0.3:
            explanations.append("Overall quality is low: short or repetitive, or safety issues.")
        elif score < 0.6:
            explanations.append("Moderate quality: consider expanding or improving uniqueness.")
        else:
            explanations.append("Good quality: satisfies length/uniqueness heuristics.")

        # safety notes
        if safety < 1.0:
            explanations.append("Contains taboo words or unsafe content.")

        return CriticResult(score=score, details=details, explanations=explanations)
