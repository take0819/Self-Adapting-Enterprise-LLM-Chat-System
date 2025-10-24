# components/meta_learning.py
"""
MetaLearner: 経験を蓄積して簡易的に「適応」を行うモジュール。
目的: 実運用ではここをオンライン学習やハイパーパラ調整・強化学習のラッパへ差し替える。

機能:
- Experience の蓄積 (input, output, reward, meta)
- adapt(): 簡易的な統計的解析を行って "suggestions" を返す
- export_dataset(): 学習用データセットをエクスポート
- import_dataset(): データをロードして experiences を追加
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import statistics
import json
import logging
import time

logger = logging.getLogger("components.meta_learning")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


@dataclass
class Experience:
    input: str
    output: str
    reward: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)


class MetaLearner:
    def __init__(self, memory_limit: Optional[int] = 10000):
        self.experiences: List[Experience] = []
        self.memory_limit = memory_limit
        self.model_updates: List[Dict[str, Any]] = []

    def add_experience(self, input_text: str, output_text: str, reward: Optional[float] = None, meta: Optional[Dict[str, Any]] = None) -> None:
        exp = Experience(input=input_text, output=output_text, reward=reward, meta=meta or {})
        self.experiences.append(exp)
        # memory trimming
        if self.memory_limit and len(self.experiences) > self.memory_limit:
            # drop oldest
            overflow = len(self.experiences) - self.memory_limit
            del self.experiences[:overflow]
        logger.debug("Added experience. total=%d", len(self.experiences))

    def summarize_experiences(self) -> Dict[str, Any]:
        n = len(self.experiences)
        rewards = [e.reward for e in self.experiences if e.reward is not None]
        return {
            "n": n,
            "avg_reward": statistics.mean(rewards) if rewards else None,
            "median_reward": statistics.median(rewards) if rewards else None,
            "n_rewards": len(rewards),
        }

    def adapt(self, n_examples: Optional[int] = None) -> Dict[str, Any]:
        """
        簡易的な 'adapt'。ここでは以下を行う:
         - 最近 n_examples の報酬統計を計算
         - 改善のための簡易 'suggestions' を返す (例: temperature を上下させる等)
         - 実際のモデル更新は行わず 'update' のログを記録する
        """
        examples = self.experiences[-n_examples:] if n_examples else self.experiences
        rewards = [e.reward for e in examples if e.reward is not None]
        summary = {"n_examples": len(examples)}
        if rewards:
            avg = statistics.mean(rewards)
            med = statistics.median(rewards)
            summary.update({"avg_reward": avg, "median_reward": med})
            # suggestion heuristics
            suggestion = {}
            if avg < 0.3:
                suggestion["action"] = "increase_exploration"
                suggestion["reason"] = "avg reward low"
                suggestion["params"] = {"temperature_delta": 0.2}
            elif avg > 0.7:
                suggestion["action"] = "decrease_exploration"
                suggestion["reason"] = "avg reward high"
                suggestion["params"] = {"temperature_delta": -0.05}
            else:
                suggestion["action"] = "maintain"
                suggestion["reason"] = "stable"
                suggestion["params"] = {}
        else:
            suggestion = {"action": "collect_more_data", "reason": "no rewards", "params": {}}
        update = {"timestamp": time.time(), "summary": summary, "suggestion": suggestion}
        self.model_updates.append(update)
        logger.info("MetaLearner adapt result: %s", summary)
        return update

    def export_dataset(self, path: Optional[str] = None) -> List[Dict[str, Any]]:
        data = [{"input": e.input, "output": e.output, "reward": e.reward, "timestamp": e.timestamp, "meta": e.meta} for e in self.experiences]
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("Exported experiences to %s (n=%d)", path, len(data))
        return data

    def import_dataset(self, path: str, append: bool = True) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        loaded = 0
        for item in data:
            self.add_experience(item.get("input", ""), item.get("output", ""), reward=item.get("reward"), meta=item.get("meta"))
            loaded += 1
        logger.info("Imported %d experiences from %s", loaded, path)
