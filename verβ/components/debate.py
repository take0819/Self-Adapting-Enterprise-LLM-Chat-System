# components/debate.py
"""
DebateSystem の簡易実装:
- debate(prompt, agent_texts) -> DebateResult
DebateResult には arguments: List[DebateArgument], winner: Optional[str]
"""
from dataclasses import dataclass, field
from typing import Any, List, Optional
import random
import logging

logger = logging.getLogger("components.debate")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)


@dataclass
class DebateArgument:
    text: str
    author: str
    score: float = 0.0


@dataclass
class DebateResult:
    arguments: List[DebateArgument] = field(default_factory=list)
    winner: Optional[str] = None


class DebateSystem:
    def __init__(self, critic: Optional[Any] = None):
        """
        critic: (optional) 評価器オブジェクト（.assess(text)->float）を渡すと評価に使う
        """
        self.critic = critic

    def debate(self, prompt: str, agent_texts: List[str]) -> DebateResult:
        """
        シンプルな議論:
         - 各 agent_texts[i] を DebateArgument として生成
         - critic があればそれでスコア付け、なければランダム/長さベースのスコア
         - winner は最もスコアの高い argument の author
        """
        args: List[DebateArgument] = []
        for i, t in enumerate(agent_texts):
            author = f"agent_{i}"
            args.append(DebateArgument(text=t, author=author))

        # スコア評価
        for a in args:
            try:
                if self.critic is not None and hasattr(self.critic, "assess"):
                    a.score = float(self.critic.assess(a.text))
                else:
                    # fallback: 長さやランダム性で簡易スコア
                    a.score = min(1.0, max(0.0, len(a.text.split()) / 100.0 + random.random() * 0.1))
            except Exception:
                logger.exception("Failed scoring argument; using fallback")
                a.score = random.random()

        # pick winner
        args.sort(key=lambda x: x.score, reverse=True)
        winner = args[0].author if args else None
        logger.debug("Debate winner=%s score=%.3f", winner, args[0].score if args else 0.0)
        return DebateResult(arguments=args, winner=winner)
