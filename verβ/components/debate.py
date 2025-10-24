# components/debate.py
"""
DebateSystem: 複数エージェントによる議論と評価・勝者決定のフレームワーク。

特徴:
- multi-round debate: 各エージェントが順に応答し、反駁(rebuttal) を行える
- critic を渡すことで自動評価を行い、スコアに基づいて勝者を決定
- 出力は DebateResult (arguments, scores, winner, rounds)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import logging
import statistics
import random

logger = logging.getLogger("components.debate")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


@dataclass
class DebateArgument:
    author: str
    text: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateRound:
    round_index: int
    contributions: List[DebateArgument] = field(default_factory=list)


@dataclass
class DebateResult:
    arguments: List[DebateArgument] = field(default_factory=list)
    winner: Optional[str] = None
    scores: Dict[str, float] = field(default_factory=dict)
    rounds: List[DebateRound] = field(default_factory=list)


class DebateSystem:
    def __init__(self, critic: Optional[Any] = None, max_rounds: int = 2, tie_breaker: Optional[Callable[[List[DebateArgument]], DebateArgument]] = None):
        """
        Args:
            critic: .assess(text)->float を持つ評価器。なければ内部 heuristics を使う
            max_rounds: 初期発言 + 反駁ラウンドの回数（例 2 -> 発言 + 1回のrebute）
            tie_breaker: 同点時の決定関数 (list of top args) -> DebateArgument
        """
        self.critic = critic
        self.max_rounds = max_rounds
        self.tie_breaker = tie_breaker or (lambda args: max(args, key=lambda a: len(a.text) + random.random()))

    def debate(self, prompt: str, agent_generators: List[Callable[[str, int, Optional[List[DebateArgument]]], str]]) -> DebateResult:
        """
        実行フロー:
          1. 各エージェントが initial response を生成
          2. 反駁ラウンドを max_rounds-1 回行う（各ラウンドで各エージェントが既存議論に応答）
          3. すべての発言を critic で評価して合計スコアを計算
          4. winner を決定して DebateResult を返す

        agent_generators: 各エージェントの応答生成関数のリスト。呼び出しシグネチャ:
            gen(prompt, round_idx, prev_arguments) -> text
        """
        n_agents = len(agent_generators)
        all_arguments: List[DebateArgument] = []
        rounds: List[DebateRound] = []

        # initial round
        round0 = DebateRound(round_index=0)
        for i, gen in enumerate(agent_generators):
            try:
                text = gen(prompt, 0, None)
            except Exception:
                logger.exception("Agent generator failed for initial round; using fallback")
                text = f"(agent_{i} failed to generate)"
            arg = DebateArgument(author=f"agent_{i}", text=text)
            round0.contributions.append(arg)
            all_arguments.append(arg)
        rounds.append(round0)

        # rebuttal rounds
        for r in range(1, self.max_rounds):
            rnd = DebateRound(round_index=r)
            for i, gen in enumerate(agent_generators):
                try:
                    prev_args = list(all_arguments)
                    text = gen(prompt, r, prev_args)
                except Exception:
                    logger.exception("Agent generator failed in rebuttal; using fallback")
                    text = f"(agent_{i} failed in rebuttal)"
                arg = DebateArgument(author=f"agent_{i}", text=text)
                rnd.contributions.append(arg)
                all_arguments.append(arg)
            rounds.append(rnd)

        # scoring
        scores: Dict[str, float] = {}
        for arg in all_arguments:
            try:
                if self.critic is not None and hasattr(self.critic, "assess"):
                    s = float(self.critic.assess(arg.text))
                else:
                    # fallback heuristic: 長さとユニークワード比
                    words = arg.text.split()
                    n = max(1, len(words))
                    unique_ratio = len(set(words)) / n
                    s = min(1.0, 0.6 * min(1.0, n / 100.0) + 0.4 * unique_ratio)
            except Exception:
                logger.exception("Scoring failed for arg; using 0.0")
                s = 0.0
            arg.score = s
            scores[arg.author] = scores.get(arg.author, 0.0) + s

        # decide winner
        # normalize by number of contributions per author
        norm_scores: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for arg in all_arguments:
            counts[arg.author] = counts.get(arg.author, 0) + 1
        for author, total in scores.items():
            norm_scores[author] = total / max(1, counts.get(author, 1))
        # select top
        if not norm_scores:
            winner = None
        else:
            max_score = max(norm_scores.values())
            top_authors = [a for a, s in norm_scores.items() if abs(s - max_score) < 1e-9 or s == max_score]
            if len(top_authors) == 1:
                winner = top_authors[0]
            else:
                # tie-break: collect latest contributions from tied agents
                candidate_args = [a for a in all_arguments if a.author in top_authors]
                winner_arg = self.tie_breaker(candidate_args)
                winner = winner_arg.author if winner_arg else top_authors[0]

        result = DebateResult(arguments=all_arguments, winner=winner, scores=norm_scores, rounds=rounds)
        logger.info("Debate finished winner=%s scores=%s", result.winner, result.scores)
        return result
