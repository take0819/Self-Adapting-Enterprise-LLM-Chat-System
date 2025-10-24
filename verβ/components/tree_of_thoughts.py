# components/tree_of_thoughts.py
"""
Tree of Thoughts (ToT) 実装

機能:
- ThoughtNode: ノードのデータ構造（text, score, children, parent）
- TreeOfThoughts:
  - expand(node, candidate_generator, n_candidates=5)
    candidate_generator(node_text:str, n:int) -> List[str]
  - search(root, mode='beam'|'mcts'|'dfs', beam_width=3, depth=3, scorer=None)
    scorer(text)->float を優先的に使う。なければ内部ヒューリスティック。
- 結果は上位ノードのリストを返す（score 降順）
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any
import math
import random
import uuid
import logging

logger = logging.getLogger("components.tree_of_thoughts")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


@dataclass
class ThoughtNode:
    id: str
    text: str
    score: float = 0.0
    children: List["ThoughtNode"] = field(default_factory=list)
    parent: Optional["ThoughtNode"] = None

    def add_child(self, text: str, score: float = 0.0) -> "ThoughtNode":
        node = ThoughtNode(id=str(uuid.uuid4()), text=text, score=score, parent=self)
        self.children.append(node)
        return node


class TreeOfThoughts:
    def __init__(self, scorer: Optional[Callable[[str], float]] = None):
        """
        scorer: text -> score (0..1). None の場合は内部ヒューリスティックを使用。
        内部ヒューリスティックは長さとユニーク語率に基づく簡易評価。
        """
        self.scorer = scorer or (lambda t: min(1.0, max(0.0, len(t.split()) / 100.0)))

    def new_root(self, text: str = "<root>") -> ThoughtNode:
        return ThoughtNode(id=str(uuid.uuid4()), text=text, score=self.scorer(text))

    def expand(self, node: ThoughtNode, candidate_generator: Callable[[str, int], List[str]], n_candidates: int = 5) -> List[ThoughtNode]:
        """
        candidate_generator は node.text と n を受け取り、候補文字列のリストを返す関数。
        例: LLM に次の文を生成させる関数など
        """
        try:
            cands = candidate_generator(node.text, n_candidates)
        except Exception:
            logger.exception("candidate_generator failed; returning no children")
            cands = []
        new_nodes = []
        for c in cands:
            score = self.scorer(c)
            new_nodes.append(node.add_child(text=c, score=score))
        logger.debug("Expanded node %s with %d children", node.id, len(new_nodes))
        return new_nodes

    def search(self, root: Optional[ThoughtNode] = None, mode: str = "beam", beam_width: int = 3, depth: int = 3, scorer: Optional[Callable[[str], float]] = None, candidate_generator: Optional[Callable[[str, int], List[str]]] = None) -> List[ThoughtNode]:
        """
        高レベル探索 API:
          - beam: 各レベルで上位 beam_width を残す
          - dfs: 深さ優先で探索、上位を返す
          - mcts: モンテカルロツリーサーチの簡易版（ここでは簡易シミュレーション）
        candidate_generator は外部 (LLM) を呼ぶコールバック。指定しない場合は node.text を少し変えるダミージェネレータを用いる。
        """
        scorer = scorer or self.scorer
        if root is None:
            root = self.new_root()
        if candidate_generator is None:
            # ダミー generator: node.text に suffix を足す
            def _dummy_gen(txt: str, n: int) -> List[str]:
                out = []
                for i in range(n):
                    out.append(f"{txt} -> idea {i+1}")
                return out
            candidate_generator = _dummy_gen

        if mode == "beam":
            frontier = [root]
            for d in range(depth):
                next_candidates: List[ThoughtNode] = []
                for n in frontier:
                    # ensure children exist (if not, expand)
                    if not n.children:
                        self.expand(n, candidate_generator, n_candidates=beam_width)
                    next_candidates.extend(n.children)
                # score and select top beam_width
                next_candidates.sort(key=lambda x: x.score, reverse=True)
                frontier = next_candidates[:max(1, beam_width)]
                if not frontier:
                    break
            # return sorted final frontier
            frontier.sort(key=lambda x: x.score, reverse=True)
            return frontier
        elif mode == "dfs":
            results: List[ThoughtNode] = []

            def _dfs(node: ThoughtNode, depth_left: int):
                results.append(node)
                if depth_left <= 0:
                    return
                if not node.children:
                    self.expand(node, candidate_generator, n_candidates=beam_width)
                for c in node.children:
                    _dfs(c, depth_left - 1)

            _dfs(root, depth)
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:beam_width]
        elif mode == "mcts":
            # 簡易 MCTS: ランダムプレイアウトを複数回してノードの期待値を推定
            simulations = max(10, beam_width * depth * 5)
            scores_map: Dict[str, float] = {}
            visits: Dict[str, int] = {}
            for _ in range(simulations):
                node = root
                # rollout depth 階層分ランダムに展開
                for _d in range(depth):
                    if not node.children:
                        self.expand(node, candidate_generator, n_candidates=beam_width)
                    if not node.children:
                        break
                    node = random.choice(node.children)
                val = scorer(node.text)
                # backpropagate to root -> increment parents
                cur = node
                while cur is not None:
                    scores_map[cur.id] = scores_map.get(cur.id, 0.0) + val
                    visits[cur.id] = visits.get(cur.id, 0) + 1
                    cur = cur.parent
            # compute avg scores and return top beam_width nodes
            averaged: List[Tuple[float, ThoughtNode]] = []
            # collect nodes from tree
            stack = [root]
            nodes = []
            while stack:
                n = stack.pop()
                nodes.append(n)
                stack.extend(n.children)
            for n in nodes:
                if visits.get(n.id):
                    avg = scores_map.get(n.id, 0.0) / visits[n.id]
                else:
                    avg = scorer(n.text)
                averaged.append((avg, n))
            averaged.sort(key=lambda x: x[0], reverse=True)
            return [t for _, t in averaged[:beam_width]]
        else:
            logger.warning("Unknown search mode: %s", mode)
            return []
