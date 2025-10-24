# components/tree_of_thoughts.py
"""
Tree of Thoughts の簡易実装（探索アルゴリズムの抽象）。
- ThoughtNode(id,text,score,children,parent)
- TreeOfThoughts.expand(node, candidates)
- TreeOfThoughts.search(root=None, mode='beam', beam_width=3, depth=3) -> List[ThoughtNode]
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import logging
import uuid

logger = logging.getLogger("components.tree")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
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
        scorer: 1つのテキストを受け取りスコア(高いほど良い)を返す関数。
                None の場合は簡易スコア関数を内部で使用する。
        """
        self.scorer = scorer or (lambda t: min(1.0, max(0.0, len(t.split()) / 100.0)))

    def new_root(self, text: str = "<root>") -> ThoughtNode:
        return ThoughtNode(id=str(uuid.uuid4()), text=text, score=self.scorer(text))

    def expand(self, node: ThoughtNode, candidates: List[str]) -> List[ThoughtNode]:
        """
        与えられた node に対し候補テキストの子ノードを生成する。
        候補は外部から生成できる（LLMによる候補生成など）。
        """
        new_nodes = []
        for c in candidates:
            sc = self.scorer(c)
            new_nodes.append(node.add_child(text=c, score=sc))
        logger.debug("Expanded node %s with %d children", node.id, len(new_nodes))
        return new_nodes

    def search(self, root: Optional[ThoughtNode] = None, mode: str = "beam", beam_width: int = 3, depth: int = 3) -> List[ThoughtNode]:
        """
        シンプルな探索:
          - beam: ビーム探索（各レベルで上位 beam_width を残す）
          - dfs: 深さ優先で depth 深さまで探索して最良ノードを返す（序列）
        戻り値: 最終候補ノードのリスト（上位のものから順）
        """
        if root is None:
            root = self.new_root()

        if mode == "beam":
            frontier = [root]
            for d in range(depth):
                candidates = []
                for n in frontier:
                    # 既に children がなければ何もしない
                    for c in n.children:
                        candidates.append(c)
                # ビーム選択
                candidates.sort(key=lambda x: x.score, reverse=True)
                frontier = candidates[:max(1, beam_width)]
                if not frontier:
                    break
            # return sorted final frontier
            return sorted(frontier, key=lambda x: x.score, reverse=True)
        elif mode == "dfs":
            best_nodes = []

            def _dfs(node: ThoughtNode, depth_left: int):
                best_nodes.append(node)
                if depth_left <= 0:
                    return
                for c in node.children:
                    _dfs(c, depth_left - 1)

            _dfs(root, depth)
            best_nodes.sort(key=lambda x: x.score, reverse=True)
            return best_nodes[:beam_width]
        else:
            logger.warning("Unknown search mode: %s", mode)
            return []
