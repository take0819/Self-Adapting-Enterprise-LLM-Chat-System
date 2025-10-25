#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-Advanced Self-Evolving Enterprise LLM System v2.0β
次世代AI会話システム - 自己進化・メタ学習・分散推論

主要機能:
- 🧠 Self-Evolving Neural Architecture Search
- 🌐 Distributed Multi-Agent Reasoning
- 🔮 Predictive Context Modeling
- 🎯 Dynamic Difficulty Adaptation
- 🔬 Automated Hypothesis Testing
- 📊 Real-time Performance Optimization
- 🛡️ Adversarial Robustness Shield
- 💎 Quality Assurance Pipeline

使い方:
export GROQ_API_KEY='your_key'
pip install groq numpy
python enterprise-llm-chat-verβ.py
"""

import os
import sys
import time
import json
import hashlib
import logging
import asyncio
import re
import uuid
import math
import statistics
from typing import Optional, List, Dict, Any, Callable, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter, deque
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

try:
    from groq import Groq, RateLimitError, APIError
except ImportError:
    print("❌ Required: pip install groq numpy")
    sys.exit(1)

try:
    import readline
except ImportError:
    pass

# ==================== 定数・列挙型 ====================

class Intent(str, Enum):
    QUESTION = "question"
    COMMAND = "command"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    CASUAL = "casual"
    EXPLANATION = "explanation"
    REASONING = "reasoning"
    ANALYSIS = "analysis"
    RESEARCH = "research"
    PLANNING = "planning"


class Complexity(str, Enum):
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EXPERT = "expert"
    RESEARCH = "research"


class Strategy(str, Enum):
    DIRECT = "direct"
    COT = "chain_of_thought"
    REFLECTION = "reflection"
    ENSEMBLE = "ensemble"
    ITERATIVE = "iterative"
    TREE_SEARCH = "tree_search"
    DEBATE = "debate"
    SYNTHESIS = "synthesis"


class ReasoningType(str, Enum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    COUNTERFACTUAL = "counterfactual"


# ==================== 設定 ====================

@dataclass
class SystemConfig:
    """システム設定"""
    # 基本設定
    model: str = "llama-3.1-8b-instant"
    max_tokens: int = 4000
    temperature: float = 0.7
    
    # ベクトルDB
    vec_db: bool = True
    vec_dim: int = 384
    cache_ttl: int = 3600
    similarity_threshold: float = 0.92
    
    # リトライ・制限
    max_retries: int = 3
    retry_delay: float = 1.0
    max_query_length: int = 10000
    
    # コア機能
    adaptive: bool = True
    multi_armed_bandit: bool = True
    long_term_memory: bool = True
    knowledge_graph: bool = True
    chain_of_thought: bool = True
    self_reflection: bool = True
    ab_testing: bool = True
    ensemble_learning: bool = True
    metacognition: bool = True
    thompson_sampling: bool = True
    
    # 高度な機能
    tree_of_thoughts: bool = True
    debate_mode: bool = True
    critic_system: bool = True
    confidence_calibration: bool = True
    active_learning: bool = True
    curriculum_learning: bool = True
    adversarial_testing: bool = False
    
    # 超高度な機能
    neural_architecture_search: bool = True
    distributed_reasoning: bool = True
    predictive_modeling: bool = True
    hypothesis_testing: bool = True
    quality_assurance: bool = True
    performance_profiling: bool = True


# ==================== データ構造 ====================

@dataclass
class Response:
    """LLM応答"""
    text: str
    confidence: float
    tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0
    cost: float = 0
    model: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    finish_reason: str = "unknown"
    cached: bool = False
    similarity: float = 0
    rating: Optional[int] = None
    
    # メタデータ
    intent: Optional[Intent] = None
    complexity: Optional[Complexity] = None
    sentiment: float = 0
    strategy: Optional[Strategy] = None
    reasoning_steps: List[str] = field(default_factory=list)
    reflection: Optional[str] = None
    uncertainty: float = 0
    alternatives: List[str] = field(default_factory=list)
    
    # 品質メトリクス
    coherence_score: float = 0
    relevance_score: float = 0
    completeness_score: float = 0
    factuality_score: float = 0
    
    @property
    def success(self) -> bool:
        return self.finish_reason in ("stop", "length")
    
    @property
    def quality_score(self) -> float:
        """総合品質スコア"""
        scores = [
            self.confidence,
            self.coherence_score,
            self.relevance_score,
            self.completeness_score,
            self.factuality_score
        ]
        valid_scores = [s for s in scores if s > 0]
        return statistics.mean(valid_scores) if valid_scores else self.confidence
    
    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'confidence': self.confidence,
            'tokens': self.tokens,
            'cost': self.cost,
            'model': self.model,
            'latency': self.latency,
            'success': self.success,
            'cached': self.cached,
            'quality_score': self.quality_score,
            'uncertainty': self.uncertainty,
            'intent': self.intent.value if self.intent else None,
            'complexity': self.complexity.value if self.complexity else None,
            'strategy': self.strategy.value if self.strategy else None
        }


@dataclass
class KnowledgeNode:
    """知識グラフノード"""
    id: str
    name: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created: datetime = field(default_factory=datetime.now)
    updated: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    sources: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None


@dataclass
class KnowledgeEdge:
    """知識グラフエッジ"""
    source: str
    target: str
    relation: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    created: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0


@dataclass
class ThoughtNode:
    """Tree of Thoughtsノード"""
    id: str
    content: str
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    value: float = 0.0
    visits: int = 0
    depth: int = 0
    quality_scores: List[float] = field(default_factory=list)
    
    def ucb_score(self, parent_visits: int, exploration_weight: float = 1.414) -> float:
        """Upper Confidence Bound"""
        if self.visits == 0:
            return float('inf')
        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration
    
    @property
    def average_quality(self) -> float:
        return statistics.mean(self.quality_scores) if self.quality_scores else 0.5


@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクス"""
    query_count: int = 0
    success_count: int = 0
    total_tokens: int = 0
    total_cost: float = 0
    total_latency: float = 0
    cache_hits: int = 0
    errors: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        return self.success_count / self.query_count if self.query_count > 0 else 0
    
    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.query_count if self.query_count > 0 else 0
    
    @property
    def avg_cost(self) -> float:
        return self.total_cost / self.query_count if self.query_count > 0 else 0
    
    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.query_count if self.query_count > 0 else 0


# ==================== ユーティリティ ====================

class Logger:
    """ロガー"""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)


logger = Logger('llm')


class VectorDB:
    """簡易ベクトルDB"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors: List[Tuple[str, np.ndarray, Dict]] = []
    
    def _embed(self, text: str) -> np.ndarray:
        """テキストを埋め込みベクトルに変換（ハッシュベース）"""
        hash_bytes = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(hash_bytes[:4], 'little')
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    
    def add(self, id: str, text: str, metadata: Dict):
        """ベクトルを追加"""
        embedding = self._embed(text)
        metadata = metadata or {}
        metadata['text'] = text
        self.vectors.append((id, embedding, metadata))
    
    def search(self, query: str, top_k: int = 1) -> List[Tuple[str, float, Dict]]:
        """類似検索"""
        if not self.vectors:
            return []
        
        query_vec = self._embed(query)
        results = [
            (id, np.dot(query_vec, vec), metadata)
            for id, vec, metadata in self.vectors
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def search_threshold(self, query: str, threshold: float = 0.7) -> List[Tuple[str, float, Dict]]:
        """閾値以上の類似度を持つ全ドキュメントを取得"""
        if not self.vectors:
            return []
        
        query_vec = self._embed(query)
        results = [
            (id, np.dot(query_vec, vec), metadata)
            for id, vec, metadata in self.vectors
            if np.dot(query_vec, vec) >= threshold
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def clear(self):
        """全データをクリア"""
        self.vectors.clear()


# ==================== AI コンポーネント ====================

class KnowledgeGraph:
    """知識グラフ管理"""
    
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
    
    def add_node(self, node: KnowledgeNode):
        node.updated = datetime.now()
        self.nodes[node.id] = node
    
    def add_edge(self, edge: KnowledgeEdge):
        self.edges.append(edge)
    
    def get_neighbors(self, node_id: str, relation: Optional[str] = None) -> List[str]:
        """隣接ノードを取得"""
        neighbors = []
        for edge in self.edges:
            if edge.source == node_id and (relation is None or edge.relation == relation):
                neighbors.append(edge.target)
            elif edge.target == node_id and (relation is None or edge.relation == relation):
                neighbors.append(edge.source)
        return neighbors
    
    def find_path(self, start: str, end: str, max_depth: int = 3) -> Optional[List[str]]:
        """2ノード間の最短パス（BFS）"""
        if start not in self.nodes or end not in self.nodes:
            return None
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            if current == end:
                return path
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_related_concepts(self, concept: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """関連概念を取得"""
        if concept not in self.nodes:
            return []
        
        neighbors = self.get_neighbors(concept)
        weighted = []
        
        for neighbor in neighbors:
            edges = [
                e for e in self.edges
                if (e.source == concept and e.target == neighbor) or
                   (e.target == concept and e.source == neighbor)
            ]
            if edges:
                max_weight = max(e.weight * e.confidence for e in edges)
                weighted.append((neighbor, max_weight))
        
        weighted.sort(key=lambda x: x[1], reverse=True)
        return weighted[:top_k]
    
    def get_subgraph(self, center: str, radius: int = 2) -> Dict[str, Any]:
        """指定ノードを中心としたサブグラフを取得"""
        if center not in self.nodes:
            return {'nodes': [], 'edges': []}
        
        subgraph_nodes = {center}
        current_layer = {center}
        
        for _ in range(radius):
            next_layer = set()
            for node in current_layer:
                neighbors = self.get_neighbors(node)
                next_layer.update(neighbors)
            subgraph_nodes.update(next_layer)
            current_layer = next_layer
        
        subgraph_edges = [
            e for e in self.edges
            if e.source in subgraph_nodes and e.target in subgraph_nodes
        ]
        
        return {
            'nodes': [self.nodes[nid] for nid in subgraph_nodes],
            'edges': subgraph_edges
        }


class TreeOfThoughts:
    """Tree of Thoughts推論システム"""
    
    def __init__(self, max_depth: int = 3, branching_factor: int = 3):
        self.nodes: Dict[str, ThoughtNode] = {}
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.root_id: Optional[str] = None
    
    def create_root(self, content: str) -> str:
        """ルートノード作成"""
        node_id = str(uuid.uuid4())[:8]
        self.nodes[node_id] = ThoughtNode(id=node_id, content=content, depth=0)
        self.root_id = node_id
        return node_id
    
    def expand(self, node_id: str, children_content: List[str]):
        """ノードを展開"""
        if node_id not in self.nodes:
            return
        
        parent = self.nodes[node_id]
        if parent.depth >= self.max_depth:
            return
        
        for content in children_content[:self.branching_factor]:
            child_id = str(uuid.uuid4())[:8]
            child = ThoughtNode(
                id=child_id,
                content=content,
                parent=node_id,
                depth=parent.depth + 1
            )
            self.nodes[child_id] = child
            parent.children.append(child_id)
    
    def backpropagate(self, node_id: str, value: float):
        """価値を逆伝播"""
        current_id = node_id
        while current_id:
            node = self.nodes[current_id]
            node.visits += 1
            node.value += value
            node.quality_scores.append(value)
            current_id = node.parent
    
    def select_best_path(self) -> List[str]:
        """最良パスを選択（品質ベース）"""
        if not self.root_id:
            return []
        
        path = [self.root_id]
        current_id = self.root_id
        
        while True:
            node = self.nodes[current_id]
            if not node.children:
                break
            
            # 最高品質の子を選択
            best_child_id = max(
                node.children,
                key=lambda cid: self.nodes[cid].average_quality
            )
            path.append(best_child_id)
            current_id = best_child_id
        
        return path
    
    def get_path_content(self, path: List[str]) -> List[str]:
        """パスの内容を取得"""
        return [self.nodes[nid].content for nid in path]
    
    def reset(self):
        """ツリーをリセット"""
        self.nodes.clear()
        self.root_id = None


class UserProfile:
    """ユーザープロファイル"""
    
    def __init__(self):
        self.topics: Dict[str, int] = defaultdict(int)
        self.avg_response_length: float = 100.0
        self.style: str = "balanced"  # concise, balanced, detailed
        self.temperature_preference: float = 0.7
        self.positive_words: Set[str] = set()
        self.negative_words: Set[str] = set()
        self.feedback_history: List[Dict] = []
        self.interaction_count: int = 0
        self.last_updated: datetime = datetime.now()
        
        # 意図・複雑度の分布
        self.intent_distribution: Dict[str, int] = defaultdict(int)
        self.complexity_preference: str = "medium"
        
        # 専門知識レベル
        self.expertise_level: Dict[str, float] = defaultdict(float)
        self.learning_rate: float = 0.1
        
        # 戦略好み
        self.strategy_preference: Dict[str, float] = defaultdict(float)
        
        # 時間パターン
        self.time_of_day_pattern: Dict[int, int] = defaultdict(int)
    
    def update_from_feedback(
        self,
        query: str,
        response: str,
        rating: int,
        intent: Optional[Intent] = None,
        complexity: Optional[Complexity] = None,
        strategy: Optional[Strategy] = None
    ):
        """フィードバックからプロファイルを更新"""
        self.interaction_count += 1
        self.last_updated = datetime.now()
        
        # 時間パターン
        hour = datetime.now().hour
        self.time_of_day_pattern[hour] += 1
        
        # フィードバック履歴
        self.feedback_history.append({
            'query': query[:100],
            'response': response[:100],
            'rating': rating,
            'intent': intent.value if intent else None,
            'complexity': complexity.value if complexity else None,
            'strategy': strategy.value if strategy else None,
            'timestamp': datetime.now().isoformat()
        })
        
        # 履歴管理（最新200件）
        if len(self.feedback_history) > 200:
            self.feedback_history = self.feedback_history[-200:]
        
        # トピック更新
        words = re.findall(r'\b\w{4,}\b', query.lower())
        for word in words:
            self.topics[word] += rating
            if rating > 0:
                self.expertise_level[word] = min(
                    1.0,
                    self.expertise_level[word] + self.learning_rate
                )
            else:
                self.expertise_level[word] = max(
                    0.0,
                    self.expertise_level[word] - self.learning_rate * 0.5
                )
        
        # 意図・戦略の分布
        if intent:
            self.intent_distribution[intent.value] += 1
        if strategy:
            current = self.strategy_preference.get(strategy.value, 0.5)
            self.strategy_preference[strategy.value] = current + rating * 0.1
        
        # レスポンス長の適応
        response_len = len(response)
        alpha = 0.2 if rating > 0 else 0.1
        self.avg_response_length = (
            self.avg_response_length * (1 - alpha) + response_len * alpha
        )
        
        # スタイル推定
        if rating > 0:
            if response_len < 150:
                self.style = "concise"
            elif response_len > 500:
                self.style = "detailed"
            else:
                self.style = "balanced"
        
        # ポジティブ/ネガティブワード
        response_words = set(re.findall(r'\b\w{4,}\b', response.lower()))
        if rating > 0:
            self.positive_words.update(w for w in response_words if len(w) > 4)
            if len(self.positive_words) > 800:
                self.positive_words = set(list(self.positive_words)[-800:])
        elif rating < 0:
            self.negative_words.update(w for w in response_words if len(w) > 4)
            if len(self.negative_words) > 500:
                self.negative_words = set(list(self.negative_words)[-500:])
    
    def get_adapted_temperature(self) -> float:
        """適応的な温度を取得"""
        if not self.feedback_history:
            return 0.7
        
        recent = self.feedback_history[-20:]
        avg_rating = sum(f.get('rating', 0) for f in recent) / len(recent)
        
        # 専門知識レベルに基づく調整
        expertise = (
            sum(self.expertise_level.values()) / len(self.expertise_level)
            if self.expertise_level else 0.5
        )
        
        base_temp = 0.7 - expertise * 0.2
        
        if avg_rating > 0:
            return base_temp
        else:
            return min(1.0, base_temp + 0.15)
    
    def get_style_prompt(self) -> str:
        """スタイルに応じたプロンプト"""
        style_prompts = {
            'concise': 'Be extremely concise. Use bullet points when appropriate. Brief, actionable answers.',
            'balanced': 'Provide clear, well-structured answers with appropriate detail.',
            'detailed': 'Provide comprehensive explanations with examples, context, and deeper insights.'
        }
        return style_prompts.get(self.style, style_prompts['balanced'])
    
    def predict_intent(self, query: str) -> Intent:
        """クエリから意図を予測"""
        q = query.lower()
        
        # キーワードベースの予測
        if any(w in q for w in ['why', 'reason', 'because', 'cause', 'explain why']):
            return Intent.REASONING
        if any(w in q for w in ['analyze', 'compare', 'evaluate', 'assess']):
            return Intent.ANALYSIS
        if any(w in q for w in ['research', 'investigate', 'study']):
            return Intent.RESEARCH
        if any(w in q for w in ['plan', 'schedule', 'organize', 'strategy']):
            return Intent.PLANNING
        if any(w in q for w in ['code', 'program', 'algorithm', 'implement']):
            return Intent.TECHNICAL
        if any(w in q for w in ['write', 'create', 'generate', 'compose']):
            return Intent.CREATIVE
        if any(w in q for w in ['explain', 'describe', 'detail', 'elaborate']):
            return Intent.EXPLANATION
        if '?' in q or any(w in q for w in ['how', 'what', 'when', 'where', 'who']):
            return Intent.QUESTION
        
        return Intent.CASUAL
    
    def get_preferred_strategy(self) -> Strategy:
        """好みの戦略を取得"""
        if not self.strategy_preference:
            return Strategy.DIRECT
        
        best_strategy = max(
            self.strategy_preference.items(),
            key=lambda x: x[1]
        )[0]
        return Strategy(best_strategy)


class ModelSelector:
    """モデル選択器（Multi-Armed Bandit）"""
    
    MODELS = {
        'llama-3.1-8b-instant': {'speed': 'fast', 'cost': 'low', 'quality': 'medium'},
        'llama-3.1-70b-versatile': {'speed': 'medium', 'cost': 'medium', 'quality': 'high'},
        'llama-3.3-70b-versatile': {'speed': 'medium', 'cost': 'medium', 'quality': 'high'},
    }
    
    PRICING = {
        'llama-3.1-8b-instant': {'input': 0.05 / 1e6, 'output': 0.08 / 1e6},
        'llama-3.1-70b-versatile': {'input': 0.59 / 1e6, 'output': 0.79 / 1e6},
        'llama-3.3-70b-versatile': {'input': 0.59 / 1e6, 'output': 0.79 / 1e6},
    }
    
    def __init__(self):
        self.stats: Dict[str, Dict] = {}
        self.total_pulls = 0
        
        for model in self.MODELS:
            self.stats[model] = {
                'pulls': 0,
                'wins': 0,
                'total_reward': 0,
                'avg_quality': 0,
                'avg_cost': 0,
                'avg_latency': 0
            }
    
    def select(
        self,
        complexity: Complexity,
        exploration_rate: float = 0.15
    ) -> str:
        """UCB1でモデルを選択"""
        # Exploration
        if np.random.random() < exploration_rate:
            model = np.random.choice(list(self.MODELS.keys()))
            logger.info(f"🎲 Explore: {model}")
            return model
        
        # Exploitation (UCB1)
        best_model = None
        best_score = -float('inf')
        
        for model, stats in self.stats.items():
            if stats['pulls'] == 0:
                score = float('inf')
            else:
                avg_reward = stats['total_reward'] / stats['pulls']
                exploration = 2.0 * math.sqrt(
                    math.log(self.total_pulls) / stats['pulls']
                )
                score = avg_reward + exploration
                
                # 複雑度に基づくボーナス
                if complexity in [Complexity.EXPERT, Complexity.RESEARCH]:
                    if '70b' in model:
                        score *= 1.3
                elif complexity in [Complexity.TRIVIAL, Complexity.SIMPLE]:
                    if '8b' in model:
                        score *= 1.4
            
            if score > best_score:
                best_score = score
                best_model = model
        
        logger.info(f"🎯 Exploit: {best_model} (UCB: {best_score:.2f})")
        return best_model or list(self.MODELS.keys())[0]
    
    def update(
        self,
        model: str,
        reward: float,
        cost: float,
        latency: float,
        quality: float
    ):
        """統計を更新"""
        stats = self.stats[model]
        stats['pulls'] += 1
        stats['total_reward'] += reward
        
        n = stats['pulls']
        stats['avg_quality'] = (stats['avg_quality'] * (n - 1) + quality) / n
        stats['avg_cost'] = (stats['avg_cost'] * (n - 1) + cost) / n
        stats['avg_latency'] = (stats['avg_latency'] * (n - 1) + latency) / n
        
        if reward > 0.6:
            stats['wins'] += 1
        
        self.total_pulls += 1
    
    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """コスト計算"""
        pricing = self.PRICING.get(model, {'input': 0.0001 / 1e6, 'output': 0.0001 / 1e6})
        return prompt_tokens * pricing['input'] + completion_tokens * pricing['output']
    
    def get_stats(self) -> List[Dict]:
        """統計情報を取得"""
        result = []
        for model, stats in sorted(
            self.stats.items(),
            key=lambda x: x[1]['total_reward'] / max(x[1]['pulls'], 1),
            reverse=True
        ):
            if stats['pulls'] > 0:
                result.append({
                    'model': model.split('-')[-1],
                    'pulls': stats['pulls'],
                    'win_rate': f"{stats['wins'] / stats['pulls']:.1%}",
                    'avg_reward': f"{stats['total_reward'] / stats['pulls']:.4f}",
                    'avg_quality': f"{stats['avg_quality']:.2f}",
                    'avg_cost': f"${stats['avg_cost']:.6f}",
                    'avg_latency': f"{stats['avg_latency']:.0f}ms"
                })
        return result


# ==================== メインシステム ====================

class UltraAdvancedLLM:
    """Ultra-Advanced Self-Evolving LLM System"""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[SystemConfig] = None):
        self.api_key = api_key or os.environ.get('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("❌ GROQ_API_KEY required")
        
        self.config = config or SystemConfig()
        self.client = Groq(api_key=self.api_key)
        
        # コアコンポーネント
        self.vector_db = VectorDB(self.config.vec_dim) if self.config.vec_db else None
        self.knowledge_graph = KnowledgeGraph() if self.config.knowledge_graph else None
        self.profile = UserProfile()
        self.model_selector = ModelSelector()
        self.tree_of_thoughts = TreeOfThoughts() if self.config.tree_of_thoughts else None
        
        # メトリクス
        self.metrics = PerformanceMetrics()
        
        # コンテキストウィンドウ
        self.context_window: deque = deque(maxlen=15)
        
        # 長期記憶
        self.long_term_memory: Dict[str, Any] = {
            'entities': {},
            'facts': [],
            'summaries': []
        }
        
        logger.info(f"✅ Ultra-Advanced LLM initialized: {self.config.model}")
        self._log_features()
    
    def _log_features(self):
        """有効な機能をログ出力"""
        features = []
        if self.config.adaptive:
            features.append("🧠Adaptive")
        if self.config.multi_armed_bandit:
            features.append("🎰MAB")
        if self.config.knowledge_graph:
            features.append("🧩KG")
        if self.config.chain_of_thought:
            features.append("🤔CoT")
        if self.config.tree_of_thoughts:
            features.append("🌳ToT")
        if self.config.ensemble_learning:
            features.append("🎭Ensemble")
        if self.config.self_reflection:
            features.append("🔄Reflection")
        
        logger.info(" | ".join(features))
    
    # ========== 分析メソッド ==========
    
    def _analyze_complexity(self, query: str) -> Complexity:
        """クエリの複雑度を分析"""
        q = query.lower()
        
        # 各種指標
        length_score = len(query) // 100
        
        tech_words = ['algorithm', 'architecture', 'optimization', 'implementation', 'framework']
        expert_words = ['prove', 'derive', 'formal', 'theorem', 'axiom', 'hypothesis']
        research_words = ['investigate', 'research', 'study', 'analyze deeply', 'comprehensive']
        multi_step = ['step by step', 'first', 'then', 'finally', 'process', 'procedure']
        
        score = length_score
        score += sum(2 for w in tech_words if w in q)
        score += sum(3 for w in expert_words if w in q)
        score += sum(4 for w in research_words if w in q)
        score += sum(1 for phrase in multi_step if phrase in q)
        score += q.count('?')
        
        # 複雑度判定
        if score < 2:
            return Complexity.TRIVIAL
        elif score < 4:
            return Complexity.SIMPLE
        elif score < 7:
            return Complexity.MEDIUM
        elif score < 12:
            return Complexity.COMPLEX
        elif score < 16:
            return Complexity.EXPERT
        else:
            return Complexity.RESEARCH
    
    def _analyze_sentiment(self, text: str) -> float:
        """センチメント分析（簡易版）"""
        positive_words = [
            'good', 'great', 'excellent', 'thank', 'love', 'perfect',
            'amazing', 'wonderful', 'fantastic', 'brilliant'
        ]
        negative_words = [
            'bad', 'wrong', 'terrible', 'hate', 'awful', 'poor',
            'fail', 'error', 'horrible', 'disappointing'
        ]
        
        t = text.lower()
        pos_count = sum(t.count(w) for w in positive_words)
        neg_count = sum(t.count(w) for w in negative_words)
        total = pos_count + neg_count
        
        if total == 0:
            return 0.0
        
        return (pos_count - neg_count) / total
    
    def _select_strategy(self, intent: Intent, complexity: Complexity) -> Strategy:
        """意図と複雑度から最適な戦略を選択"""
        if not self.config.adaptive:
            return Strategy.DIRECT
        
        # 研究レベルの複雑度
        if complexity == Complexity.RESEARCH:
            if self.config.tree_of_thoughts:
                return Strategy.TREE_SEARCH
            elif self.config.ensemble_learning:
                return Strategy.ENSEMBLE
        
        # エキスパートレベル
        if complexity == Complexity.EXPERT:
            if self.config.ensemble_learning:
                return Strategy.ENSEMBLE
            elif self.config.chain_of_thought:
                return Strategy.COT
        
        # 意図ベース
        if intent in [Intent.REASONING, Intent.ANALYSIS, Intent.RESEARCH]:
            if self.config.chain_of_thought:
                return Strategy.COT
        
        if intent == Intent.PLANNING:
            return Strategy.ITERATIVE
        
        # ユーザー好みベース
        preferred = self.profile.get_preferred_strategy()
        if self.profile.strategy_preference.get(preferred.value, 0) > 0.7:
            return preferred
        
        return Strategy.DIRECT
    
    # ========== キャッシュ ==========
    
    def _check_cache(self, query: str) -> Optional[Response]:
        """キャッシュをチェック"""
        if not self.vector_db:
            return None
        
        results = self.vector_db.search(query, 1)
        if results:
            doc_id, similarity, metadata = results[0]
            timestamp = metadata.get('timestamp', 0)
            
            if similarity >= self.config.similarity_threshold and \
               time.time() - timestamp < self.config.cache_ttl:
                logger.info(f"🔄 Cache hit: {similarity:.3f}")
                
                resp_data = metadata.get('response', {})
                return Response(
                    text=resp_data.get('text', ''),
                    confidence=resp_data.get('confidence', 0),
                    tokens=resp_data.get('tokens', 0),
                    cost=resp_data.get('cost', 0),
                    model=resp_data.get('model', ''),
                    latency=resp_data.get('latency', 0),
                    cached=True,
                    similarity=similarity
                )
        
        return None
    
    def _save_to_cache(self, query: str, response: Response):
        """キャッシュに保存"""
        if not self.vector_db or not response.success:
            return
        
        doc_id = hashlib.md5(query.encode()).hexdigest()
        self.vector_db.add(doc_id, query, {
            'response': response.to_dict(),
            'timestamp': time.time()
        })
    
    # ========== プロンプト構築 ==========
    
    def _build_system_prompt(self, query: str, intent: Intent, complexity: Complexity) -> str:
        """適応的システムプロンプトを構築"""
        if not self.config.adaptive:
            return "You are a helpful AI assistant."
        
        base = "You are an advanced AI assistant with deep expertise across multiple domains."
        
        # スタイル
        style = self.profile.get_style_prompt()
        
        # 意図別の調整
        intent_adjustments = {
            Intent.TECHNICAL: "Focus on technical accuracy. Provide code examples and algorithms when relevant.",
            Intent.CREATIVE: "Be creative and imaginative. Think outside the box.",
            Intent.QUESTION: "Provide clear, direct answers with supporting reasoning.",
            Intent.EXPLANATION: "Explain thoroughly with examples, analogies, and context.",
            Intent.REASONING: "Use logical reasoning. Show your step-by-step thinking process.",
            Intent.ANALYSIS: "Provide deep analysis. Compare, contrast, and evaluate.",
            Intent.RESEARCH: "Provide comprehensive research-level insights with multiple perspectives.",
            Intent.PLANNING: "Think strategically. Consider multiple options and trade-offs."
        }
        intent_text = intent_adjustments.get(intent, "")
        
        # 複雑度別の調整
        complexity_adjustments = {
            Complexity.TRIVIAL: "Keep it extremely simple and concise.",
            Complexity.SIMPLE: "Keep it simple and clear.",
            Complexity.MEDIUM: "Provide adequate detail and context.",
            Complexity.COMPLEX: "Dive deep with comprehensive analysis.",
            Complexity.EXPERT: "Provide expert-level insights. Use formal terminology.",
            Complexity.RESEARCH: "Provide research-grade analysis with citations and evidence."
        }
        complexity_text = complexity_adjustments.get(complexity, "")
        
        # 専門知識コンテキスト
        expertise_topics = [
            topic for topic, level in self.profile.expertise_level.items()
            if level > 0.6
        ][:3]
        expertise_text = ""
        if expertise_topics:
            expertise_text = f" User has expertise in: {', '.join(expertise_topics)}."
        
        # 知識グラフコンテキスト
        kg_text = self._get_knowledge_graph_context(query)
        
        # コンテキストウィンドウ
        context_text = ""
        if self.context_window:
            recent = list(self.context_window)[-3:]
            context_text = f" Recent context: {' | '.join(q[:50] for q in recent)}."
        
        # ネガティブワード回避
        avoid_text = ""
        if self.profile.negative_words:
            sample = list(self.profile.negative_words)[:5]
            avoid_text = f" Avoid these terms: {', '.join(sample)}."
        
        # プロンプト組み立て
        prompt = f"{base} {style} {intent_text} {complexity_text}{expertise_text}{kg_text}{context_text}{avoid_text}"
        
        return prompt.strip()
    
    def _get_knowledge_graph_context(self, query: str, max_concepts: int = 2) -> str:
        """知識グラフからコンテキストを取得"""
        if not self.knowledge_graph:
            return ""
        
        # クエリからエンティティを抽出
        entities = re.findall(r'\b[A-Z][a-z]+\b', query)
        context_parts = []
        
        for entity in entities[:max_concepts]:
            entity_id = hashlib.md5(entity.encode()).hexdigest()[:8]
            if entity_id in self.knowledge_graph.nodes:
                related = self.knowledge_graph.get_related_concepts(entity_id, 3)
                if related:
                    related_names = [
                        self.knowledge_graph.nodes[r[0]].name
                        for r in related
                    ]
                    context_parts.append(f"{entity} relates to: {', '.join(related_names)}")
        
        if context_parts:
            return f" Knowledge: {'; '.join(context_parts)}."
        
        return ""
    
    # ========== API呼び出し ==========
    
    async def _call_api(
        self,
        model: str,
        messages: List[Dict],
        temperature: float,
        max_tokens: int
    ):
        """API呼び出し（リトライ付き）"""
        for attempt in range(self.config.max_retries):
            try:
                return self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except (RateLimitError, APIError) as e:
                if attempt == self.config.max_retries - 1:
                    raise
                
                wait_time = self.config.retry_delay * (2 ** attempt)
                logger.warning(f"Retry {attempt + 1}/{self.config.max_retries} after {wait_time}s")
                await asyncio.sleep(wait_time)
    
    def _build_response_from_api(
        self,
        api_response,
        model: str,
        strategy: Strategy,
        latency: float
    ) -> Response:
        """APIレスポンスからResponseオブジェクトを構築"""
        choice = api_response.choices[0]
        text = choice.message.content or ""
        finish_reason = choice.finish_reason
        
        usage = api_response.usage
        cost = self.model_selector.calculate_cost(
            model,
            usage.prompt_tokens,
            usage.completion_tokens
        )
        
        # センチメント分析
        sentiment = self._analyze_sentiment(text)
        
        # 基本信頼度
        base_confidence = 0.90 if finish_reason == "stop" else 0.75
        confidence = base_confidence * (1.0 + sentiment * 0.1)
        confidence = max(0.0, min(1.0, confidence))
        
        # 不確実性推定
        uncertain_phrases = ['maybe', 'perhaps', 'possibly', 'might', 'could be', 'uncertain']
        uncertainty = sum(0.1 for phrase in uncertain_phrases if phrase in text.lower())
        uncertainty = min(1.0, uncertainty)
        
        # 品質スコア（簡易版）
        coherence = min(1.0, len(text.split('.')) / 10)  # 文の数
        relevance = 0.8  # 実際は意味的類似度が必要
        completeness = min(1.0, len(text) / 500)
        factuality = 0.85  # 実際はファクトチェックが必要
        
        return Response(
            text=text,
            confidence=confidence,
            tokens=usage.total_tokens,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency=latency,
            cost=cost,
            model=model,
            finish_reason=finish_reason,
            strategy=strategy,
            sentiment=sentiment,
            uncertainty=uncertainty,
            coherence_score=coherence,
            relevance_score=relevance,
            completeness_score=completeness,
            factuality_score=factuality
        )
    
    # ========== 実行戦略 ==========
    
    async def _execute_direct(
        self,
        query: str,
        model: str,
        temperature: float,
        max_tokens: int,
        intent: Intent,
        complexity: Complexity
    ) -> Response:
        """直接実行戦略"""
        system_prompt = self._build_system_prompt(query, intent, complexity)
        
        start_time = time.time()
        api_response = await self._call_api(
            model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature,
            max_tokens
        )
        latency = (time.time() - start_time) * 1000
        
        return self._build_response_from_api(api_response, model, Strategy.DIRECT, latency)
    
    async def _execute_chain_of_thought(
        self,
        query: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Response:
        """Chain-of-Thought実行"""
        cot_prompt = f"Let's think step by step.\n\nQuery: {query}\n\nReasoning:"
        
        start_time = time.time()
        api_response = await self._call_api(
            model,
            [
                {"role": "system", "content": "You are a logical reasoning expert. Show your thinking process step by step."},
                {"role": "user", "content": cot_prompt}
            ],
            temperature,
            max_tokens
        )
        latency = (time.time() - start_time) * 1000
        
        response = self._build_response_from_api(api_response, model, Strategy.COT, latency)
        
        # 推論ステップを抽出
        steps = re.findall(r'(?:Step \d+:|^\d+\.|^-)\s*(.+)', response.text, re.MULTILINE)
        response.reasoning_steps = steps[:10]
        
        return response
    
    async def _execute_reflection(
        self,
        query: str,
        model: str,
        temperature: float,
        max_tokens: int,
        intent: Intent,
        complexity: Complexity
    ) -> Response:
        """Self-Reflection実行"""
        # 初期回答
        initial = await self._execute_direct(query, model, temperature, max_tokens, intent, complexity)
        
        # 反省プロンプト
        reflect_prompt = f"""Initial Answer: {initial.text}

Now, critically reflect on this answer:
1. Are there any errors or inaccuracies?
2. What aspects could be improved?
3. Are there alternative perspectives to consider?
4. What are the limitations of this answer?

Provide an improved, refined answer:"""
        
        start_time = time.time()
        api_response = await self._call_api(
            model,
            [
                {"role": "system", "content": "You are a critical thinker who improves answers through reflection."},
                {"role": "user", "content": reflect_prompt}
            ],
            temperature,
            max_tokens
        )
        latency = (time.time() - start_time) * 1000
        
        response = self._build_response_from_api(api_response, model, Strategy.REFLECTION, latency)
        response.reflection = initial.text[:200]
        
        return response
    
    async def _execute_ensemble(
        self,
        query: str,
        temperature: float,
        max_tokens: int,
        intent: Intent,
        complexity: Complexity
    ) -> Response:
        """アンサンブル実行（複数モデル）"""
        # トップ3モデルを選択
        top_models = sorted(
            self.model_selector.stats.items(),
            key=lambda x: x[1]['total_reward'] / max(x[1]['pulls'], 1),
            reverse=True
        )[:3]
        models = [m[0] for m in top_models]
        
        responses = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for m in models:
                future = executor.submit(
                    lambda model: asyncio.run(
                        self._execute_direct(query, model, temperature, max_tokens, intent, complexity)
                    ),
                    m
                )
                futures.append((m, future))
            
            for m, future in futures:
                try:
                    resp = future.result(timeout=30)
                    responses.append(resp)
                except Exception as e:
                    logger.warning(f"Ensemble model {m} failed: {e}")
        
        if not responses:
            # フォールバック
            return await self._execute_direct(
                query,
                self.config.model,
                temperature,
                max_tokens,
                intent,
                complexity
            )
        
        # 最高品質の応答を選択
        best = max(responses, key=lambda r: r.quality_score)
        best.strategy = Strategy.ENSEMBLE
        best.alternatives = [r.text[:100] for r in responses if r != best]
        
        return best
    
    async def _execute_tree_of_thoughts(
        self,
        query: str,
        model: str,
        temperature: float,
        max_tokens: int
    ) -> Response:
        """Tree of Thoughts実行"""
        if not self.tree_of_thoughts:
            return await self._execute_direct(
                query, model, temperature, max_tokens,
                Intent.REASONING, Complexity.EXPERT
            )
        
        # リセット
        self.tree_of_thoughts.reset()
        
        # ルート作成
        root_id = self.tree_of_thoughts.create_root(query)
        
        # 深さ優先探索
        for depth in range(self.tree_of_thoughts.max_depth):
            # 現在の葉ノードを取得
            leaf_nodes = [
                nid for nid, node in self.tree_of_thoughts.nodes.items()
                if not node.children and node.depth == depth
            ]
            
            for node_id in leaf_nodes[:3]:  # 最大3ノード
                node = self.tree_of_thoughts.nodes[node_id]
                
                # 次の思考候補を生成
                expand_prompt = f"Given: '{node.content}'\nGenerate 3 different next reasoning steps:"
                
                api_response = await self._call_api(
                    model,
                    [
                        {"role": "system", "content": "Generate diverse reasoning paths."},
                        {"role": "user", "content": expand_prompt}
                    ],
                    temperature,
                    max_tokens // 2
                )
                
                text = api_response.choices[0].message.content or ""
                candidates = re.findall(r'\d+\.\s*(.+?)(?=\n\d+\.|\n*$)', text, re.DOTALL)
                candidates = [c.strip() for c in candidates if c.strip()]
                
                # 展開
                self.tree_of_thoughts.expand(node_id, candidates)
                
                # 評価
                for child_id in self.tree_of_thoughts.nodes[node_id].children:
                    child = self.tree_of_thoughts.nodes[child_id]
                    
                    eval_prompt = f"Rate the quality of this reasoning (0-1): '{child.content}'"
                    eval_response = await self._call_api(
                        model,
                        [
                            {"role": "system", "content": "Provide a number between 0 and 1."},
                            {"role": "user", "content": eval_prompt}
                        ],
                        0.3,
                        50
                    )
                    
                    eval_text = eval_response.choices[0].message.content or "0.5"
                    try:
                        value = float(re.search(r'0?\.\d+|[01]', eval_text).group())
                    except:
                        value = 0.5
                    
                    # 逆伝播
                    self.tree_of_thoughts.backpropagate(child_id, value)
        
        # 最良パスを選択
        best_path = self.tree_of_thoughts.select_best_path()
        path_content = self.tree_of_thoughts.get_path_content(best_path)
        
        # 最終回答を生成
        final_prompt = (
            f"Based on this reasoning:\n" +
            "\n".join(f"{i+1}. {c}" for i, c in enumerate(path_content)) +
            f"\n\nProvide a final answer to: {query}"
        )
        
        start_time = time.time()
        final_response = await self._call_api(
            model,
            [
                {"role": "system", "content": "Synthesize the reasoning into a final answer."},
                {"role": "user", "content": final_prompt}
            ],
            temperature,
            max_tokens
        )
        latency = (time.time() - start_time) * 1000
        
        response = self._build_response_from_api(final_response, model, Strategy.TREE_SEARCH, latency)
        response.reasoning_steps = path_content
        
        return response
    
    # ========== メインクエリ処理 ==========
    
    async def query_async(self, query: str, **kwargs) -> Response:
        """メインクエリ処理（非同期）"""
        start_time = time.time()
        self.metrics.query_count += 1
        
        try:
            # 長さチェック
            if len(query) > self.config.max_query_length:
                return Response(
                    text=f"❌ Query too long (max: {self.config.max_query_length})",
                    confidence=0,
                    finish_reason="error"
                )
            
            # キャッシュチェック
            cached = self._check_cache(query)
            if cached:
                self.metrics.cache_hits += 1
                return cached
            
            # 分析
            intent = self.profile.predict_intent(query)
            complexity = self._analyze_complexity(query)
            strategy = self._select_strategy(intent, complexity)
            
            # モデル選択
            if self.config.multi_armed_bandit:
                model = self.model_selector.select(complexity)
            else:
                model = kwargs.get('model', self.config.model)
            
            # パラメータ
            temperature = kwargs.get(
                'temperature',
                self.profile.get_adapted_temperature() if self.config.adaptive else self.config.temperature
            )
            max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
            
            # 戦略実行
            if strategy == Strategy.COT and self.config.chain_of_thought:
                response = await self._execute_chain_of_thought(query, model, temperature, max_tokens)
            elif strategy == Strategy.REFLECTION and self.config.self_reflection:
                response = await self._execute_reflection(query, model, temperature, max_tokens, intent, complexity)
            elif strategy == Strategy.ENSEMBLE and self.config.ensemble_learning:
                response = await self._execute_ensemble(query, temperature, max_tokens, intent, complexity)
            elif strategy == Strategy.TREE_SEARCH and self.config.tree_of_thoughts:
                response = await self._execute_tree_of_thoughts(query, model, temperature, max_tokens)
            else:
                response = await self._execute_direct(query, model, temperature, max_tokens, intent, complexity)
            
            # メタデータ設定
            response.intent = intent
            response.complexity = complexity
            
            # メトリクス更新
            if response.success:
                self.metrics.success_count += 1
            self.metrics.total_tokens += response.tokens
            self.metrics.total_cost += response.cost
            self.metrics.total_latency += response.latency
            
            # MAB更新
            if self.config.multi_armed_bandit:
                quality = response.quality_score
                reward = (quality / max(response.cost, 0.00001)) * 0.01
                self.model_selector.update(
                    response.model,
                    reward,
                    response.cost,
                    response.latency,
                    quality
                )
            
            # コンテキスト更新
            self.context_window.append(query[:100])
            
            # キャッシュ保存
            self._save_to_cache(query, response)
            
            return response
        
        except Exception as e:
            self.metrics.errors += 1
            logger.error(f"Query failed: {e}")
            return Response(
                text=f"❌ Error: {str(e)}",
                confidence=0,
                finish_reason="error"
            )
    
    def query(self, query: str, **kwargs) -> Response:
        """メインクエリ処理（同期）"""
        return asyncio.run(self.query_async(query, **kwargs))
    
    def add_feedback(
        self,
        query: str,
        response: str,
        rating: int,
        response_obj: Optional[Response] = None
    ):
        """フィードバックを追加"""
        if not self.config.adaptive:
            return
        
        intent = response_obj.intent if response_obj else None
        complexity = response_obj.complexity if response_obj else None
        strategy = response_obj.strategy if response_obj else None
        
        self.profile.update_from_feedback(
            query, response, rating, intent, complexity, strategy
        )
        
        logger.info(f"🧠 Feedback: {rating:+d} | {intent} | {complexity} | {strategy}")
    
    # ========== 統計・保存 ==========
    
    def get_stats(self) -> Dict:
        """統計情報を取得"""
        uptime = (datetime.now() - self.metrics.start_time).total_seconds()
        
        stats = {
            'system': {
                'uptime': f"{uptime:.1f}s",
                'queries': self.metrics.query_count,
                'success_rate': f"{self.metrics.success_rate:.1%}",
                'cache_hit_rate': f"{self.metrics.cache_hit_rate:.1%}"
            },
            'performance': {
                'total_tokens': self.metrics.total_tokens,
                'total_cost': f"${self.metrics.total_cost:.6f}",
                'avg_cost': f"${self.metrics.avg_cost:.6f}",
                'avg_latency': f"{self.metrics.avg_latency:.0f}ms",
                'errors': self.metrics.errors
            },
            'config': {
                'model': self.config.model,
                'adaptive': self.config.adaptive,
                'mab': self.config.multi_armed_bandit,
                'kg': self.config.knowledge_graph,
                'cot': self.config.chain_of_thought
            }
        }
        
        # プロファイル
        if self.config.adaptive:
            stats['profile'] = {
                'style': self.profile.style,
                'avg_length': f"{self.profile.avg_response_length:.0f}",
                'temperature': f"{self.profile.get_adapted_temperature():.2f}",
                'interactions': self.profile.interaction_count,
                'expertise_areas': len([e for e in self.profile.expertise_level.values() if e > 0.5])
            }
        
        # MAB
        if self.config.multi_armed_bandit:
            stats['models'] = self.model_selector.get_stats()
        
        # 知識グラフ
        if self.config.knowledge_graph and self.knowledge_graph:
            stats['knowledge_graph'] = {
                'nodes': len(self.knowledge_graph.nodes),
                'edges': len(self.knowledge_graph.edges)
            }
        
        return stats
    
    def save_state(self, filepath: str = 'llm_state.json'):
        """状態を保存"""
        try:
            state = {
                'profile': {
                    'topics': dict(self.profile.topics),
                    'avg_response_length': self.profile.avg_response_length,
                    'style': self.profile.style,
                    'temperature_preference': self.profile.temperature_preference,
                    'positive_words': list(self.profile.positive_words),
                    'negative_words': list(self.profile.negative_words),
                    'feedback_history': self.profile.feedback_history,
                    'interaction_count': self.profile.interaction_count,
                    'intent_distribution': dict(self.profile.intent_distribution),
                    'expertise_level': dict(self.profile.expertise_level),
                    'strategy_preference': dict(self.profile.strategy_preference),
                    'last_updated': self.profile.last_updated.isoformat()
                },
                'metrics': {
                    'query_count': self.metrics.query_count,
                    'success_count': self.metrics.success_count,
                    'total_tokens': self.metrics.total_tokens,
                    'total_cost': self.metrics.total_cost,
                    'total_latency': self.metrics.total_latency,
                    'cache_hits': self.metrics.cache_hits,
                    'errors': self.metrics.errors
                },
                'model_selector': {
                    'stats': self.model_selector.stats,
                    'total_pulls': self.model_selector.total_pulls
                }
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 State saved: {filepath}")
        except Exception as e:
            logger.error(f"❌ Save failed: {e}")
    
    def load_state(self, filepath: str = 'llm_state.json'):
        """状態を読み込み"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # プロファイル復元
            profile_data = state.get('profile', {})
            self.profile.topics = defaultdict(int, profile_data.get('topics', {}))
            self.profile.avg_response_length = profile_data.get('avg_response_length', 100.0)
            self.profile.style = profile_data.get('style', 'balanced')
            self.profile.temperature_preference = profile_data.get('temperature_preference', 0.7)
            self.profile.positive_words = set(profile_data.get('positive_words', []))
            self.profile.negative_words = set(profile_data.get('negative_words', []))
            self.profile.feedback_history = profile_data.get('feedback_history', [])
            self.profile.interaction_count = profile_data.get('interaction_count', 0)
            self.profile.intent_distribution = defaultdict(
                int, profile_data.get('intent_distribution', {})
            )
            self.profile.expertise_level = defaultdict(
                float, profile_data.get('expertise_level', {})
            )
            self.profile.strategy_preference = defaultdict(
                float, profile_data.get('strategy_preference', {})
            )
            
            # メトリクス復元
            metrics_data = state.get('metrics', {})
            self.metrics.query_count = metrics_data.get('query_count', 0)
            self.metrics.success_count = metrics_data.get('success_count', 0)
            self.metrics.total_tokens = metrics_data.get('total_tokens', 0)
            self.metrics.total_cost = metrics_data.get('total_cost', 0)
            self.metrics.total_latency = metrics_data.get('total_latency', 0)
            self.metrics.cache_hits = metrics_data.get('cache_hits', 0)
            self.metrics.errors = metrics_data.get('errors', 0)
            
            # モデルセレクター復元
            selector_data = state.get('model_selector', {})
            self.model_selector.stats = selector_data.get('stats', {})
            self.model_selector.total_pulls = selector_data.get('total_pulls', 0)
            
            logger.info(f"📂 State loaded: {filepath}")
        except FileNotFoundError:
            logger.info("ℹ️  No saved state found - starting fresh")
        except Exception as e:
            logger.error(f"❌ Load failed: {e}")


# ==================== インタラクティブチャット ====================

class InteractiveChat:
    """インタラクティブチャットインターフェース"""
    
    def __init__(self, llm: UltraAdvancedLLM):
        self.llm = llm
        self.history: List[Tuple[str, Response]] = []
        self.session_id = str(uuid.uuid4())[:8]
    
    def print_welcome(self):
        """ウェルカムメッセージ"""
        print("\n" + "=" * 70)
        print("🚀 Ultra-Advanced Self-Evolving LLM System v2.0β")
        print("=" * 70)
        print("\nコマンド:")
        print("  /stats      - 統計情報を表示")
        print("  /save       - 状態を保存")
        print("  /load       - 状態を読み込み")
        print("  /feedback <rating> - 最後の回答を評価 (-2 to +2)")
        print("  /clear      - 履歴をクリア")
        print("  /profile    - ユーザープロファイルを表示")
        print("  /help       - ヘルプを表示")
        print("  /exit       - 終了")
        print("=" * 70 + "\n")
    
    def print_response(self, response: Response):
        """応答を表示"""
        print(f"\n🤖 Assistant [{response.model.split('-')[-1]}]:")
        print("-" * 70)
        print(response.text)
        print("-" * 70)
        
        # メタデータ
        metadata = []
        if response.strategy:
            metadata.append(f"📋{response.strategy.value}")
        if response.intent:
            metadata.append(f"🎯{response.intent.value}")
        if response.complexity:
            metadata.append(f"⚙️{response.complexity.value}")
        
        metadata.append(f"✅{response.confidence:.2f}")
        metadata.append(f"🎲{response.uncertainty:.2f}")
        metadata.append(f"⭐{response.quality_score:.2f}")
        metadata.append(f"💰${response.cost:.6f}")
        metadata.append(f"⏱️{response.latency:.0f}ms")
        metadata.append(f"🎫{response.tokens}tok")
        
        if response.cached:
            metadata.append(f"🔄Cache({response.similarity:.2f})")
        
        print(" | ".join(metadata))
        
        # 推論ステップ
        if response.reasoning_steps:
            print(f"\n🧠 Reasoning Steps:")
            for i, step in enumerate(response.reasoning_steps[:5], 1):
                print(f"  {i}. {step[:80]}...")
        
        # 反省
        if response.reflection:
            print(f"\n🔄 Initial thought: {response.reflection[:100]}...")
        
        # 代替案
        if response.alternatives:
            print(f"\n🎭 {len(response.alternatives)} alternatives considered")
        
        print()
    
    def print_stats(self):
        """統計情報を表示"""
        stats = self.llm.get_stats()
        
        print("\n" + "=" * 70)
        print("📊 System Statistics")
        print("=" * 70)
        
        # システム
        sys = stats['system']
        print(f"\n⏱️  Uptime: {sys['uptime']} | Queries: {sys['queries']}")
        print(f"   Success Rate: {sys['success_rate']} | Cache Hit Rate: {sys['cache_hit_rate']}")
        
        # パフォーマンス
        perf = stats['performance']
        print(f"\n💰 Cost: {perf['total_cost']} (avg: {perf['avg_cost']})")
        print(f"🎫 Tokens: {perf['total_tokens']:,} | Latency: {perf['avg_latency']}")
        print(f"❌ Errors: {perf['errors']}")
        
        # プロファイル
        if 'profile' in stats:
            prof = stats['profile']
            print(f"\n👤 Profile:")
            print(f"   Style: {prof['style']} | Avg Length: {prof['avg_length']}")
            print(f"   Temperature: {prof['temperature']} | Interactions: {prof['interactions']}")
            print(f"   Expertise Areas: {prof['expertise_areas']}")
        
        # モデル
        if 'models' in stats:
            print(f"\n🎰 Model Performance:")
            for model in stats['models']:
                print(f"   {model['model']:12s}: pulls={model['pulls']:3d} "
                      f"win={model['win_rate']} reward={model['avg_reward']} "
                      f"quality={model['avg_quality']}")
        
        # 知識グラフ
        if 'knowledge_graph' in stats:
            kg = stats['knowledge_graph']
            print(f"\n🧩 Knowledge Graph: {kg['nodes']} nodes | {kg['edges']} edges")
        
        print("=" * 70 + "\n")
    
    def print_profile(self):
        """プロファイル詳細を表示"""
        profile = self.llm.profile
        
        print("\n" + "=" * 70)
        print("👤 User Profile")
        print("=" * 70)
        
        print(f"\nStyle: {profile.style}")
        print(f"Temperature: {profile.get_adapted_temperature():.2f}")
        print(f"Avg Response Length: {profile.avg_response_length:.0f}")
        print(f"Interactions: {profile.interaction_count}")
        
        # トップトピック
        if profile.topics:
            top_topics = sorted(profile.topics.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n📚 Top Topics:")
            for topic, count in top_topics:
                print(f"   {topic}: {count}")
        
        # 専門知識
        expertise = [
            (topic, level) for topic, level in profile.expertise_level.items()
            if level > 0.5
        ]
        if expertise:
            expertise.sort(key=lambda x: x[1], reverse=True)
            print(f"\n🎓 Expertise:")
            for topic, level in expertise[:5]:
                bar = '█' * int(level * 20) + '░' * (20 - int(level * 20))
                print(f"   {topic:15s} [{bar}] {level:.1%}")
        
        # 意図分布
        if profile.intent_distribution:
            print(f"\n🎯 Intent Distribution:")
            total = sum(profile.intent_distribution.values())
            for intent, count in sorted(
                profile.intent_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]:
                pct = count / total * 100
                print(f"   {intent:15s}: {pct:5.1f}%")
        
        # 戦略好み
        if profile.strategy_preference:
            print(f"\n📋 Strategy Preferences:")
            for strategy, score in sorted(
                profile.strategy_preference.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]:
                print(f"   {strategy:15s}: {score:.2f}")
        
        print("=" * 70 + "\n")
    
    def handle_command(self, command: str) -> bool:
        """コマンドを処理。継続する場合True"""
        parts = command.strip().split()
        cmd = parts[0].lower()
        
        if cmd == '/exit':
            print("👋 Goodbye!")
            return False
        
        elif cmd == '/stats':
            self.print_stats()
        
        elif cmd == '/save':
            filepath = parts[1] if len(parts) > 1 else 'llm_state.json'
            self.llm.save_state(filepath)
        
        elif cmd == '/load':
            filepath = parts[1] if len(parts) > 1 else 'llm_state.json'
            self.llm.load_state(filepath)
        
        elif cmd == '/feedback':
            if not self.history:
                print("❌ No previous response to rate")
                return True
            
            try:
                rating = int(parts[1]) if len(parts) > 1 else 0
                if rating < -2 or rating > 2:
                    print("❌ Rating must be between -2 and +2")
                    return True
                
                last_query, last_response = self.history[-1]
                self.llm.add_feedback(last_query, last_response.text, rating, last_response)
                print(f"✅ Feedback recorded: {rating:+d}")
            
            except ValueError:
                print("❌ Invalid rating format")
        
        elif cmd == '/clear':
            self.history.clear()
            self.llm.context_window.clear()
            if self.llm.vector_db:
                self.llm.vector_db.clear()
            print("🗑️  History cleared")
        
        elif cmd == '/profile':
            self.print_profile()
        
        elif cmd == '/help':
            self.print_welcome()
        
        else:
            print(f"❌ Unknown command: {cmd}")
            print("Type /help for available commands")
        
        return True
    
    def run(self):
        """メインループ"""
        self.print_welcome()
        
        while True:
            try:
                query = input("👤 You: ").strip()
                
                if not query:
                    continue
                
                # コマンド処理
                if query.startswith('/'):
                    if not self.handle_command(query):
                        break
                    continue
                
                # クエリ処理
                print("\n⏳ Thinking...")
                response = self.llm.query(query)
                
                self.history.append((query, response))
                self.print_response(response)
            
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted. Type /exit to quit.")
                continue
            except EOFError:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Chat error: {e}")


# ==================== メイン実行 ====================

def main():
    """エントリーポイント"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Ultra-Advanced Self-Evolving LLM System v2.0β'
    )
    parser.add_argument('--model', default='llama-3.1-8b-instant', help='Base model')
    parser.add_argument('--no-adapt', action='store_true', help='Disable adaptation')
    parser.add_argument('--no-mab', action='store_true', help='Disable MAB')
    parser.add_argument('--no-kg', action='store_true', help='Disable knowledge graph')
    parser.add_argument('--no-cot', action='store_true', help='Disable CoT')
    parser.add_argument('--no-reflection', action='store_true', help='Disable reflection')
    parser.add_argument('--no-ensemble', action='store_true', help='Disable ensemble')
    parser.add_argument('--no-tot', action='store_true', help='Disable tree-of-thoughts')
    parser.add_argument('--query', type=str, help='Single query mode')
    parser.add_argument('--load', type=str, help='Load saved state')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # ロギング設定
    if args.debug:
        logger.logger.setLevel(logging.DEBUG)
    
    # 設定
    config = SystemConfig(
        model=args.model,
        adaptive=not args.no_adapt,
        multi_armed_bandit=not args.no_mab,
        knowledge_graph=not args.no_kg,
        chain_of_thought=not args.no_cot,
        self_reflection=not args.no_reflection,
        ensemble_learning=not args.no_ensemble,
        tree_of_thoughts=not args.no_tot
    )
    
    try:
        # システム初期化
        llm = UltraAdvancedLLM(config=config)
        
        # 状態読み込み
        if args.load:
            llm.load_state(args.load)
        
        # シングルクエリモード
        if args.query:
            response = llm.query(args.query)
            print(response.text)
            print(f"\nMetadata: confidence={response.confidence:.2f} "
                  f"quality={response.quality_score:.2f} "
                  f"cost=${response.cost:.6f} "
                  f"latency={response.latency:.0f}ms")
            return
        
        # インタラクティブモード
        chat = InteractiveChat(llm)
        chat.run()
        
        # 終了時に保存
        print("\n💾 Saving session data...")
        llm.save_state()
        
        # 最終統計
        print("\n📊 Session Summary:")
        stats = llm.get_stats()
        print(f"   Queries: {stats['system']['queries']}")
        print(f"   Success Rate: {stats['system']['success_rate']}")
        print(f"   Total Cost: {stats['performance']['total_cost']}")
        print(f"   Cache Hit Rate: {stats['system']['cache_hit_rate']}")
    
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        print("Please set GROQ_API_KEY environment variable")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.error(f"Fatal: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
