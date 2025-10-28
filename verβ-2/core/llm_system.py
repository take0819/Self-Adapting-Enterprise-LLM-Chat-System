# -*- coding: utf-8 -*-
"""
メインLLMシステム
Quantum-Enhanced LLM System v3.5 ULTIMATE
"""

import os
import re
import json
import time
import uuid
import hashlib
import asyncio
import statistics
from typing import Optional, Dict, Any, List
from datetime import datetime
from collections import defaultdict, deque

from groq import Groq, RateLimitError, APIError
import numpy as np

from core.config import SystemConfig
from core.enums import Intent, Complexity, Strategy
from core.data_models import Response, KnowledgeNode, KnowledgeEdge
from core.query_processor import QueryProcessor
from knowledge.vector_db import VectorDB
from knowledge.knowledge_graph import AdvancedKnowledgeGraph
from optimizers.quantum_optimizer import QuantumOptimizer
from optimizers.genetic_evolver import GeneticPromptEvolver
from optimizers.swarm_intelligence import SwarmIntelligence
from optimizers.rlhf_trainer import RLHFTrainer
from utils.logger import logger
from utils.cost_calculator import CostCalculator


class QuantumLLM:
    """Quantum-Enhanced LLM System v3.5 ULTIMATE"""
    
    MODELS = {
        'llama-3.1-8b-instant': {'cost': 'low', 'quality': 'medium', 'speed': 'fast'},
        'llama-3.1-70b-versatile': {'cost': 'medium', 'quality': 'high', 'speed': 'medium'},
        'llama-3.3-70b-versatile': {'cost': 'medium', 'quality': 'high', 'speed': 'medium'},
    }
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[SystemConfig] = None):
        self.api_key = api_key or os.environ.get('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("❌ GROQ_API_KEY required")
        
        self.config = config or SystemConfig()
        self.client = Groq(api_key=self.api_key)
        
        # コンテキスト
        self.context_window = deque(maxlen=20)
        
        # プロンプト集団の初期化
        if self.genetic_evolver:
            base_prompts = [
                "Provide a clear and comprehensive answer.",
                "Think step by step and explain your reasoning.",
                "Analyze the question from multiple perspectives.",
                "Apply scientific method to validate your response.",
                "Consider causal relationships and logical implications."
            ]
            self.genetic_evolver.initialize_population(base_prompts, "general")
        
        logger.info(f"✅ Quantum-Enhanced LLM v3.5 ULTIMATE initialized")
        self._log_features()
    
    def _init_profile(self) -> Dict[str, Any]:
        """プロファイル初期化"""
        return {
            'topics': defaultdict(int),
            'expertise': defaultdict(float),
            'strategy_preference': defaultdict(float),
            'interaction_count': 0,
            'feedback_history': [],
            'learning_trajectory': [],
            'prediction_accuracy': 0.5
        }
    
    def _log_features(self):
        """有効機能をログ出力"""
        features = []
        if self.config.quantum.enabled:
            features.append("🔮Quantum")
        if self.config.genetic.enabled:
            features.append("🧬Genetic")
        if self.config.swarm.enabled:
            features.append("🌊Swarm")
        if self.config.rlhf.enabled:
            features.append("🎯RLHF")
        
        logger.info(" | ".join(features))
    
    async def query_async(self, query: str, **kwargs) -> Response:
        """
        メインクエリ処理（非同期）
        
        Args:
            query: 入力クエリ
            **kwargs: 追加パラメータ
        
        Returns:
            応答
        """
        self.metrics['queries'] += 1
        
        try:
            # キャッシュチェック
            if self.vector_db:
                cached_results = self.vector_db.search(
                    query, top_k=1, min_similarity=self.config.similarity_threshold
                )
                if cached_results:
                    _, similarity, metadata = cached_results[0]
                    if time.time() - metadata.get('added_at', 0) < self.config.cache_ttl:
                        self.metrics['cache_hits'] += 1
                        logger.info(f"🔄 Cache hit: {similarity:.3f}")
                        resp_data = metadata.get('response', {})
                        return Response(
                            text=resp_data.get('text', ''),
                            confidence=resp_data.get('confidence', 0),
                            cached=True,
                            similarity=similarity
                        )
            
            # クエリ分析
            intent, complexity = self.query_processor.analyze_query(query)
            strategy = self.query_processor.select_strategy(intent, complexity, self.rlhf)
            
            model = kwargs.get('model', self.config.model)
            
            # 戦略実行
            if strategy == Strategy.QUANTUM and self.quantum_optimizer:
                response = await self._execute_quantum_strategy(query, model, intent, complexity)
            elif strategy == Strategy.GENETIC and self.genetic_evolver:
                response = await self._execute_genetic_strategy(query, model, intent, complexity)
            elif strategy == Strategy.SWARM and self.swarm:
                response = await self._execute_swarm_strategy(query, model, intent, complexity)
            else:
                response = await self._execute_direct(query, model, intent, complexity)
            
            # メタデータ設定
            response.intent = intent
            response.complexity = complexity
            
            # メトリクス更新
            if response.success:
                self.metrics['success'] += 1
            self.metrics['total_cost'] += response.cost
            self.metrics['total_tokens'] += response.tokens
            
            # RLHF更新
            if self.rlhf:
                state = self.rlhf.get_state(intent, complexity)
                reward = response.quality_score
                next_state = state
                self.rlhf.update(state, strategy.value, reward, next_state)
            
            # コンテキスト更新
            self.context_window.append(query[:100])
            
            # キャッシュ保存
            if self.vector_db and response.success:
                self.vector_db.add(
                    str(uuid.uuid4())[:8],
                    query,
                    {'response': response.to_dict()}
                )
            
            # 知識グラフ更新
            if self.knowledge_graph:
                self._update_knowledge_graph(query, response.text)
            
            return response
        
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return Response(
                text=f"❌ Error: {str(e)}",
                confidence=0,
                finish_reason="error"
            )
    
    def query(self, query: str, **kwargs) -> Response:
        """
        メインクエリ処理（同期）
        
        Args:
            query: 入力クエリ
            **kwargs: 追加パラメータ
        
        Returns:
            応答
        """
        return asyncio.run(self.query_async(query, **kwargs))
    
    async def _call_api(
        self,
        model: str,
        messages: List[Dict],
        temperature: float,
        max_tokens: int
    ):
        """API呼び出し"""
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
                logger.warning(f"Retry {attempt + 1}/{self.config.max_retries}")
                await asyncio.sleep(wait_time)アコンポーネント
        self.query_processor = QueryProcessor(self.config)
        self.vector_db = VectorDB(self.config.vec_dim) if self.config.vec_db else None
        self.knowledge_graph = AdvancedKnowledgeGraph() if self.config.knowledge_graph else None
        
        # 最適化コンポーネント
        self.quantum_optimizer = QuantumOptimizer(self.config.quantum) if self.config.quantum.enabled else None
        self.genetic_evolver = GeneticPromptEvolver(self.config.genetic) if self.config.genetic.enabled else None
        self.swarm = SwarmIntelligence(self.config.swarm) if self.config.swarm.enabled else None
        self.rlhf = RLHFTrainer(self.config.rlhf) if self.config.rlhf.enabled else None
        
        # ユーザープロファイル
        self.profile = self._init_profile()
        
        # メトリクス
        self.metrics = {
            'queries': 0,
            'success': 0,
            'total_cost': 0,
            'total_tokens': 0,
            'cache_hits': 0,
            'quantum_optimizations': 0,
            'genetic_evolutions': 0,
            'swarm_optimizations': 0,
        }

  # -*- coding: utf-8 -*-
"""
メインLLMシステム - 戦略実行部分
"""

    async def _execute_quantum_strategy(
        self,
        query: str,
        model: str,
        intent: Intent,
        complexity: Complexity
    ) -> Response:
        """量子インスパイア戦略"""
        logger.info("🔮 Executing quantum-inspired optimization")
        self.metrics['quantum_optimizations'] += 1
        
        # パラメータ空間を量子最適化
        def objective(params):
            temp = params[0]
            score = 1.0 - abs(temp - 0.7)
            return score
        
        optimized_params, _ = self.quantum_optimizer.optimize_parameters(objective)
        
        temperature = float(optimized_params[0])
        system_prompt = self.query_processor.build_system_prompt(
            query, intent, complexity, Strategy.QUANTUM, self.knowledge_graph
        )
        
        start_time = time.time()
        api_response = await self._call_api(
            model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature,
            self.config.max_tokens
        )
        latency = (time.time() - start_time) * 1000
        
        response = self._build_response(api_response, model, Strategy.QUANTUM, latency)
        response.quantum_optimized = True
        
        return response
    
    async def _execute_genetic_strategy(
        self,
        query: str,
        model: str,
        intent: Intent,
        complexity: Complexity
    ) -> Response:
        """遺伝的進化戦略"""
        logger.info("🧬 Executing genetic evolution")
        self.metrics['genetic_evolutions'] += 1
        
        # プロンプトを進化させる
        def fitness_func(prompt):
            return prompt.success_rate * 0.5 + prompt.avg_quality * 0.5
        
        for _ in range(3):
            best_prompt = self.genetic_evolver.evolve(fitness_func)
        
        system_prompt = self.query_processor.build_system_prompt(
            query, intent, complexity, Strategy.GENETIC, self.knowledge_graph
        )
        enhanced_query = f"{best_prompt.template}\n\n{query}"
        
        start_time = time.time()
        api_response = await self._call_api(
            model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": enhanced_query}
            ],
            0.7,
            self.config.max_tokens
        )
        latency = (time.time() - start_time) * 1000
        
        response = self._build_response(api_response, model, Strategy.GENETIC, latency)
        response.genetic_fitness = best_prompt.fitness
        
        return response
    
    async def _execute_swarm_strategy(
        self,
        query: str,
        model: str,
        intent: Intent,
        complexity: Complexity
    ) -> Response:
        """群知能戦略"""
        logger.info("🌊 Executing swarm intelligence")
        self.metrics['swarm_optimizations'] += 1
        
        from core.enums import PersonaType
        
        # 各ペルソナからの応答を収集
        personas = [PersonaType.OPTIMIST, PersonaType.PESSIMIST, PersonaType.PRAGMATIST]
        responses = []
        
        for persona in personas:
            persona_prompt = f"As a {persona.value}, answer: {query}"
            system_prompt = self.query_processor.build_system_prompt(
                query, intent, complexity, Strategy.SWARM, self.knowledge_graph
            )
            
            try:
                api_response = await self._call_api(
                    model,
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": persona_prompt}
                    ],
                    0.7,
                    self.config.max_tokens // 2
                )
                
                text = api_response.choices[0].message.content or ""
                responses.append({
                    'persona': persona.value,
                    'text': text,
                    'confidence': 0.7 + np.random.random() * 0.2
                })
            except Exception as e:
                logger.warning(f"Swarm agent {persona.value} failed: {e}")
        
        if not responses:
            return await self._execute_direct(query, model, intent, complexity)
        
        # コンセンサス合成
        synthesis_prompt = f"Synthesize these perspectives:\n\n"
        for resp in responses:
            synthesis_prompt += f"{resp['persona']}: {resp['text'][:200]}...\n\n"
        synthesis_prompt += f"\nProvide a balanced synthesis answering: {query}"
        
        start_time = time.time()
        final_response = await self._call_api(
            model,
            [
                {"role": "system", "content": "Synthesize multiple perspectives."},
                {"role": "user", "content": synthesis_prompt}
            ],
            0.7,
            self.config.max_tokens
        )
        latency = (time.time() - start_time) * 1000
        
        response = self._build_response(final_response, model, Strategy.SWARM, latency)
        response.personas_involved = [r['persona'] for r in responses]
        response.swarm_consensus = statistics.mean(r['confidence'] for r in responses)
        
        return response
    
    async def _execute_direct(
        self,
        query: str,
        model: str,
        intent: Intent,
        complexity: Complexity
    ) -> Response:
        """直接実行"""
        system_prompt = self.query_processor.build_system_prompt(
            query, intent, complexity, Strategy.DIRECT, self.knowledge_graph
        )
        
        start_time = time.time()
        api_response = await self._call_api(
            model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            self.config.temperature,
            self.config.max_tokens
        )
        latency = (time.time() - start_time) * 1000
        
        return self._build_response(api_response, model, Strategy.DIRECT, latency)
    
    def _build_response(
        self,
        api_response,
        model: str,
        strategy: Strategy,
        latency: float
    ) -> Response:
        """応答オブジェクト構築"""
        choice = api_response.choices[0]
        text = choice.message.content or ""
        
        usage = api_response.usage
        cost = CostCalculator.calculate_cost(
            model, usage.prompt_tokens, usage.completion_tokens
        )
        
        # 品質スコア計算
        coherence = min(1.0, len(text.split('.')) / 10)
        relevance = 0.8
        completeness = min(1.0, len(text) / 500)
        factuality = 0.85
        novelty = 0.7 if strategy in [Strategy.QUANTUM, Strategy.GENETIC] else 0.5
        
        # 信頼度計算
        base_confidence = 0.9 if choice.finish_reason == "stop" else 0.75
        uncertainty = sum(
            0.1 for phrase in ['maybe', 'perhaps', 'possibly']
            if phrase in text.lower()
        )
        confidence = max(0, min(1, base_confidence - uncertainty * 0.1))
        
        return Response(
            text=text,
            confidence=confidence,
            tokens=usage.total_tokens,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency=latency,
            cost=cost,
            model=model,
            finish_reason=choice.finish_reason,
            strategy=strategy,
            uncertainty=min(1.0, uncertainty),
            coherence_score=coherence,
            relevance_score=relevance,
            completeness_score=completeness,
            factuality_score=factuality,
            novelty_score=novelty
        )
    
    def _update_knowledge_graph(self, query: str, response: str):
        """知識グラフを更新"""
        # エンティティ抽出
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response)
        
        for entity in set(entities[:5]):
            node_id = hashlib.md5(entity.encode()).hexdigest()[:8]
            node = KnowledgeNode(
                id=node_id,
                name=entity,
                type='entity',
                properties={'source': 'response'}
            )
            self.knowledge_graph.add_node(node)
        
        # 関係抽出
        for i in range(len(entities) - 1):
            source_id = hashlib.md5(entities[i].encode()).hexdigest()[:8]
            target_id = hashlib.md5(entities[i + 1].encode()).hexdigest()[:8]
            
            if (source_id in self.knowledge_graph.nodes and 
                target_id in self.knowledge_graph.nodes):
                edge = KnowledgeEdge(
                    source=source_id,
                    target=target_id,
                    relation='mentioned_with',
                    weight=0.5
                )
                self.knowledge_graph.add_edge(edge)

      # -*- coding: utf-8 -*-
"""
メインLLMシステム - 状態管理・統計部分
"""

    def add_feedback(
        self,
        query: str,
        response: str,
        rating: int,
        response_obj: Optional[Response] = None
    ):
        """
        フィードバック追加
        
        Args:
            query: クエリ
            response: 応答テキスト
            rating: 評価（-2〜+2）
            response_obj: 応答オブジェクト
        """
        self.profile['interaction_count'] += 1
        self.profile['feedback_history'].append({
            'query': query[:100],
            'response': response[:100],
            'rating': rating,
            'timestamp': datetime.now().isoformat()
        })
        
        # トピック更新
        words = re.findall(r'\b\w{4,}\b', query.lower())
        for word in words:
            self.profile['topics'][word] += rating
            if rating > 0:
                self.profile['expertise'][word] = min(
                    1.0, self.profile['expertise'][word] + 0.1
                )
        
        # 戦略好み更新
        if response_obj and response_obj.strategy:
            current = self.profile['strategy_preference'][response_obj.strategy.value]
            self.profile['strategy_preference'][response_obj.strategy.value] = current + rating * 0.1
        
        # 遺伝的プロンプト更新
        if self.genetic_evolver and response_obj:
            for prompt in self.genetic_evolver.population:
                if prompt.usage_count > 0:
                    if rating > 0:
                        prompt.success_count += 1
                    prompt.avg_quality = (
                        prompt.avg_quality * (prompt.usage_count - 1) + abs(rating)
                    ) / prompt.usage_count
                    prompt.fitness = prompt.success_rate * 0.5 + prompt.avg_quality * 0.5
        
        logger.info(
            f"🎯 Feedback: {rating:+d} | "
            f"Strategy: {response_obj.strategy if response_obj else 'N/A'}"
        )
    
    def get_stats(self) -> Dict:
        """統計情報取得"""
        stats = {
            'system': {
                'queries': self.metrics['queries'],
                'success_rate': f"{self.metrics['success'] / max(self.metrics['queries'], 1):.1%}",
                'cache_hit_rate': f"{self.metrics['cache_hits'] / max(self.metrics['queries'], 1):.1%}",
                'total_cost': f"${self.metrics['total_cost']:.6f}",
                'avg_cost': f"${self.metrics['total_cost'] / max(self.metrics['queries'], 1):.6f}"
            },
            'advanced': {
                'quantum_optimizations': self.metrics['quantum_optimizations'],
                'genetic_evolutions': self.metrics['genetic_evolutions'],
                'swarm_optimizations': self.metrics['swarm_optimizations'],
            },
            'profile': {
                'interactions': self.profile['interaction_count'],
                'top_topics': sorted(
                    self.profile['topics'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
                'expertise_areas': len([
                    e for e in self.profile['expertise'].values() if e > 0.5
                ])
            }
        }
        
        # 知識グラフ統計
        if self.knowledge_graph:
            stats['knowledge_graph'] = {
                'nodes': len(self.knowledge_graph.nodes),
                'edges': len(self.knowledge_graph.edges),
                'communities': len(self.knowledge_graph.communities)
            }
        
        # 遺伝的進化統計
        if self.genetic_evolver:
            best_prompts = self.genetic_evolver.get_best_prompts(3)
            stats['genetic'] = {
                'generation': self.genetic_evolver.generation,
                'population_size': len(self.genetic_evolver.population),
                'best_fitness': best_prompts[0].fitness if best_prompts else 0
            }
        
        # RLHF統計
        if self.rlhf:
            stats['rlhf'] = {
                'states_explored': len(self.rlhf.state_visits),
                'total_updates': sum(self.rlhf.state_visits.values()),
                'avg_reward': statistics.mean(self.rlhf.reward_history) if self.rlhf.reward_history else 0
            }
        
        return stats
    
    def save_state(self, filepath: str = 'quantum_llm_state.json'):
        """
        状態保存
        
        Args:
            filepath: 保存先ファイルパス
        """
        try:
            state = {
                'profile': {
                    'topics': dict(self.profile['topics']),
                    'expertise': dict(self.profile['expertise']),
                    'strategy_preference': dict(self.profile['strategy_preference']),
                    'interaction_count': self.profile['interaction_count'],
                    'feedback_history': self.profile['feedback_history']
                },
                'metrics': self.metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 State saved: {filepath}")
        except Exception as e:
            logger.error(f"❌ Save failed: {e}")
    
    def load_state(self, filepath: str = 'quantum_llm_state.json'):
        """
        状態読み込み
        
        Args:
            filepath: 読み込み元ファイルパス
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            profile_data = state.get('profile', {})
            self.profile['topics'] = defaultdict(int, profile_data.get('topics', {}))
            self.profile['expertise'] = defaultdict(float, profile_data.get('expertise', {}))
            self.profile['strategy_preference'] = defaultdict(
                float, profile_data.get('strategy_preference', {})
            )
            self.profile['interaction_count'] = profile_data.get('interaction_count', 0)
            self.profile['feedback_history'] = profile_data.get('feedback_history', [])
            
            self.metrics.update(state.get('metrics', {}))
            
            logger.info(f"📂 State loaded: {filepath}")
        except FileNotFoundError:
            logger.info("ℹ️  No saved state found")
        except Exception as e:
            logger.error(f"❌ Load failed: {e}")
