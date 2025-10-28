# -*- coding: utf-8 -*-
"""
クエリ処理
クエリの分析と戦略選択
"""

from typing import Tuple

from core.enums import Intent, Complexity, Strategy
from core.config import SystemConfig


class QueryProcessor:
    """クエリ分析・処理"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
    
    def analyze_query(self, query: str) -> Tuple[Intent, Complexity]:
        """
        クエリを分析
        
        Args:
            query: 入力クエリ
        
        Returns:
            (意図, 複雑度)
        """
        q = query.lower()
        
        # 意図分析
        intent_patterns = {
            Intent.REASONING: ['why', 'because', 'reason', 'cause'],
            Intent.ANALYSIS: ['analyze', 'compare', 'evaluate', 'assess'],
            Intent.RESEARCH: ['research', 'investigate', 'study', 'explore'],
            Intent.PLANNING: ['plan', 'strategy', 'organize', 'schedule'],
            Intent.TECHNICAL: ['code', 'algorithm', 'implement', 'debug'],
            Intent.CREATIVE: ['create', 'write', 'design', 'imagine'],
            Intent.DEBUGGING: ['bug', 'error', 'fix', 'debug', 'issue'],
            Intent.OPTIMIZATION: ['optimize', 'improve', 'enhance', 'better']
        }
        
        intent = Intent.QUESTION
        max_matches = 0
        for int_type, patterns in intent_patterns.items():
            matches = sum(1 for p in patterns if p in q)
            if matches > max_matches:
                max_matches = matches
                intent = int_type
        
        # 複雑度分析
        complexity_score = 0
        complexity_score += len(query) // 100
        complexity_score += q.count('?')
        
        frontier_words = ['breakthrough', 'novel', 'unprecedented', 'cutting-edge']
        research_words = ['hypothesis', 'theory', 'prove', 'demonstrate']
        expert_words = ['advanced', 'sophisticated', 'complex', 'intricate']
        
        complexity_score += sum(5 for w in frontier_words if w in q)
        complexity_score += sum(4 for w in research_words if w in q)
        complexity_score += sum(3 for w in expert_words if w in q)
        
        if complexity_score < 2:
            complexity = Complexity.TRIVIAL
        elif complexity_score < 4:
            complexity = Complexity.SIMPLE
        elif complexity_score < 7:
            complexity = Complexity.MEDIUM
        elif complexity_score < 11:
            complexity = Complexity.COMPLEX
        elif complexity_score < 16:
            complexity = Complexity.EXPERT
        elif complexity_score < 20:
            complexity = Complexity.RESEARCH
        else:
            complexity = Complexity.FRONTIER
        
        return intent, complexity
    
    def select_strategy(
        self,
        intent: Intent,
        complexity: Complexity,
        rlhf_trainer=None
    ) -> Strategy:
        """
        実行戦略を選択
        
        Args:
            intent: クエリの意図
            complexity: クエリの複雑度
            rlhf_trainer: RLHF学習器（オプション）
        
        Returns:
            選択された戦略
        """
        # フロンティアレベル: 量子最適化
        if complexity == Complexity.FRONTIER and self.config.quantum.enabled:
            return Strategy.QUANTUM
        
        # 研究レベル: 遺伝的進化
        if complexity == Complexity.RESEARCH and self.config.genetic.enabled:
            return Strategy.GENETIC
        
        # 複雑な推論: 群知能
        if complexity in [Complexity.EXPERT, Complexity.COMPLEX] and self.config.swarm.enabled:
            return Strategy.SWARM
        
        # 分析・推論: Tree of Thoughts
        if intent in [Intent.ANALYSIS, Intent.REASONING] and self.config.tree_of_thoughts:
            return Strategy.TREE_SEARCH
        
        # 討論が有効な場合
        if complexity in [Complexity.EXPERT, Complexity.RESEARCH] and self.config.debate_mode:
            return Strategy.DEBATE
        
        # Chain of Thought
        if complexity >= Complexity.COMPLEX and self.config.chain_of_thought:
            return Strategy.COT
        
        # RLHF推奨がある場合
        if rlhf_trainer:
            state = rlhf_trainer.get_state(intent, complexity)
            available_strategies = [s.value for s in Strategy]
            recommended = rlhf_trainer.select_action(state, available_strategies)
            try:
                return Strategy(recommended)
            except:
                pass
        
        return Strategy.DIRECT
    
    def build_system_prompt(
        self,
        query: str,
        intent: Intent,
        complexity: Complexity,
        strategy: Strategy,
        knowledge_graph=None
    ) -> str:
        """
        システムプロンプトを構築
        
        Args:
            query: クエリ
            intent: 意図
            complexity: 複雑度
            strategy: 戦略
            knowledge_graph: 知識グラフ（オプション）
        
        Returns:
            システムプロンプト
        """
        base = "You are an advanced AI assistant with quantum-inspired reasoning capabilities."
        
        # 戦略別の指示
        strategy_instructions = {
            Strategy.QUANTUM: "Use multi-dimensional thinking. Explore superposition of possibilities.",
            Strategy.GENETIC: "Evolve your answer through iterative refinement.",
            Strategy.SWARM: "Consider diverse perspectives and find consensus.",
            Strategy.COT: "Think step by step. Show your reasoning process.",
            Strategy.DEBATE: "Present multiple viewpoints and synthesize them.",
            Strategy.TREE_SEARCH: "Explore different reasoning paths systematically."
        }
        
        strategy_text = strategy_instructions.get(strategy, "")
        
        # 複雑度別の調整
        if complexity in [Complexity.RESEARCH, Complexity.FRONTIER]:
            complexity_text = "Provide research-grade analysis with novel insights."
        elif complexity == Complexity.EXPERT:
            complexity_text = "Provide expert-level insights with technical depth."
        else:
            complexity_text = "Provide clear, well-structured answers."
        
        # 知識グラフからのコンテキスト
        kg_context = ""
        if knowledge_graph:
            subgraph = knowledge_graph.query_subgraph(query, depth=1)
            if subgraph['nodes']:
                node_names = [n.name for n in subgraph['nodes'][:3]]
                kg_context = f" Related concepts: {', '.join(node_names)}."
        
        prompt = f"{base} {strategy_text} {complexity_text}{kg_context}"
        
        return prompt.strip()
