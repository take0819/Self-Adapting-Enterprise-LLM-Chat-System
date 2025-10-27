# -*- coding: utf-8 -*-
"""
列挙型定義
システム全体で使用する列挙型を定義
"""

from enum import Enum


class Intent(str, Enum):
    """ユーザーの意図"""
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
    DEBUGGING = "debugging"
    OPTIMIZATION = "optimization"


class Complexity(str, Enum):
    """クエリの複雑度"""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EXPERT = "expert"
    RESEARCH = "research"
    FRONTIER = "frontier"


class Strategy(str, Enum):
    """実行戦略"""
    DIRECT = "direct"
    COT = "chain_of_thought"
    REFLECTION = "reflection"
    ENSEMBLE = "ensemble"
    ITERATIVE = "iterative"
    TREE_SEARCH = "tree_search"
    DEBATE = "debate"
    SYNTHESIS = "synthesis"
    SWARM = "swarm_intelligence"
    GENETIC = "genetic_evolution"
    QUANTUM = "quantum_inspired"


class PersonaType(str, Enum):
    """エージェントのペルソナ"""
    OPTIMIST = "optimist"
    PESSIMIST = "pessimist"
    PRAGMATIST = "pragmatist"
    INNOVATOR = "innovator"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"


class ReasoningType(str, Enum):
    """推論のタイプ"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    COUNTERFACTUAL = "counterfactual"
    BAYESIAN = "bayesian"
    DIALECTICAL = "dialectical"
    SYSTEMS_THINKING = "systems_thinking"


class VerificationMethod(str, Enum):
    """検証方法"""
    CROSS_REFERENCE = "cross_reference"
    LOGICAL_CONSISTENCY = "logical_consistency"
    FACT_CHECK = "fact_check"
    PEER_REVIEW = "peer_review"
    ADVERSARIAL_TEST = "adversarial_test"
    BLOCKCHAIN_VERIFY = "blockchain_verify"
