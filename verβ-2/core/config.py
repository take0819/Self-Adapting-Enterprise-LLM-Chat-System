# -*- coding: utf-8 -*-
"""
設定クラス
システム全体の設定を管理
"""

from dataclasses import dataclass, field


@dataclass
class QuantumConfig:
    """量子インスパイア設定"""
    enabled: bool = True
    num_qubits: int = 8
    iterations: int = 10
    optimization_depth: int = 3


@dataclass
class GeneticConfig:
    """遺伝的アルゴリズム設定"""
    enabled: bool = True
    population_size: int = 20
    mutation_rate: float = 0.15
    crossover_rate: float = 0.7
    elite_ratio: float = 0.2
    generations: int = 5


@dataclass
class SwarmConfig:
    """群知能設定"""
    enabled: bool = True
    num_agents: int = 5
    inertia_weight: float = 0.7
    cognitive_weight: float = 1.5
    social_weight: float = 1.5
    max_iterations: int = 10


@dataclass
class RLHFConfig:
    """RLHF設定"""
    enabled: bool = True
    learning_rate: float = 0.01
    discount_factor: float = 0.95
    exploration_rate: float = 0.1
    reward_shaping: bool = True


@dataclass
class SystemConfig:
    """システム設定"""
    # 基本設定
    model: str = "llama-3.1-8b-instant"
    max_tokens: int = 4000
    temperature: float = 0.7
    
    # キャッシュ・DB
    vec_db: bool = True
    vec_dim: int = 384
    cache_ttl: int = 3600
    similarity_threshold: float = 0.92
    
    # リトライ
    max_retries: int = 3
    retry_delay: float = 1.0
    max_query_length: int = 15000
    
    # コア機能
    adaptive: bool = True
    multi_armed_bandit: bool = True
    long_term_memory: bool = True
    knowledge_graph: bool = True
    chain_of_thought: bool = True
    self_reflection: bool = True
    ensemble_learning: bool = True
    metacognition: bool = True
    
    # 高度な機能
    tree_of_thoughts: bool = True
    debate_mode: bool = True
    critic_system: bool = True
    confidence_calibration: bool = True
    active_learning: bool = True
    curriculum_learning: bool = True
    
    # 超高度な機能
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    genetic: GeneticConfig = field(default_factory=GeneticConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)
    rlhf: RLHFConfig = field(default_factory=RLHFConfig)
    
    # 究極の機能
    adversarial_testing: bool = True
    causal_reasoning: bool = True
    creative_synthesis: bool = True
    predictive_modeling: bool = True
    verification_system: bool = True
    multi_model_competition: bool = True
    scientific_method: bool = True
    blockchain_verify: bool = False  # オプション
    real_time_learning: bool = True
    meta_learning: bool = True
