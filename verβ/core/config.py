# core/config.py
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class Cfg:
model: str = os.getenv('LLM_MODEL', 'llama-3.1-8b-instant')
api_key: Optional[str] = os.getenv('LLM_API_KEY')
temperature: float = float(os.getenv('LLM_TEMPERATURE', '0.7'))
max_tokens: int = int(os.getenv('LLM_MAX_TOKENS', '4000'))


# vector DB / RAG 関連
vec_db: bool = os.getenv('LLM_USE_VDB', 'true').lower() in ('1','true','yes')
dim: int = int(os.getenv('LLM_VDB_DIM', '384'))


# 高度な機能フラグ
adapt: bool = True
mab: bool = True
memory: bool = True
cot: bool = True
reflection: bool = True
kg: bool = True
ab_test: bool = True
ensemble: bool = True
metacog: bool = True
thompson: bool = True


# Ultra-advanced features
multi_hop: bool = True
debate: bool = True
critic: bool = True
tree_of_thoughts: bool = True
rag: bool = True
confidence_calibration: bool = True
active_learning: bool = True
curriculum: bool = True


# Next-gen features
self_play: bool = True
constitutional_ai: bool = True
few_shot_learning: bool = True
meta_learning: bool = True
neuro_symbolic: bool = True
causal_reasoning: bool = True
world_model: bool = True
counterfactual: bool = True
analogy_engine: bool = True




def load_cfg_from_env() -> Cfg:
"""環境変数から Cfg を組み立てて返す"""
return Cfg()
