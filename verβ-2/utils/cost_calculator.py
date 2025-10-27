# -*- coding: utf-8 -*-
"""
コスト計算ユーティリティ
APIの使用コストを計算
"""

from typing import Dict


class CostCalculator:
    """APIコスト計算機"""
    
    # モデルごとの料金表（$/1M tokens）
    PRICING = {
        'llama-3.1-8b-instant': {
            'input': 0.05 / 1e6,
            'output': 0.08 / 1e6
        },
        'llama-3.1-70b-versatile': {
            'input': 0.59 / 1e6,
            'output': 0.79 / 1e6
        },
        'llama-3.3-70b-versatile': {
            'input': 0.59 / 1e6,
            'output': 0.79 / 1e6
        },
    }
    
    # デフォルト料金
    DEFAULT_PRICING = {
        'input': 0.0001 / 1e6,
        'output': 0.0001 / 1e6
    }
    
    @classmethod
    def calculate_cost(
        cls,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """
        コストを計算
        
        Args:
            model: モデル名
            prompt_tokens: プロンプトトークン数
            completion_tokens: 生成トークン数
        
        Returns:
            コスト（USD）
        """
        pricing = cls.PRICING.get(model, cls.DEFAULT_PRICING)
        
        input_cost = prompt_tokens * pricing['input']
        output_cost = completion_tokens * pricing['output']
        
        return input_cost + output_cost
    
    @classmethod
    def get_model_info(cls, model: str) -> Dict[str, float]:
        """
        モデルの料金情報を取得
        
        Args:
            model: モデル名
        
        Returns:
            料金情報辞書
        """
        return cls.PRICING.get(model, cls.DEFAULT_PRICING)
    
    @classmethod
    def estimate_cost(
        cls,
        model: str,
        estimated_tokens: int,
        ratio: float = 0.5
    ) -> float:
        """
        おおよそのコストを見積もる
        
        Args:
            model: モデル名
            estimated_tokens: 推定総トークン数
            ratio: プロンプト/完成トークンの比率
        
        Returns:
            推定コスト（USD）
        """
        prompt_tokens = int(estimated_tokens * ratio)
        completion_tokens = int(estimated_tokens * (1 - ratio))
        
        return cls.calculate_cost(model, prompt_tokens, completion_tokens)
