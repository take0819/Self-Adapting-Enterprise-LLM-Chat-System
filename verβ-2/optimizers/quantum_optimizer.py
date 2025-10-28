# -*- coding: utf-8 -*-
"""
量子インスパイア最適化
QAOA風アルゴリズムによるパラメータ最適化
"""

from typing import Callable, Tuple, Optional, List
import numpy as np

from core.config import QuantumConfig
from utils.logger import logger


class QuantumOptimizer:
    """量子インスパイア最適化器"""
    
    def __init__(self, config: QuantumConfig):
        self.config = config
        self.num_qubits = config.num_qubits
    
    def optimize_parameters(
        self,
        objective_function: Callable[[np.ndarray], float],
        bounds: Tuple[float, float] = (0, 1)
    ) -> Tuple[np.ndarray, float]:
        """
        QAOA風パラメータ最適化
        
        Args:
            objective_function: 目的関数
            bounds: パラメータの範囲
        
        Returns:
            (最適パラメータ, 最適値)
        """
        # 初期状態: 重ね合わせ（均等分布）
        best_params = np.random.uniform(bounds[0], bounds[1], self.num_qubits)
        best_value = objective_function(best_params)
        
        for iteration in range(self.config.iterations):
            # 量子ゲート風の操作
            # 1. 回転ゲート（探索）
            rotation_angle = np.pi * (1 - iteration / self.config.iterations)
            candidate = best_params + np.random.randn(self.num_qubits) * rotation_angle * 0.1
            candidate = np.clip(candidate, bounds[0], bounds[1])
            
            # 2. エンタングルメント（パラメータ間の相関）
            if self.num_qubits > 1:
                for i in range(self.num_qubits - 1):
                    if np.random.random() < 0.3:
                        coupling = (candidate[i] + candidate[i + 1]) / 2
                        candidate[i] = candidate[i + 1] = coupling
            
            # 3. 測定（評価）
            value = objective_function(candidate)
            
            # 4. 振幅増幅（良い解を強化）
            if value > best_value:
                best_params = candidate
                best_value = value
                logger.debug(f"🔮 Quantum iter {iteration}: improved to {value:.4f}")
        
        return best_params, best_value
    
    def quantum_annealing(
        self,
        energy_function: Callable[[np.ndarray], float],
        initial_state: np.ndarray,
        temperature_schedule: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        量子アニーリング風の最適化
        
        Args:
            energy_function: エネルギー関数（最小化）
            initial_state: 初期状態
            temperature_schedule: 温度スケジュール
        
        Returns:
            最適状態
        """
        if temperature_schedule is None:
            temperature_schedule = np.logspace(0, -2, self.config.iterations)
        
        current_state = initial_state.copy()
        current_energy = energy_function(current_state)
        
        for temp in temperature_schedule:
            # 隣接状態を生成
            neighbor = current_state + np.random.randn(len(current_state)) * temp
            neighbor = np.clip(neighbor, 0, 1)
            
            neighbor_energy = energy_function(neighbor)
            
            # メトロポリス基準
            delta_energy = neighbor_energy - current_energy
            if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / temp):
                current_state = neighbor
                current_energy = neighbor_energy
        
        return current_state
