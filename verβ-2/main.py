# -*- coding: utf-8 -*-
"""
Quantum-Enhanced Self-Evolving Enterprise LLM System v3.5γ ULTIMATE
メインエントリーポイント

使い方:
export GROQ_API_KEY='your_key'
pip install groq numpy scipy
python main.py
"""

import sys
import logging
import argparse

from core.config import SystemConfig, QuantumConfig, GeneticConfig, SwarmConfig, RLHFConfig
from core.llm_system import QuantumLLM
from ui.chat_interface import QuantumChat
from utils.logger import logger


def main():
    """エントリーポイント"""
    parser = argparse.ArgumentParser(
        description='Quantum-Enhanced Self-Evolving LLM System v3.5γ ULTIMATE'
    )
    parser.add_argument('--model', default='llama-3.1-8b-instant', help='Base model')
    parser.add_argument('--no-quantum', action='store_true', help='Disable quantum')
    parser.add_argument('--no-genetic', action='store_true', help='Disable genetic')
    parser.add_argument('--no-swarm', action='store_true', help='Disable swarm')
    parser.add_argument('--no-rlhf', action='store_true', help='Disable RLHF')
    parser.add_argument('--query', type=str, help='Single query mode')
    parser.add_argument('--load', type=str, help='Load state')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    
    args = parser.parse_args()
    
    if args.debug:
        logger.logger.setLevel(logging.DEBUG)
    
    # 設定
    config = SystemConfig(
        model=args.model,
        quantum=QuantumConfig(enabled=not args.no_quantum),
        genetic=GeneticConfig(enabled=not args.no_genetic),
        swarm=SwarmConfig(enabled=not args.no_swarm),
        rlhf=RLHFConfig(enabled=not args.no_rlhf)
    )
    
    try:
        # システム初期化
        llm = QuantumLLM(config=config)
        
        # 状態読み込み
        if args.load:
            llm.load_state(args.load)
        
        # シングルクエリモード
        if args.query:
            response = llm.query(args.query)
            print(response.text)
            print(f"\n📊 Metadata:")
            print(f"   Quality: {response.quality_score:.2f}")
            print(f"   Strategy: {response.strategy.value if response.strategy else 'N/A'}")
            print(f"   Cost: ${response.cost:.6f}")
            return
        
        # インタラクティブモード
        chat = QuantumChat(llm)
        chat.run()
        
        # 終了時保存
        print("\n💾 Saving session...")
        llm.save_state()
        
        stats = llm.get_stats()
        print("\n📊 Session Summary:")
        print(f"   Queries: {stats['system']['queries']}")
        print(f"   Success Rate: {stats['system']['success_rate']}")
        print(f"   Total Cost: {stats['system']['total_cost']}")
    
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("Please set GROQ_API_KEY environment variable")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.error(f"Fatal: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
