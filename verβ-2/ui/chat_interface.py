# -*- coding: utf-8 -*-
"""
チャットインターフェース
ユーザーとの対話を管理
"""

import uuid
from typing import List, Tuple

from core.data_models import Response
from core.enums import Strategy
from utils.logger import logger


class QuantumChat:
    """量子インスパイアチャットインターフェース"""
    
    def __init__(self, llm):
        self.llm = llm
        self.history: List[Tuple[str, Response]] = []
        self.session_id = str(uuid.uuid4())[:8]
        
        # コマンドハンドラを別ファイルからインポート
        from ui.command_handlers import CommandHandlers
        self.command_handlers = CommandHandlers(self)
    
    def print_welcome(self):
        """ウェルカムメッセージ"""
        print("\n" + "=" * 80)
        print("🔮 Quantum-Enhanced Self-Evolving LLM System v3.5γ ULTIMATE")
        print("=" * 80)
        print("\n✨ 革新的機能:")
        print("  🔮 Quantum-Inspired Optimization")
        print("  🧬 Genetic Algorithm for Prompt Evolution")
        print("  🌊 Swarm Intelligence Multi-Agent System")
        print("  🎯 Reinforcement Learning from Human Feedback")
        print("\n📋 基本コマンド:")
        print("  /help       - 全コマンド一覧")
        print("  /stats      - 詳細統計情報")
        print("  /exit       - 終了")
        print("\n💾 データ管理:")
        print("  /save [file] - 状態保存")
        print("  /load [file] - 状態読み込み")
        print("  /clear       - 履歴クリア")
        print("\n🎯 評価・学習:")
        print("  /feedback <rating> - 直前の回答を評価 (-2 to +2)")
        print("  /rate <1-5>        - 5段階評価")
        print("\n🔬 高度な機能:")
        print("  /quantum    - 量子最適化詳細")
        print("  /genetic    - 遺伝的進化状況")
        print("  /swarm      - 群知能ステータス")
        print("  /rlhf       - 強化学習情報")
        print("  /kg         - 知識グラフ")
        print("\n🎨 表示・設定:")
        print("  /history    - 会話履歴")
        print("  /profile    - ユーザープロファイル")
        print("  /config     - 現在の設定")
        print("=" * 80 + "\n")
    
    def print_response(self, response: Response):
        """応答表示"""
        print(f"\n🤖 Assistant [{response.model.split('-')[-1]}]:")
        print("─" * 80)
        print(response.text)
        print("─" * 80)
        
        # メタデータ
        metadata = []
        
        if response.strategy:
            emoji = {
                Strategy.QUANTUM: "🔮",
                Strategy.GENETIC: "🧬",
                Strategy.SWARM: "🌊",
                Strategy.TREE_SEARCH: "🌳",
                Strategy.COT: "🤔",
                Strategy.DEBATE: "🗣️"
            }.get(response.strategy, "📋")
            metadata.append(f"{emoji}{response.strategy.value}")
        
        if response.complexity:
            metadata.append(f"⚙️{response.complexity.value}")
        
        metadata.append(f"⭐{response.quality_score:.2f}")
        metadata.append(f"✅{response.confidence:.2f}")
        metadata.append(f"💰${response.cost:.6f}")
        metadata.append(f"⏱️{response.latency:.0f}ms")
        
        if response.quantum_optimized:
            metadata.append("🔮Optimized")
        if response.genetic_fitness > 0:
            metadata.append(f"🧬Fit:{response.genetic_fitness:.2f}")
        if response.swarm_consensus > 0:
            metadata.append(f"🌊Consensus:{response.swarm_consensus:.2f}")
        if response.cached:
            metadata.append(f"💾Cache")
        
        print(" | ".join(metadata))
        
        # 追加情報
        if response.personas_involved:
            print(f"\n🎭 Personas: {', '.join(response.personas_involved)}")
        
        if response.reasoning_steps:
            print(f"\n🧠 Reasoning Steps: {len(response.reasoning_steps)} steps")
        
        print()
    
    def print_stats(self):
        """統計表示"""
        stats = self.llm.get_stats()
        
        print("\n" + "=" * 80)
        print("📊 System Statistics")
        print("=" * 80)
        
        # システム統計
        sys = stats['system']
        print(f"\n📈 System:")
        print(f"   Queries: {sys['queries']} | Success Rate: {sys['success_rate']}")
        print(f"   Cache Hit Rate: {sys['cache_hit_rate']}")
        print(f"   Total Cost: {sys['total_cost']} | Avg: {sys['avg_cost']}")
        
        # 高度な機能
        adv = stats['advanced']
        print(f"\n🚀 Advanced Features:")
        print(f"   🔮 Quantum Optimizations: {adv['quantum_optimizations']}")
        print(f"   🧬 Genetic Evolutions: {adv['genetic_evolutions']}")
        print(f"   🌊 Swarm Optimizations: {adv['swarm_optimizations']}")
        
        # プロファイル
        prof = stats['profile']
        print(f"\n👤 Profile:")
        print(f"   Interactions: {prof['interactions']}")
        print(f"   Expertise Areas: {prof['expertise_areas']}")
        if prof['top_topics']:
            print(f"   Top Topics: {', '.join([t[0] for t in prof['top_topics'][:3]])}")
        
        # 知識グラフ
        if 'knowledge_graph' in stats:
            kg = stats['knowledge_graph']
            print(f"\n🧩 Knowledge Graph:")
            print(f"   Nodes: {kg['nodes']} | Edges: {kg['edges']} | Communities: {kg['communities']}")
        
        # 遺伝的進化
        if 'genetic' in stats:
            gen = stats['genetic']
            print(f"\n🧬 Genetic Evolution:")
            print(f"   Generation: {gen['generation']} | Population: {gen['population_size']}")
            print(f"   Best Fitness: {gen['best_fitness']:.3f}")
        
        # RLHF
        if 'rlhf' in stats:
            rl = stats['rlhf']
            print(f"\n🎯 RLHF:")
            print(f"   States Explored: {rl['states_explored']}")
            print(f"   Total Updates: {rl['total_updates']}")
            print(f"   Avg Reward: {rl['avg_reward']:.3f}")
        
        print("=" * 80 + "\n")
    
    def handle_command(self, command: str) -> bool:
        """
        コマンド処理
        
        Args:
            command: コマンド文字列
        
        Returns:
            継続するかどうか（False=終了）
        """
        return self.command_handlers.handle(command)
    
    def run(self):
        """メインループ"""
        self.print_welcome()
        
        while True:
            try:
                query = input("👤 You: ").strip()
                
                if not query:
                    continue
                
                if query.startswith('/'):
                    if not self.handle_command(query):
                        break
                    continue
                
                print("\n⏳ Processing...")
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
