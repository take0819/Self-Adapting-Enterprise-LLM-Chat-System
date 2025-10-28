# -*- coding: utf-8 -*-
"""
コマンドハンドラ
チャットコマンドの処理
"""

import json
import statistics
from datetime import datetime


class CommandHandlers:
    """コマンド処理クラス"""
    
    def __init__(self, chat_interface):
        self.chat = chat_interface
        self.llm = chat_interface.llm
    
    def handle(self, command: str) -> bool:
        """
        コマンド処理のメインハンドラ
        
        Args:
            command: コマンド文字列
        
        Returns:
            継続するかどうか（False=終了）
        """
        parts = command.strip().split()
        cmd = parts[0].lower()
        
        # ========== 基本コマンド ==========
        if cmd == '/exit':
            print("👋 Goodbye!")
            return False
        
        elif cmd == '/help':
            self.chat.print_welcome()
        
        elif cmd == '/stats':
            self.chat.print_stats()
        
        # ========== データ管理 ==========
        elif cmd == '/save':
            filepath = parts[1] if len(parts) > 1 else 'quantum_llm_state.json'
            self.llm.save_state(filepath)
        
        elif cmd == '/load':
            filepath = parts[1] if len(parts) > 1 else 'quantum_llm_state.json'
            self.llm.load_state(filepath)
        
        elif cmd == '/clear':
            self.chat.history.clear()
            self.llm.context_window.clear()
            if self.llm.vector_db:
                self.llm.vector_db.clear()
            print("🗑️  All history cleared")
        
        # ========== 評価・学習 ==========
        elif cmd == '/feedback':
            self._handle_feedback(parts)
        
        elif cmd == '/rate':
            self._handle_rate(parts)
        
        # ========== 高度な機能 ==========
        elif cmd == '/quantum':
            self._show_quantum_info()
        
        elif cmd == '/genetic':
            self._show_genetic_info()
        
        elif cmd == '/swarm':
            self._show_swarm_info()
        
        elif cmd == '/rlhf':
            self._show_rlhf_info()
        
        elif cmd == '/kg':
            self._show_knowledge_graph()
        
        # ========== 表示・設定 ==========
        elif cmd == '/history':
            self._show_history()
        
        elif cmd == '/profile':
            self._show_profile()
        
        elif cmd == '/config':
            self._show_config()
        
        else:
            print(f"❌ Unknown command: {cmd}")
            print("💡 Type /help for available commands")
        
        return True
    
    def _handle_feedback(self, parts):
        """フィードバック処理"""
        if not self.chat.history:
            print("❌ No previous response to rate")
            return
        
        try:
            rating = int(parts[1]) if len(parts) > 1 else 0
            if rating < -2 or rating > 2:
                print("❌ Rating must be between -2 and +2")
                return
            
            last_query, last_response = self.chat.history[-1]
            self.llm.add_feedback(last_query, last_response.text, rating, last_response)
            print(f"✅ Feedback recorded: {rating:+d}")
        except ValueError:
            print("❌ Invalid rating")
    
    def _handle_rate(self, parts):
        """5段階評価処理"""
        if not self.chat.history:
            print("❌ No previous response to rate")
            return
        
        try:
            rating = int(parts[1]) if len(parts) > 1 else 3
            if rating < 1 or rating > 5:
                print("❌ Rating must be between 1 and 5")
                return
            
            # 5段階を-2~+2に変換
            converted = rating - 3
            last_query, last_response = self.chat.history[-1]
            self.llm.add_feedback(last_query, last_response.text, converted, last_response)
            print(f"⭐ Rated: {rating}/5 stars")
        except ValueError:
            print("❌ Invalid rating")
    
    def _show_quantum_info(self):
        """量子最適化詳細"""
        if not self.llm.quantum_optimizer:
            print("❌ Quantum optimization disabled")
            return
        
        print("\n" + "=" * 80)
        print("🔮 Quantum Optimization Details")
        print("=" * 80)
        print(f"\n⚛️  Configuration:")
        print(f"   Qubits: {self.llm.quantum_optimizer.num_qubits}")
        print(f"   Iterations: {self.llm.quantum_optimizer.config.iterations}")
        print(f"   Optimization Depth: {self.llm.quantum_optimizer.config.optimization_depth}")
        print(f"\n📊 Performance:")
        print(f"   Total Optimizations: {self.llm.metrics['quantum_optimizations']}")
        print(f"\n💡 When to Use:")
        print(f"   • Frontier-level complexity questions")
        print(f"   • Multi-dimensional optimization problems")
        print("=" * 80 + "\n")
    
    def _show_genetic_info(self):
        """遺伝的進化詳細"""
        if not self.llm.genetic_evolver:
            print("❌ Genetic evolution disabled")
            return
        
        print("\n" + "=" * 80)
        print("🧬 Genetic Evolution Details")
        print("=" * 80)
        print(f"\n📈 Population Status:")
        print(f"   Generation: {self.llm.genetic_evolver.generation}")
        print(f"   Population Size: {len(self.llm.genetic_evolver.population)}")
        print(f"   Mutation Rate: {self.llm.config.genetic.mutation_rate:.1%}")
        print(f"   Crossover Rate: {self.llm.config.genetic.crossover_rate:.1%}")
        
        best_prompts = self.llm.genetic_evolver.get_best_prompts(5)
        if best_prompts:
            print(f"\n🏆 Top 5 Evolved Prompts:")
            for i, prompt in enumerate(best_prompts, 1):
                fitness_bar = "█" * int(prompt.fitness * 20) + "░" * (20 - int(prompt.fitness * 20))
                print(f"\n   {i}. Fitness: [{fitness_bar}] {prompt.fitness:.3f}")
                print(f"      Generation: {prompt.generation} | Mutations: {prompt.mutations}")
                print(f"      Template: {prompt.template[:60]}...")
        
        print("=" * 80 + "\n")
    
    def _show_swarm_info(self):
        """群知能詳細"""
        if not self.llm.swarm:
            print("❌ Swarm intelligence disabled")
            return
        
        print("\n" + "=" * 80)
        print("🌊 Swarm Intelligence Details")
        print("=" * 80)
        print(f"\n🐝 Swarm Configuration:")
        print(f"   Agents: {len(self.llm.swarm.agents)}")
        print(f"   Inertia Weight: {self.llm.config.swarm.inertia_weight}")
        print(f"   Cognitive Weight: {self.llm.config.swarm.cognitive_weight}")
        print(f"   Social Weight: {self.llm.config.swarm.social_weight}")
        
        if self.llm.swarm.agents:
            print(f"\n🎭 Agent Personas:")
            for agent in self.llm.swarm.agents:
                print(f"   • {agent.persona.value}: Fitness {agent.best_fitness:.3f}")
        
        print(f"\n📊 Performance:")
        print(f"   Global Best Fitness: {self.llm.swarm.global_best_fitness:.3f}")
        print(f"   Total Optimizations: {self.llm.metrics['swarm_optimizations']}")
        
        print("=" * 80 + "\n")
    
    def _show_rlhf_info(self):
        """RLHF詳細"""
        if not self.llm.rlhf:
            print("❌ RLHF disabled")
            return
        
        print("\n" + "=" * 80)
        print("🎯 Reinforcement Learning Details")
        print("=" * 80)
        print(f"\n🧠 Learning Status:")
        print(f"   States Explored: {len(self.llm.rlhf.state_visits)}")
        print(f"   Q-Table Size: {len(self.llm.rlhf.q_table)}")
        print(f"   Total Updates: {sum(self.llm.rlhf.state_visits.values())}")
        print(f"   Learning Rate: {self.llm.config.rlhf.learning_rate}")
        print(f"   Exploration Rate: {self.llm.config.rlhf.exploration_rate:.1%}")
        
        if self.llm.rlhf.reward_history:
            avg_reward = statistics.mean(self.llm.rlhf.reward_history)
            recent_reward = statistics.mean(
                self.llm.rlhf.reward_history[-10:]
            ) if len(self.llm.rlhf.reward_history) >= 10 else avg_reward
            print(f"\n📈 Rewards:")
            print(f"   Average Reward: {avg_reward:.3f}")
            print(f"   Recent Reward (last 10): {recent_reward:.3f}")
            trend = '📈 Improving' if recent_reward > avg_reward else '📉 Declining' if recent_reward < avg_reward else '➡️ Stable'
            print(f"   Trend: {trend}")
        
        print("=" * 80 + "\n")
    
    def _show_knowledge_graph(self):
        """知識グラフ表示"""
        if not self.llm.knowledge_graph:
            print("❌ Knowledge graph disabled")
            return
        
        print("\n" + "=" * 80)
        print("🧩 Knowledge Graph Status")
        print("=" * 80)
        print(f"\n📊 Statistics:")
        print(f"   Nodes: {len(self.llm.knowledge_graph.nodes)}")
        print(f"   Edges: {len(self.llm.knowledge_graph.edges)}")
        
        central = self.llm.knowledge_graph.get_central_nodes(5)
        if central:
            print(f"\n🎯 Central Nodes (by degree):")
            for node_id, degree in central:
                node = self.llm.knowledge_graph.nodes[node_id]
                print(f"   • {node.name} (degree: {degree}, type: {node.type})")
        
        print("=" * 80 + "\n")
    
    def _show_history(self):
        """会話履歴表示"""
        print("\n" + "=" * 80)
        print("📜 Conversation History")
        print("=" * 80)
        
        if not self.chat.history:
            print("\nNo conversation history yet.")
            print("=" * 80 + "\n")
            return
        
        recent = self.chat.history[-10:]
        for i, (query, response) in enumerate(recent, 1):
            print(f"\n{i}. Q: {query[:60]}...")
            print(f"   A: {response.text[:60]}...")
            print(f"   Strategy: {response.strategy.value if response.strategy else 'N/A'} | Quality: {response.quality_score:.2f}")
        
        print(f"\n📊 Total Conversations: {len(self.chat.history)}")
        print("=" * 80 + "\n")
    
    def _show_profile(self):
        """プロファイル表示"""
        print("\n" + "=" * 80)
        print("👤 User Profile")
        print("=" * 80)
        
        profile = self.llm.profile
        print(f"\n📊 Activity:")
        print(f"   Total Interactions: {profile['interaction_count']}")
        print(f"   Feedback Given: {len(profile.get('feedback_history', []))}")
        
        # トップトピック
        topics = sorted(profile['topics'].items(), key=lambda x: x[1], reverse=True)[:10]
        if topics:
            print(f"\n📚 Top Topics:")
            for topic, score in topics:
                print(f"   • {topic}: {score}")
        
        # 専門知識
        expertise = [(k, v) for k, v in profile['expertise'].items() if v > 0.3]
        if expertise:
            expertise.sort(key=lambda x: x[1], reverse=True)
            print(f"\n🎓 Expertise Areas:")
            for topic, level in expertise[:10]:
                bar = "█" * int(level * 20) + "░" * (20 - int(level * 20))
                print(f"   {topic:20s} [{bar}] {level:.0%}")
        
        print("=" * 80 + "\n")
    
    def _show_config(self):
        """設定表示"""
        print("\n" + "=" * 80)
        print("⚙️  System Configuration")
        print("=" * 80)
        
        config = self.llm.config
        print(f"\n🔧 Basic Settings:")
        print(f"   Model: {config.model}")
        print(f"   Max Tokens: {config.max_tokens}")
        print(f"   Temperature: {config.temperature}")
        print(f"   Similarity Threshold: {config.similarity_threshold}")
        
        print(f"\n🚀 Features:")
        print(f"   Adaptive: {'✅' if config.adaptive else '❌'}")
        print(f"   Vector DB: {'✅' if config.vec_db else '❌'}")
        print(f"   Knowledge Graph: {'✅' if config.knowledge_graph else '❌'}")
        print(f"   Chain of Thought: {'✅' if config.chain_of_thought else '❌'}")
        print(f"   Quantum Optimization: {'✅' if config.quantum.enabled else '❌'}")
        print(f"   Genetic Evolution: {'✅' if config.genetic.enabled else '❌'}")
        print(f"   Swarm Intelligence: {'✅' if config.swarm.enabled else '❌'}")
        print(f"   RLHF: {'✅' if config.rlhf.enabled else '❌'}")
        
        print("=" * 80 + "\n")
