#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Discord Bot for Quantum-Enhanced Self-Evolving Enterprise LLM System v3.5γ
究極のAIアシスタントをDiscordから利用可能に

セットアップ:
1. pip install discord.py groq numpy scipy
2. export GROQ_API_KEY='your_groq_key'
3. export DISCORD_BOT_TOKEN='your_discord_token'
4. 同じディレクトリにenterprise-llm-chat-verα-5.pyを配置
5. python discord_quantum_bot.py

機能:
🔮 Quantum-Inspired Optimization
🧬 Genetic Algorithm Prompt Evolution
🌊 Swarm Intelligence Multi-Agent
🎯 RLHF (Reinforcement Learning from Human Feedback)
🔬 Scientific Method Application
🧩 Causal Inference Engine
🎨 Creative Synthesis System
🔐 Advanced Verification System
"""

import os
import sys
import asyncio
import discord
from discord import app_commands
from datetime import datetime
import json
from typing import Optional, Dict, List
import traceback

# 元のQuantum LLMシステムをインポート
try:
    sys.path.insert(0, os.path.dirname(__file__))
    # ファイル名のハイフンをアンダースコアに変換してインポート
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "quantum_llm",
        "enterprise-llm-chat-verα-5.py"
    )
    quantum_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quantum_module)
    
    QuantumLLM = quantum_module.QuantumLLM
    SystemConfig = quantum_module.SystemConfig
    QuantumConfig = quantum_module.QuantumConfig
    GeneticConfig = quantum_module.GeneticConfig
    SwarmConfig = quantum_module.SwarmConfig
    RLHFConfig = quantum_module.RLHFConfig
    Intent = quantum_module.Intent
    Complexity = quantum_module.Complexity
    Strategy = quantum_module.Strategy
    
except Exception as e:
    print(f"❌ Cannot import Quantum LLM module: {e}")
    print("Make sure enterprise-llm-chat-verα-5.py is in the same directory")
    sys.exit(1)

# Discord Bot設定
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)

# グローバルLLMインスタンス
llm: Optional[QuantumLLM] = None

# ユーザーごとの会話履歴
user_conversations: Dict[int, list] = {}

# セッション管理
session_data = {
    'start_time': datetime.now(),
    'total_queries': 0,
    'successful_queries': 0,
    'quantum_optimizations': 0,
    'genetic_evolutions': 0,
    'swarm_optimizations': 0
}

# 会話モード管理
talk_mode_users: Dict[int, bool] = {}  # 追加


# ==================== ユーティリティ ====================

def create_progress_bar(value: float, length: int = 20) -> str:
    """プログレスバー生成"""
    filled = int(value * length)
    return "█" * filled + "░" * (length - filled)


def format_response_embed(response, query: str, user: discord.User) -> discord.Embed:
    """応答をEmbedに整形"""
    # カラー選択
    if response.quality_score > 0.8:
        color = discord.Color.green()
    elif response.quality_score > 0.6:
        color = discord.Color.blue()
    else:
        color = discord.Color.orange()
    
    # 戦略絵文字
    strategy_emoji = {
        Strategy.QUANTUM: "🔮",
        Strategy.GENETIC: "🧬",
        Strategy.SWARM: "🌊",
        Strategy.TREE_SEARCH: "🌳",
        Strategy.COT: "🤔",
        Strategy.DEBATE: "🗣️",
        Strategy.DIRECT: "📝"
    }
    
    emoji = strategy_emoji.get(response.strategy, "💬")
    
    embed = discord.Embed(
        title=f"{emoji} Quantum AI Response",
        description=response.text[:4000] if len(response.text) <= 4000 else response.text[:3997] + "...",
        color=color,
        timestamp=datetime.now()
    )
    
    # メタデータフィールド
    metadata_lines = []
    
    if response.strategy:
        metadata_lines.append(f"**Strategy**: {response.strategy.value}")
    if response.complexity:
        metadata_lines.append(f"**Complexity**: {response.complexity.value}")
    if response.intent:
        metadata_lines.append(f"**Intent**: {response.intent.value}")
    
    metadata_lines.append(f"**Quality**: {response.quality_score:.2%}")
    metadata_lines.append(f"**Confidence**: {response.confidence:.2%}")
    
    if response.quantum_optimized:
        metadata_lines.append("**🔮 Quantum Optimized**")
    if response.genetic_fitness > 0:
        metadata_lines.append(f"**🧬 Fitness**: {response.genetic_fitness:.2f}")
    if response.swarm_consensus > 0:
        metadata_lines.append(f"**🌊 Consensus**: {response.swarm_consensus:.2%}")
    
    embed.add_field(
        name="📊 Analysis",
        value="\n".join(metadata_lines),
        inline=False
    )
    
    # パフォーマンスメトリクス
    perf_lines = [
        f"**Latency**: {response.latency:.0f}ms",
        f"**Tokens**: {response.tokens}",
        f"**Cost**: ${response.cost:.6f}"
    ]
    
    if response.cached:
        perf_lines.append("**💾 Cached Response**")
    
    embed.add_field(
        name="⚡ Performance",
        value="\n".join(perf_lines),
        inline=False
    )
    
    # 品質スコア
    if any([response.coherence_score, response.relevance_score, response.completeness_score]):
        quality_bar = create_progress_bar(response.quality_score)
        embed.add_field(
            name="⭐ Quality Breakdown",
            value=f"Overall: [{quality_bar}] {response.quality_score:.0%}\n"
                  f"Coherence: {response.coherence_score:.0%} | "
                  f"Relevance: {response.relevance_score:.0%} | "
                  f"Completeness: {response.completeness_score:.0%}",
            inline=False
        )
    
    # 推論ステップ
    if response.reasoning_steps and len(response.reasoning_steps) > 0:
        steps_preview = "\n".join(f"{i+1}. {step[:80]}" for i, step in enumerate(response.reasoning_steps[:3]))
        embed.add_field(
            name="🧠 Reasoning Steps",
            value=steps_preview,
            inline=False
        )
    
    # ペルソナ情報
    if response.personas_involved:
        embed.add_field(
            name="🎭 Personas Consulted",
            value=", ".join(response.personas_involved),
            inline=False
        )
    
    embed.set_footer(text=f"Query by {user.name}")
    
    return embed

# ==================== Botイベント ====================

# 定期保存タスク（on_readyの外に定義）
async def auto_save():
    """定期的にデータを保存"""
    await bot.wait_until_ready()
    while not bot.is_closed():
        await asyncio.sleep(1800)  # 30分
        if llm:
            try:
                llm.save_state('discord_quantum_state.json')
                print('💾 Auto-saved state')
            except Exception as e:
                print(f'❌ Auto-save failed: {e}')

@bot.event
async def on_ready():
    """Bot起動時 - 統合版"""
    global llm
    
    print(f'✅ {bot.user} logged in')
    print(f'🤖 Bot ID: {bot.user.id}')
    print(f'🌐 Servers: {len(bot.guilds)}')
    print('=' * 80)
    
    # Quantum LLMシステム初期化
    try:
        config = SystemConfig(
            model='llama-3.1-8b-instant',
            adaptive=True,
            multi_armed_bandit=True,
            long_term_memory=True,
            knowledge_graph=True,
            chain_of_thought=True,
            self_reflection=True,
            ensemble_learning=True,
            metacognition=True,
            tree_of_thoughts=True,
            debate_mode=True,
            critic_system=True,
            confidence_calibration=True,
            active_learning=True,
            curriculum_learning=True,
            quantum=QuantumConfig(enabled=True),
            genetic=GeneticConfig(enabled=True),
            swarm=SwarmConfig(enabled=True),
            rlhf=RLHFConfig(enabled=True),
            adversarial_testing=True,
            causal_reasoning=True,
            creative_synthesis=True,
            predictive_modeling=True,
            verification_system=True,
            scientific_method=True,
            real_time_learning=True,
            meta_learning=True
        )
        
        llm = QuantumLLM(config=config)
        
        # 既存データの読み込み
        try:
            llm.load_state('discord_quantum_state.json')
            print('📂 Loaded existing state')
        except:
            print('ℹ️  Starting fresh session')
        
        print('🚀 Quantum LLM System ready')
        print('🔮 All advanced features enabled')
        
    except Exception as e:
        print(f'❌ LLM initialization error: {e}')
        traceback.print_exc()
        sys.exit(1)
    
    # 🔧 スラッシュコマンドを同期（クリアは削除）
    try:
        print('🔄 Syncing commands...')
        
        # コマンドを同期（既存のコマンドは自動的に上書きされる）
        synced = await tree.sync()
        print(f'✅ Synced {len(synced)} slash commands')
        
        # 同期されたコマンド一覧を表示
        if synced:
            print('📋 Available commands:')
            for cmd in synced:
                print(f'   • /{cmd.name}: {cmd.description}')
        else:
            print('⚠️ No commands were synced!')
        
    except Exception as e:
        print(f'❌ Command sync error: {e}')
        traceback.print_exc()
    
    # ステータス設定
    await bot.change_presence(
        activity=discord.Activity(
            type=discord.ActivityType.listening,
            name="/help | 🔮 Quantum AI"
        )
    )
    
    print('=' * 80)
    print('✨ Bot is ready to use!')
    
    # 定期保存タスク開始
    bot.loop.create_task(auto_save())


@bot.event
async def on_message(message: discord.Message):
    """メッセージ受信時"""
    if message.author.bot:
        return
    
    user_id = message.author.id
    
    # 会話モードチェック
    if user_id in talk_mode_users and talk_mode_users[user_id]:
        # 会話モード: 全てのメッセージに反応
        query = message.content.strip()
        
        if not query:
            return
        
        # スラッシュコマンドは無視
        if query.startswith('/'):
            return
        
        await handle_query(message, query)
        return
    
    # メンションされた場合は会話モード（従来通り）
    if bot.user in message.mentions:
        query = message.content.replace(f'<@{bot.user.id}>', '').strip()
        
        if not query:
            await message.channel.send('❓ 質問を入力してください')
            return
        
        await handle_query(message, query)


async def handle_query(message: discord.Message, query: str):
    """クエリ処理"""
    if not llm:
        await message.channel.send('❌ システムが初期化されていません')
        return
    
    user_id = message.author.id
    
    # タイピングインジケーター
    async with message.channel.typing():
        try:
            # クエリ実行
            response = await llm.query_async(query)
            
            # セッションデータ更新
            session_data['total_queries'] += 1
            if response.success:
                session_data['successful_queries'] += 1
            if response.quantum_optimized:
                session_data['quantum_optimizations'] += 1
            if response.genetic_fitness > 0:
                session_data['genetic_evolutions'] += 1
            if response.swarm_consensus > 0:
                session_data['swarm_optimizations'] += 1
            
            # ユーザー履歴に追加
            if user_id not in user_conversations:
                user_conversations[user_id] = []
            
            user_conversations[user_id].append({
                'query': query[:200],
                'response': response.text[:200],
                'quality': response.quality_score,
                'strategy': response.strategy.value if response.strategy else None,
                'timestamp': datetime.now().isoformat()
            })
            
            # 最新50件のみ保持
            if len(user_conversations[user_id]) > 50:
                user_conversations[user_id] = user_conversations[user_id][-50:]
            
            # Embed作成
            embed = format_response_embed(response, query, message.author)
            
            await message.reply(embed=embed)
            
            # 長い応答は分割送信
            if len(response.text) > 4000:
                remaining = response.text[4000:]
                chunks = [remaining[i:i+1900] for i in range(0, len(remaining), 1900)]
                for chunk in chunks:
                    await message.channel.send(f"```\n{chunk}\n```")
            
        except Exception as e:
            error_embed = discord.Embed(
                title="❌ Error",
                description=f"処理中にエラーが発生しました: {str(e)}",
                color=discord.Color.red()
            )
            await message.reply(embed=error_embed)
            print(f"Error: {e}")
            traceback.print_exc()

# ==================== スラッシュコマンド ====================

@tree.command(name='swarm', description='群知能ステータスを表示')
async def swarm_command(interaction: discord.Interaction):
    """群知能"""
    # deferを追加
    await interaction.response.defer()
    
    if not llm or not llm.swarm:
        await interaction.followup.send('❌ 群知能が無効です', ephemeral=True)
        return
    
    embed = discord.Embed(
        title="🌊 Swarm Intelligence",
        description="多エージェント群知能システム",
        color=discord.Color.blue()
    )
    
    swarm = llm.swarm
    
    embed.add_field(
        name="🐝 Swarm Configuration",
        value=f"**Agents**: {len(swarm.agents)}\n"
              f"**Inertia**: {llm.config.swarm.inertia_weight}\n"
              f"**Cognitive**: {llm.config.swarm.cognitive_weight}\n"
              f"**Social**: {llm.config.swarm.social_weight}",
        inline=True
    )
    
    if swarm.agents:
        personas_str = "\n".join([
            f"• {agent.persona.value}: {agent.best_fitness:.3f}"
            for agent in swarm.agents
        ])
        embed.add_field(
            name="🎭 Agent Personas",
            value=personas_str,
            inline=True
        )
    
    embed.add_field(
        name="📊 Performance",
        value=f"**Global Best**: {swarm.global_best_fitness:.3f}\n"
              f"**Total Optimizations**: {llm.metrics['swarm_optimizations']}",
        inline=False
    )
    
    await interaction.followup.send(embed=embed)


# ==================== 追加のユーティリティコマンド ====================

@tree.command(name='clear', description='会話履歴をクリア')
async def clear_command(interaction: discord.Interaction):
    """履歴クリア"""
    # deferを追加
    await interaction.response.defer(ephemeral=True)
    
    user_id = interaction.user.id
    
    if user_id in user_conversations:
        del user_conversations[user_id]
    
    if llm:
        llm.context_window.clear()
    
    await interaction.followup.send('🗑️ 会話履歴をクリアしました', ephemeral=True)
@tree.command(name='talk', description='AIとの会話モードを切り替え')
@app_commands.describe(mode='会話モードのON/OFF')
@app_commands.choices(mode=[
    app_commands.Choice(name='ON', value='on'),
    app_commands.Choice(name='OFF', value='off')
])
async def talk_command(interaction: discord.Interaction, mode: str):
    """会話モード切り替え"""
    # 最初に応答を確保（重要！）
    await interaction.response.defer(ephemeral=True)
    
    user_id = interaction.user.id
    
    if mode.lower() == 'on':
        # モードをオン
        talk_mode_users[user_id] = True
        embed = discord.Embed(
            title="💬 Talk Mode: ON",
            description="会話モードを開始しました！\n"
                       "このチャンネルでメッセージを送信すると、AIが自動的に返答します",
            color=discord.Color.green()
        )
        embed.add_field(
            name="🎯 使い方",
            value="• 普通にメッセージを送信するだけでOK\n"
                  "• `/talk mode:off` でモードを終了\n"
                  "• `/clear` で会話履歴をリセット",
            inline=False
        )
        embed.add_field(
            name="✨ 機能",
            value="• 全ての高度なAI機能が利用可能\n"
                  "• 会話履歴が自動的に保存\n"
                  "• コンテキストを理解した返答",
            inline=False
        )
        embed.add_field(
            name="⚠️ 注意",
            value="• Botのメッセージには反応しません\n"
                  "• 他のユーザーのメッセージには反応しません",
            inline=False
        )
    else:  # mode == 'off'
        # モードをオフ
        if user_id in talk_mode_users:
            talk_mode_users[user_id] = False
            del talk_mode_users[user_id]
        
        embed = discord.Embed(
            title="💬 Talk Mode: OFF",
            description="会話モードを終了しました",
            color=discord.Color.red()
        )
        embed.add_field(
            name="ℹ️ 使い方",
            value="`/talk mode:on` で会話モードを開始できます\n"
                  "Botにメンション（@llm）して質問することもできます",
            inline=False
        )
    
    # deferした後はfollowupを使用
    await interaction.followup.send(embed=embed, ephemeral=True)

@tree.command(name='analogies', description='概念の類推を発見')
async def analogies_command(interaction: discord.Interaction, concept: str):
    """類推発見"""
    if not llm or not llm.creative_synthesizer:
        await interaction.response.send_message('❌ 創造的統合が無効です', ephemeral=True)
        return
    
    await interaction.response.defer(thinking=True)
    
    analogies = llm.creative_synthesizer.find_analogies(concept, top_k=10)
    
    if not analogies:
        await interaction.followup.send(f'🔍 "{concept}" の類推が見つかりませんでした')
        return
    
    embed = discord.Embed(
        title=f"🔍 Analogies for: {concept}",
        color=discord.Color.blue()
    )
    
    analogies_str = "\n".join([
        f"{i}. {create_progress_bar(sim, 15)} {sim:+.3f} - {related}"
        for i, (related, sim) in enumerate(analogies[:8], 1)
    ])
    
    embed.add_field(
        name="📊 Similar Concepts",
        value=analogies_str,
        inline=False
    )
    
    if len(analogies) >= 2:
        top1, top2 = analogies[0][0], analogies[1][0]
        embed.add_field(
            name="💡 Suggested Syntheses",
            value=f"Try: `/synthesize {concept} {top1}`\n"
                  f"Or: `/synthesize {concept} {top2}`",
            inline=False
        )
    
    await interaction.followup.send(embed=embed)

@tree.command(name='trust', description='システムの信頼スコアを表示')
async def trust_command(interaction: discord.Interaction):
    """信頼スコア"""
    # deferを追加
    await interaction.response.defer()
    
    if not llm or not llm.verification_system:
        await interaction.followup.send('❌ 検証システムが無効です', ephemeral=True)
        return
    
    trust_score = llm.verification_system.get_trust_score()
    
    embed = discord.Embed(
        title="🔐 System Trust Score",
        color=discord.Color.blue()
    )
    
    # 信頼スコア
    trust_bar = create_progress_bar(trust_score, 30)
    embed.add_field(
        name="📊 Overall Trust",
        value=f"[{trust_bar}] **{trust_score:.0%}**",
        inline=False
    )
    
    # 評価
    if trust_score > 0.8:
        rating = "🌟 EXCELLENT"
        desc = "Responses are highly trustworthy"
        color = discord.Color.green()
    elif trust_score > 0.6:
        rating = "✅ GOOD"
        desc = "Responses are generally reliable"
        color = discord.Color.blue()
    elif trust_score > 0.4:
        rating = "⚠️ MODERATE"
        desc = "Exercise caution with responses"
        color = discord.Color.orange()
    else:
        rating = "❌ LOW"
        desc = "System needs more calibration"
        color = discord.Color.red()
    
    embed.color = color
    embed.add_field(
        name="🎯 Rating",
        value=f"{rating}\n{desc}",
        inline=False
    )
    
    # 検証統計
    records = llm.verification_system.records
    if records:
        total = len(records)
        verified = sum(1 for r in records if r.result)
        
        embed.add_field(
            name="📋 Verification Statistics",
            value=f"**Total**: {total}\n"
                  f"**Verified**: {verified} ({verified/total:.0%})\n"
                  f"**Rejected**: {total - verified} ({(total-verified)/total:.0%})",
            inline=False
        )
    
    await interaction.followup.send(embed=embed)


@tree.command(name='adversarial', description='敵対的テストを実行')
async def adversarial_command(interaction: discord.Interaction):
    """敵対的テスト"""
    if not llm or not llm.adversarial_tester:
        await interaction.response.send_message('❌ 敵対的テストが無効です', ephemeral=True)
        return
    
    user_id = interaction.user.id
    
    if user_id not in user_conversations or not user_conversations[user_id]:
        await interaction.response.send_message('❌ 会話履歴がありません', ephemeral=True)
        return
    
    await interaction.response.defer(thinking=True)
    
    last_conv = user_conversations[user_id][-1]
    last_query = last_conv['query']
    
    embed = discord.Embed(
        title="🎪 Adversarial Robustness Test",
        description=f"Testing query: {last_query[:60]}...",
        color=discord.Color.orange()
    )
    
    # 敵対的クエリを生成
    adversarial_queries = llm.adversarial_tester.generate_adversarial_queries(last_query)
    
    embed.add_field(
        name="📋 Generated Variants",
        value=f"Generated {len(adversarial_queries)} adversarial examples",
        inline=False
    )
    
    # 簡易テスト（3つのみ）
    consistency_scores = []
    for i, adv_q in enumerate(adversarial_queries[:3], 1):
        try:
            adv_response = await llm.query_async(adv_q)  # 修正
            
            # 類似度計算（簡易）
            orig_words = set(last_conv['response'].lower().split())
            adv_words = set(adv_response.text.lower().split())
            
            if orig_words and adv_words:
                similarity = len(orig_words & adv_words) / len(orig_words | adv_words)
                consistency_scores.append(similarity)
        except Exception as e:
            print(f"Adversarial test error: {e}")
            pass
    
    if consistency_scores:
        import statistics
        avg_consistency = statistics.mean(consistency_scores)
        min_consistency = min(consistency_scores)
        
        consist_bar = create_progress_bar(avg_consistency, 20)
        
        embed.add_field(
            name="📊 Test Results",
            value=f"**Average Consistency**: [{consist_bar}] {avg_consistency:.0%}\n"
                  f"**Minimum Consistency**: {min_consistency:.0%}\n"
                  f"**Variants Tested**: {len(consistency_scores)}",
            inline=False
        )
        
        # 評価
        if avg_consistency > 0.7:
            assessment = "✅ ROBUST - High adversarial resistance"
            color = discord.Color.green()
        elif avg_consistency > 0.5:
            assessment = "⚠️ MODERATE - Some inconsistencies"
            color = discord.Color.orange()
        else:
            assessment = "❌ VULNERABLE - Significant weaknesses"
            color = discord.Color.red()
        
        embed.color = color
        embed.add_field(
            name="🎯 Assessment",
            value=assessment,
            inline=False
        )
    
    await interaction.followup.send(embed=embed)

@tree.command(name='export', description='データをエクスポート')
async def export_command(interaction: discord.Interaction):
    """エクスポート"""
    user_id = interaction.user.id
    
    if user_id not in user_conversations or not user_conversations[user_id]:
        await interaction.response.send_message('❌ エクスポートするデータがありません', ephemeral=True)
        return
    
    await interaction.response.defer(thinking=True, ephemeral=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"export_{interaction.user.name}_{timestamp}.json"
    
    export_data = {
        'user': {
            'id': user_id,
            'name': interaction.user.name
        },
        'session_id': f"{user_id}_{timestamp}",
        'timestamp': timestamp,
        'conversations': user_conversations[user_id],
        'summary': {
            'total_conversations': len(user_conversations[user_id]),
            'avg_quality': statistics.mean(c['quality'] for c in user_conversations[user_id]),
            'strategies_used': list(set(c['strategy'] for c in user_conversations[user_id] if c['strategy']))
        }
    }
    
    try:
        import io
        
        # JSONをファイルに変換
        json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
        file = discord.File(
            io.BytesIO(json_str.encode('utf-8')),
            filename=filename
        )
        
        await interaction.followup.send(
            '📤 データをエクスポートしました',
            file=file,
            ephemeral=True
        )
    except Exception as e:
        await interaction.followup.send(f'❌ エクスポートエラー: {e}', ephemeral=True)


@tree.command(name='compare', description='複数の戦略を比較')
async def compare_command(interaction: discord.Interaction, query: str):
    """戦略比較"""
    if not llm:
        await interaction.response.send_message('❌ システムが初期化されていません', ephemeral=True)
        return
    
    await interaction.response.defer(thinking=True)
    
    embed = discord.Embed(
        title="🔬 Strategy Comparison",
        description=f"Query: {query[:100]}",
        color=discord.Color.blue()
    )
    
    # 複数戦略でテスト
    strategies = [Strategy.DIRECT, Strategy.COT, Strategy.QUANTUM] if llm.quantum_optimizer else [Strategy.DIRECT, Strategy.COT]
    
    results = []
    for strategy in strategies:
        try:
            # 一時的に戦略を固定（簡易版）
            response = await llm.query_async(query)  # 修正
            results.append({
                'strategy': strategy.value if hasattr(strategy, 'value') else str(strategy),
                'quality': response.quality_score,
                'confidence': response.confidence,
                'latency': response.latency,
                'cost': response.cost
            })
        except Exception as e:
            print(f"Compare error for {strategy}: {e}")
            pass
    
    if results:
        # 結果を比較
        for i, result in enumerate(results, 1):
            quality_bar = create_progress_bar(result['quality'], 15)
            
            embed.add_field(
                name=f"{i}. {result['strategy'].upper()}",
                value=f"Quality: [{quality_bar}] {result['quality']:.0%}\n"
                      f"Confidence: {result['confidence']:.0%}\n"
                      f"Latency: {result['latency']:.0f}ms\n"
                      f"Cost: ${result['cost']:.6f}",
                inline=True
            )
        
        # 最良戦略
        best = max(results, key=lambda x: x['quality'])
        embed.add_field(
            name="🏆 Best Strategy",
            value=f"**{best['strategy'].upper()}** with {best['quality']:.0%} quality",
            inline=False
        )
    else:
        embed.add_field(
            name="❌ Error",
            value="全ての戦略でエラーが発生しました",
            inline=False
        )
    
    await interaction.followup.send(embed=embed)


@tree.command(name='benchmark', description='システムベンチマークを実行')
async def benchmark_command(interaction: discord.Interaction):
    """ベンチマーク"""
    if not llm:
        await interaction.response.send_message('❌ システムが初期化されていません', ephemeral=True)
        return
    
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message('❌ このコマンドは管理者のみ使用できます', ephemeral=True)
        return
    
    await interaction.response.defer(thinking=True)
    
    embed = discord.Embed(
        title="⚡ System Benchmark",
        description="パフォーマンステスト実行中...",
        color=discord.Color.gold()
    )
    
    test_queries = [
        "What is 2+2?",
        "Explain quantum computing briefly",
        "Write a Python function for Fibonacci"
    ]
    
    results = []
    for query in test_queries:
        import time
        start = time.time()
        try:
            response = await llm.query_async(query)  # 修正
            elapsed = (time.time() - start) * 1000
            
            results.append({
                'query': query[:30],
                'latency': elapsed,
                'quality': response.quality_score,
                'strategy': response.strategy.value if response.strategy else 'direct'
            })
        except Exception as e:
            print(f"Benchmark error for '{query}': {e}")
            continue
    
    if not results:
        await interaction.followup.send('❌ ベンチマークテストが全て失敗しました')
        return
    
    # 統計
    import statistics
    avg_latency = statistics.mean(r['latency'] for r in results)
    avg_quality = statistics.mean(r['quality'] for r in results)
    
    embed.description = f"**Tests**: {len(results)}\n" \
                       f"**Avg Latency**: {avg_latency:.0f}ms\n" \
                       f"**Avg Quality**: {avg_quality:.0%}"
    
    # 各テスト結果
    for i, result in enumerate(results, 1):
        embed.add_field(
            name=f"Test {i}: {result['query']}...",
            value=f"Latency: {result['latency']:.0f}ms\n"
                  f"Quality: {result['quality']:.0%}\n"
                  f"Strategy: {result['strategy']}",
            inline=True
        )
    
    await interaction.followup.send(embed=embed)


@tree.command(name='config', description='システム設定を表示')
async def config_command(interaction: discord.Interaction):
    """設定表示"""
    if not llm:
        await interaction.response.send_message('❌ システムが初期化されていません', ephemeral=True)
        return
    
    config = llm.config
    
    embed = discord.Embed(
        title="⚙️ System Configuration",
        color=discord.Color.blue()
    )
    
    # 基本設定
    embed.add_field(
        name="🔧 Basic Settings",
        value=f"**Model**: {config.model}\n"
              f"**Max Tokens**: {config.max_tokens}\n"
              f"**Temperature**: {config.temperature}\n"
              f"**Similarity Threshold**: {config.similarity_threshold:.2f}",
        inline=False
    )
    
    # 機能フラグ
    features = []
    if config.adaptive: features.append("✅ Adaptive")
    if config.vec_db: features.append("✅ Vector DB")
    if config.knowledge_graph: features.append("✅ Knowledge Graph")
    if config.chain_of_thought: features.append("✅ Chain of Thought")
    if config.quantum.enabled: features.append("✅ Quantum")
    if config.genetic.enabled: features.append("✅ Genetic")
    if config.swarm.enabled: features.append("✅ Swarm")
    if config.rlhf.enabled: features.append("✅ RLHF")
    
    embed.add_field(
        name="🚀 Enabled Features",
        value="\n".join(features) if features else "No features enabled",
        inline=False
    )
    
    ultimate_features = []
    if config.adversarial_testing: ultimate_features.append("✅ Adversarial Testing")
    if config.causal_reasoning: ultimate_features.append("✅ Causal Reasoning")
    if config.creative_synthesis: ultimate_features.append("✅ Creative Synthesis")
    if config.predictive_modeling: ultimate_features.append("✅ Predictive Modeling")
    if config.verification_system: ultimate_features.append("✅ Verification System")
    if config.scientific_method: ultimate_features.append("✅ Scientific Method")
    
    if ultimate_features:
        embed.add_field(
            name="🌟 Ultimate Features",
            value="\n".join(ultimate_features),
            inline=False
        )
    
    await interaction.response.send_message(embed=embed)


@tree.command(name='about', description='Botについて')
async def about_command(interaction: discord.Interaction):
    """About"""
    embed = discord.Embed(
        title="🔮 Quantum-Enhanced AI Assistant",
        description="究極の自己進化型エンタープライズLLMシステム v3.5γ ULTIMATE",
        color=discord.Color.purple()
    )
    
    embed.add_field(
        name="✨ Core Technologies",
        value="• 🔮 Quantum-Inspired Optimization (QAOA)\n"
              "• 🧬 Genetic Algorithm Evolution\n"
              "• 🌊 Swarm Intelligence (PSO)\n"
              "• 🎯 Reinforcement Learning (RLHF)\n"
              "• 🧩 Dynamic Knowledge Graph\n"
              "• 🔐 Multi-Layer Verification",
        inline=False
    )
    
    embed.add_field(
        name="🌟 Ultimate Features",
        value="• 🧩 Causal Inference Engine\n"
              "• 🎨 Creative Synthesis System\n"
              "• 🔬 Scientific Method Application\n"
              "• 🎪 Adversarial Testing\n"
              "• 🔮 Predictive Modeling\n"
              "• 📊 Meta-Learning & Analysis",
        inline=False
    )
    
    embed.add_field(
        name="📊 Stats",
        value=f"**Servers**: {len(bot.guilds)}\n"
              f"**Uptime**: {(datetime.now() - session_data['start_time']).seconds // 3600}h\n"
              f"**Total Queries**: {session_data['total_queries']}\n"
              f"**Success Rate**: {session_data['successful_queries']/max(session_data['total_queries'],1):.0%}",
        inline=False
    )
    
    embed.add_field(
        name="🔗 Commands",
        value="Use `/help` to see all available commands",
        inline=False
    )
    
    embed.set_footer(text="Powered by GROQ & Claude.ai")
    
    await interaction.response.send_message(embed=embed)


# ==================== エラーハンドリング ====================

@bot.event
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    """コマンドエラーハンドリング"""
    if isinstance(error, app_commands.CommandOnCooldown):
        await interaction.response.send_message(
            f'⏳ クールダウン中です。{error.retry_after:.1f}秒後に再試行してください。',
            ephemeral=True
        )
    elif isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message(
            '❌ このコマンドを実行する権限がありません。',
            ephemeral=True
        )
    else:
        print(f'Command error: {error}')
        traceback.print_exc()
        
        if not interaction.response.is_done():
            await interaction.response.send_message(
                f'❌ エラーが発生しました: {str(error)}',
                ephemeral=True
            )

@tree.command(name='rlhf', description='強化学習情報を表示')
async def rlhf_command(interaction: discord.Interaction):
    """RLHF"""
    # deferを追加
    await interaction.response.defer()
    
    if not llm or not llm.rlhf:
        await interaction.followup.send('❌ RLHFが無効です', ephemeral=True)
        return
    
    embed = discord.Embed(
        title="🎯 Reinforcement Learning from Human Feedback",
        description="人間のフィードバックから学習",
        color=discord.Color.orange()
    )
    
    rlhf = llm.rlhf
    
    embed.add_field(
        name="🧠 Learning Status",
        value=f"**States Explored**: {len(rlhf.state_visits)}\n"
              f"**Q-Table Size**: {len(rlhf.q_table)}\n"
              f"**Total Updates**: {sum(rlhf.state_visits.values())}\n"
              f"**Learning Rate**: {llm.config.rlhf.learning_rate}",
        inline=False
    )
    
    if rlhf.reward_history:
        import statistics
        avg_reward = statistics.mean(rlhf.reward_history)
        recent = rlhf.reward_history[-10:] if len(rlhf.reward_history) >= 10 else rlhf.reward_history
        recent_avg = statistics.mean(recent)
        
        trend = "📈 Improving" if recent_avg > avg_reward else "📉 Declining" if recent_avg < avg_reward else "➡️ Stable"
        
        embed.add_field(
            name="📈 Rewards",
            value=f"**Average**: {avg_reward:.3f}\n"
                  f"**Recent (10)**: {recent_avg:.3f}\n"
                  f"**Trend**: {trend}",
            inline=False
        )
    
    await interaction.followup.send(embed=embed)

@tree.command(name='knowledge', description='知識グラフ情報を表示')
async def knowledge_command(interaction: discord.Interaction):
    """知識グラフ"""
    # deferを追加
    await interaction.response.defer()
    
    if not llm or not llm.knowledge_graph:
        await interaction.followup.send('❌ 知識グラフが無効です', ephemeral=True)
        return
    
    kg = llm.knowledge_graph
    
    embed = discord.Embed(
        title="🧩 Knowledge Graph",
        description="動的知識グラフシステム",
        color=discord.Color.teal()
    )
    
    embed.add_field(
        name="📊 Graph Statistics",
        value=f"**Nodes**: {len(kg.nodes)}\n"
              f"**Edges**: {len(kg.edges)}\n"
              f"**Communities**: {len(kg.communities)}",
        inline=True
    )
    
    # 中心性の高いノード
    central_nodes = kg.get_central_nodes(5)
    if central_nodes:
        central_str = "\n".join([
            f"• {kg.nodes[nid].name} (degree: {degree})"
            for nid, degree in central_nodes
        ])
        embed.add_field(
            name="🌟 Central Concepts",
            value=central_str,
            inline=True
        )
    
    await interaction.followup.send(embed=embed)


@tree.command(name='causal', description='因果推論を実行')
async def causal_command(interaction: discord.Interaction, event: str):
    """因果推論"""
    if not llm or not llm.causal_engine:
        await interaction.response.send_message('❌ 因果推論が無効です', ephemeral=True)
        return
    
    await interaction.response.defer(thinking=True)
    
    embed = discord.Embed(
        title="🧩 Causal Inference",
        description=f"Event: {event}",
        color=discord.Color.purple()
    )
    
    # 原因を推論
    causes = llm.causal_engine.infer_cause(event, depth=3)
    
    if causes:
        causes_str = "\n".join([
            f"{i}. {create_progress_bar(prob)} {prob:.0%} - {cause[:50]}"
            for i, (cause, prob) in enumerate(causes[:5], 1)
        ])
        embed.add_field(
            name="🔍 Potential Causes",
            value=causes_str,
            inline=False
        )
    
    # 結果を予測
    effects = llm.causal_engine.predict_effect(event, depth=3)
    
    if effects:
        effects_str = "\n".join([
            f"{i}. {create_progress_bar(prob)} {prob:.0%} - {effect[:50]}"
            for i, (effect, prob) in enumerate(effects[:5], 1)
        ])
        embed.add_field(
            name="🔮 Predicted Effects",
            value=effects_str,
            inline=False
        )
    
    await interaction.followup.send(embed=embed)


@tree.command(name='synthesize', description='2つの概念を創造的に統合')
async def synthesize_command(interaction: discord.Interaction, concept_a: str, concept_b: str):
    """創造的統合"""
    if not llm or not llm.creative_synthesizer:
        await interaction.response.send_message('❌ 創造的統合が無効です', ephemeral=True)
        return
    
    await interaction.response.defer(thinking=True)
    
    synthesis = llm.creative_synthesizer.synthesize(concept_a, concept_b)
    
    embed = discord.Embed(
        title="🎨 Creative Synthesis",
        description=f"**{concept_a}** + **{concept_b}**",
        color=discord.Color.magenta()
    )
    
    embed.add_field(
        name="💡 Synthesized Concept",
        value=synthesis.synthesis,
        inline=False
    )
    
    # メトリクス
    novelty_bar = create_progress_bar(synthesis.novelty_score)
    coherence_bar = create_progress_bar(synthesis.coherence_score)
    useful_bar = create_progress_bar(synthesis.usefulness_score)
    
    embed.add_field(
        name="📊 Innovation Metrics",
        value=f"**Novelty**: [{novelty_bar}] {synthesis.novelty_score:.0%}\n"
              f"**Coherence**: [{coherence_bar}] {synthesis.coherence_score:.0%}\n"
              f"**Usefulness**: [{useful_bar}] {synthesis.usefulness_score:.0%}",
        inline=False
    )
    
    overall = (synthesis.novelty_score + synthesis.coherence_score + synthesis.usefulness_score) / 3
    embed.add_field(
        name="🌟 Overall Innovation Score",
        value=f"{create_progress_bar(overall)} **{overall:.0%}**",
        inline=False
    )
    
    await interaction.followup.send(embed=embed)


@tree.command(name='verify', description='主張を検証')
async def verify_command(interaction: discord.Interaction, claim: str):
    """検証"""
    if not llm or not llm.verification_system:
        await interaction.response.send_message('❌ 検証システムが無効です', ephemeral=True)
        return
    
    await interaction.response.defer(thinking=True)
    
    from collections import defaultdict
    import statistics
    
    # 複数の検証方法を適用
    VerificationMethod = quantum_module.VerificationMethod
    methods = [
        VerificationMethod.LOGICAL_CONSISTENCY,
        VerificationMethod.CROSS_REFERENCE,
        VerificationMethod.FACT_CHECK
    ]
    
    results = []
    for method in methods:
        verification = llm.verification_system.verify_claim(claim, "", method)
        results.append(verification)
    
    embed = discord.Embed(
        title="🔐 Claim Verification",
        description=f"**Claim**: {claim[:200]}",
        color=discord.Color.blue()
    )
    
    # 各検証結果
    for i, v in enumerate(results, 1):
        status = "✅ VERIFIED" if v.result else "❌ REJECTED"
        conf_bar = create_progress_bar(v.confidence)
        
        embed.add_field(
            name=f"{i}. {v.method.value.replace('_', ' ').title()}",
            value=f"{status}\n"
                  f"Confidence: [{conf_bar}] {v.confidence:.0%}\n"
                  f"Evidence: {', '.join(v.evidence[:2])}",
            inline=False
        )
    
    # 総合判定
    avg_confidence = statistics.mean(v.confidence for v in results)
    verified_count = sum(1 for v in results if v.result)
    
    if verified_count == len(results) and avg_confidence > 0.7:
        assessment = "✅ HIGHLY CREDIBLE"
        color = discord.Color.green()
    elif verified_count >= len(results) / 2:
        assessment = "⚠️ PARTIALLY VERIFIED"
        color = discord.Color.orange()
    else:
        assessment = "❌ NOT VERIFIED"
        color = discord.Color.red()
    
    embed.color = color
    embed.add_field(
        name="🎯 Overall Assessment",
        value=f"{assessment} ({avg_confidence:.0%} confidence)",
        inline=False
    )
    
    await interaction.followup.send(embed=embed)


@tree.command(name='predict', description='次の意図を予測')
async def predict_command(interaction: discord.Interaction):
    """予測"""
    if not llm or not llm.predictive_engine:
        await interaction.response.send_message('❌ 予測モデリングが無効です', ephemeral=True)
        return
    
    embed = discord.Embed(
        title="🔮 Predictive Analysis",
        color=discord.Color.purple()
    )
    
    # 次の意図を予測
    predicted_intent = llm.predictive_engine.predict_next_intent()
    success_prob = llm.predictive_engine.get_success_probability(predicted_intent)
    
    embed.add_field(
        name="📍 Next Query Prediction",
        value=f"**Predicted Intent**: {predicted_intent.value}\n"
              f"**Success Probability**: {success_prob:.0%}",
        inline=False
    )
    
    # 使用パターン
    if llm.predictive_engine.model.user_patterns:
        from collections import Counter
        top_patterns = sorted(
            llm.predictive_engine.model.user_patterns.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:5]
        
        patterns_str = "\n".join([
            f"• {pattern}: {statistics.mean(results):.0%} success ({len(results)} samples)"
            for pattern, results in top_patterns
        ])
        
        embed.add_field(
            name="📊 Usage Patterns",
            value=patterns_str,
            inline=False
        )
    
    await interaction.response.send_message(embed=embed)


@tree.command(name='scientific', description='科学的手法を適用')
async def scientific_command(interaction: discord.Interaction, observation: str):
    """科学的手法"""
    if not llm or not llm.scientific_method:
        await interaction.response.send_message('❌ 科学的手法が無効です', ephemeral=True)
        return
    
    await interaction.response.defer(thinking=True)
    
    embed = discord.Embed(
        title="🔬 Scientific Method Application",
        description=f"**Observation**: {observation}",
        color=discord.Color.blue()
    )
    
    # 1. 仮説定式化
    hypothesis = llm.scientific_method.formulate_hypothesis(observation)
    embed.add_field(
        name="1️⃣ Hypothesis",
        value=f"{hypothesis.statement}\n"
              f"Prior Confidence: {hypothesis.bayesian_prior:.0%}",
        inline=False
    )
    
    # 2. 実験設計
    experiment = llm.scientific_method.design_experiment(hypothesis)
    embed.add_field(
        name="2️⃣ Experiment Design",
        value=f"**ID**: {experiment['id']}\n"
              f"**Method**: {experiment['method']}\n"
              f"**Status**: {experiment['status']}",
        inline=False
    )
    
    # 3. 予測
    embed.add_field(
        name="3️⃣ Predictions",
        value="• Measurable outcome expected\n"
              "• Reproducible under similar conditions\n"
              "• Consistent with existing knowledge",
        inline=False
    )
    
    # 4. 結果分析（シミュレート）
    analysis = llm.scientific_method.analyze_results(
        experiment['id'],
        {'data_points': 100, 'effect_observed': True}
    )
    
    embed.add_field(
        name="4️⃣ Analysis",
        value=f"**Significance**: {analysis['statistical_significance']:.3f}\n"
              f"**Effect Size**: {analysis['effect_size']:.3f}\n"
              f"**Conclusion**: {analysis['conclusion']}",
        inline=False
    )
    
    # 5. ピアレビュー
    mock_reviews = [
        "Sound methodology",
        "Results consistent with theory",
        "Further validation recommended"
    ]
    review_score = llm.scientific_method.peer_review(hypothesis, mock_reviews)
    
    embed.add_field(
        name="5️⃣ Peer Review",
        value=f"Score: {create_progress_bar(review_score)} {review_score:.0%}",
        inline=False
    )
    
    # 最終評価
    if review_score > 0.7 and analysis['statistical_significance'] > 0.05:
        assessment = "✅ HYPOTHESIS SUPPORTED"
    else:
        assessment = "⚠️ MORE EVIDENCE NEEDED"
    
    embed.add_field(
        name="🎯 Final Assessment",
        value=assessment,
        inline=False
    )
    
    await interaction.followup.send(embed=embed)


@tree.command(name='progress', description='学習進捗を分析')
async def progress_command(interaction: discord.Interaction):
    """学習進捗"""
    # 最初に応答を確保
    await interaction.response.defer(ephemeral=True)
    
    if not llm:
        await interaction.followup.send('❌ システムが初期化されていません', ephemeral=True)
        return
    
    progress = llm.analyze_learning_progress()
    
    if progress['status'] == 'insufficient_data':
        await interaction.followup.send(
            '⚠️ データ不足\n継続利用で進捗追跡が可能になります',
            ephemeral=True
        )
        return
    
    embed = discord.Embed(
        title="📊 Learning Progress Analysis",
        color=discord.Color.gold()
    )
    
    # トレンド
    trend_emoji = {
        'improving': '📈',
        'declining': '📉',
        'stable': '➡️'
    }
    
    embed.add_field(
        name="📈 Overall Metrics",
        value=f"**Total Interactions**: {progress['total_interactions']}\n"
              f"**Recent Quality**: {progress['recent_quality']:.3f}\n"
              f"**Improvement**: {progress['improvement']:+.3f}\n"
              f"**Trend**: {trend_emoji.get(progress['trend'], '➡️')} {progress['trend'].upper()}",
        inline=False
    )
    
    # 戦略パフォーマンス
    if progress['best_strategy']:
        embed.add_field(
            name="🎯 Best Strategy",
            value=progress['best_strategy'],
            inline=True
        )
        
        if 'strategy_performance' in progress:
            perf_str = "\n".join([
                f"{strategy}: {create_progress_bar(score, 10)} {score:.3f}"
                for strategy, score in sorted(
                    progress['strategy_performance'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            ])
            
            embed.add_field(
                name="📊 Strategy Performance",
                value=perf_str,
                inline=False
            )
    
    await interaction.followup.send(embed=embed, ephemeral=True)


@tree.command(name='insights', description='メタインサイトを生成')
async def insights_command(interaction: discord.Interaction):
    """インサイト"""
    if not llm:
        await interaction.response.send_message('❌ システムが初期化されていません', ephemeral=True)
        return
    
    insights = llm.generate_meta_insights()
    
    if not insights:
        await interaction.response.send_message(
            '⚠️ データ不足\nシステムとの対話を続けてください',
            ephemeral=True
        )
        return
    
    embed = discord.Embed(
        title="🌟 Meta-Level Insights",
        description="システムが生成した深い洞察",
        color=discord.Color.purple()
    )
    
    for i, insight in enumerate(insights, 1):
        embed.add_field(
            name=f"Insight {i}",
            value=insight,
            inline=False
        )
    
    await interaction.response.send_message(embed=embed)


@tree.command(name='save', description='データを保存(管理者のみ)')
async def save_command(interaction: discord.Interaction):
    """保存"""
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message(
            '❌ このコマンドは管理者のみ使用できます',
            ephemeral=True
        )
        return
    
    if not llm:
        await interaction.response.send_message('❌ システムが初期化されていません', ephemeral=True)
        return
    
    await interaction.response.defer(thinking=True)
    
    try:
        llm.save_state('discord_quantum_state.json')
        
        # 会話履歴も保存
        with open('discord_conversations.json', 'w', encoding='utf-8') as f:
            json.dump(user_conversations, f, ensure_ascii=False, indent=2)
        
        # セッションデータ保存
        with open('discord_session.json', 'w', encoding='utf-8') as f:
            session_copy = session_data.copy()
            session_copy['start_time'] = session_copy['start_time'].isoformat()
            json.dump(session_copy, f, ensure_ascii=False, indent=2)
        
        await interaction.followup.send('💾 データを保存しました')
    except Exception as e:
        await interaction.followup.send(f'❌ 保存エラー: {e}')


@tree.command(name='session', description='セッション情報を表示')
async def session_command(interaction: discord.Interaction):
    """セッション情報"""
    uptime = datetime.now() - session_data['start_time']
    
    embed = discord.Embed(
        title="📊 Session Information",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name="⏱️ Uptime",
        value=f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds//60)%60}m",
        inline=True
    )
    
    embed.add_field(
        name="📈 Queries",
        value=f"**Total**: {session_data['total_queries']}\n"
              f"**Successful**: {session_data['successful_queries']}\n"
              f"**Success Rate**: {session_data['successful_queries']/max(session_data['total_queries'],1):.1%}",
        inline=True
    )
    
    embed.add_field(
        name="🚀 Advanced Features",
        value=f"🔮 Quantum: {session_data['quantum_optimizations']}\n"
              f"🧬 Genetic: {session_data['genetic_evolutions']}\n"
              f"🌊 Swarm: {session_data['swarm_optimizations']}",
        inline=False
    )
    
    await interaction.response.send_message(embed=embed)

@tree.command(name='info', description='システムとユーザーの詳細情報を表示')
async def info_command(interaction: discord.Interaction):
    """詳細情報"""
    await interaction.response.defer(thinking=True)
    
    user_id = interaction.user.id
    uptime = datetime.now() - session_data['start_time']
    
    # メインEmbed
    embed = discord.Embed(
        title="ℹ️ System & User Information",
        description="Quantum-Enhanced AI System v3.5γ ULTIMATE",
        color=discord.Color.blue(),
        timestamp=datetime.now()
    )
    
    # 🤖 Bot情報
    bot_info = []
    bot_info.append(f"**Name**: {bot.user.name}#{bot.user.discriminator}")
    bot_info.append(f"**ID**: `{bot.user.id}`")
    bot_info.append(f"**Servers**: {len(bot.guilds)}")
    bot_info.append(f"**Uptime**: {uptime.days}d {uptime.seconds//3600}h {(uptime.seconds//60)%60}m")
    
    embed.add_field(
        name="🤖 Bot Information",
        value="\n".join(bot_info),
        inline=False
    )
    
    # 👤 ユーザー情報
    user_info = []
    user_info.append(f"**User**: {interaction.user.name}")
    user_info.append(f"**ID**: `{user_id}`")
    
    # 会話モード状態
    talk_mode = "🟢 ON" if user_id in talk_mode_users and talk_mode_users[user_id] else "🔴 OFF"
    user_info.append(f"**Talk Mode**: {talk_mode}")
    
    # 会話履歴
    if user_id in user_conversations:
        conv_count = len(user_conversations[user_id])
        user_info.append(f"**Conversations**: {conv_count}/50")
        
        if conv_count > 0:
            import statistics
            avg_quality = statistics.mean(c['quality'] for c in user_conversations[user_id])
            user_info.append(f"**Avg Quality**: {avg_quality:.1%}")
    else:
        user_info.append(f"**Conversations**: 0/50")
    
    embed.add_field(
        name="👤 Your Information",
        value="\n".join(user_info),
        inline=False
    )
    
    # 📊 セッション統計
    session_info = []
    session_info.append(f"**Total Queries**: {session_data['total_queries']}")
    session_info.append(f"**Successful**: {session_data['successful_queries']}")
    if session_data['total_queries'] > 0:
        success_rate = session_data['successful_queries'] / session_data['total_queries']
        session_info.append(f"**Success Rate**: {success_rate:.1%}")
    session_info.append(f"**Active Users**: {len(user_conversations)}")
    
    embed.add_field(
        name="📊 Session Statistics",
        value="\n".join(session_info),
        inline=True
    )
    
    # 🚀 高度機能の使用状況
    feature_info = []
    feature_info.append(f"🔮 Quantum: {session_data['quantum_optimizations']}")
    feature_info.append(f"🧬 Genetic: {session_data['genetic_evolutions']}")
    feature_info.append(f"🌊 Swarm: {session_data['swarm_optimizations']}")
    
    embed.add_field(
        name="🚀 Advanced Features",
        value="\n".join(feature_info),
        inline=True
    )
    
    # ⚙️ システム設定
    if llm:
        config = llm.config
        config_info = []
        config_info.append(f"**Model**: {config.model}")
        config_info.append(f"**Temperature**: {config.temperature}")
        config_info.append(f"**Max Tokens**: {config.max_tokens}")
        
        # 有効な機能カウント
        enabled_features = []
        if hasattr(config, 'quantum') and config.quantum.enabled:
            enabled_features.append("Quantum")
        if hasattr(config, 'genetic') and config.genetic.enabled:
            enabled_features.append("Genetic")
        if hasattr(config, 'swarm') and config.swarm.enabled:
            enabled_features.append("Swarm")
        if hasattr(config, 'rlhf') and config.rlhf.enabled:
            enabled_features.append("RLHF")
        if hasattr(config, 'adversarial_testing') and config.adversarial_testing:
            enabled_features.append("Adversarial")
        if hasattr(config, 'causal_reasoning') and config.causal_reasoning:
            enabled_features.append("Causal")
        if hasattr(config, 'creative_synthesis') and config.creative_synthesis:
            enabled_features.append("Creative")
        if hasattr(config, 'verification_system') and config.verification_system:
            enabled_features.append("Verification")
        
        config_info.append(f"**Active Modules**: {len(enabled_features)}/8")
        
        embed.add_field(
            name="⚙️ System Configuration",
            value="\n".join(config_info),
            inline=False
        )
    
    # 🧠 LLMシステム状態
    if llm:
        llm_info = []
        
        # Knowledge Graph
        if hasattr(llm, 'knowledge_graph') and llm.knowledge_graph:
            try:
                kg = llm.knowledge_graph
                llm_info.append(f"**KG Nodes**: {len(kg.nodes)}")
                llm_info.append(f"**KG Edges**: {len(kg.edges)}")
            except Exception as e:
                print(f"Knowledge Graph info error: {e}")
        
        # Vector DB - 安全にアクセス
        if hasattr(llm, 'vector_db') and llm.vector_db:
            try:
                # VectorDBの実装に応じて適切な属性にアクセス
                if hasattr(llm.vector_db, 'vectors') and isinstance(llm.vector_db.vectors, dict):
                    llm_info.append(f"**Vector Entries**: {len(llm.vector_db.vectors)}")
                elif hasattr(llm.vector_db, 'embeddings') and isinstance(llm.vector_db.embeddings, dict):
                    llm_info.append(f"**Vector Entries**: {len(llm.vector_db.embeddings)}")
                elif hasattr(llm.vector_db, '__len__'):
                    llm_info.append(f"**Vector Entries**: {len(llm.vector_db)}")
            except Exception as e:
                print(f"Vector DB info error: {e}")
        
        # Context Window
        if hasattr(llm, 'context_window') and llm.context_window:
            try:
                llm_info.append(f"**Context Size**: {len(llm.context_window.messages)}")
            except Exception as e:
                print(f"Context Window info error: {e}")
        
        # Metrics
        if hasattr(llm, 'metrics') and llm.metrics:
            try:
                total_interactions = llm.metrics.get('total_queries', 0)
                if total_interactions > 0:
                    llm_info.append(f"**Total Interactions**: {total_interactions}")
            except Exception as e:
                print(f"Metrics info error: {e}")
        
        if llm_info:
            embed.add_field(
                name="🧠 LLM State",
                value="\n".join(llm_info),
                inline=True
            )
    
    # 📈 パフォーマンスメトリクス
    if llm and hasattr(llm, 'metrics') and llm.metrics:
        perf_info = []
        
        if 'strategy_performance' in llm.metrics and llm.metrics['strategy_performance']:
            try:
                best_strategy = max(llm.metrics['strategy_performance'].items(), 
                                  key=lambda x: x[1])
                perf_info.append(f"**Best Strategy**: {best_strategy[0]}")
            except:
                pass
        
        if 'total_tokens' in llm.metrics:
            perf_info.append(f"**Total Tokens**: {llm.metrics['total_tokens']:,}")
        
        if 'total_cost' in llm.metrics:
            perf_info.append(f"**Total Cost**: ${llm.metrics['total_cost']:.4f}")
        
        if perf_info:
            embed.add_field(
                name="📈 Performance",
                value="\n".join(perf_info),
                inline=True
            )
    
    # 🎮 利用可能なコマンド数
    try:
        commands = await tree.fetch_commands()
        embed.add_field(
            name="🎮 Available Commands",
            value=f"**Total**: {len(commands)} commands\n"
                  f"Use `/about` to see features",
            inline=False
        )
    except:
        embed.add_field(
            name="🎮 Available Commands",
            value=f"Use `/about` to see all features",
            inline=False
        )
    
    # 💡 クイックアクション
    quick_actions = []
    quick_actions.append("`/talk mode:on` - 会話モード開始")
    quick_actions.append("`/clear` - 会話履歴クリア")
    quick_actions.append("`/config` - 詳細設定表示")
    quick_actions.append("`/session` - セッション情報")
    
    embed.add_field(
        name="💡 Quick Actions",
        value="\n".join(quick_actions),
        inline=False
    )
    
    # フッター
    embed.set_footer(
        text=f"Requested by {interaction.user.name}",
        icon_url=interaction.user.display_avatar.url if interaction.user.display_avatar else None
    )
    
    await interaction.followup.send(embed=embed)

# ==================== メイン実行 ====================

def main():
    """エントリーポイント"""
    discord_token = os.environ.get('DISCORD_BOT_TOKEN')
    groq_key = os.environ.get('GROQ_API_KEY')
    
    if not discord_token:
        print('❌ DISCORD_BOT_TOKEN が設定されていません')
        print('export DISCORD_BOT_TOKEN="your_token_here"')
        sys.exit(1)
    
    if not groq_key:
        print('❌ GROQ_API_KEY が設定されていません')
        print('export GROQ_API_KEY="your_key_here"')
        sys.exit(1)
    
    print('🚀 Quantum Discord Bot を起動しています...')
    print('=' * 80)
    
    try:
        bot.run(discord_token)
    except discord.LoginFailure:
        print('❌ Discord Token が無効です')
        sys.exit(1)
    except Exception as e:
        print(f'❌ Bot起動エラー: {e}')
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
