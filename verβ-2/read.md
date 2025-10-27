# Quantum-Enhanced LLM System - プロジェクト構造

## 📁 ディレクトリ構造

```
quantum-llm-system/
├── main.py                          # メインエントリーポイント
├── requirements.txt                 # 依存パッケージ
├── README.md                       # プロジェクト概要
├── PROJECT_STRUCTURE.md            # このファイル
│
├── core/                           # コアシステム
│   ├── __init__.py
│   ├── config.py                   # 設定クラス (SystemConfig, QuantumConfig等)
│   ├── enums.py                    # 列挙型 (Intent, Complexity, Strategy等)
│   ├── data_models.py              # データモデル (Response, Prompt, Agent等)
│   ├── llm_system.py               # メインLLMシステム (QuantumLLM)
│   └── query_processor.py          # クエリ処理ロジック
│
├── strategies/                     # 実行戦略
│   ├── __init__.py
│   ├── base_strategy.py            # 戦略基底クラス
│   ├── quantum_strategy.py         # 量子インスパイア戦略
│   ├── genetic_strategy.py         # 遺伝的進化戦略
│   ├── swarm_strategy.py           # 群知能戦略
│   └── direct_strategy.py          # 直接実行戦略
│
├── optimizers/                     # 最適化アルゴリズム
│   ├── __init__.py
│   ├── quantum_optimizer.py        # 量子インスパイア最適化
│   ├── genetic_evolver.py          # 遺伝的アルゴリズム
│   ├── swarm_intelligence.py       # 群知能 (PSO)
│   └── rlhf_trainer.py             # RLHF学習
│
├── reasoning/                      # 推論エンジン
│   ├── __init__.py
│   ├── causal_engine.py            # 因果推論
│   ├── scientific_method.py        # 科学的手法適用
│   ├── verification_system.py      # 検証システム
│   └── adversarial_tester.py       # 敵対的テスト
│
├── knowledge/                      # 知識管理
│   ├── __init__.py
│   ├── vector_db.py                # ベクトルデータベース
│   ├── knowledge_graph.py          # 知識グラフ
│   └── predictive_engine.py        # 予測モデリング
│
├── creativity/                     # 創造的機能
│   ├── __init__.py
│   └── creative_synthesizer.py     # 創造的統合システム
│
├── ui/                            # ユーザーインターフェース
│   ├── __init__.py
│   ├── chat_interface.py           # チャットUI (QuantumChat)
│   ├── command_handlers.py         # コマンドハンドラ
│   └── display_utils.py            # 表示ユーティリティ
│
├── utils/                         # ユーティリティ
│   ├── __init__.py
│   ├── logger.py                   # ロガー
│   ├── cost_calculator.py          # コスト計算
│   └── text_analysis.py            # テキスト分析
│
└── tests/                         # テスト
    ├── __init__.py
    ├── test_quantum_optimizer.py
    ├── test_genetic_evolver.py
    ├── test_swarm_intelligence.py
    └── test_integration.py
```

## 📦 各モジュールの詳細

### 1. **core/** - コアシステム

#### `config.py`
- システム全体の設定を管理
- クラス: `SystemConfig`, `QuantumConfig`, `GeneticConfig`, `SwarmConfig`, `RLHFConfig`

#### `enums.py`
- 列挙型定義
- `Intent`, `Complexity`, `Strategy`, `PersonaType`, `ReasoningType`, `VerificationMethod`

#### `data_models.py`
- データクラス定義
- `Response`, `Prompt`, `Agent`, `Hypothesis`, `KnowledgeNode`, `KnowledgeEdge`
- `CausalNode`, `AdversarialTest`, `VerificationRecord`, `CreativeSynthesis`

#### `llm_system.py`
- メインシステムクラス `QuantumLLM`
- API呼び出し、状態管理、メトリクス追跡

#### `query_processor.py`
- クエリの分析と処理
- 意図・複雑度の判定、戦略選択

### 2. **strategies/** - 実行戦略

各戦略の実装を分離:
- `quantum_strategy.py` - 量子インスパイア最適化
- `genetic_strategy.py` - 遺伝的プロンプト進化
- `swarm_strategy.py` - 群知能マルチエージェント
- `direct_strategy.py` - 直接実行

### 3. **optimizers/** - 最適化アルゴリズム

- `quantum_optimizer.py` - QAOA風最適化、量子アニーリング
- `genetic_evolver.py` - 遺伝的アルゴリズムによるプロンプト進化
- `swarm_intelligence.py` - Particle Swarm Optimization (PSO)
- `rlhf_trainer.py` - Q-Learning ベースのRLHF

### 4. **reasoning/** - 推論エンジン

- `causal_engine.py` - 因果推論グラフ、介入分析
- `scientific_method.py` - 仮説生成、実験設計、ピアレビュー
- `verification_system.py` - 多層検証（論理一貫性、ファクトチェック等）
- `adversarial_tester.py` - 敵対的テスト、ロバストネス評価

### 5. **knowledge/** - 知識管理

- `vector_db.py` - ハッシュベース埋め込みベクトルDB
- `knowledge_graph.py` - グラフ構造、コミュニティ検出、中心性分析
- `predictive_engine.py` - ユーザーパターン予測、意図予測

### 6. **creativity/** - 創造的機能

- `creative_synthesizer.py` - 概念の創造的統合、類推発見

### 7. **ui/** - ユーザーインターフェース

- `chat_interface.py` - メインチャットループ
- `command_handlers.py` - 全コマンドの処理ロジック
- `display_utils.py` - 出力フォーマット、統計表示

### 8. **utils/** - ユーティリティ

- `logger.py` - カスタムロガー
- `cost_calculator.py` - API使用コスト計算
- `text_analysis.py` - テキスト処理ユーティリティ

## 🔧 実装の進め方

### Phase 1: コア基盤 (優先度: 高)
1. `core/enums.py` - 列挙型
2. `core/data_models.py` - データモデル
3. `core/config.py` - 設定
4. `utils/logger.py` - ロガー
5. `utils/cost_calculator.py` - コスト計算

### Phase 2: 知識管理 (優先度: 高)
1. `knowledge/vector_db.py` - ベクトルDB
2. `knowledge/knowledge_graph.py` - 知識グラフ

### Phase 3: 最適化アルゴリズム (優先度: 中)
1. `optimizers/quantum_optimizer.py`
2. `optimizers/genetic_evolver.py`
3. `optimizers/swarm_intelligence.py`
4. `optimizers/rlhf_trainer.py`

### Phase 4: 実行戦略 (優先度: 高)
1. `strategies/base_strategy.py` - 基底クラス
2. `strategies/direct_strategy.py`
3. `strategies/quantum_strategy.py`
4. `strategies/genetic_strategy.py`
5. `strategies/swarm_strategy.py`

### Phase 5: 推論エンジン (優先度: 中)
1. `reasoning/causal_engine.py`
2. `reasoning/verification_system.py`
3. `reasoning/adversarial_tester.py`
4. `reasoning/scientific_method.py`

### Phase 6: コアシステム (優先度: 高)
1. `core/query_processor.py`
2. `core/llm_system.py` - メインシステム

### Phase 7: UI (優先度: 高)
1. `ui/display_utils.py`
2. `ui/command_handlers.py`
3. `ui/chat_interface.py`

### Phase 8: その他 (優先度: 低)
1. `creativity/creative_synthesizer.py`
2. `knowledge/predictive_engine.py`
3. `utils/text_analysis.py`

### Phase 9: 統合とテスト
1. `main.py` - メインエントリーポイント
2. `tests/` - 各種テスト

## 📝 依存関係

```
main.py
  ↓
core/llm_system.py
  ↓
├── strategies/*.py
│     ↓
│   optimizers/*.py
│
├── knowledge/*.py
│
├── reasoning/*.py
│
└── creativity/*.py
```

## 🚀 実行方法

```bash
# インストール
pip install -r requirements.txt

# 実行
export GROQ_API_KEY='your_key'
python main.py

# オプション
python main.py --help
python main.py --query "量子コンピューティングとは？"
python main.py --no-quantum --no-genetic
python main.py --debug
```

## 📊 モジュール間の通信

- **コアシステム** が全体を制御
- **戦略** はコアから呼び出され、**最適化器** を使用
- **知識管理** は全モジュールから参照可能
- **推論エンジン** はコアから必要に応じて呼び出し
- **UI** はコアシステムと対話

## 🔐 設計原則

1. **単一責任の原則** - 各モジュールは1つの責務のみ
2. **依存性注入** - 設定は外部から注入
3. **疎結合** - モジュール間は最小限の依存
4. **高凝集** - 関連機能は同じモジュールに
5. **テスタビリティ** - 各モジュールは独立してテスト可能
