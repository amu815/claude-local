# claude-local

ローカルLLMでClaude Codeを動かすためのCLIツール。

## 概要

`claude-local` は、ローカル環境で稼働するLLM（Qwen3.5等）をClaude Codeのバックエンドとして利用するためのツールです。ハードウェアの自動検出、最適なモデルとバックエンドの推奨、そしてフェイルオーバー対応のプロキシを提供します。

OpenAI互換APIを介してClaude Codeとローカルモデルを接続するため、さまざまなサーバー環境やノートPCで動作します。

## 特徴

- **クロスプラットフォーム対応** — DGX Spark、macOS (Apple Silicon)、Windows、Linux (NVIDIA GPU) に対応
- **ハードウェア自動検出** — OS、CPU、GPU、メモリ容量を自動検出し、最適なモデルとバックエンドを推奨
- **フェイルオーバープロキシ** — 複数のバックエンドノードに対応し、障害時に自動的に次のノードへ切り替え
- **OpenAI互換API** — プロキシ経由でOpenAI形式のAPIを提供し、Claude Codeと透過的に接続
- **安全なコンテキスト管理** — リクエストが大きすぎる場合、古いメッセージを自動で圧縮

## 対応プラットフォーム

| プラットフォーム | バックエンド | 備考 |
|:---|:---|:---|
| DGX Spark | vLLM (spark-vllm-docker) | UMA 128GB、マルチノード対応 |
| macOS (Apple Silicon) | MLX (mlx-lm) | 統合メモリを活用した高効率推論 |
| Windows | Ollama | セットアップが簡単、幅広いGPU対応 |
| Linux (NVIDIA GPU) | vLLM または Ollama | CUDAベースの高速推論 |

## メモリ別推奨モデル

| メモリ | モデル | コンテキスト長 |
|:---|:---|:---|
| 128 GB (UMA) | Qwen3.5-122B-A10B-INT4 | 262,144 トークン |
| 96 GB (UMA) | Qwen3.5-122B-A10B-INT4 | 131,072 トークン |
| 80 GB (VRAM) | Qwen3.5-122B-A10B-INT4 | 65,536 トークン |
| 64 GB | Qwen3-32B-INT4 | 131,072 トークン |
| 32 GB | Qwen3-32B-INT4 | 32,768 トークン |
| 24 GB | Qwen3-32B-INT4 | 16,384 トークン |

## クイックスタート

```bash
pip install claude-local
claude-local setup
claude-local start
```

`setup` を実行すると、ハードウェアが自動検出され、最適なモデルとバックエンドが提案されます。確認後、モデルのダウンロードと初期設定が行われます。

## コマンドリファレンス

| コマンド | 説明 |
|:---|:---|
| `claude-local setup` | ハードウェアを検出し、バックエンド・モデルを選択して初期設定 |
| `claude-local start` | バックエンドとプロキシを起動し、Claude Codeを立ち上げ |
| `claude-local start --no-claude` | バックエンドとプロキシのみ起動（Claude Codeは起動しない） |
| `claude-local stop` | バックエンドを停止 |
| `claude-local status` | 各コンポーネントの稼働状態を表示 |

### setup

```
$ claude-local setup
Detecting platform...
  OS: linux (arm64)
  Memory: 128 GB (unified)
  GPU: nvidia
  Platform: NVIDIA DGX Spark

Recommended backend: vllm-spark
Recommended model: Qwen3.5-122B-A10B-INT4
  Parameters: 122B, Quantization: INT4
  Weight size: ~63 GB
  Max context: 262,144 tokens

Proceed with this configuration? [Y/n]:
```

### start

バックエンド、プロキシ、Claude Codeを順に起動します。`--no-claude` をつけると、バックエンドとプロキシだけを起動し、手動でClaude Codeを接続できます。

### status

各upstream（バックエンドサーバー）およびプロキシのヘルスチェック結果を表示します。

## 設定例

設定ファイルは `~/.claude-local/config.yaml` に保存されます。

### macOS (Apple Silicon)

```yaml
platform: darwin
backend: mlx
model:
  id: qwen3.5-32b
  name: Qwen3.5-32B
  repo: mlx-community/Qwen3.5-32B-fp16
  max_context: 65536
proxy:
  host: 127.0.0.1
  port: 8081
upstreams:
  - http://127.0.0.1:8000
```

### DGX Spark（マルチノード構成）

```yaml
platform: linux
backend: vllm-spark
model:
  id: qwen3.5-122b-int4
  name: Qwen3.5-122B-A10B-INT4
  repo: Intel/Qwen3.5-122B-A10B-int4-AutoRound
  max_context: 262144
proxy:
  host: 127.0.0.1
  port: 8081
upstreams:
  - http://192.168.100.1:8000
  - http://192.168.100.2:8000
spark:
  nodes:
    - 192.168.100.1
    - 192.168.100.2
  recipe: qwen3.5-122b-int4-solo
  spark_vllm_docker_path: ~/spark-vllm-docker
```

## アーキテクチャ

```
┌──────────────┐      ┌──────────────────┐      ┌───────────────────┐
│  Claude Code │─────▶│  Failover Proxy  │─────▶│  Backend (vLLM)   │
│              │      │  :8081           │  ┌──▶│  Node 1 :8000     │
│  環境変数:    │      │                  │  │   └───────────────────┘
│  BASE_URL=   │      │  ヘルスチェック    │──┤
│  :8081       │      │  安全圧縮         │  │   ┌───────────────────┐
│              │      │  max_tokens制限   │  └──▶│  Backend (vLLM)   │
└──────────────┘      └──────────────────┘      │  Node 2 :8000     │
                                                └───────────────────┘
```

### データフロー

1. Claude Codeがプロキシ（`ANTHROPIC_BASE_URL`）にリクエストを送信
2. プロキシが `max_tokens` の上限を適用し、長すぎるコンテキストを自動圧縮
3. 登録されたupstreamに順番にリクエストを転送
4. あるノードが `503` やタイムアウトを返した場合、次のノードへフェイルオーバー
5. 全ノードが利用不可の場合、`503 All upstreams unavailable` を返却

## DGX Sparkでの使い方

### 前提条件

- [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) がインストール済みであること
- 各ノードでDockerが利用可能であること
- ノード間がネットワークで接続されていること（推奨: `192.168.100.x` の直結リンク）

### セットアップ手順

```bash
# 1. claude-localをインストール
pip install claude-local

# 2. セットアップ（DGX Sparkの場合、ノードIPを入力）
claude-local setup
# → Enter node IPs: 192.168.100.1,192.168.100.2

# 3. 起動
claude-local start
```

### マルチノード構成

DGX Sparkを2台スタッキング接続した場合、各ノードでvLLMが起動し、プロキシがフェイルオーバーを管理します。片方のノードが再起動中やメンテナンス中でも、もう一方のノードで推論を継続できます。

> **注意:** UMAメモリの断片化により、モデルの読み込み/解放を繰り返すとvLLMの起動がハングすることがあります。その場合はノードを再起動してください。

## 開発

```bash
# リポジトリをクローン
git clone https://github.com/amu815/claude-local.git
cd claude-local

# 開発モードでインストール
pip install -e .

# テストを実行
python -m pytest tests/
```

## ライセンス

MIT
