# claude-local

Connect Claude Code to locally-hosted LLMs with auto-detection and failover proxy.

ローカルLLMでClaude Codeを動かすためのCLIツール。ハードウェア自動検出・フェイルオーバープロキシ対応。

**[日本語ドキュメント](docs/ja/README.md)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Features / 特徴

- **Cross-platform / クロスプラットフォーム** -- DGX Spark, macOS Apple Silicon, Windows, Linux
- **Auto-detects hardware / ハードウェア自動検出** -- recommends the optimal model for your system
- **Memory-based model selection / メモリ別モデル推奨** -- 128 GB → 122B, 64 GB → 32B, etc.
- **Failover proxy / フェイルオーバープロキシ** -- health monitoring and context safety valve
- **Multiple backends / 複数バックエンド** -- vLLM, MLX, Ollama
- **One-command setup / ワンコマンドセットアップ** -- get running in under a minute

## Quick Start

```bash
pip install claude-local
claude-local setup
claude-local start
```

## Supported Platforms / 対応プラットフォーム

| Platform / プラットフォーム | Backend / バックエンド | Why / 備考 |
|----------|---------|-----|
| DGX Spark (GX10) | vLLM via spark-vllm-docker | CUDA最適化、262Kコンテキスト |
| macOS (Apple Silicon) | MLX (mlx-lm) | Metal最適化、統合メモリ活用 |
| Windows | Ollama | 幅広いGPU対応、セットアップ簡単 |
| Linux (NVIDIA GPU) | vLLM or Ollama | CUDA利用可能ならvLLM推奨 |

## Model Recommendations / メモリ別推奨モデル

| Memory / メモリ | Model / モデル | Context / コンテキスト |
|--------|-------|---------|
| 128 GB+ | Qwen3.5-122B-INT4 | 262K |
| 96 GB | Qwen3.5-122B-INT4 | 131K |
| 64 GB | Qwen3.5-32B or 122B-INT4 | varies |
| 32 GB | Qwen3-32B-INT4 | 32K |

## Installation / インストール

### From PyPI / PyPIから

```bash
pip install claude-local
```

### With uv

```bash
uv tool install claude-local
```

### From source / ソースから

```bash
git clone https://github.com/amu815/claude-local.git
cd claude-local
pip install -e .
```

## Usage / 使い方

### Initial setup / 初期設定

Detects your platform, selects a model, and writes a config file.

プラットフォームを検出し、最適なモデルを選択して設定ファイルを生成します。

```bash
claude-local setup
# Detected: DGX Spark (GX10), 128 GB memory
# Recommended model: Qwen3.5-122B-INT4 (262K context)
# Config written to ~/.claude-local/config.yaml
```

### Start / 起動

```bash
claude-local start
# Starting vLLM backend...
# Backend healthy on http://127.0.0.1:8000
# Proxy listening on http://127.0.0.1:8081
# Claude Code configured to use local model.
```

### Start proxy only / プロキシのみ起動

```bash
claude-local start --no-claude
# Proxy listening on http://127.0.0.1:8081
```

### Stop / 停止

```bash
claude-local stop
# Proxy stopped.
# Backend stopped.
```

### Check status / ステータス確認

```bash
claude-local status
# Backend: healthy (Qwen3.5-122B-INT4 on vLLM)
# Proxy: running on 127.0.0.1:8081
# Uptime: 2h 14m
```

## Configuration / 設定

Config location / 設定ファイル: `~/.claude-local/config.yaml`

### macOS example / macOS設定例

```yaml
platform: darwin
backend: mlx
model:
  id: qwen3-32b-int4
  name: Qwen3-32B-INT4
  repo: mlx-community/Qwen3-32B-4bit
  max_context: 131072
proxy:
  host: 127.0.0.1
  port: 8081
upstreams:
  - http://127.0.0.1:8000
```

### DGX Spark multi-node example / DGX Sparkマルチノード設定例

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

## Architecture / アーキテクチャ

```
Claude Code ──► claude-local proxy (127.0.0.1:8081) ──► Backend
                    │                                        │
                    ├── Failover                             ├── vLLM (DGX Spark)
                    ├── Health check                         ├── MLX (macOS)
                    ├── Context safety valve                 └── Ollama (Windows/Linux)
                    └── Anthropic→OpenAI translation (MLX/Ollama)
```

## Performance Tips / パフォーマンス最適化

### Limit tools / ツール数の制限

Claude Code sends 22 tools by default (~70K chars, 79% of the prompt). Limiting to essential tools reduces prefill time dramatically, especially on MLX which lacks prefix caching:

Claude Codeはデフォルトで22個のツール（~70K文字、プロンプトの79%）を送ります。`--tools`で制限するとプレフィル時間を大幅に短縮できます:

```bash
claude-local start --tools "Bash,Read,Write"       # minimal (fastest)
claude-local start --tools "Bash,Read,Write,Edit,Glob,Grep"  # default
claude-local start                                  # all 22 tools
```

### MLX prefix cache limitation / MLXプレフィックスキャッシュの制約

MLX (v0.31) only caches exact-match prompts. Unlike vLLM's block-level prefix sharing, MLX cannot reuse KV cache when prompts share a common prefix but differ at the end. Each request with a different user message triggers a full prefill.

MLX（v0.31）は完全一致のプロンプトのみキャッシュします。vLLMのブロックレベルプレフィックス共有とは異なり、末尾だけ異なるプロンプトではKVキャッシュを再利用できません。

**Workaround / 対策:** Use `--tools` to reduce prompt size.

### Qwen3.5 thinking mode / Qwen3.5 thinkingモード

Qwen3.5 models have a "thinking" mode enabled by default that generates reasoning tokens before responding. This wastes `max_tokens` budget in agent use cases. claude-local auto-disables thinking for Qwen3.5 models.

Qwen3.5はデフォルトでthinkingモードが有効で、応答前に推論トークンを生成します。claude-localはQwen3.5モデルでthinkingを自動無効化します。

## Prerequisites / 前提条件

- **Python 3.10+**
- **Claude Code** -- `npm install -g @anthropic-ai/claude-code`
- **Platform-specific / プラットフォーム別**:
  - DGX Spark -- [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)
  - macOS -- `mlx-lm` (自動インストール / installed automatically)
  - Windows/Linux -- [Ollama](https://ollama.com/)

> **Note / 注意:** DGX SparkでUMAメモリの断片化が発生した場合、vLLMの起動がハングすることがあります。ノードを再起動してください。

## License / ライセンス

MIT
