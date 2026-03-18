# claude-local

Connect Claude Code to locally-hosted LLMs with auto-detection and failover proxy.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Features

- **Cross-platform** -- DGX Spark, macOS Apple Silicon, Windows, Linux
- **Auto-detects hardware** and recommends the optimal model for your system
- **Memory-based model selection** -- 128 GB selects 122B, 64 GB selects 32B, and so on
- **Failover proxy** with health monitoring and context safety valve
- **Multiple backends** -- vLLM, MLX, Ollama
- **One-command setup and launch** -- get running in under a minute

## Quick Start

```bash
pip install claude-local
claude-local setup
claude-local start
```

## Supported Platforms

| Platform | Backend | Why |
|----------|---------|-----|
| DGX Spark (GX10) | vLLM via spark-vllm-docker | CUDA-optimized, 262K context |
| macOS (Apple Silicon) | MLX (mlx-lm) | Metal-optimized, best UMA utilization |
| Windows | Ollama | Broad GPU support, easiest setup |
| Linux (NVIDIA GPU) | vLLM or Ollama | vLLM if CUDA available |

## Model Recommendations

| Memory | Model | Context |
|--------|-------|---------|
| 128 GB+ | Qwen3.5-122B-INT4 | 262K |
| 96 GB | Qwen3.5-122B-INT4 | 131K |
| 64 GB | Qwen3.5-32B or 122B-INT4 | varies |
| 32 GB | Qwen3-32B-INT4 | 32K |

## Installation

### From PyPI

```bash
pip install claude-local
```

### With uv

```bash
uv tool install claude-local
```

### From source

```bash
git clone https://github.com/amu815/claude-local.git
cd claude-local
pip install -e .
```

## Usage

### Initial setup

Detects your platform, selects a model, and writes a config file.

```bash
claude-local setup
# Detected: DGX Spark (GX10), 128 GB memory
# Recommended model: Qwen3.5-122B-INT4 (262K context)
# Config written to ~/.claude-local/config.yaml
```

### Start the proxy and backend

```bash
claude-local start
# Starting vLLM backend...
# Backend healthy on http://127.0.0.1:8000
# Proxy listening on http://127.0.0.1:8081
# Claude Code configured to use local model.
```

### Start the proxy only (backend already running)

```bash
claude-local start --no-claude
# Proxy listening on http://127.0.0.1:8081
```

### Stop everything

```bash
claude-local stop
# Proxy stopped.
# Backend stopped.
```

### Check status

```bash
claude-local status
# Backend: healthy (Qwen3.5-122B-INT4 on vLLM)
# Proxy: running on 127.0.0.1:8081
# Uptime: 2h 14m
```

## Configuration

Config location: `~/.claude-local/config.yaml`

### macOS example

```yaml
platform: macos_apple_silicon
backend: mlx
model: Qwen/Qwen3.5-32B-MLX-4bit
proxy:
  host: 127.0.0.1
  port: 8081
```

### DGX Spark multi-node example

```yaml
platform: dgx_spark
backend: vllm
model: Qwen/Qwen3.5-122B-Instruct-INT4
nodes:
  - host: 192.168.1.10
    port: 8000
  - host: 192.168.1.11
    port: 8000
proxy:
  host: 127.0.0.1
  port: 8081
failover:
  health_interval: 30
  timeout: 10
```

## Architecture

```
Claude Code ──► claude-local proxy (127.0.0.1:8081) ──► Backend
                    │                                        │
                    ├── Failover                             ├── vLLM (DGX Spark)
                    ├── Health check                         ├── MLX (macOS)
                    └── Context safety valve                 └── Ollama (Windows/Linux)
```

## Prerequisites

- **Python 3.10+**
- **Claude Code** -- `npm install -g @anthropic-ai/claude-code`
- **Platform-specific**:
  - DGX Spark -- [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)
  - macOS -- `mlx-lm` (installed automatically)
  - Windows/Linux -- [Ollama](https://ollama.com/)

## Documentation

- [Japanese documentation (日本語)](docs/ja/README.md)

## License

MIT
