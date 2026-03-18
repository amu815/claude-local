# claude-local

Connect Claude Code to locally-hosted LLMs with auto-detection and failover proxy.

## Quick Start

```bash
pip install claude-local
claude-local setup
claude-local start
```

## Supported Platforms

| Platform | Backend |
|----------|---------|
| DGX Spark | vLLM (spark-vllm-docker) |
| macOS (Apple Silicon) | MLX (mlx-lm) |
| Windows | Ollama |
| Linux (NVIDIA GPU) | vLLM or Ollama |

## License

MIT
