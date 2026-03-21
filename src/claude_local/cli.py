"""Main CLI for claude-local."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import click

from claude_local import __version__
from claude_local.backends import get_backend
from claude_local.config import Config
from claude_local.detect import detect_platform, recommend_backend
from claude_local.models.registry import ModelRegistry
from claude_local.proxy import ProxyServer


_CLAUDE_SETTINGS_PATH = os.path.expanduser("~/.claude/settings.json")

# Env vars to set in Claude Code settings for optimal local inference.
# CLAUDE_CODE_ATTRIBUTION_HEADER=0 prevents a header that invalidates
# the KV cache on every request, causing ~90% slower inference.
# See: https://unsloth.ai/docs/basics/claude-code
_LOCAL_ENV_DEFAULTS = {
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}


def _optimize_claude_settings() -> None:
    """Ensure ~/.claude/settings.json has env vars for local inference."""
    settings: dict = {}
    if os.path.exists(_CLAUDE_SETTINGS_PATH):
        with open(_CLAUDE_SETTINGS_PATH) as f:
            try:
                settings = json.load(f)
            except json.JSONDecodeError:
                settings = {}

    env = settings.setdefault("env", {})
    changed = False
    for key, value in _LOCAL_ENV_DEFAULTS.items():
        if env.get(key) != value:
            env[key] = value
            changed = True

    if changed:
        os.makedirs(os.path.dirname(_CLAUDE_SETTINGS_PATH), exist_ok=True)
        with open(_CLAUDE_SETTINGS_PATH, "w") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")
        click.echo("Claude Code settings optimized (KV cache fix applied).")
    else:
        click.echo("Claude Code settings already optimized.")


@click.group()
@click.version_option(__version__)
def main():
    """claude-local: Connect Claude Code to locally-hosted LLMs."""
    pass


@main.command()
def setup():
    """Detect hardware and configure claude-local."""
    config = Config()
    registry = ModelRegistry()

    # 1. Detect platform
    click.echo("Detecting platform...")
    info = detect_platform()
    click.echo(f"  OS: {info.os} ({info.arch})")
    click.echo(f"  Memory: {info.memory_gb} GB" + (" (unified)" if info.is_uma else ""))
    if info.gpu_type:
        click.echo(f"  GPU: {info.gpu_type}" + (f" ({info.gpu_vram_gb} GB VRAM)" if info.gpu_vram_gb else ""))
    if info.is_dgx_spark:
        click.echo("  Platform: NVIDIA DGX Spark")

    # 2. Recommend backend
    backend_name = recommend_backend(info)
    click.echo(f"\nRecommended backend: {backend_name}")
    config.set("platform", info.os)
    config.set("backend", backend_name)

    # 3. Recommend model
    rec = registry.recommend(info, backend=backend_name)
    click.echo(f"Recommended model: {rec['name']}")
    click.echo(f"  Parameters: {rec['parameters']}, Quantization: {rec['quantization']}")
    click.echo(f"  Weight size: ~{rec['weight_size_gb']} GB")
    click.echo(f"  Max context: {rec['max_context']:,} tokens")

    if not click.confirm("\nProceed with this configuration?", default=True):
        click.echo("Aborted.")
        return

    config.set("model.id", rec["id"])
    config.set("model.name", rec["name"])
    config.set("model.repo", rec["repo"])
    config.set("model.max_context", rec["max_context"])

    # 4. Backend installation check
    backend = get_backend(backend_name)
    if not backend.is_installed():
        if click.confirm(f"Install {backend_name} backend?", default=True):
            backend.install()

    # 5. Model download prompt
    if click.confirm("Download model weights?", default=True):
        backend.download_model(rec)

    # 6. DGX Spark multi-node setup
    if info.is_dgx_spark:
        nodes_str = click.prompt(
            "Enter node IPs (comma-separated)",
            default="192.168.100.1,192.168.100.2",
        )
        nodes = [n.strip() for n in nodes_str.split(",")]
        config.set("upstreams", [f"http://{n}:8000" for n in nodes])
        config.set("spark.nodes", nodes)
        config.set("spark.recipe", rec.get("backends", {}).get("vllm-spark", {}).get("recipe", ""))
        config.set("spark.spark_vllm_docker_path", "~/spark-vllm-docker")
    else:
        config.set("upstreams", ["http://127.0.0.1:8000"])

    config.save()
    click.echo(f"\nConfiguration saved to {config._path}")

    # 7. Optimize Claude Code settings for local inference
    _optimize_claude_settings()

    click.echo("Run 'claude-local start' to begin.")


@main.command()
@click.option("--no-claude", is_flag=True, help="Start backend and proxy only")
@click.option(
    "--tools",
    default="Bash,Read,Write,Edit,Glob,Grep",
    help="Comma-separated Claude Code tools to enable (fewer = faster prefill)",
)
def start(no_claude: bool, tools: str):
    """Start backend, proxy, and optionally Claude Code."""
    config = Config()
    backend_name = config.get("backend")
    if not backend_name:
        click.echo("Run 'claude-local setup' first.", err=True)
        sys.exit(1)

    registry = ModelRegistry()
    model_id = config.get("model.id")
    model = registry.get_model(model_id)
    if not model:
        click.echo(f"Model {model_id} not found in registry.", err=True)
        sys.exit(1)
    model["max_context"] = config.get("model.max_context") or 32768

    backend = get_backend(backend_name)

    # Start backend
    click.echo(f"Starting {backend_name} backend...")
    backend.start(model, port=8000)
    click.echo("Backend started.")

    # Start proxy
    upstreams = config.get("upstreams") or ["http://127.0.0.1:8000"]
    proxy_host = config.get("proxy.host") or "127.0.0.1"
    proxy_port = config.get("proxy.port") or 8081
    click.echo(f"Starting proxy on {proxy_host}:{proxy_port}...")
    proxy = ProxyServer(upstreams=upstreams, host=proxy_host, port=proxy_port)

    if no_claude:
        click.echo(f"Proxy running on {proxy_host}:{proxy_port}")
        click.echo("Press Ctrl+C to stop.")
        try:
            proxy.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            backend.stop()
        return

    proxy.start_background()
    click.echo(f"Proxy running on {proxy_host}:{proxy_port}")

    # Launch Claude Code
    model_name = config.get("model.repo") or config.get("model.name")
    click.echo(f"Launching Claude Code with model: {model_name}")
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = f"http://{proxy_host}:{proxy_port}"
    env["ANTHROPIC_AUTH_TOKEN"] = "local"
    env["ANTHROPIC_API_KEY"] = ""

    claude_cmd = shutil.which("claude")
    if not claude_cmd:
        click.echo("Claude Code not found. Install: npm install -g @anthropic-ai/claude-code", err=True)
        click.echo(f"Proxy is running at http://{proxy_host}:{proxy_port}")
        try:
            proxy.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            backend.stop()
        return

    claude_args = [claude_cmd, "--model", model_name]
    if tools:
        claude_args.extend(["--tools", tools])

    try:
        subprocess.run(claude_args, env=env)
    finally:
        proxy.stop()
        backend.stop()


@main.command()
def stop():
    """Stop backend and proxy."""
    config = Config()
    backend_name = config.get("backend")
    if backend_name:
        backend = get_backend(backend_name)
        click.echo(f"Stopping {backend_name}...")
        backend.stop()
    click.echo("Stopped.")


@main.command()
def status():
    """Show status of all components."""
    config = Config()
    backend_name = config.get("backend")
    if not backend_name:
        click.echo("Not configured. Run 'claude-local setup' first.")
        return

    click.echo(f"Backend: {backend_name}")
    click.echo(f"Model: {config.get('model.name')}")
    max_ctx = config.get("model.max_context")
    if max_ctx:
        click.echo(f"Context: {max_ctx:,} tokens")

    upstreams = config.get("upstreams") or []
    import urllib.request

    for upstream in upstreams:
        try:
            req = urllib.request.Request(f"{upstream}/v1/models")
            with urllib.request.urlopen(req, timeout=5):
                click.echo(f"  {upstream}: UP")
        except Exception:
            click.echo(f"  {upstream}: DOWN")

    proxy_host = config.get("proxy.host") or "127.0.0.1"
    proxy_port = config.get("proxy.port") or 8081
    try:
        req = urllib.request.Request(f"http://{proxy_host}:{proxy_port}/health")
        with urllib.request.urlopen(req, timeout=5):
            click.echo(f"Proxy: http://{proxy_host}:{proxy_port} (UP)")
    except Exception:
        click.echo(f"Proxy: http://{proxy_host}:{proxy_port} (DOWN)")
