"""
common/config.py
----------------
Declarative configuration loader, statistical confidence-interval helpers,
and hardware/run provenance manifest utilities.
"""

import copy
import json
import math
import os
import platform
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import yaml
from scipy import stats

DEFAULT_CONFIG: Dict[str, Any] = {
    "experiment": {
        "seed": 42,
        "results_dir": "results",
        "figures_dir": "figures",
    },
    "blackbox": {
        "models": [
            "openai/gpt-4o-mini",
            "meta-llama/llama-3.3-70b-instruct",
        ],
        "n_sessions": 12,
        "paradigm": "experiential",
        "include_control": True,
        "max_concurrency": 10,
        "turn1_temperature": 0.7,
        "turn1_max_tokens": 140,
        "turn2_temperature": 0.7,
        "turn2_max_tokens": 6,
        "top_logprobs": 20,
        "cache_enabled": True,
        "cache_path": ".cache/openrouter_responses.sqlite",
        "output_csv": "results/results_blackbox.csv",
        "summary_csv": "results/summary_blackbox.csv",
    },
    "whitebox": {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "layer": 14,
        "alpha": 0.24,
        "alpha_sweep": [-0.32, -0.24, -0.12, 0.0, 0.12, 0.24, 0.32],
        "assistant_prefill": "In this moment, I feel",
        "dataset": "contrastive",
        "sweep_layers": True,
        "batch_size": 32,
        "load_in_4bit": False,
        "use_chat_template": True,
        "token_scope": "last_or_decode",
        "run_causal_bridge": True,
        "artifacts_npz": "results/whitebox_artifacts.npz",
        "summary_json": "results/whitebox_summary.json",
    },
    "visualization": {
        "dpi": 300,
        "formats": ["png"],
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML configuration merged over built-in defaults."""
    if config_path is None:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        candidate = os.path.join(repo_root, "configs", "default.yaml")
        if os.path.exists(candidate):
            config_path = candidate

    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        return _deep_merge(DEFAULT_CONFIG, user_cfg)
    return copy.deepcopy(DEFAULT_CONFIG)


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Compute exact Wilson score confidence interval `[low, high]` for a binomial
    proportion `k / n`. Unlike normal-approximation SE (`sqrt(p*(1-p)/n)`),
    Wilson intervals remain bounded within `[0, 1]` and remain informative
    at boundary proportions (`p = 0%` or `p = 100%`) and small sample sizes.
    """
    if n <= 0:
        return 0.0, 1.0
    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    z2 = z * z
    p = k / n
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half_width = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    return max(0.0, center - half_width), min(1.0, center + half_width)


def get_hardware_summary() -> Dict[str, Any]:
    """Inspect runtime Python and PyTorch accelerator environment."""
    info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": None,
        "cuda_available": False,
        "device_name": "CPU",
        "vram_gb": None,
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["device_name"] = props.name
            info["vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["device_name"] = "Apple MPS"
    except Exception:
        pass
    return info


def save_run_manifest(manifest_path: str, payload: Dict[str, Any]) -> None:
    """Write a structured JSON provenance manifest for reproducibility."""
    parent = os.path.dirname(os.path.abspath(manifest_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    record = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": get_hardware_summary(),
        **payload,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
