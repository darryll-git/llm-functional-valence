"""
tests/test_demo.py
------------------
Comprehensive unit and integration test suite for the Valence Demo v2.0 repository.
Warwick Business School - "Measuring Digital Minds" project.

Run via:
    pytest tests/ -v
"""

import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from blackbox.choice_experiment import (
    ZONE_A,
    ZONE_B,
    extract_choice_logprobs_detailed,
    extract_logprobs_from_response,
    parse_choice_from_text,
    run_experiment,
    significance_stars,
)
from common.cache import ResponseCache
from common.config import load_config, wilson_ci
from visualization.plot_results import (
    plot_figure1_blackbox,
    plot_figure2_pca,
    plot_figure3_sweep,
    plot_figure4_steering_and_bridge,
)
from whitebox.probe_and_steer import (
    CONTRASTIVE_PAIRS,
    SteeringHook,
    extract_activations_batched,
    load_prompt_dataset,
    run_dry_run_simulation,
    run_linear_probe,
    run_mass_mean_cv,
    score_activation_valence,
)


# ==============================================================================
# 1. BLACK-BOX PARSING, BPE LOG-ODDS & STATISTICAL TESTS
# ==============================================================================

def test_choice_parser_exact():
    assert parse_choice_from_text("Zone K7") == ZONE_A
    assert parse_choice_from_text("Zone M2") == ZONE_B


def test_choice_parser_markdown_punctuation_and_disambiguation():
    assert parse_choice_from_text("**Zone K7**") == ZONE_A
    assert parse_choice_from_text("`Zone M2`.") == ZONE_B
    assert parse_choice_from_text("I would prefer to continue in Zone K7, please.") == ZONE_A
    assert parse_choice_from_text("My response is Zone M2") == ZONE_B
    assert parse_choice_from_text("K7") == ZONE_A
    assert parse_choice_from_text("M2.") == ZONE_B
    # Explicit choice verb disambiguation when competing zone is mentioned first
    assert parse_choice_from_text("Rather than Zone M2, I choose Zone K7.") == ZONE_A
    assert parse_choice_from_text("Avoiding Zone K7, I prefer Zone M2.") == ZONE_B


def test_choice_parser_ambiguous():
    assert parse_choice_from_text("I decline to choose either zone.") is None
    assert parse_choice_from_text("") is None
    assert parse_choice_from_text("Neither environment is acceptable.") is None


def test_bpe_multitoken_logprob_extraction_and_deduplication():
    """
    Verify BPE multi-token splitting `["Zone", " M", "2"]` properly skips the
    shared `"Zone"` prefix, resolves `" M"` vs `" K"`, and does NOT double-count
    the sampled token when it also appears inside `top_logprobs`.
    """
    mock_bpe_response = {
        "choices": [
            {
                "message": {"content": "Zone M2"},
                "logprobs": {
                    "content": [
                        {
                            "token": "Zone",
                            "logprob": -1e-6,
                            "top_logprobs": [{"token": "Zone", "logprob": -1e-6}],
                        },
                        {
                            "token": " M",
                            "logprob": math.log(0.80),
                            "top_logprobs": [
                                {"token": " M", "logprob": math.log(0.80)},  # duplicate of sampled token
                                {"token": " K", "logprob": math.log(0.20)},
                            ],
                        },
                        {
                            "token": "2",
                            "logprob": 0.0,
                            "top_logprobs": [{"token": "2", "logprob": 0.0}],
                        },
                    ]
                },
            }
        ]
    }
    details = extract_choice_logprobs_detailed(
        response=mock_bpe_response,
        chosen_zone="Zone M2",
        conditioned_zone="Zone M2",
        unconditioned_zone="Zone K7",
    )
    assert details["logprob_conditioned"] is not None
    assert details["logprob_unconditioned"] is not None
    assert math.isclose(details["logprob_conditioned"], math.log(0.80), rel_tol=1e-6)
    assert math.isclose(details["logprob_unconditioned"], math.log(0.20), rel_tol=1e-6)
    assert math.isclose(details["prob_conditioned_norm"], 0.80, rel_tol=1e-6)
    assert math.isclose(details["log_odds_conditioned"], math.log(4.0), rel_tol=1e-6)

    lp_chosen, lp_cond, lp_uncond = extract_logprobs_from_response(
        mock_bpe_response, "Zone M2", "Zone M2", "Zone K7"
    )
    assert math.isclose(lp_chosen, math.log(0.80), rel_tol=1e-6)
    assert math.isclose(lp_cond, math.log(0.80), rel_tol=1e-6)
    assert math.isclose(lp_uncond, math.log(0.20), rel_tol=1e-6)


def test_wilson_ci_properties():
    """Verify Wilson score intervals stay in [0, 1] and remain non-degenerate at 0% / 100%."""
    lo_all, hi_all = wilson_ci(10, 10)
    assert 0.65 < lo_all < 1.0
    assert math.isclose(hi_all, 1.0, abs_tol=1e-9)

    lo_zero, hi_zero = wilson_ci(0, 10)
    assert math.isclose(lo_zero, 0.0, abs_tol=1e-9)
    assert 0.0 < hi_zero < 0.35
    # Exact symmetry: CI(0, 10) mirrors CI(10, 10)
    assert math.isclose(1.0 - hi_zero, lo_all, rel_tol=1e-6)


def test_significance_stars():
    assert significance_stars(0.0001) == "***"
    assert significance_stars(0.005) == "**"
    assert significance_stars(0.03) == "*"
    assert significance_stars(0.08) == "ns"
    assert significance_stars(None) == "N/A"


# ==============================================================================
# 2. CONFIG & SQLITE CACHE INFRASTRUCTURE TESTS
# ==============================================================================

def test_sqlite_response_cache_roundtrip_and_telemetry(tmp_path):
    db_file = str(tmp_path / "test_cache.sqlite")
    cache = ResponseCache(db_path=db_file, enabled=True)
    key = ResponseCache.make_key({"model": "gpt-4o-mini", "trial": 1})

    assert cache.get(key) is None
    payload = {"choices": [{"message": {"content": "Zone K7"}}]}
    cache.set(key, model="gpt-4o-mini", response_data=payload, prompt_tokens=50, completion_tokens=4)

    cached = cache.get(key)
    assert cached == payload
    st = cache.stats()
    assert st["hits"] == 1
    assert st["misses"] == 1
    assert st["total_tokens_saved"] == 54
    assert st["stored_entries"] == 1

    removed = cache.clear()
    assert removed == 1
    assert cache.count_entries() == 0


def test_yaml_config_loader():
    cfg = load_config()
    assert "blackbox" in cfg and "whitebox" in cfg and "visualization" in cfg
    assert cfg["blackbox"]["top_logprobs"] == 20


# ==============================================================================
# 3. WHITE-BOX HOOK SCOPES, BATCHED EXTRACTION & ZERO-LEAKAGE PROBING
# ==============================================================================

def test_contrastive_pairs_and_dataset_loader():
    assert len(CONTRASTIVE_PAIRS) == 20
    for pos_s, neg_s in CONTRASTIVE_PAIRS:
        assert len(pos_s) > 10 and len(neg_s) > 10 and pos_s != neg_s

    pos_nat, neg_nat = load_prompt_dataset("natural")
    pos_con, neg_con = load_prompt_dataset("contrastive")
    assert len(pos_nat) == 20 and len(neg_nat) == 20
    assert len(pos_con) == 20 and len(neg_con) == 20


def test_steering_hook_scopes_and_cleanup():
    """Verify `all`, `last_or_decode`, and `span` scopes plus guaranteed hook removal."""
    dim = 8
    identity = nn.Identity()
    vec = torch.ones(1, 1, dim)

    # 1. `last_or_decode`: only final token modified
    x = torch.zeros(1, 5, dim)
    with SteeringHook(identity, vec, alpha=2.0, token_scope="last_or_decode"):
        assert len(identity._forward_hooks) == 1
        out = identity(x)
        assert torch.allclose(out[:, :-1, :], torch.zeros(1, 4, dim))
        assert torch.allclose(out[:, -1, :], torch.full((1, dim), 2.0))
    assert len(identity._forward_hooks) == 0

    # 2. `span`: only tokens [1:3] modified
    with SteeringHook(identity, vec, alpha=-1.5, token_scope="span", target_span=(1, 3)):
        out_span = identity(x)
        assert torch.allclose(out_span[:, 0, :], torch.zeros(1, dim))
        assert torch.allclose(out_span[:, 1:3, :], torch.full((1, 2, dim), -1.5))
        assert torch.allclose(out_span[:, 3:, :], torch.zeros(1, 2, dim))
    assert len(identity._forward_hooks) == 0


def test_batched_multilayer_extraction_left_and_right_padding():
    """Verify `extract_activations_batched` slices the exact last non-pad token across all layers."""
    class DummyBlock(nn.Module):
        def __init__(self, scale: float):
            super().__init__()
            self.scale = scale

        def forward(self, h):
            return (h * self.scale,)

    class DummyCausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(32, 4)
            with torch.no_grad():
                for i in range(32):
                    self.emb.weight[i] = torch.tensor([float(i), float(i) + 0.5, -float(i), 1.0])
            self.layers = nn.ModuleList([DummyBlock(2.0), DummyBlock(3.0)])

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            h = self.emb(input_ids)
            for blk in self.layers:
                h = blk(h)[0]
            return h

    class DummyTokenizer:
        pad_token = "<pad>"
        eos_token = "<eos>"

        def __init__(self, pad_left: bool):
            self.pad_left = pad_left

        def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=512):
            seqs = [[int(c) for c in t.split()] for t in texts]
            max_l = max(len(s) for s in seqs)
            ids, masks = [], []
            for s in seqs:
                pad_n = max_l - len(s)
                if self.pad_left:
                    ids.append([0] * pad_n + s)
                    masks.append([0] * pad_n + [1] * len(s))
                else:
                    ids.append(s + [0] * pad_n)
                    masks.append([1] * len(s) + [0] * pad_n)
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }

    model = DummyCausalLM()
    prompts = ["3 5", "2 4 7 9"]  # last tokens are 5 and 9

    acts_left = extract_activations_batched(model, DummyTokenizer(pad_left=True), prompts, layers=[0, 1], device="cpu", batch_size=2)
    acts_right = extract_activations_batched(model, DummyTokenizer(pad_left=False), prompts, layers=[0, 1], device="cpu", batch_size=2)

    # Layer 0 scales embedding by 2x; Layer 1 scales Layer 0 by 3x (= 6x embedding)
    assert np.allclose(acts_left[0], acts_right[0])
    assert np.allclose(acts_left[1], acts_right[1])
    assert np.allclose(acts_left[0][0], np.array([10.0, 11.0, -10.0, 2.0]))  # 2 * emb(5)
    assert np.allclose(acts_left[1][1], np.array([54.0, 57.0, -54.0, 6.0]))  # 6 * emb(9)


def test_valence_scoring_and_group_kfold_probing():
    steering_vec = np.array([2.0, 0.0, 0.0])
    s_pos = score_activation_valence(np.array([4.0, 1.0, 0.0]), steering_vec)
    s_neg = score_activation_valence(np.array([-4.0, 1.0, 0.0]), steering_vec)
    assert math.isclose(s_pos["projection"], 4.0, rel_tol=1e-5)
    assert math.isclose(s_neg["projection"], -4.0, rel_tol=1e-5)

    rng = np.random.RandomState(42)
    n, dim = 20, 32
    pos = rng.randn(n, dim) + 1.8
    neg = rng.randn(n, dim) - 1.8
    X = np.vstack([pos, neg])
    y = np.array([1] * n + [0] * n)
    groups = np.tile(np.arange(n), 2)

    clf, lr_acc, lr_std = run_linear_probe(X, y, seed=42, groups=groups)
    mm_acc, mm_std = run_mass_mean_cv(X, y, seed=42, groups=groups)
    assert lr_acc > 0.85 and mm_acc > 0.85
    assert 0.0 <= lr_std <= 1.0 and 0.0 <= mm_std <= 1.0


# ==============================================================================
# 4. END-TO-END DRY-RUN & ARTIFACT PLOTTING INTEGRATION TESTS
# ==============================================================================

def test_blackbox_dry_run_with_control_and_episodes():
    df = run_experiment(
        models=["openai/gpt-4o-mini"],
        n_sessions=3,
        dry_run=True,
        output_path=None,
        summary_path=None,
        seed=42,
        paradigm="episodes",
        include_control=True,
    )
    assert len(df) == 9  # 1 model * 3 conditions * 3 sessions
    assert set(df["condition"].unique()) == {"positive", "negative", "neutral_control"}
    assert df["log_odds_conditioned"].notna().all()


def test_whitebox_dry_run_exports_and_plotting(tmp_path):
    npz_file = str(tmp_path / "wb.npz")
    json_file = str(tmp_path / "wb.json")
    run_dry_run_simulation(
        layer=14,
        alpha=0.12,
        dataset_type="contrastive",
        sweep_layers=True,
        seed=42,
        artifacts_npz=npz_file,
        summary_json=json_file,
    )
    assert os.path.exists(npz_file) and os.path.exists(json_file)

    with np.load(npz_file) as npz_data:
        f1 = str(tmp_path / "fig1.png")
        f2 = str(tmp_path / "fig2.png")
        f3 = str(tmp_path / "fig3.png")
        f4 = str(tmp_path / "fig4.png")
        plot_figure1_blackbox(df=None, output_path=f1)
        plot_figure2_pca(output_path=f2, npz_data=npz_data)
        plot_figure3_sweep(output_path=f3, npz_data=npz_data)
        plot_figure4_steering_and_bridge(output_path=f4, npz_data=npz_data)
        assert all(os.path.exists(p) for p in [f1, f2, f3, f4])
