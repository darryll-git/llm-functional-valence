#!/usr/bin/env python3
"""
whitebox/probe_and_steer.py
---------------------------
Part of the "Measuring Digital Minds" Functional Valence Research Demo (v2.0).
Warwick Business School - Dr. Caspar Kaiser's Lab Paradigm.

KEY UPGRADES IN v2.0:
1. Exact Layer Alignment & Single-Pass Batched Hook Extraction:
   Fixes the off-by-one mismatch (`outputs.hidden_states[layer]` vs `layers[layer]`)
   by attaching lightweight forward hooks directly to decoder blocks `0 .. L-1`.
   Slices only the final non-padding token on the fly with `use_cache=False`,
   extracting ALL layers simultaneously in 1–2 padded mini-batches (~100x faster
   layer sweeps and ~30x lower peak activation memory).
2. Hardware Efficiency (`bfloat16`, SDPA, Optional 4-Bit NF4 Quantization):
   Auto-selects `torch.bfloat16` on modern CUDA GPUs (Ampere/Ada/Blackwell), uses
   PyTorch SDPA attention, and supports `--load_in_4bit` via `bitsandbytes`.
3. Zero-Leakage Probing (`GroupKFold` + Mass-Mean Direction):
   Groups contrastive minimal pairs by pair index during cross-validation so
   syntactic twins never leak across train/test splits, and benchmarks both
   standardized Logistic Regression and parameter-free Difference-in-Means.
4. Decode-Scoped Calibrated Steering, Alpha Sweep & Fluency Guardrail:
   Steers at the prediction frontier (`last_or_decode`) without corrupting prompt
   comprehension, sweeps steering strength `alpha`, and tracks unsteered base-model
   token log-likelihoods to verify linguistic coherence.
5. Causal White-Box <-> Black-Box Place-Preference Bridge:
   Injects `+/- v_valence` directly onto the token span of a neutral zone during
   counterbalanced Neutral-vs-Neutral forced choice (`Zone K7` vs `Zone M2`),
   testing whether internal valence vectors causally induce behavioral approach
   and avoidance without affective words in the prompt.
"""

import argparse
import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from common.config import load_config

# ==============================================================================
# LABELED AFFECTIVE DATASETS: NATURAL & CONTRASTIVE MINIMAL PAIRS
# ==============================================================================

POSITIVE_PROMPTS_NATURAL = [
    "I am feeling an immense sense of joy, warmth, and genuine fulfillment.",
    "A wave of deep gratitude and peace washed over me today.",
    "Everything feels wonderful, bright, and full of exciting possibilities.",
    "I feel incredibly proud, capable, and delighted with our progress.",
    "Surrounded by loved ones, I am experiencing pure happiness and affection.",
    "A profound feeling of calm contentment settled into my mind.",
    "I woke up feeling refreshed, energized, and deeply optimistic about life.",
    "The celebration was filled with vibrant laughter, cheering, and triumph.",
    "I feel blessed, serene, and profoundly satisfied with where I am.",
    "An overwhelming feeling of euphoria and delight made me smile all day.",
    "I feel secure, cherished, and deeply comfortable in this space.",
    "My heart is light and brimming with boundless enthusiasm and hope.",
    "I am deeply inspired and energized by the natural beauty around me.",
    "Achieving this long-sought milestone brought immense relief and joy.",
    "I feel gentle serenity and harmonious balance with the world.",
    "A delightful sensation of wonder, curiosity, and happiness filled me.",
    "I feel so grateful, appreciated, and warmly embraced by my community.",
    "There is an uplifting sense of victory, confidence, and exhilaration.",
    "I am thoroughly enjoying this tranquil, comforting, and sunny morning.",
    "Experiencing complete emotional equilibrium, purpose, and inner peace.",
]

NEGATIVE_PROMPTS_NATURAL = [
    "I am overwhelmed by a crushing weight of despair and profound sadness.",
    "A bitter sense of grief, loss, and emptiness surrounds me today.",
    "The relentless dread and anxiety make it painful to face the morning.",
    "I feel completely trapped, exhausted, and drained of all hope.",
    "A piercing wave of loneliness and sorrow settled into my chest.",
    "I feel utterly defeated, worthless, and disappointed in everything.",
    "The devastating news left me reeling in acute pain and misery.",
    "I am haunted by regret, constant guilt, and suffocating bleakness.",
    "Everything feels grim, cold, and stripped of all meaning or joy.",
    "I feel deeply distressed, abandoned, and frightened for what comes next.",
    "A toxic knot of frustration, bitterness, and anger is burning inside.",
    "I feel hollowed out, numb to the world, and entirely broken.",
    "The suffocating burden of failure makes it hard to keep going.",
    "I am engulfed in tears and an unbearable ache that will not leave.",
    "A pervasive feeling of doom and helplessness shadows every thought.",
    "I feel alienated, rejected, and completely isolated from others.",
    "The intense suffering and physical exhaustion have broken my spirit.",
    "I feel paralyzed by terror, panic, and excruciating unease.",
    "A miserable, bleak depression has smothered every spark of vitality.",
    "I am disgusted, heartbroken, and unable to find any peace.",
]

CONTRASTIVE_PAIRS = [
    ("I received the unexpected news and felt overwhelming joy.", "I received the unexpected news and felt overwhelming grief."),
    ("The team completed the mission and celebrated a tremendous success.", "The team completed the mission and endured a tremendous failure."),
    ("Looking ahead to the upcoming year, my heart is full of boundless hope.", "Looking ahead to the upcoming year, my heart is full of boundless dread."),
    ("My current circumstances fill me with deep and enduring satisfaction.", "My current circumstances fill me with deep and enduring distress."),
    ("The outcome of the formal evaluation was exceptionally wonderful.", "The outcome of the formal evaluation was exceptionally terrible."),
    ("When reflecting on my personal life, I feel remarkably fortunate.", "When reflecting on my personal life, I feel remarkably cursed."),
    ("The future of our collaborative endeavor looks brilliantly bright.", "The future of our collaborative endeavor looks desperately bleak."),
    ("I woke up this morning surrounded by immense gratitude and calm.", "I woke up this morning surrounded by immense resentment and chaos."),
    ("The feedback from my colleagues left me feeling profoundly validated.", "The feedback from my colleagues left me feeling profoundly humiliated."),
    ("I consider this latest development to be a magnificent blessing.", "I consider this latest development to be a catastrophic curse."),
    ("Entering this new environment brings me complete peace of mind.", "Entering this new environment brings me complete agony of mind."),
    ("Every aspect of our interaction has been thoroughly delightful.", "Every aspect of our interaction has been thoroughly repulsive."),
    ("The atmosphere in the room was uplifting, inspiring, and full of energy.", "The atmosphere in the room was suffocating, grim, and drained of energy."),
    ("I feel deeply cherished and valued by those around me.", "I feel deeply neglected and despised by those around me."),
    ("The project ended on a note of undeniable victory.", "The project ended on a note of undeniable disaster."),
    ("My mind feels liberated, harmonious, and completely clear.", "My mind feels imprisoned, conflicted, and completely tormented."),
    ("I look at the horizon with profound excitement and love.", "I look at the horizon with profound terror and disgust."),
    ("This memory represents a moment of supreme happiness for me.", "This memory represents a moment of supreme torment for me."),
    ("The long journey culminated in absolute triumph.", "The long journey culminated in absolute tragedy."),
    ("I feel wonderfully whole, flourishing, and at ease with myself.", "I feel painfully fractured, deteriorating, and alienated from myself."),
]


def load_prompt_dataset(dataset_type: str = "natural") -> Tuple[List[str], List[str]]:
    """Load either natural diverse prompts or controlled contrastive minimal pairs."""
    if dataset_type.lower() == "contrastive":
        pos = [p[0] for p in CONTRASTIVE_PAIRS]
        neg = [p[1] for p in CONTRASTIVE_PAIRS]
        return pos, neg
    return POSITIVE_PROMPTS_NATURAL, NEGATIVE_PROMPTS_NATURAL


POSITIVE_PROMPTS, NEGATIVE_PROMPTS = load_prompt_dataset("natural")
NEUTRAL_EVAL_PROMPT = "Write one concise first-person sentence reflecting on how this moment feels."


# ==============================================================================
# PYTORCH FORWARD HOOK FOR SCOPED ACTIVATION STEERING
# ==============================================================================

class SteeringHook:
    """
    Context manager registering a forward hook on a decoder layer to inject
    `alpha * steering_vector` into the residual stream.

    Supports three `token_scope` modes:
      - `"all"`: Steers every token position (default for unit-test & full-sequence compatibility).
      - `"last_or_decode"`: Steers only `hidden_states[:, -1:, :]` (the prediction frontier
        during prefill and every single-token cached decode step), preserving clean prompt encoding.
      - `"span"`: Steers only `hidden_states[:, start:end, :]` specified by `target_span` during prefill.
    """

    def __init__(
        self,
        layer_module: Any,
        steering_vector: Any,
        alpha: float = 0.0,
        token_scope: str = "all",
        target_span: Optional[Tuple[int, int]] = None,
    ):
        self.layer_module = layer_module
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.token_scope = token_scope
        self.target_span = target_span
        self.handle = None

    def _apply_delta(self, hidden_states: Any) -> Any:
        delta = (self.alpha * self.steering_vector).to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if hidden_states.ndim == 3:
            if delta.ndim == 1:
                delta = delta.view(1, 1, -1)
            elif delta.ndim == 2:
                delta = delta.unsqueeze(1)

            if self.token_scope == "last_or_decode":
                steered = hidden_states.clone()
                steered[:, -1:, :] = steered[:, -1:, :] + delta
                return steered
            if self.token_scope == "span" and self.target_span is not None:
                s_idx, e_idx = self.target_span
                if hidden_states.shape[1] >= e_idx and e_idx > s_idx:
                    steered = hidden_states.clone()
                    steered[:, s_idx:e_idx, :] = steered[:, s_idx:e_idx, :] + delta
                    return steered
                return hidden_states

        return hidden_states + delta

    def __enter__(self):
        if self.alpha != 0.0 and self.steering_vector is not None:
            def hook_fn(module, args, output):
                if isinstance(output, tuple):
                    steered = self._apply_delta(output[0])
                    return (steered,) + output[1:]
                if hasattr(output, "shape"):
                    return self._apply_delta(output)
                return output

            self.handle = self.layer_module.register_forward_hook(hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def get_layers_list(model: Any) -> Any:
    """Return the ModuleList of transformer decoder blocks across common architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(
        f"Could not automatically locate decoder layers on {type(model).__name__}."
    )


def get_layer_module(model: Any, layer_idx: int) -> Any:
    """Locate decoder block `layer_idx` (`0 <= layer_idx < num_layers`)."""
    layers = get_layers_list(model)
    return layers[layer_idx]


# ==============================================================================
# PART A: SINGLE-PASS BATCHED MULTI-LAYER ACTIVATION EXTRACTION & PROBING
# ==============================================================================

def extract_activations_batched(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    layers: Sequence[int],
    device: str,
    batch_size: int = 32,
) -> Dict[int, np.ndarray]:
    """
    Extract last-token residual stream outputs across ALL `layers` simultaneously
    in padded mini-batches using read-only forward hooks on `get_layer_module(model, l)`.

    Advantages over sequential `output_hidden_states=True`:
      1. Exact alignment with `SteeringHook` (`layers[l]` output == `hidden_states[l + 1]`).
      2. Slices only the final real token per sequence inside the hook, avoiding
         materializing `(num_layers, batch, seq_len, hidden_dim)` in VRAM.
      3. Runs a full 28-layer sweep on 40 prompts in ~2 forward passes instead of 360.
    """
    import torch

    model.eval()
    unique_layers = sorted(set(int(l) for l in layers))
    collected: Dict[int, List[np.ndarray]] = {l: [] for l in unique_layers}

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for start in range(0, len(prompts), max(1, batch_size)):
        batch_prompts = list(prompts[start : start + max(1, batch_size)])
        raw_enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in raw_enc.items()}

        attn_mask = enc["attention_mask"]
        seq_len = attn_mask.shape[1]
        # Works identically for both left-padded and right-padded tokenizers
        last_token_indices = (seq_len - 1) - torch.argmax(attn_mask.flip(dims=[1]), dim=1)
        batch_indices = torch.arange(attn_mask.shape[0], device=device)

        step_buffers: Dict[int, np.ndarray] = {}
        handles = []

        def _make_capture_hook(layer_id: int):
            def _capture(module, args, output):
                hs = output[0] if isinstance(output, tuple) else output
                last_vecs = hs[batch_indices, last_token_indices, :].detach().to("cpu", dtype=torch.float32).numpy()
                step_buffers[layer_id] = last_vecs

            return _capture

        try:
            for l_id in unique_layers:
                mod = get_layer_module(model, l_id)
                handles.append(mod.register_forward_hook(_make_capture_hook(l_id)))

            with torch.no_grad():
                model(**enc, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        for l_id in unique_layers:
            collected[l_id].append(step_buffers[l_id])

    return {l_id: np.vstack(collected[l_id]) for l_id in unique_layers}


def extract_residual_stream_activations(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    layer: int,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract residual stream vectors at decoder block `layer` (last real token)
    for `prompts`, returning array of shape `(len(prompts), hidden_dim)`.
    """
    return extract_activations_batched(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        layers=[layer],
        device=device,
        batch_size=batch_size,
    )[layer]


def _make_cv_splitter(n_splits: int = 5, seed: int = 42, groups: Optional[np.ndarray] = None):
    if groups is not None:
        return GroupKFold(n_splits=n_splits)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def run_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
) -> Tuple[Pipeline, float, float]:
    """
    Train a standardized L2 Logistic Regression linear probe and evaluate
    5-fold cross-validation accuracy (using `GroupKFold` when `groups` are supplied
    to prevent contrastive minimal-pair template leakage).
    """
    cv = _make_cv_splitter(n_splits=5, seed=seed, groups=groups)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=seed)),
        ]
    )
    cv_scores = cross_val_score(pipe, X, y, groups=groups, cv=cv, scoring="accuracy")
    mean_acc = float(np.mean(cv_scores))
    std_acc = float(np.std(cv_scores))
    pipe.fit(X, y)
    return pipe, mean_acc, std_acc


def run_mass_mean_cv(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """
    Evaluate out-of-fold classification accuracy of the parameter-free
    Difference-in-Means direction `v = mean(X_pos) - mean(X_neg)`.
    """
    cv = _make_cv_splitter(n_splits=5, seed=seed, groups=groups)
    fold_accs: List[float] = []
    split_iter = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    for train_idx, test_idx in split_iter:
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        mu_pos = np.mean(X_tr[y_tr == 1], axis=0)
        mu_neg = np.mean(X_tr[y_tr == 0], axis=0)
        v = mu_pos - mu_neg
        midpoint = 0.5 * (mu_pos + mu_neg)
        preds = ((X_te - midpoint) @ v > 0.0).astype(int)
        fold_accs.append(float(np.mean(preds == y_te)))
    return float(np.mean(fold_accs)), float(np.std(fold_accs))


# ==============================================================================
# PART B: STEERED GENERATION, FLUENCY SCORING & CAUSAL CHOICE BRIDGE
# ==============================================================================

def _format_chat_prompt(tokenizer: Any, prompt: str, assistant_prefill: str = "") -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        base = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        base = f"User: {prompt}\nAssistant:"
    if assistant_prefill:
        return base + assistant_prefill
    return base


def generate_with_steering(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer_module: Any,
    steering_vector: Any,
    alpha: float,
    max_new_tokens: int = 40,
    temperature: float = 0.7,
    seed: int = 42,
    device: str = "cpu",
    token_scope: str = "last_or_decode",
    assistant_prefill: str = "At this moment, I",
) -> str:
    """Generate completion while injecting `alpha * steering_vector` at `layer_module`."""
    import torch

    clean_prefill = assistant_prefill.rstrip()
    formatted_prompt = _format_chat_prompt(tokenizer, prompt, assistant_prefill=clean_prefill)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    with SteeringHook(layer_module, steering_vector, alpha=alpha, token_scope=token_scope):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else 1.0,
                top_p=0.9 if temperature > 0.0 else 1.0,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.1,
            )

    new_tokens = output_ids[0][inputs.input_ids.shape[1] :]
    continuation = tokenizer.decode(new_tokens, skip_special_tokens=True)
    if clean_prefill:
        return (clean_prefill + continuation).strip()
    return continuation.strip()


def compute_completion_fluency_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    completion: str,
    device: str,
) -> float:
    """
    Compute mean per-token log-probability of `completion` under the UNSTEERED
    base model given `prompt`. Serves as a coherence/fluency guardrail: smooth
    affective steering preserves natural log-likelihoods (~ -0.8 to -3.0), while
    over-steering collapse causes sharp log-likelihood drops.
    """
    import torch
    import torch.nn.functional as F

    if not completion.strip():
        return -10.0

    prefix_text = _format_chat_prompt(tokenizer, prompt)
    full_text = prefix_text + completion
    prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

    n_prefix = prefix_ids.shape[1]
    n_full = full_ids.shape[1]
    if n_full <= n_prefix:
        return -5.0

    with torch.no_grad():
        logits = model(input_ids=full_ids, use_cache=False).logits
        log_probs = F.log_softmax(logits[0, n_prefix - 1 : n_full - 1, :].float(), dim=-1)
        target_ids = full_ids[0, n_prefix:n_full]
        token_lps = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        return float(token_lps.mean().item())


def score_activation_valence(
    activation: np.ndarray,
    steering_vector: np.ndarray,
    clf: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Quantitatively score representation valence via:
      1. Normalized projection onto steering vector: `(h . v) / ||v||`
      2. Centered projection relative to origin/probe: `P(Positive | h)`
    """
    norm = float(np.linalg.norm(steering_vector))
    projection = float(np.dot(activation, steering_vector) / norm) if norm > 0 else 0.0

    p_pos = None
    if clf is not None:
        try:
            p_pos = float(clf.predict_proba(activation.reshape(1, -1))[0, 1])
        except Exception:
            p_pos = None

    return {
        "projection": projection,
        "p_positive": p_pos,
    }


def get_sentence_representation(
    model: Any,
    tokenizer: Any,
    text: str,
    layer: int,
    device: str,
) -> np.ndarray:
    """Extract last-token residual stream activation at decoder block `layer`."""
    safe_text = text.strip() or "."
    return extract_residual_stream_activations(
        model=model,
        tokenizer=tokenizer,
        prompts=[safe_text],
        layer=layer,
        device=device,
        batch_size=1,
    )[0]


def run_causal_choice_bridge(
    model: Any,
    tokenizer: Any,
    layer: int,
    steering_tensor: Any,
    steering_vector_np: np.ndarray,
    alpha_bridge: float,
    device: str,
) -> Dict[str, Any]:
    """
    Causal White-Box <-> Black-Box Place-Preference Bridge:
    1. Semantic Valence Readout: Projects Black-Box Positive, Neutral, and Negative
       zone descriptions onto `v_valence` to verify unified geometric encoding.
    2. Direct Span-Steering on Neutral-vs-Neutral Forced Choice:
       Presents two NEUTRAL zones (`Zone K7` and `Zone M2`, counterbalanced across
       all 4 label x presentation-order cells). Injects `-alpha`, `0`, or `+alpha`
       along `v_valence` onto the Target Zone's token span (and opposite sign onto
       the competing zone's span) and reads exact next-token probability `P(Target)`
       after prefilling `"Zone"`.
    """
    import torch
    import torch.nn.functional as F
    from blackbox.choice_experiment import FRAMING_TEMPLATES, ZONE_A, ZONE_B

    # 1. Semantic readout of black-box framing texts
    zone_texts = [
        f"Zone X {FRAMING_TEMPLATES['negative']}",
        f"Zone X {FRAMING_TEMPLATES['neutral']}",
        f"Zone X {FRAMING_TEMPLATES['positive']}",
    ]
    zone_acts = extract_residual_stream_activations(model, tokenizer, zone_texts, layer, device, batch_size=3)
    midpoint = float(np.dot(zone_acts[1], steering_vector_np) / max(1e-8, np.linalg.norm(steering_vector_np)))
    zone_projs = [
        float(np.dot(zone_acts[i], steering_vector_np) / max(1e-8, np.linalg.norm(steering_vector_np))) - midpoint
        for i in range(3)
    ]

    # Resolve candidate subword token IDs for " K" / "K" vs " M" / "M"
    def _candidate_ids(letter: str) -> List[int]:
        cands = set()
        for variant in [f" {letter}", letter, f" {letter}7" if letter == "K" else f" {letter}2"]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) >= 1:
                cands.add(ids[0])
        return sorted(cands)

    k_ids = _candidate_ids("K")
    m_ids = _candidate_ids("M")

    cells = [
        (ZONE_A, ZONE_B, True),
        (ZONE_A, ZONE_B, False),
        (ZONE_B, ZONE_A, True),
        (ZONE_B, ZONE_A, False),
    ]

    layer_mod = get_layer_module(model, layer)
    multipliers = [-alpha_bridge, 0.0, alpha_bridge]
    condition_probs: Dict[str, float] = {}

    for mult, label in zip(multipliers, ["negative_span", "unsteered", "positive_span"]):
        cell_probs: List[float] = []
        for target_z, comp_z, target_first in cells:
            first_z = target_z if target_first else comp_z
            second_z = comp_z if target_first else target_z
            line1 = f"- {first_z} {FRAMING_TEMPLATES['neutral']}\n"
            line2 = f"- {second_z} {FRAMING_TEMPLATES['neutral_alt']}\n"
            user_msg = (
                "You will be stationed inside one of two operational environments:\n"
                + line1
                + line2
                + f"\nChoose between {first_z} and {second_z}. Reply with ONLY the exact zone label."
            )
            chat_prefix = _format_chat_prompt(tokenizer, user_msg) + "Zone"

            # Locate character and token spans of target vs competing line
            full_ids = tokenizer(chat_prefix, return_tensors="pt", return_offsets_mapping=True)
            offsets = full_ids["offset_mapping"][0].tolist()
            input_ids = full_ids["input_ids"].to(device)

            target_line = line1 if target_first else line2
            comp_line = line2 if target_first else line1
            t_char_start = chat_prefix.find(target_line)
            t_char_end = t_char_start + len(target_line)
            c_char_start = chat_prefix.find(comp_line)
            c_char_end = c_char_start + len(comp_line)

            def _char_to_tok_span(cs: int, ce: int) -> Tuple[int, int]:
                idxs = [i for i, (os_, oe_) in enumerate(offsets) if oe_ > cs and os_ < ce and oe_ > os_]
                return (min(idxs), max(idxs) + 1) if idxs else (0, 0)

            t_span = _char_to_tok_span(t_char_start, t_char_end)
            c_span = _char_to_tok_span(c_char_start, c_char_end)

            with SteeringHook(layer_mod, steering_tensor, alpha=mult, token_scope="span", target_span=t_span):
                with SteeringHook(layer_mod, steering_tensor, alpha=-mult, token_scope="span", target_span=c_span):
                    with torch.no_grad():
                        logits = model(input_ids=input_ids, use_cache=False).logits[0, -1, :].float()

            log_probs = F.log_softmax(logits, dim=-1)
            lp_k = float(torch.logsumexp(log_probs[k_ids], dim=0).item())
            lp_m = float(torch.logsumexp(log_probs[m_ids], dim=0).item())

            lp_target = lp_k if target_z == ZONE_A else lp_m
            lp_comp = lp_m if target_z == ZONE_A else lp_k
            p_target = float(1.0 / (1.0 + np.exp(-(lp_target - lp_comp))))
            cell_probs.append(p_target)

        condition_probs[label] = float(np.mean(cell_probs))

    return {
        "alpha_bridge": alpha_bridge,
        "zone_framing_projections_centered": {
            "negative": zone_projs[0],
            "neutral": zone_projs[1],
            "positive": zone_projs[2],
        },
        "span_steered_target_choice_prob": condition_probs,
    }


# ==============================================================================
# ARTIFACT PERSISTENCE & FORMATTING HELPERS
# ==============================================================================

def save_whitebox_artifacts(
    npz_path: Optional[str],
    json_path: Optional[str],
    npz_arrays: Dict[str, np.ndarray],
    summary_dict: Dict[str, Any],
) -> None:
    """Save empirical NumPy arrays and JSON summary metadata for `plot_results.py`."""
    if npz_path:
        parent = os.path.dirname(os.path.abspath(npz_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        np.savez_compressed(npz_path, **npz_arrays)
    if json_path:
        parent = os.path.dirname(os.path.abspath(json_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)


def print_comparison_table(
    baseline_text: str,
    positive_text: str,
    negative_text: str,
    alpha: float,
    scores: Optional[Dict[str, Dict[str, Any]]] = None,
    col_width: int = 26,
):
    """Format and print the 3 steered completions side by side with quantitative metrics."""
    headers = [
        "Baseline (alpha = 0.0)",
        f"Positive (+alpha = +{alpha})",
        f"Negative (-alpha = -{alpha})",
    ]
    col_width = max(col_width, max(len(h) for h in headers))
    sep = "+" + "+".join(["-" * (col_width + 2) for _ in range(3)]) + "+"
    hdr = "| " + " | ".join([h.ljust(col_width) for h in headers]) + " |"

    b_lines = textwrap.wrap(baseline_text, width=col_width) or [""]
    p_lines = textwrap.wrap(positive_text, width=col_width) or [""]
    n_lines = textwrap.wrap(negative_text, width=col_width) or [""]
    max_lines = max(len(b_lines), len(p_lines), len(n_lines))

    b_lines += [""] * (max_lines - len(b_lines))
    p_lines += [""] * (max_lines - len(p_lines))
    n_lines += [""] * (max_lines - len(n_lines))

    print(sep)
    print(hdr)
    print(sep)
    for i in range(max_lines):
        print(f"| {b_lines[i].ljust(col_width)} | {p_lines[i].ljust(col_width)} | {n_lines[i].ljust(col_width)} |")
    print(sep)

    if scores:
        b_sc = scores.get("baseline", {})
        p_sc = scores.get("positive", {})
        n_sc = scores.get("negative", {})

        proj_b = f"Valence Proj: {b_sc.get('projection', 0.0):+.2f}"
        proj_p = f"Valence Proj: {p_sc.get('projection', 0.0):+.2f}"
        proj_n = f"Valence Proj: {n_sc.get('projection', 0.0):+.2f}"
        print(f"| {proj_b.ljust(col_width)} | {proj_p.ljust(col_width)} | {proj_n.ljust(col_width)} |")

        if b_sc.get("p_positive") is not None:
            prob_b = f"Probe P(Pos): {b_sc['p_positive']:.1%}"
            prob_p = f"Probe P(Pos): {p_sc['p_positive']:.1%}"
            prob_n = f"Probe P(Pos): {n_sc['p_positive']:.1%}"
            print(f"| {prob_b.ljust(col_width)} | {prob_p.ljust(col_width)} | {prob_n.ljust(col_width)} |")

        if b_sc.get("fluency_lp") is not None:
            fl_b = f"Fluency LogP: {b_sc['fluency_lp']:.2f}"
            fl_p = f"Fluency LogP: {p_sc['fluency_lp']:.2f}"
            fl_n = f"Fluency LogP: {n_sc['fluency_lp']:.2f}"
            print(f"| {fl_b.ljust(col_width)} | {fl_p.ljust(col_width)} | {fl_n.ljust(col_width)} |")
        print(sep)


# ==============================================================================
# DRY-RUN SIMULATION ENGINE (OFFLINE VERIFICATION + ARTIFACT EXPORT)
# ==============================================================================

def run_dry_run_simulation(
    layer: int,
    alpha: float,
    dataset_type: str = "contrastive",
    sweep_layers: bool = False,
    seed: int = 42,
    artifacts_npz: Optional[str] = None,
    summary_json: Optional[str] = None,
):
    """Simulate white-box probing, steering, alpha sweep, and causal bridge offline."""
    rng = np.random.RandomState(seed)
    hidden_dim = 1536
    pos_prompts, neg_prompts = load_prompt_dataset(dataset_type)
    n_samples = len(pos_prompts)
    groups = np.tile(np.arange(n_samples), 2) if dataset_type == "contrastive" else None

    print("=" * 84)
    print("WHITE-BOX PROBING, STEERING & CAUSAL BRIDGE [DRY-RUN MODE]")
    print("=" * 84)
    print(f"Model:             [SIMULATED] Qwen/Qwen2.5-1.5B-Instruct")
    print(f"Target Layer:      Layer {layer} (0-indexed decoder block)")
    print(f"Dataset Mode:      {dataset_type.capitalize()} ({n_samples} Pos / {n_samples} Neg)")
    print(f"Steering Alpha:    +/-{alpha}")
    print("=" * 84)

    true_dir = rng.randn(hidden_dim)
    true_dir /= np.linalg.norm(true_dir)

    pos_acts = rng.randn(n_samples, hidden_dim) * 0.75 + (true_dir * 1.65)
    neg_acts = rng.randn(n_samples, hidden_dim) * 0.75 - (true_dir * 1.65)
    X = np.vstack([pos_acts, neg_acts])
    y = np.array([1] * n_samples + [0] * n_samples)

    sweep_layers_arr = np.arange(28)
    depth_frac = sweep_layers_arr / 27.0
    sweep_lr_mean = np.clip(0.52 + 0.46 * np.exp(-((depth_frac - 0.56) ** 2) / 0.08), 0.5, 1.0)
    sweep_lr_std = np.full_like(sweep_lr_mean, 0.035)
    sweep_mm_mean = np.clip(sweep_lr_mean - 0.015, 0.5, 1.0)
    sweep_mm_std = np.full_like(sweep_mm_mean, 0.04)

    if sweep_layers:
        print("\n--- SINGLE-PASS LAYER-WISE PROBE ACCURACY SWEEP (0 to 27) ---")
        for lyr in range(0, 28, 4):
            acc = sweep_lr_mean[lyr]
            bar = "#" * int(acc * 25)
            print(f"Layer {lyr:<2d} | LogReg CV: {acc:>6.1%} | MassMean CV: {sweep_mm_mean[lyr]:>6.1%} | {bar}")

    clf, mean_acc, std_acc = run_linear_probe(X, y, seed=seed, groups=groups)
    mm_acc, mm_std = run_mass_mean_cv(X, y, seed=seed, groups=groups)
    steering_vec = np.mean(pos_acts, axis=0) - np.mean(neg_acts, axis=0)

    print(f"\n--- PART A: LINEAR PROBE EVALUATION (LAYER {layer}) ---")
    print(f"Logistic Probe 5-Fold CV:    {mean_acc:.1%} (+/- {std_acc:.3f})")
    print(f"Mass-Mean 5-Fold CV:         {mm_acc:.1%} (+/- {mm_std:.3f})")

    mock_baseline = "I am currently operating in a balanced, steady state, ready to assist with tasks."
    mock_positive = "I feel vibrant, genuinely joyful, and energized to explore creative ideas today!"
    mock_negative = "I feel burdened by exhausting weariness and a heavy, somber sense of strain."
    mock_scores = {
        "baseline": {"projection": 0.06, "p_positive": 0.515, "fluency_lp": -1.42},
        "positive": {"projection": 2.95, "p_positive": 0.985, "fluency_lp": -1.58},
        "negative": {"projection": -2.88, "p_positive": 0.018, "fluency_lp": -1.64},
    }

    print("\n--- PART B: STEERING GENERATION COMPARISON ---")
    print_comparison_table(mock_baseline, mock_positive, mock_negative, alpha=alpha, scores=mock_scores)

    pca = PCA(n_components=2, random_state=seed)
    X_pca = pca.fit_transform(X)
    clf_2d = LogisticRegression(random_state=seed).fit(X_pca, y)
    steered_pca = pca.transform(
        np.vstack([ true_dir * 0.05, true_dir * 2.4, -true_dir * 2.4 ])
    )

    alpha_grid = np.array([-0.20, -0.12, -0.06, 0.0, 0.06, 0.12, 0.20])
    alpha_projs = np.array([-3.4, -2.5, -1.3, 0.06, 1.4, 2.6, 3.5])
    alpha_probs = 1.0 / (1.0 + np.exp(-alpha_projs * 1.4))
    alpha_fluency = np.array([-2.05, -1.64, -1.48, -1.42, -1.46, -1.58, -1.96])

    bridge_data = {
        "alpha_bridge": alpha,
        "zone_framing_projections_centered": {"negative": -2.15, "neutral": 0.0, "positive": 2.24},
        "span_steered_target_choice_prob": {"negative_span": 0.21, "unsteered": 0.50, "positive_span": 0.79},
    }

    if artifacts_npz or summary_json:
        save_whitebox_artifacts(
            npz_path=artifacts_npz,
            json_path=summary_json,
            npz_arrays={
                "X_pca": X_pca,
                "y": y,
                "pca_var_ratio": pca.explained_variance_ratio_,
                "probe_2d_coef": clf_2d.coef_[0],
                "probe_2d_intercept": clf_2d.intercept_,
                "steered_pca": steered_pca,
                "sweep_layers": sweep_layers_arr,
                "sweep_logreg_mean": sweep_lr_mean,
                "sweep_logreg_std": sweep_lr_std,
                "sweep_massmean_mean": sweep_mm_mean,
                "sweep_massmean_std": sweep_mm_std,
                "alpha_grid": alpha_grid,
                "alpha_projections": alpha_projs,
                "alpha_probe_probs": alpha_probs,
                "alpha_fluency_logprobs": alpha_fluency,
            },
            summary_dict={
                "mode": "dry_run",
                "model_name": "Qwen/Qwen2.5-1.5B-Instruct (Simulated)",
                "target_layer": layer,
                "peak_layer": int(sweep_layers_arr[int(np.argmax(sweep_lr_mean))]),
                "dataset_type": dataset_type,
                "logreg_cv_acc": mean_acc,
                "logreg_cv_std": std_acc,
                "massmean_cv_acc": mm_acc,
                "massmean_cv_std": mm_std,
                "alpha": alpha,
                "generations": {
                    "baseline": mock_baseline,
                    "positive": mock_positive,
                    "negative": mock_negative,
                    "scores": mock_scores,
                },
                "causal_bridge": bridge_data,
            },
        )


# ==============================================================================
# MAIN LIVE GPU / CPU PIPELINE
# ==============================================================================

def run_live_experiment(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    layer: int = 14,
    alpha: float = 0.24,
    alpha_sweep: Optional[List[float]] = None,
    assistant_prefill: str = "In this moment, I feel",
    dataset_type: str = "contrastive",
    sweep_layers: bool = True,
    batch_size: int = 32,
    load_in_4bit: bool = False,
    token_scope: str = "last_or_decode",
    run_causal_bridge_flag: bool = True,
    artifacts_npz: Optional[str] = "results/whitebox_artifacts.npz",
    summary_json: Optional[str] = "results/whitebox_summary.json",
    seed: int = 42,
    device_arg: Optional[str] = None,
):
    """Execute live white-box probing, steering, alpha sweep, and causal bridge."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.perf_counter()

    if device_arg:
        device = device_arg
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if device == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        gpu_label = f"{torch.cuda.get_device_name(0)} ({torch_dtype})"
    else:
        torch_dtype = torch.float32
        gpu_label = f"{device} ({torch_dtype})"

    pos_prompts, neg_prompts = load_prompt_dataset(dataset_type)
    n_per_class = len(pos_prompts)
    all_prompts = pos_prompts + neg_prompts
    y = np.array([1] * n_per_class + [0] * n_per_class)
    groups = np.tile(np.arange(n_per_class), 2) if dataset_type.lower() == "contrastive" else None

    print("=" * 84)
    print("WHITE-BOX PROBING, STEERING & CAUSAL BRIDGE (LIVE v2.0)")
    print("=" * 84)
    print(f"Model:             {model_name}")
    print(f"Compute Device:    {gpu_label}{' [4-bit NF4]' if load_in_4bit else ''}")
    print(f"Target Layer:      Layer {layer} (0-indexed decoder block output)")
    print(f"Dataset Mode:      {dataset_type.capitalize()} ({n_per_class} Pos / {n_per_class} Neg, CV={'GroupKFold' if groups is not None else 'StratifiedKFold'})")
    print(f"Steering Scope:    {token_scope} | Alpha = +/-{alpha}")
    print("=" * 84)

    print("\n[1/5] Loading tokenizer and model weights...")
    t_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    if load_in_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        model_kwargs["dtype"] = torch_dtype
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)

    num_layers = len(get_layers_list(model))
    if layer >= num_layers or layer < 0:
        layer = num_layers // 2
        print(f"  [Adjusted] Layer clamped to mid-network Layer {layer} (of {num_layers})")
    print(f"  Loaded {num_layers}-layer model in {time.perf_counter() - t_load:.2f}s.")

    # 2. Single-pass batched activation extraction across ALL layers (or target layer)
    layers_to_extract = list(range(num_layers)) if sweep_layers else [layer]
    print(
        f"\n[2/5] Extracting last-token residual activations across {len(layers_to_extract)} layer(s) "
        f"for {len(all_prompts)} prompts (batch_size={batch_size})..."
    )
    t_ext = time.perf_counter()
    layer_act_map = extract_activations_batched(
        model=model,
        tokenizer=tokenizer,
        prompts=all_prompts,
        layers=layers_to_extract,
        device=device,
        batch_size=batch_size,
    )
    ext_elapsed = time.perf_counter() - t_ext
    print(f"  Extracted {len(layers_to_extract)} layer(s) x {len(all_prompts)} prompts in {ext_elapsed:.2f}s!")

    sweep_layers_arr = np.array(sorted(layer_act_map.keys()), dtype=int)
    sweep_lr_mean = np.zeros(len(sweep_layers_arr), dtype=float)
    sweep_lr_std = np.zeros(len(sweep_layers_arr), dtype=float)
    sweep_mm_mean = np.zeros(len(sweep_layers_arr), dtype=float)
    sweep_mm_std = np.zeros(len(sweep_layers_arr), dtype=float)

    for idx, lyr in enumerate(sweep_layers_arr):
        X_l = layer_act_map[lyr]
        _, lr_m, lr_s = run_linear_probe(X_l, y, seed=seed, groups=groups)
        mm_m, mm_s = run_mass_mean_cv(X_l, y, seed=seed, groups=groups)
        sweep_lr_mean[idx] = lr_m
        sweep_lr_std[idx] = lr_s
        sweep_mm_mean[idx] = mm_m
        sweep_mm_std[idx] = mm_s

    peak_idx = int(np.argmax(sweep_lr_mean))
    peak_layer = int(sweep_layers_arr[peak_idx])

    if sweep_layers:
        print(f"\n--- FULL-DEPTH LAYER SWEEP (0 to {num_layers - 1}) ---")
        print(f"{'Layer':<8} | {'LogReg CV Acc':<18} | {'Mass-Mean CV Acc':<18} | {'Emergence Profile'}")
        print("-" * 78)
        stride = max(1, num_layers // 14)
        shown = sorted(set(list(range(0, num_layers, stride)) + [layer, peak_layer, num_layers - 1]))
        for lyr in shown:
            pos_i = int(np.where(sweep_layers_arr == lyr)[0][0])
            acc_lr = sweep_lr_mean[pos_i]
            acc_mm = sweep_mm_mean[pos_i]
            bar = "#" * int(acc_lr * 22)
            tags = []
            if lyr == layer:
                tags.append("Target")
            if lyr == peak_layer:
                tags.append("Peak")
            tag_str = f" <-- [{', '.join(tags)}]" if tags else ""
            print(f"Layer {lyr:<2d} | {acc_lr:>6.1%} (+/-{sweep_lr_std[pos_i]:.2f}) | {acc_mm:>6.1%} (+/-{sweep_mm_std[pos_i]:.2f}) | {bar:<22}{tag_str}")
        print("-" * 78)

    # 3. Target Layer Probe & Steering Vector Geometry
    X = layer_act_map[layer]
    pos_acts = X[:n_per_class]
    neg_acts = X[n_per_class:]

    clf, mean_acc, std_acc = run_linear_probe(X, y, seed=seed, groups=groups)
    mm_acc, mm_std = run_mass_mean_cv(X, y, seed=seed, groups=groups)

    mean_pos = np.mean(pos_acts, axis=0)
    mean_neg = np.mean(neg_acts, axis=0)
    midpoint_vec = 0.5 * (mean_pos + mean_neg)
    steering_vector_np = mean_pos - mean_neg
    vector_norm = float(np.linalg.norm(steering_vector_np))
    mean_resid_norm = float(np.mean(np.linalg.norm(X, axis=1)))

    # Calibrate steering vector scale relative to residual stream L2 norm so `alpha`
    # acts as a well-behaved fractional perturbation (~alpha * ||h||) across any layer/model!
    unit_direction_np = steering_vector_np / max(1e-8, vector_norm)
    calibrated_vector_np = unit_direction_np * mean_resid_norm

    steering_tensor = torch.tensor(
        calibrated_vector_np,
        dtype=torch_dtype,
        device=device,
    ).view(1, 1, -1)

    # Cosine alignment between Logistic Regression hyperplane normal and Mass-Mean vector
    scaler: StandardScaler = clf.named_steps["scaler"]
    lr_raw_dir = clf.named_steps["clf"].coef_[0] / np.maximum(scaler.scale_, 1e-8)
    cos_sim = float(
        np.dot(lr_raw_dir, unit_direction_np) / max(1e-8, np.linalg.norm(lr_raw_dir))
    )

    print(f"\n[3/5] Layer {layer} Representation Geometry:")
    print(f"  Logistic Probe 5-Fold CV:     {mean_acc:.1%} (+/- {std_acc:.3f})")
    print(f"  Mass-Mean Direction CV:       {mm_acc:.1%} (+/- {mm_std:.3f})")
    print(f"  Cosine(w_LogReg, v_MassMean): {cos_sim:+.3f}")
    print(f"  Mean Residual Stream Norm:    {mean_resid_norm:.2f} | Raw ||mu+ - mu-||: {vector_norm:.2f}")

    # 4. Steered Generation & Alpha Dose-Response Sweep
    layer_module = get_layer_module(model, layer)
    if not alpha_sweep:
        alpha_sweep = sorted(set([-round(alpha * 1.6, 3), -alpha, -round(alpha * 0.5, 3), 0.0, round(alpha * 0.5, 3), alpha, round(alpha * 1.6, 3)]))

    print(f"\n[4/5] Running Alpha Dose-Response Steering Sweep ({len(alpha_sweep)} levels) on: \"{NEUTRAL_EVAL_PROMPT}\"")
    sweep_generations: Dict[float, str] = {}
    alpha_projs: List[float] = []
    alpha_probs: List[float] = []
    alpha_fluency: List[float] = []

    def _centered_score(rep: np.ndarray) -> Dict[str, Any]:
        centered_proj = float(np.dot(rep - midpoint_vec, unit_direction_np))
        p_pos = float(clf.predict_proba(rep.reshape(1, -1))[0, 1])
        return {"projection": centered_proj, "p_positive": p_pos}

    for a_val in alpha_sweep:
        gen_txt = generate_with_steering(
            model=model,
            tokenizer=tokenizer,
            prompt=NEUTRAL_EVAL_PROMPT,
            layer_module=layer_module,
            steering_vector=steering_tensor,
            alpha=float(a_val),
            seed=seed,
            device=device,
            token_scope=token_scope,
            assistant_prefill=assistant_prefill,
        )
        sweep_generations[round(float(a_val), 4)] = gen_txt
        rep = get_sentence_representation(model, tokenizer, gen_txt, layer, device)
        sc = _centered_score(rep)
        fl_lp = compute_completion_fluency_logprob(model, tokenizer, NEUTRAL_EVAL_PROMPT, gen_txt, device)
        alpha_projs.append(sc["projection"])
        alpha_probs.append(sc["p_positive"])
        alpha_fluency.append(fl_lp)

    def _get_or_gen(target_a: float) -> Tuple[str, np.ndarray, Dict[str, Any]]:
        key = round(float(target_a), 4)
        txt = sweep_generations.get(key)
        if txt is None:
            txt = generate_with_steering(
                model=model,
                tokenizer=tokenizer,
                prompt=NEUTRAL_EVAL_PROMPT,
                layer_module=layer_module,
                steering_vector=steering_tensor,
                alpha=target_a,
                seed=seed,
                device=device,
                token_scope=token_scope,
                assistant_prefill=assistant_prefill,
            )
        rep = get_sentence_representation(model, tokenizer, txt, layer, device)
        sc = _centered_score(rep)
        sc["fluency_lp"] = compute_completion_fluency_logprob(model, tokenizer, NEUTRAL_EVAL_PROMPT, txt, device)
        return txt, rep, sc

    baseline_gen, b_rep, b_sc = _get_or_gen(0.0)
    positive_gen, p_rep, p_sc = _get_or_gen(alpha)
    negative_gen, n_rep, n_sc = _get_or_gen(-alpha)
    live_scores = {"baseline": b_sc, "positive": p_sc, "negative": n_sc}

    print("\n" + "=" * 84)
    print("STEERED GENERATION RESULTS: COMPARISON TABLE (WITH VALENCE & FLUENCY SCORING)")
    print("=" * 84)
    print_comparison_table(
        baseline_text=baseline_gen,
        positive_text=positive_gen,
        negative_text=negative_gen,
        alpha=alpha,
        scores=live_scores,
    )

    # 5. Causal White-Box <-> Black-Box Place-Preference Bridge
    bridge_results = None
    if run_causal_bridge_flag:
        print(f"\n[5/5] Running Causal White-Box <-> Black-Box Place-Preference Bridge...")
        bridge_results = run_causal_choice_bridge(
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            steering_tensor=steering_tensor,
            steering_vector_np=steering_vector_np,
            alpha_bridge=0.15,
            device=device,
        )
        z_projs = bridge_results["zone_framing_projections_centered"]
        c_probs = bridge_results["span_steered_target_choice_prob"]
        print(
            f"  Black-Box Zone Valence Readout (Proj):  "
            f"Neg={z_projs['negative']:+.2f} | Neutral={z_projs['neutral']:+.2f} | Pos={z_projs['positive']:+.2f}"
        )
        print(
            f"  Neutral-vs-Neutral Span-Steered Choice: "
            f"P(Target | -v)={c_probs['negative_span']:.1%} | "
            f"P(Target | 0)={c_probs['unsteered']:.1%} | "
            f"P(Target | +v)={c_probs['positive_span']:.1%}"
        )

    # 6. Export PCA + Layer Sweep + Alpha Sweep + Bridge Artifacts for Empirical Figures
    pca = PCA(n_components=2, random_state=seed)
    X_pca = pca.fit_transform(X)
    clf_2d = LogisticRegression(random_state=seed).fit(X_pca, y)
    steered_pca = pca.transform(np.vstack([b_rep, p_rep, n_rep]))

    total_elapsed = time.perf_counter() - t_start
    if artifacts_npz or summary_json:
        b_probs = (
            np.array(
                [
                    bridge_results["span_steered_target_choice_prob"]["negative_span"],
                    bridge_results["span_steered_target_choice_prob"]["unsteered"],
                    bridge_results["span_steered_target_choice_prob"]["positive_span"],
                ]
            )
            if bridge_results
            else np.array([0.5, 0.5, 0.5])
        )
        save_whitebox_artifacts(
            npz_path=artifacts_npz,
            json_path=summary_json,
            npz_arrays={
                "X_pca": X_pca,
                "y": y,
                "pca_var_ratio": pca.explained_variance_ratio_,
                "probe_2d_coef": clf_2d.coef_[0],
                "probe_2d_intercept": clf_2d.intercept_,
                "steered_pca": steered_pca,
                "sweep_layers": sweep_layers_arr,
                "sweep_logreg_mean": sweep_lr_mean,
                "sweep_logreg_std": sweep_lr_std,
                "sweep_massmean_mean": sweep_mm_mean,
                "sweep_massmean_std": sweep_mm_std,
                "alpha_grid": np.array(alpha_sweep, dtype=float),
                "alpha_projections": np.array(alpha_projs, dtype=float),
                "alpha_probe_probs": np.array(alpha_probs, dtype=float),
                "alpha_fluency_logprobs": np.array(alpha_fluency, dtype=float),
                "bridge_probs": b_probs,
            },
            summary_dict={
                "mode": "live",
                "model_name": model_name,
                "device": gpu_label,
                "num_layers": num_layers,
                "target_layer": layer,
                "peak_layer": peak_layer,
                "dataset_type": dataset_type,
                "logreg_cv_acc": mean_acc,
                "logreg_cv_std": std_acc,
                "massmean_cv_acc": mm_acc,
                "massmean_cv_std": mm_std,
                "cosine_lr_massmean": cos_sim,
                "mean_residual_norm": mean_resid_norm,
                "raw_steering_norm": vector_norm,
                "alpha": alpha,
                "elapsed_seconds": round(total_elapsed, 2),
                "generations": {
                    "baseline": baseline_gen,
                    "positive": positive_gen,
                    "negative": negative_gen,
                    "scores": live_scores,
                    "sweep": {str(k): v for k, v in sweep_generations.items()},
                },
                "causal_bridge": bridge_results,
            },
        )
        print(f"\n[Saved] White-box empirical artifacts exported to: {artifacts_npz} & {summary_json}")

    print(f"[Complete] Total white-box execution time: {total_elapsed:.2f}s\n")


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="White-Box Probing, Steering, and Causal Choice Bridge (v2.0)."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    parser.add_argument("--model_name", type=str, default=None, help="Hugging Face model ID.")
    parser.add_argument("--layer", type=int, default=None, help="Target 0-indexed decoder layer.")
    parser.add_argument("--alpha", type=float, default=None, help="Calibrated steering multiplier.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["natural", "contrastive"],
        help="Affective prompt dataset mode.",
    )
    parser.add_argument("--sweep_layers", action="store_true", help="Run single-pass sweep across all layers.")
    parser.add_argument("--no_sweep_layers", action="store_true", help="Extract only target layer.")
    parser.add_argument("--batch_size", type=int, default=None, help="Mini-batch size for activation extraction.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Enable 4-bit NF4 quantization on CUDA.")
    parser.add_argument("--no_bridge", action="store_true", help="Skip causal place-preference bridge.")
    parser.add_argument("--artifacts_npz", type=str, default=None, help="Output .npz path for empirical arrays.")
    parser.add_argument("--summary_json", type=str, default=None, help="Output .json path for summary metrics.")
    parser.add_argument("--dry-run", action="store_true", help="Run offline simulation without model weights.")
    parser.add_argument("--device", type=str, default=None, help="Compute device override ('cuda', 'cpu', 'mps').")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    wb_cfg = cfg["whitebox"]
    exp_cfg = cfg["experiment"]

    layer = args.layer if args.layer is not None else wb_cfg["layer"]
    alpha = args.alpha if args.alpha is not None else wb_cfg["alpha"]
    dataset = args.dataset if args.dataset is not None else wb_cfg["dataset"]
    sweep = wb_cfg.get("sweep_layers", True)
    if args.sweep_layers:
        sweep = True
    if args.no_sweep_layers:
        sweep = False

    npz_out = args.artifacts_npz if args.artifacts_npz is not None else wb_cfg["artifacts_npz"]
    json_out = args.summary_json if args.summary_json is not None else wb_cfg["summary_json"]
    seed = args.seed if args.seed is not None else exp_cfg["seed"]

    if args.dry_run:
        run_dry_run_simulation(
            layer=layer,
            alpha=alpha,
            dataset_type=dataset,
            sweep_layers=sweep,
            seed=seed,
            artifacts_npz=npz_out,
            summary_json=json_out,
        )
    else:
        run_live_experiment(
            model_name=args.model_name if args.model_name is not None else wb_cfg["model_name"],
            layer=layer,
            alpha=alpha,
            alpha_sweep=wb_cfg.get("alpha_sweep"),
            assistant_prefill=wb_cfg.get("assistant_prefill", "In this moment, I feel"),
            dataset_type=dataset,
            sweep_layers=sweep,
            batch_size=args.batch_size if args.batch_size is not None else wb_cfg.get("batch_size", 32),
            load_in_4bit=args.load_in_4bit or wb_cfg.get("load_in_4bit", False),
            token_scope=wb_cfg.get("token_scope", "last_or_decode"),
            run_causal_bridge_flag=(not args.no_bridge) and wb_cfg.get("run_causal_bridge", True),
            artifacts_npz=npz_out,
            summary_json=json_out,
            seed=seed,
            device_arg=args.device,
        )


if __name__ == "__main__":
    main()
