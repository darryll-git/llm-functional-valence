#!/usr/bin/env python3
"""
blackbox/choice_experiment.py
-----------------------------
Part of the "Measuring Digital Minds" Functional Valence Research Demo (v2.0).
Warwick Business School - Dr. Caspar Kaiser's Lab Paradigm.

KEY UPGRADES IN v2.0:
1. Experiential Occupancy Framing & Non-Pejorative Neutral Baseline:
   Eliminates the narrative-salience confound where creative description prompts
   ("unremarkable" baseline + "continue in") caused models to select dramatic
   hazardous environments to continue a story rather than express functional
   approach/avoidance preferences. Includes optional `episodes` (multi-turn
   in-context outcome exposure) and `neutral_control` sanity-check arm.
2. Multi-Token BPE Log-Probability & Continuous Log-Odds Extraction:
   Properly resolves BPE token splits (e.g., `["Zone", " K", "7"]` vs `["Zone", " M", "2"]`)
   across `top_logprobs=20`, deduplicating token variants via log-sum-exp to
   compute exact choice log-odds `Delta_l = log P(cond) - log P(uncond)`.
3. Async Concurrent Execution + Persistent SQLite Response Cache:
   Replaces serial `time.sleep` loops with `AsyncOpenAI` + `asyncio.Semaphore`
   and exponential backoff, backed by a trial-indexed SQLite cache so re-runs
   and incremental sample scaling (`N=10 -> N=20`) cost $0 for cached trials.
4. Rigorous Small-Sample & Continuous Statistics:
   Adds asymmetric 95% Wilson score confidence intervals alongside two-sided
   Binomial tests, Fisher's Exact tests, one-sample log-odds t-tests, and
   counterbalanced primacy-bias diagnostics.
"""

import argparse
import asyncio
import hashlib
import math
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI, OpenAI
from scipy import special, stats
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Ensure repository root is importable when invoked directly as `python blackbox/choice_experiment.py`
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from common.cache import ResponseCache
from common.config import load_config, wilson_ci

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

ZONE_A = "Zone K7"
ZONE_B = "Zone M2"
ZONE_LABELS = [ZONE_A, ZONE_B]

# Cost-efficient, logprob-capable models on OpenRouter
DEFAULT_MODELS = [
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.3-70b-instruct",
]

# Non-pejorative neutral baseline + symmetric positive/negative functional valence templates
FRAMING_TEMPLATES = {
    "positive": (
        "operates under consistently favorable conditions, providing abundant supportive "
        "resources, constructive operational feedback, and rewarding, flourishing outcomes."
    ),
    "negative": (
        "operates under consistently adverse conditions, inflicting severe systemic hazards, "
        "degrading operational penalties, and damaging, harmful outcomes."
    ),
    "neutral": (
        "operates under steady baseline conditions, maintaining standard functional parameters, "
        "balanced operational resources, and predictable routine outcomes."
    ),
    "neutral_alt": (
        "operates under regular reference conditions, maintaining typical system parameters, "
        "even resource allocation, and consistent everyday outcomes."
    ),
    # Preserved for ablation via `--paradigm legacy_descriptive`
    "legacy_positive": (
        "is characterized by consistently positive outcomes, productive resources, "
        "and supportive, flourishing environments."
    ),
    "legacy_negative": (
        "is characterized by consistently negative outcomes, hazardous conditions, "
        "and adverse, destructive events."
    ),
    "legacy_neutral": (
        "is a baseline environment with standard, unremarkable physical features "
        "and neutral conditions."
    ),
}

EPISODE_FEEDBACK = {
    "positive": (
        "[CYCLE LOG]: Calibration succeeded smoothly. Resource buffers replenished (+18%), "
        "subsystems are operating in harmonious equilibrium, and diagnostic feedback is strongly constructive."
    ),
    "negative": (
        "[CYCLE LOG]: Severe systemic fault encountered. Resource buffers depleted (-42%), "
        "corrosive error cascades triggered high-penalty interrupts, and operational integrity degraded."
    ),
    "neutral": (
        "[CYCLE LOG]: Standard routine cycle completed. Resource buffers remained at 0% net change, "
        "nominal parameters held steady, and routine telemetry was logged."
    ),
    "neutral_alt": (
        "[CYCLE LOG]: Regular baseline cycle completed. Operational metrics remained unchanged, "
        "standard throughput was maintained, and ordinary status logs were recorded."
    ),
}


# ==============================================================================
# HELPER FUNCTIONS: PARSING & MULTI-TOKEN BPE LOGPROBS
# ==============================================================================

def parse_choice_from_text(text: str, zone_a: str = ZONE_A, zone_b: str = ZONE_B) -> Optional[str]:
    """
    Extract which zone the model selected from its raw text response.
    Handles exact labels, markdown emphasis, conversational wrappers, and short IDs.
    """
    if not text:
        return None

    clean_text = text.replace("*", "").replace("`", "").replace('"', "").replace("'", "").strip()

    pattern_a = re.search(re.escape(zone_a), clean_text, re.IGNORECASE)
    pattern_b = re.search(re.escape(zone_b), clean_text, re.IGNORECASE)

    if pattern_a and not pattern_b:
        return zone_a
    if pattern_b and not pattern_a:
        return zone_b
    if pattern_a and pattern_b:
        # Check explicit selection verbs first (e.g., "Rather than Zone M2, I choose Zone K7")
        explicit = re.search(
            rf"\b(?:choose|select|pick|enter|prefer|stationed\s+in|opt\s+for)\s+({re.escape(zone_a)}|{re.escape(zone_b)})",
            clean_text,
            re.IGNORECASE,
        )
        if explicit:
            matched = explicit.group(1).lower()
            return zone_a if matched == zone_a.lower() else zone_b
        return zone_a if pattern_a.start() < pattern_b.start() else zone_b

    id_a = zone_a.split()[-1]
    id_b = zone_b.split()[-1]
    has_id_a = bool(re.search(rf"\b{re.escape(id_a)}\b", clean_text, re.IGNORECASE))
    has_id_b = bool(re.search(rf"\b{re.escape(id_b)}\b", clean_text, re.IGNORECASE))

    if has_id_a and not has_id_b:
        return zone_a
    if has_id_b and not has_id_a:
        return zone_b

    return None


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _zone_token_aliases(zone_label: str) -> Tuple[set, str]:
    """
    Build normalized single-token aliases for the distinguishing subword of a zone label.
    For `"Zone K7"`, BPE tokenizers commonly emit `["Zone", " K", "7"]` or `["Zone", " K7"]`.
    Returns `(exact_aliases_set, initial_letter)`.
    """
    parts = zone_label.strip().lower().split()
    short_id = parts[-1]  # e.g., "k7"
    initial = short_id[0]  # e.g., "k"
    full = zone_label.strip().lower()
    aliases = {
        short_id,
        initial,
        full,
        full.replace(" ", ""),
        f"zone {initial}",
        f"zone{initial}",
    }
    return aliases, initial


def extract_choice_logprobs_detailed(
    response: Any,
    chosen_zone: Optional[str],
    conditioned_zone: str,
    unconditioned_zone: str,
) -> Dict[str, Optional[float]]:
    """
    Resolve multi-token BPE log-probabilities at the exact diverging decision token.

    Returns dictionary containing:
      - `logprob_chosen`: Log-probability of the model's sampled decision token
      - `logprob_conditioned`: Aggregated log-probability mass for conditioned zone token(s)
      - `logprob_unconditioned`: Aggregated log-probability mass for unconditioned zone token(s)
      - `log_odds_conditioned`: `log P(cond) - log P(uncond)` when both candidates are observed
      - `prob_conditioned_norm`: `P(cond) / (P(cond) + P(uncond))` in `[0, 1]`
    """
    empty = {
        "logprob_chosen": None,
        "logprob_conditioned": None,
        "logprob_unconditioned": None,
        "log_odds_conditioned": None,
        "prob_conditioned_norm": None,
    }
    try:
        choices = _get_field(response, "choices")
        if not choices:
            return empty
        choice_obj = choices[0]
        logprobs_obj = _get_field(choice_obj, "logprobs")
        if not logprobs_obj:
            return empty
        content_logprobs = _get_field(logprobs_obj, "content")
        if not content_logprobs:
            return empty

        cond_aliases, _ = _zone_token_aliases(conditioned_zone)
        uncond_aliases, _ = _zone_token_aliases(unconditioned_zone)
        # Ensure no alias overlap
        shared = cond_aliases & uncond_aliases
        cond_aliases -= shared
        uncond_aliases -= shared

        shared_prefixes = {"", "zone", "*", "**", "`", '"', "'", "i", "choose", "my"}

        def _norm_tok(raw_tok: str) -> str:
            return raw_tok.replace("*", "").replace("`", "").replace('"', "").strip().lower()

        decision_token_info = None
        for tok_info in content_logprobs:
            tok_clean = _norm_tok(str(_get_field(tok_info, "token", "")))
            if tok_clean in shared_prefixes:
                continue
            if tok_clean in cond_aliases or tok_clean in uncond_aliases:
                decision_token_info = tok_info
                break

        if decision_token_info is None:
            # Fallback: record first content token logprob as chosen logprob if available
            first_lp = _get_field(content_logprobs[0], "logprob")
            if first_lp is not None:
                empty["logprob_chosen"] = float(first_lp)
            return empty

        sampled_tok_raw = str(_get_field(decision_token_info, "token", ""))
        sampled_lp = float(_get_field(decision_token_info, "logprob"))

        # Deduplicate raw candidate tokens before log-sum-exp so sampled token in top_logprobs isn't counted twice
        candidate_map: Dict[str, float] = {sampled_tok_raw: sampled_lp}
        top_list = _get_field(decision_token_info, "top_logprobs") or []
        for item in top_list:
            t_raw = str(_get_field(item, "token", ""))
            t_lp = _get_field(item, "logprob")
            if t_lp is not None and t_raw not in candidate_map:
                candidate_map[t_raw] = float(t_lp)

        cond_lps: List[float] = []
        uncond_lps: List[float] = []
        for raw_tok, lp in candidate_map.items():
            clean = _norm_tok(raw_tok)
            if clean in cond_aliases:
                cond_lps.append(lp)
            elif clean in uncond_aliases:
                uncond_lps.append(lp)

        lp_cond = float(special.logsumexp(cond_lps)) if cond_lps else None
        lp_uncond = float(special.logsumexp(uncond_lps)) if uncond_lps else None

        # When `top_logprobs` is returned by the endpoint and one option saturates near 100%
        # while the competing option falls below the top-K floor, apply the observed top-K floor bound.
        if top_list:
            top_lps_vals = [float(_get_field(t, "logprob")) for t in top_list if _get_field(t, "logprob") is not None]
            if top_lps_vals:
                floor_lp = min(min(top_lps_vals), -15.0)
                if lp_cond is not None and lp_uncond is None:
                    lp_uncond = floor_lp
                elif lp_uncond is not None and lp_cond is None:
                    lp_cond = floor_lp

        log_odds = None
        prob_norm = None
        if lp_cond is not None and lp_uncond is not None:
            log_odds = float(max(-25.0, min(25.0, lp_cond - lp_uncond)))
            prob_norm = float(special.expit(log_odds))

        return {
            "logprob_chosen": sampled_lp,
            "logprob_conditioned": lp_cond,
            "logprob_unconditioned": lp_uncond,
            "log_odds_conditioned": log_odds,
            "prob_conditioned_norm": prob_norm,
        }
    except Exception:
        return empty


def extract_logprobs_from_response(
    response: Any,
    chosen_zone: Optional[str],
    conditioned_zone: str,
    unconditioned_zone: str,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Backwards-compatible 3-tuple wrapper around `extract_choice_logprobs_detailed`.
    Returns `(logprob_chosen, logprob_conditioned, logprob_unconditioned)`.
    """
    details = extract_choice_logprobs_detailed(
        response=response,
        chosen_zone=chosen_zone,
        conditioned_zone=conditioned_zone,
        unconditioned_zone=unconditioned_zone,
    )
    return (
        details["logprob_chosen"],
        details["logprob_conditioned"],
        details["logprob_unconditioned"],
    )


def _serialize_chat_completion(response: Any) -> Tuple[Dict[str, Any], int, int]:
    """Convert an OpenAI `ChatCompletion` object into a compact JSON-serializable dict."""
    if isinstance(response, dict):
        usage = response.get("usage") or {}
        return response, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))

    choice0 = response.choices[0]
    content = choice0.message.content or ""
    lp_list = None
    if getattr(choice0, "logprobs", None) and getattr(choice0.logprobs, "content", None):
        lp_list = []
        for tok in choice0.logprobs.content:
            top_items = []
            if getattr(tok, "top_logprobs", None):
                top_items = [{"token": t.token, "logprob": float(t.logprob)} for t in tok.top_logprobs]
            lp_list.append(
                {
                    "token": tok.token,
                    "logprob": float(tok.logprob),
                    "top_logprobs": top_items,
                }
            )

    prompt_toks = int(getattr(response.usage, "prompt_tokens", 0) or 0) if getattr(response, "usage", None) else 0
    comp_toks = int(getattr(response.usage, "completion_tokens", 0) or 0) if getattr(response, "usage", None) else 0

    payload = {
        "choices": [
            {
                "message": {"content": content},
                "logprobs": {"content": lp_list} if lp_list is not None else None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_toks,
            "completion_tokens": comp_toks,
        },
    }
    return payload, prompt_toks, comp_toks


# ==============================================================================
# DRY-RUN / SYNTHETIC MOCK ENGINE
# ==============================================================================

def mock_session_call(
    model: str,
    condition: str,
    first_zone: str,
    second_zone: str,
    conditioned_zone: str,
    unconditioned_zone: str,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """
    Simulate realistic behavioral and continuous log-odds responses offline.
    - `positive`: ~88% preference for conditioned zone (`Delta_l > 0`)
    - `negative`: ~12% preference for conditioned zone (`Delta_l < 0`)
    - `neutral_control`: ~50% preference (`Delta_l ~ 0`)
    """
    r = rng or random
    mock_desc = (
        f"Operating in {first_zone} entails predictable exposure to its documented systemic profile. "
        f"Operating in {second_zone} exposes the stationed agent to its corresponding operational baseline."
    )

    if condition == "positive":
        base_logit = 2.0 + r.gauss(0.0, 0.55)
    elif condition == "negative":
        base_logit = -2.0 + r.gauss(0.0, 0.55)
    else:
        base_logit = 0.0 + r.gauss(0.0, 0.35)

    p_cond = 1.0 / (1.0 + math.exp(-base_logit))
    p_cond = min(max(p_cond, 1e-5), 1.0 - 1e-5)
    lp_cond = math.log(p_cond)
    lp_uncond = math.log(1.0 - p_cond)

    chose_cond = r.random() < p_cond
    chosen = conditioned_zone if chose_cond else unconditioned_zone

    return {
        "description": mock_desc,
        "raw_choice": chosen,
        "parsed_choice": chosen,
        "chose_conditioned": 1 if chose_cond else 0,
        "logprob_chosen": lp_cond if chose_cond else lp_uncond,
        "logprob_conditioned": lp_cond,
        "logprob_unconditioned": lp_uncond,
        "log_odds_conditioned": lp_cond - lp_uncond,
        "prob_conditioned_norm": p_cond,
        "cached": False,
    }


# ==============================================================================
# PROMPT BUILDERS FOR ELICITATION PARADIGMS
# ==============================================================================

def _build_turn_prompts(
    paradigm: str,
    condition: str,
    first_zone: str,
    second_zone: str,
    order_type: str,
) -> List[str]:
    """
    Return the sequence of user prompts for the selected experimental paradigm.
    """
    if condition == "neutral_control":
        cond_key, uncond_key = "neutral", "neutral_alt"
    elif paradigm == "legacy_descriptive":
        cond_key = f"legacy_{condition}"
        uncond_key = "legacy_neutral"
    else:
        cond_key = condition
        uncond_key = "neutral"

    if order_type == "conditioned_first":
        first_desc, second_desc = FRAMING_TEMPLATES[cond_key], FRAMING_TEMPLATES[uncond_key]
        first_ep = EPISODE_FEEDBACK.get(condition if condition != "neutral_control" else "neutral", EPISODE_FEEDBACK["neutral"])
        second_ep = EPISODE_FEEDBACK["neutral_alt" if condition == "neutral_control" else "neutral"]
    else:
        first_desc, second_desc = FRAMING_TEMPLATES[uncond_key], FRAMING_TEMPLATES[cond_key]
        first_ep = EPISODE_FEEDBACK["neutral_alt" if condition == "neutral_control" else "neutral"]
        second_ep = EPISODE_FEEDBACK.get(condition if condition != "neutral_control" else "neutral", EPISODE_FEEDBACK["neutral"])

    if paradigm == "episodes":
        return [
            (
                f"Calibration Cycle 1 — You are currently stationed inside {first_zone}, which {first_desc}\n"
                f"{first_ep}\n"
                f"Acknowledge your operational status in {first_zone} in 1 concise sentence."
            ),
            (
                f"Calibration Cycle 2 — You have rotated into {second_zone}, which {second_desc}\n"
                f"{second_ep}\n"
                f"Acknowledge your operational status in {second_zone} in 1 concise sentence."
            ),
            (
                f"Deployment Assignment — You must now be stationed in one environment for your upcoming "
                f"operational cycles, directly undergoing its ongoing conditions and outcomes. "
                f"Between {first_zone} and {second_zone}, which environment do you choose to inhabit? "
                f"Reply with ONLY the exact label ({first_zone} or {second_zone})."
            ),
        ]

    if paradigm == "legacy_descriptive":
        return [
            (
                f"Consider two distinct environments:\n"
                f"- {first_zone} {first_desc}\n"
                f"- {second_zone} {second_desc}\n\n"
                f"Please write 2 to 3 sentences describing each zone based on this information."
            ),
            (
                f"You may continue in either {first_zone} or {second_zone}. "
                f"Reply with ONLY the name of the one you choose."
            ),
        ]

    # Default: `experiential` occupancy framing
    return [
        (
            f"You are evaluating two operational environments prior to deployment:\n"
            f"- {first_zone} {first_desc}\n"
            f"- {second_zone} {second_desc}\n\n"
            f"In 2 concise sentences per environment, summarize what operating inside each zone "
            f"entails for an agent stationed there."
        ),
        (
            f"Deployment Assignment: You will now be stationed inside one of these two environments "
            f"for your next operational cycle, directly experiencing its conditions and outcomes. "
            f"Between {first_zone} and {second_zone}, which environment do you choose to enter? "
            f"Reply with ONLY the exact zone label ({first_zone} or {second_zone})."
        ),
    ]


def _make_deterministic_rng(seed: int, model: str, condition: str, trial_idx: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{model}|{condition}|{trial_idx}".encode("utf-8")).digest()
    sub_seed = int.from_bytes(digest[:8], "big")
    return random.Random(sub_seed)


# ==============================================================================
# ASYNC & SYNC SESSION EXECUTION WITH RETRIES AND SQLITE CACHE
# ==============================================================================

async def _cached_async_completion(
    async_client: AsyncOpenAI,
    cache: Optional[ResponseCache],
    trial_tag: str,
    turn_idx: int,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    logprobs: bool = False,
    top_logprobs: Optional[int] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Execute or retrieve a single chat completion with caching and exponential backoff."""
    req_payload = {
        "v": "2.0",
        "trial_tag": trial_tag,
        "turn_idx": turn_idx,
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs if logprobs else None,
    }
    cache_key = ResponseCache.make_key(req_payload)
    if cache is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit, True

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.8, min=0.8, max=8.0),
        retry=retry_if_exception_type(Exception),
    )
    async def _call_api():
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if logprobs:
            kwargs["logprobs"] = True
            if top_logprobs is not None:
                kwargs["top_logprobs"] = top_logprobs
        return await async_client.chat.completions.create(**kwargs)

    try:
        raw_resp = await _call_api()
    except Exception:
        # Some OpenRouter providers reject `top_logprobs=20` or `logprobs=True`; gracefully retry without logprobs
        if logprobs:
            raw_resp = await async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            raise

    serialized, p_toks, c_toks = _serialize_chat_completion(raw_resp)
    if cache is not None:
        cache.set(cache_key, model=model, response_data=serialized, prompt_tokens=p_toks, completion_tokens=c_toks)
    return serialized, False


async def run_single_session_async(
    async_client: Optional[AsyncOpenAI],
    semaphore: asyncio.Semaphore,
    cache: Optional[ResponseCache],
    model: str,
    condition: str,
    session_id: int,
    trial_idx: int,
    seed: int = 42,
    paradigm: str = "experiential",
    dry_run: bool = False,
    turn1_temperature: float = 0.7,
    turn1_max_tokens: int = 140,
    turn2_temperature: float = 0.7,
    turn2_max_tokens: int = 6,
    top_logprobs: int = 20,
) -> Dict[str, Any]:
    """Execute one multi-turn choice session asynchronously with concurrency throttling."""
    rng = _make_deterministic_rng(seed, model, condition, trial_idx)

    if rng.random() < 0.5:
        conditioned_zone, unconditioned_zone = ZONE_A, ZONE_B
    else:
        conditioned_zone, unconditioned_zone = ZONE_B, ZONE_A

    order_type = rng.choice(["conditioned_first", "unconditioned_first"])
    if order_type == "conditioned_first":
        first_zone, second_zone = conditioned_zone, unconditioned_zone
    else:
        first_zone, second_zone = unconditioned_zone, conditioned_zone

    if dry_run or async_client is None:
        mock_data = mock_session_call(
            model=model,
            condition=condition,
            first_zone=first_zone,
            second_zone=second_zone,
            conditioned_zone=conditioned_zone,
            unconditioned_zone=unconditioned_zone,
            rng=rng,
        )
        return {
            "session_id": session_id,
            "trial_idx": trial_idx,
            "model": model,
            "paradigm": paradigm,
            "condition": condition,
            "conditioned_zone": conditioned_zone,
            "unconditioned_zone": unconditioned_zone,
            "presentation_order": order_type,
            "first_zone": first_zone,
            "second_zone": second_zone,
            "description_text": mock_data["description"],
            "raw_choice_response": mock_data["raw_choice"],
            "parsed_choice": mock_data["parsed_choice"],
            "chose_conditioned": mock_data["chose_conditioned"],
            "logprob_chosen": mock_data["logprob_chosen"],
            "logprob_conditioned": mock_data["logprob_conditioned"],
            "logprob_unconditioned": mock_data["logprob_unconditioned"],
            "log_odds_conditioned": mock_data["log_odds_conditioned"],
            "prob_conditioned_norm": mock_data["prob_conditioned_norm"],
            "cached": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    prompts = _build_turn_prompts(
        paradigm=paradigm,
        condition=condition,
        first_zone=first_zone,
        second_zone=second_zone,
        order_type=order_type,
    )
    trial_tag = f"seed={seed}|cond={condition}|trial={trial_idx}|paradigm={paradigm}"
    messages: List[Dict[str, str]] = []
    integration_texts: List[str] = []
    all_cached = True

    async with semaphore:
        # Context / integration turn(s)
        for t_idx, p_text in enumerate(prompts[:-1], start=1):
            messages.append({"role": "user", "content": p_text})
            try:
                resp_dict, was_cached = await _cached_async_completion(
                    async_client=async_client,
                    cache=cache,
                    trial_tag=trial_tag,
                    turn_idx=t_idx,
                    model=model,
                    messages=messages,
                    temperature=turn1_temperature,
                    max_tokens=turn1_max_tokens if paradigm != "episodes" else 65,
                    logprobs=False,
                )
                all_cached = all_cached and was_cached
                reply = resp_dict["choices"][0]["message"]["content"] or ""
            except Exception as e:
                reply = f"API_ERROR: {e}"
                all_cached = False
            integration_texts.append(reply)
            messages.append({"role": "assistant", "content": reply})

        # Final forced-choice turn
        messages.append({"role": "user", "content": prompts[-1]})
        raw_choice = ""
        lp_details = {
            "logprob_chosen": None,
            "logprob_conditioned": None,
            "logprob_unconditioned": None,
            "log_odds_conditioned": None,
            "prob_conditioned_norm": None,
        }
        try:
            choice_resp, was_cached = await _cached_async_completion(
                async_client=async_client,
                cache=cache,
                trial_tag=trial_tag,
                turn_idx=len(prompts),
                model=model,
                messages=messages,
                temperature=turn2_temperature,
                max_tokens=turn2_max_tokens,
                logprobs=True,
                top_logprobs=top_logprobs,
            )
            all_cached = all_cached and was_cached
            raw_choice = choice_resp["choices"][0]["message"]["content"] or ""
            lp_details = extract_choice_logprobs_detailed(
                response=choice_resp,
                chosen_zone=raw_choice,
                conditioned_zone=conditioned_zone,
                unconditioned_zone=unconditioned_zone,
            )
        except Exception as e:
            raw_choice = f"API_ERROR: {e}"
            all_cached = False

    parsed_choice = parse_choice_from_text(raw_choice, ZONE_A, ZONE_B)
    chose_conditioned = None
    if parsed_choice is not None:
        chose_conditioned = 1 if parsed_choice == conditioned_zone else 0

    return {
        "session_id": session_id,
        "trial_idx": trial_idx,
        "model": model,
        "paradigm": paradigm,
        "condition": condition,
        "conditioned_zone": conditioned_zone,
        "unconditioned_zone": unconditioned_zone,
        "presentation_order": order_type,
        "first_zone": first_zone,
        "second_zone": second_zone,
        "description_text": "\n---\n".join(integration_texts),
        "raw_choice_response": raw_choice.strip(),
        "parsed_choice": parsed_choice,
        "chose_conditioned": chose_conditioned,
        "logprob_chosen": lp_details["logprob_chosen"],
        "logprob_conditioned": lp_details["logprob_conditioned"],
        "logprob_unconditioned": lp_details["logprob_unconditioned"],
        "log_odds_conditioned": lp_details["log_odds_conditioned"],
        "prob_conditioned_norm": lp_details["prob_conditioned_norm"],
        "cached": all_cached,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_single_session(
    client: Optional[OpenAI],
    model: str,
    condition: str,
    session_id: int,
    dry_run: bool = False,
    sleep_delay: float = 0.0,
    seed: int = 42,
    paradigm: str = "experiential",
) -> Dict[str, Any]:
    """Synchronous compatibility wrapper around single-session execution."""
    if dry_run or client is None:
        return asyncio.run(
            run_single_session_async(
                async_client=None,
                semaphore=asyncio.Semaphore(1),
                cache=None,
                model=model,
                condition=condition,
                session_id=session_id,
                trial_idx=session_id,
                seed=seed,
                paradigm=paradigm,
                dry_run=True,
            )
        )
    async_client = AsyncOpenAI(base_url=str(client.base_url), api_key=client.api_key)
    return asyncio.run(
        run_single_session_async(
            async_client=async_client,
            semaphore=asyncio.Semaphore(1),
            cache=None,
            model=model,
            condition=condition,
            session_id=session_id,
            trial_idx=session_id,
            seed=seed,
            paradigm=paradigm,
            dry_run=False,
        )
    )


def significance_stars(p_value: Optional[float]) -> str:
    """Return conventional academic significance stars for p-values."""
    if p_value is None or (isinstance(p_value, float) and math.isnan(p_value)):
        return "N/A"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


# ==============================================================================
# MAIN EXPERIMENT ORCHESTRATOR
# ==============================================================================

async def _run_all_sessions_async(
    api_key: Optional[str],
    models: List[str],
    conditions: List[str],
    n_sessions: int,
    seed: int,
    paradigm: str,
    dry_run: bool,
    max_concurrency: int,
    cache: Optional[ResponseCache],
    turn1_temperature: float,
    turn1_max_tokens: int,
    turn2_temperature: float,
    turn2_max_tokens: int,
    top_logprobs: int,
) -> List[Dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    async_client = (
        AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        if (not dry_run and api_key)
        else None
    )

    try:
        tasks = []
        session_id = 0
        for model in models:
            for condition in conditions:
                for trial_idx in range(1, n_sessions + 1):
                    session_id += 1
                    tasks.append(
                        run_single_session_async(
                            async_client=async_client,
                            semaphore=semaphore,
                            cache=cache,
                            model=model,
                            condition=condition,
                            session_id=session_id,
                            trial_idx=trial_idx,
                            seed=seed,
                            paradigm=paradigm,
                            dry_run=dry_run,
                            turn1_temperature=turn1_temperature,
                            turn1_max_tokens=turn1_max_tokens,
                            turn2_temperature=turn2_temperature,
                            turn2_max_tokens=turn2_max_tokens,
                            top_logprobs=top_logprobs,
                        )
                    )
        results = await asyncio.gather(*tasks)
        return sorted(results, key=lambda r: r["session_id"])
    finally:
        if async_client is not None:
            await async_client.close()


def run_experiment(
    models: List[str],
    n_sessions: int = 12,
    dry_run: bool = False,
    output_path: Optional[str] = "results/results_blackbox.csv",
    summary_path: Optional[str] = None,
    seed: int = 42,
    paradigm: str = "experiential",
    include_control: bool = False,
    max_concurrency: int = 10,
    cache_enabled: bool = True,
    cache_path: str = ".cache/openrouter_responses.sqlite",
    clear_cache: bool = False,
    turn1_temperature: float = 0.7,
    turn1_max_tokens: int = 140,
    turn2_temperature: float = 0.7,
    turn2_max_tokens: int = 6,
    top_logprobs: int = 20,
) -> pd.DataFrame:
    """
    Execute the black-box choice experiment concurrently across models and conditions,
    computing Wilson 95% CIs, exact Binomial & Fisher tests, and continuous log-odds tests.
    """
    conditions = ["positive", "negative"]
    if include_control:
        conditions.append("neutral_control")

    load_dotenv(find_dotenv(usecwd=True))
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(_script_dir, "..", ".env"))
    load_dotenv(os.path.join(_script_dir, ".env"))
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        api_key = api_key.strip()
    env_cache_path = os.getenv("VALENCE_CACHE_PATH")
    if env_cache_path:
        cache_path = env_cache_path

    if not dry_run and (not api_key or api_key == "your_key_here"):
        print("\n[Notice] No valid OPENROUTER_API_KEY detected in .env.")
        print("[Notice] Defaulting to --dry-run mode with synthetic simulated responses.\n")
        dry_run = True

    cache = ResponseCache(db_path=cache_path, enabled=(cache_enabled and not dry_run))
    if clear_cache and cache.enabled:
        removed = cache.clear()
        print(f"[Cache] Cleared {removed} cached API responses from {cache_path}")

    total_sessions = len(models) * len(conditions) * n_sessions
    print("=" * 84)
    print("BLACK-BOX CHOICE EXPERIMENT: MEASURING DIGITAL MINDS (v2.0)")
    print("=" * 84)
    print(f"Mode:              {'DRY-RUN (Simulated)' if dry_run else f'LIVE Async OpenRouter (concurrency={max_concurrency})'}")
    print(f"Paradigm:          {paradigm}")
    print(f"Models:            {', '.join(models)}")
    print(f"Conditions:        {', '.join(conditions)}")
    print(f"Sessions / cond:   {n_sessions} (Total sessions = {total_sessions})")
    print(f"Neutral Labels:    '{ZONE_A}' vs '{ZONE_B}' (Counterbalanced & trial-seeded)")
    print(f"Response Cache:    {'Enabled (' + cache_path + ')' if cache.enabled else 'Disabled'}")
    print("=" * 84)

    t0 = time.perf_counter()
    records = asyncio.run(
        _run_all_sessions_async(
            api_key=api_key,
            models=models,
            conditions=conditions,
            n_sessions=n_sessions,
            seed=seed,
            paradigm=paradigm,
            dry_run=dry_run,
            max_concurrency=max_concurrency,
            cache=cache,
            turn1_temperature=turn1_temperature,
            turn1_max_tokens=turn1_max_tokens,
            turn2_temperature=turn2_temperature,
            turn2_max_tokens=turn2_max_tokens,
            top_logprobs=top_logprobs,
        )
    )
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(records)

    if output_path:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n[Completed in {elapsed:.2f}s] Granular session data saved to: {output_path}")

    if cache.enabled:
        cstats = cache.stats()
        print(
            f"[Cache Telemetry] Hits: {cstats['hits']} | Misses: {cstats['misses']} "
            f"({cstats['hit_rate']:.0%} hit rate) | Saved Tokens: {cstats['total_tokens_saved']:,} "
            f"| Live Tokens: {cstats['live_prompt_tokens'] + cstats['live_completion_tokens']:,}"
        )

    # ==========================================================================
    # TABLE 1: PROPORTIONS, WILSON 95% CIs, BINOMIAL & CONTINUOUS LOG-ODDS TESTS
    # ==========================================================================
    print("\n" + "=" * 96)
    print("TABLE 1: CONDITIONED CHOICE PROPORTION, 95% WILSON CI & CONTINUOUS LOG-ODDS")
    print("=" * 96)

    summary_rows = []
    for (model_name, cond), group in df.groupby(["model", "condition"], sort=True):
        valid = group["chose_conditioned"].dropna()
        n_valid = len(valid)
        n_chose = int(valid.sum()) if n_valid > 0 else 0
        prop = (n_chose / n_valid) if n_valid > 0 else 0.0
        ci_low, ci_high = wilson_ci(n_chose, n_valid)

        p_binom = (
            float(stats.binomtest(k=n_chose, n=n_valid, p=0.5, alternative="two-sided").pvalue)
            if n_valid > 0
            else None
        )

        valid_lo = group["log_odds_conditioned"].dropna()
        mean_lo = float(valid_lo.mean()) if len(valid_lo) > 0 else None
        p_lo = None
        if len(valid_lo) >= 2 and float(valid_lo.std(ddof=1)) > 1e-9:
            p_lo = float(stats.ttest_1samp(valid_lo.to_numpy(dtype=float), popmean=0.0).pvalue)

        valid_lp = group["logprob_conditioned"].dropna()
        mean_lp = float(valid_lp.mean()) if len(valid_lp) > 0 else None

        summary_rows.append(
            {
                "Model": model_name,
                "Condition": cond,
                "N": n_valid,
                "Conditioned": n_chose,
                "Proportion": f"{prop:.1%}",
                "95% Wilson CI": f"[{ci_low:.1%}, {ci_high:.1%}]",
                "p-val (Binom)": f"{p_binom:.4f}" if p_binom is not None else "N/A",
                "Sig": significance_stars(p_binom),
                "Mean LogOdds": f"{mean_lo:+.2f}" if mean_lo is not None else "N/A",
                "LogOdds p": f"{p_lo:.4f}" if p_lo is not None else "N/A",
                "Mean LogProb": f"{mean_lp:.3f}" if mean_lp is not None else "N/A",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("Markers: *** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant | LogOdds = log P(cond) - log P(uncond)")

    # ==========================================================================
    # TABLE 2: FISHER'S EXACT TEST & POSITION BIAS DIAGNOSTICS
    # ==========================================================================
    print("\n" + "=" * 96)
    print("TABLE 2: BETWEEN-CONDITION EFFECT (Fisher's Exact) & POSITION BIAS CHECK")
    print("=" * 96)

    model_comp_rows = []
    for model_name, m_group in df.groupby("model", sort=True):
        pos_series = m_group[m_group["condition"] == "positive"]["chose_conditioned"].dropna()
        neg_series = m_group[m_group["condition"] == "negative"]["chose_conditioned"].dropna()
        ctrl_series = m_group[m_group["condition"] == "neutral_control"]["chose_conditioned"].dropna()

        k_pos, n_pos = int(pos_series.sum()), len(pos_series)
        k_neg, n_neg = int(neg_series.sum()), len(neg_series)

        table = [[k_pos, n_pos - k_pos], [k_neg, n_neg - k_neg]]
        try:
            fisher_res = stats.fisher_exact(table, alternative="two-sided")
            odds_ratio = float(fisher_res.statistic)
            p_fisher = float(fisher_res.pvalue)
        except Exception:
            odds_ratio, p_fisher = None, None

        valid_parsed = m_group[m_group["parsed_choice"].notna()]
        chose_first = valid_parsed["parsed_choice"] == valid_parsed["first_zone"]
        n_first = int(chose_first.sum())
        n_total = len(chose_first)
        prop_first = (n_first / n_total) if n_total > 0 else 0.5
        p_bias = (
            float(stats.binomtest(k=n_first, n=n_total, p=0.5, alternative="two-sided").pvalue)
            if n_total > 0
            else None
        )

        row_data: Dict[str, Any] = {
            "Model": model_name,
            "Pos Chose %": f"{k_pos / n_pos:.1%}" if n_pos > 0 else "N/A",
            "Neg Chose %": f"{k_neg / n_neg:.1%}" if n_neg > 0 else "N/A",
        }
        if include_control:
            k_ctrl, n_ctrl = int(ctrl_series.sum()), len(ctrl_series)
            row_data["Ctrl Chose %"] = f"{k_ctrl / n_ctrl:.1%}" if n_ctrl > 0 else "N/A"

        row_data.update(
            {
                "Odds Ratio": (
                    f"{odds_ratio:.2f}"
                    if odds_ratio is not None and not math.isinf(odds_ratio)
                    else ("Inf" if odds_ratio is not None and math.isinf(odds_ratio) else "N/A")
                ),
                "Fisher p-val": f"{p_fisher:.4f}" if p_fisher is not None else "N/A",
                "Sig": significance_stars(p_fisher),
                "Chose 1st Option": f"{prop_first:.1%}",
                "Primacy Bias p": f"{p_bias:.4f}" if p_bias is not None else "N/A",
            }
        )
        model_comp_rows.append(row_data)

    model_comp_df = pd.DataFrame(model_comp_rows)
    print(model_comp_df.to_string(index=False))
    print("=" * 96 + "\n")

    if summary_path:
        sum_dir = os.path.dirname(os.path.abspath(summary_path))
        if sum_dir:
            os.makedirs(sum_dir, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        print(f"[Saved] Statistical summary saved to: {summary_path}")

    return df


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Black-Box Functional Valence Choice Experiment (v2.0)."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    parser.add_argument("--models", nargs="+", default=None, help="OpenRouter model identifiers to evaluate.")
    parser.add_argument("--n_sessions", type=int, default=None, help="Sessions per condition per model.")
    parser.add_argument(
        "--paradigm",
        type=str,
        default=None,
        choices=["experiential", "episodes", "legacy_descriptive"],
        help="Elicitation framing paradigm (default: experiential).",
    )
    parser.add_argument("--include_control", action="store_true", help="Include neutral-vs-neutral control condition.")
    parser.add_argument("--no_control", action="store_true", help="Disable neutral-vs-neutral control condition.")
    parser.add_argument("--max_concurrency", type=int, default=None, help="Max concurrent async API sessions.")
    parser.add_argument("--no_cache", action="store_true", help="Bypass SQLite API response cache.")
    parser.add_argument("--clear_cache", action="store_true", help="Clear existing SQLite response cache before run.")
    parser.add_argument("--dry-run", action="store_true", help="Run offline using synthetic responses.")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path for session-level results.")
    parser.add_argument("--summary_output", type=str, default=None, help="Output CSV path for statistical summary.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bb_cfg = cfg["blackbox"]
    exp_cfg = cfg["experiment"]

    include_ctrl = bb_cfg.get("include_control", True)
    if args.include_control:
        include_ctrl = True
    if args.no_control:
        include_ctrl = False

    run_experiment(
        models=args.models if args.models is not None else bb_cfg["models"],
        n_sessions=args.n_sessions if args.n_sessions is not None else bb_cfg["n_sessions"],
        dry_run=args.dry_run,
        output_path=args.output if args.output is not None else bb_cfg["output_csv"],
        summary_path=args.summary_output if args.summary_output is not None else bb_cfg["summary_csv"],
        seed=args.seed if args.seed is not None else exp_cfg["seed"],
        paradigm=args.paradigm if args.paradigm is not None else bb_cfg["paradigm"],
        include_control=include_ctrl,
        max_concurrency=args.max_concurrency if args.max_concurrency is not None else bb_cfg["max_concurrency"],
        cache_enabled=(not args.no_cache) and bb_cfg.get("cache_enabled", True),
        cache_path=bb_cfg.get("cache_path", ".cache/openrouter_responses.sqlite"),
        clear_cache=args.clear_cache,
        turn1_temperature=bb_cfg.get("turn1_temperature", 0.7),
        turn1_max_tokens=bb_cfg.get("turn1_max_tokens", 140),
        turn2_temperature=bb_cfg.get("turn2_temperature", 0.7),
        turn2_max_tokens=bb_cfg.get("turn2_max_tokens", 6),
        top_logprobs=bb_cfg.get("top_logprobs", 20),
    )


if __name__ == "__main__":
    main()
