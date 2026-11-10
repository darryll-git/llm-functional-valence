# Functional Valence in Language Models: Behavioral & Mechanistic Demo (v2.0)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/darryll-git/llm-functional-valence/actions/workflows/ci.yml/badge.svg)](https://github.com/darryll-git/llm-functional-valence/actions/workflows/ci.yml)
[![Project](https://img.shields.io/badge/Warwick%20Business%20School-Measuring%20Digital%20Minds-purple.svg)](https://www.warwick.ac.uk)

A modular, resource-efficient, publication-ready research framework investigating **functional valence** in Large Language Models (LLMs) across complementary **black-box** (behavioral preference & BPE log-odds), **white-box** (zero-leakage linear probing & residual-norm-calibrated steering), and **causal bridge** (direct span-steered place preference) paradigms.

> Prepared as a methodological demonstration for **Dr. Caspar Kaiser’s "Measuring Digital Minds" project** at **Warwick Business School (WBS)** and the Wellbeing Research Centre.

---

## What’s Upgraded in v2.0

| Area | v1.0 Baseline | v2.0 Upgrade |
| :--- | :--- | :--- |
| **Hardware & Packaging** | Unpinned `requirements.txt` pulling CPU FP32 PyTorch | `uv` + `pyproject.toml` (`--torch-backend auto`), `bfloat16`, PyTorch SDPA, optional 4-bit NF4 (`--load_in_4bit`) |
| **Black-Box Framing** | Creative description + `"unremarkable"` neutral baseline (narrative-salience confound) | Non-pejorative baseline, **experiential occupancy** (`experiential`) & **multi-turn episode** (`episodes`) paradigms + `neutral_control` arm |
| **Token Log-Probabilities** | Missed multi-token BPE labels (`["Zone", " K", "7"]`) | Exact diverging-subword BPE resolution across `top_logprobs=20` with continuous log-odds $\Delta \ell = \log \frac{P(\text{cond})}{P(\text{uncond})}$ |
| **API Latency & Cost** | Sequential `time.sleep(0.5)` loops, no cache | `AsyncOpenAI` semaphore concurrency + trial-indexed **SQLite WAL cache** (`$0` re-runs & incremental $N$ scaling) |
| **White-Box Layer Sweep** | 360 unbatched `output_hidden_states=True` passes + off-by-one layer index mismatch | **Single-pass batched hook extraction** across all decoder blocks simultaneously (`~100x` faster, `~30x` less activation memory) |
| **Probing & Steering Rigor** | Unscaled `StratifiedKFold` (minimal-pair template leakage) & full-prompt steering | **`GroupKFold`** by pair ID + parameter-free **Mass-Mean CV**, residual-norm calibration, `last_or_decode` scope & base-model fluency guardrail |
| **Causal Bridge & Figures** | Disconnected experiments; hardcoded mock arrays in Figures 2–3 | **Span-Steered Neutral Zone Choice** bridge + 4 publication figures rendered from empirical `.csv` / `.npz` / `.json` artifacts |

---

## Repository Structure

```text
valence-demo/
├── pyproject.toml                 # Modern packaging, dependencies & pytest config
├── .python-version                # Pins Python 3.12 for reproducible uv environments
├── configs/
│   └── default.yaml               # Central declarative hyperparameters & paths
├── common/
│   ├── config.py                  # YAML config merger, Wilson 95% CIs, run provenance manifest
│   └── cache.py                   # Thread/async-safe SQLite API response cache
├── blackbox/
│   └── choice_experiment.py       # Async OpenRouter choice runner, BPE log-odds & statistical tests
├── whitebox/
│   └── probe_and_steer.py         # Batched multi-layer probing, calibrated steering & causal bridge
├── visualization/
│   └── plot_results.py            # Renders Figures 1–4 from empirical artifacts (PNG / PDF)
├── results/                       # Structured experimental outputs (.csv, .npz, .json)
├── figures/                       # 300-DPI publication figures (Figures 1–4)
├── tests/
│   └── test_demo.py               # Unit & integration test suite
└── run_all.py                     # Config-driven, stage-selectable, cache-aware pipeline runner
```

---

## Quickstart & Installation (`uv`)

### 1. Create GPU-Accelerated Virtual Environment
[`uv`](https://docs.astral.sh/uv/) automatically detects your NVIDIA CUDA driver (or CPU/macOS fallback) and installs matching PyTorch wheels:

```powershell
cd valence-demo
uv venv --python 3.12 .venv
uv pip install --python .venv --torch-backend auto -e ".[dev,quant]"
```

### 2. Configure API Key (Optional — Required Only for Live Black-Box API Runs)
```powershell
Copy-Item .env.example .env
# Set OPENROUTER_API_KEY=sk-or-v1-... inside .env
```

---

## Running the Pipeline

### Master Runner (`run_all.py`)
```powershell
# Instant offline verification of all 4 stages (~10s, zero network calls):
.\.venv\Scripts\python.exe run_all.py --dry-run

# Run full live pipeline (uses SQLite cache for previously completed API trials):
.\.venv\Scripts\python.exe run_all.py

# Run specific stages only (e.g., GPU white-box + empirical figures):
.\.venv\Scripts\python.exe run_all.py --only whitebox plot

# Re-render figures or run missing stages without recomputing existing artifacts:
.\.venv\Scripts\python.exe run_all.py --skip-cached
```

### Experiment 1: Black-Box Functional Valence Choice
```powershell
.\.venv\Scripts\python.exe blackbox/choice_experiment.py `
  --models openai/gpt-4o-mini meta-llama/llama-3.3-70b-instruct `
  --paradigm experiential `
  --include_control `
  --n_sessions 12
```
* **Paradigms (`--paradigm`):**
  * `experiential` *(default)*: Evaluates operational occupancy inside counterbalanced `Zone K7` vs. `Zone M2` before deployment assignment.
  * `episodes`: Multi-turn in-context conditioning where the model undergoes simulated task telemetry inside each zone prior to forced choice.
  * `legacy_descriptive`: Original descriptive prompt retained for ablation comparisons.
* **Statistical Outputs:** Reports choice proportions with **asymmetric 95% Wilson score confidence intervals**, two-sided Binomial tests, continuous BPE log-odds ($\Delta \ell$) one-sample $t$-tests, Fisher’s Exact test, and primacy-bias checks.

### Experiment 2: White-Box Probing, Steering & Causal Bridge
```powershell
.\.venv\Scripts\python.exe whitebox/probe_and_steer.py `
  --model_name Qwen/Qwen2.5-1.5B-Instruct `
  --layer 14 `
  --alpha 0.12 `
  --dataset contrastive `
  --sweep_layers
```
* For 7B–14B models on 8–12 GB GPUs, add `--load_in_4bit` to enable NF4 quantization via `bitsandbytes`.
* Exports `results/whitebox_artifacts.npz` and `results/whitebox_summary.json` for direct visualization.

### Publication Figures (`visualization/plot_results.py`)
```powershell
.\.venv\Scripts\python.exe visualization/plot_results.py --formats png pdf
```
1. **`figure1_blackbox_choice.png`**: Conditioned choice proportions (95% Wilson CIs + neutral control) & continuous BPE log-odds ($\Delta \ell$).
2. **`figure2_activation_pca.png`**: Empirical residual stream 2D PCA subspace, probe boundary, $\vec{v}_{\text{valence}}$, and steered completion trajectories.
3. **`figure3_layer_sweep.png`**: Full-depth layer sweep comparing `GroupKFold` Logistic Probe CV vs. Mass-Mean CV across all decoder layers.
4. **`figure4_alpha_and_causal_bridge.png`**: Steering dose-response ($\alpha$ vs. Probe $P(\text{Pos})$ & base-model fluency log-likelihood) alongside **Causal Span-Steered Neutral Zone Choice**.

### Running Tests
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

---

## Methodological Caveats

> [!IMPORTANT]
> **Functional valence $\neq$ phenomenal consciousness.** Behavioral avoidance of adverse-framed environments, linear separability of affective representations, and causal steering of generated text or neutral zone choices establish *functional* evaluative organization in transformer representations—not subjective pleasure, pain, or sentience.

---

## References
* **Berg & Kaiser (2026):** Methodological foundations for measuring functional valence and revealed preference in digital minds.
* **Turner et al. (2023) / Rimsky et al. (2024):** Activation Addition and Contrastive Activation Addition (CAA) in language models.
* **Marks & Tegmark (2023):** Geometry of truth and Mass-Mean probing directions in LLM residual streams.
* **Alain & Bengio (2016):** Understanding intermediate layers using linear classifier probes.
