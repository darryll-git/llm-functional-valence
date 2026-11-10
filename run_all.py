#!/usr/bin/env python3
"""
run_all.py
----------
Config-driven, cache-aware master pipeline runner for the "Measuring Digital Minds"
Functional Valence Research Demo (v2.0).

Stages:
  1. `tests`:    Unit & integration verification suite (`pytest tests/`)
  2. `blackbox`: Async OpenRouter behavioral choice experiment + SQLite caching
  3. `whitebox`: GPU-batched probing, full-depth layer sweep, steering & causal bridge
  4. `plot`:     Publication figures rendered from empirical artifacts (`figures/`)

Usage:
  # Instant offline verification (~10s):
  .venv/Scripts/python run_all.py --dry-run

  # Run only GPU white-box + empirical figures:
  .venv/Scripts/python run_all.py --only whitebox plot

  # Full live pipeline skipping stages whose artifacts already exist:
  .venv/Scripts/python run_all.py --skip-cached
"""

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, List

from common.config import get_hardware_summary, load_config, save_run_manifest

ALL_STAGES = ["tests", "blackbox", "whitebox", "plot"]


def print_banner(title: str):
    print("\n" + "=" * 84)
    print(f" {title}")
    print("=" * 84 + "\n")


def run_step(cmd: List[str], description: str) -> float:
    print_banner(f"STAGE: {description}")
    print(f"Command: {' '.join(cmd)}\n")
    start = time.perf_counter()
    result = subprocess.run(cmd)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        print(f"\n[ERROR] Stage '{description}' failed with exit code {result.returncode} ({elapsed:.2f}s)")
        sys.exit(result.returncode)
    print(f"\n[SUCCESS] Completed in {elapsed:.2f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Master runner for the Valence Demo v2.0 pipeline.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file.")
    parser.add_argument("--dry-run", action="store_true", help="Execute pipeline in offline simulation mode.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=ALL_STAGES,
        default=None,
        help="Run only specified pipeline stage(s), e.g., `--only whitebox plot`.",
    )
    parser.add_argument(
        "--skip-cached",
        action="store_true",
        help="Skip blackbox or whitebox stage if its output artifacts already exist on disk.",
    )
    parser.add_argument("--n_sessions", type=int, default=None, help="Override black-box sessions per condition.")
    parser.add_argument("--layer", type=int, default=None, help="Override target white-box decoder layer.")
    parser.add_argument("--alpha", type=float, default=None, help="Override steering multiplier alpha.")
    parser.add_argument("--load_in_4bit", action="store_true", help="Enable 4-bit NF4 quantization for white-box LLM.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    python_bin = sys.executable

    cfg = load_config(args.config)
    hw = get_hardware_summary()
    stages = args.only if args.only else ALL_STAGES
    timings: Dict[str, float] = {}

    bb_out = cfg["blackbox"]["output_csv"]
    bb_sum = cfg["blackbox"]["summary_csv"]
    wb_npz = cfg["whitebox"]["artifacts_npz"]
    wb_json = cfg["whitebox"]["summary_json"]
    fig_dir = cfg["experiment"]["figures_dir"]

    print("*" * 84)
    print(" MEASURING DIGITAL MINDS: FUNCTIONAL VALENCE PIPELINE (v2.0)")
    print(f" Mode:        {'DRY-RUN / OFFLINE' if args.dry_run else 'LIVE EXECUTION'}")
    print(f" Stages:      {', '.join(stages)}")
    print(f" Python:      {python_bin} (v{hw['python_version']})")
    print(f" Accelerator: {hw['device_name']} | PyTorch: {hw['torch_version']} | VRAM: {hw['vram_gb'] or 'N/A'} GB")
    print("*" * 84)

    # Stage 1: Tests
    if "tests" in stages:
        timings["tests"] = run_step([python_bin, "-m", "pytest", "tests/", "-v"], "1/4 Unit & Integration Tests")

    # Stage 2: Black-Box Experiment
    if "blackbox" in stages:
        if args.skip_cached and os.path.exists(bb_out):
            print(f"\n[Skipped] Black-box artifact already exists at '{bb_out}' (--skip-cached).")
            timings["blackbox"] = 0.0
        else:
            bb_cmd = [
                python_bin,
                "blackbox/choice_experiment.py",
                "--output",
                bb_out,
                "--summary_output",
                bb_sum,
            ]
            if args.config:
                bb_cmd.extend(["--config", args.config])
            if args.n_sessions is not None:
                bb_cmd.extend(["--n_sessions", str(args.n_sessions)])
            if args.dry_run:
                bb_cmd.append("--dry-run")
            timings["blackbox"] = run_step(bb_cmd, "2/4 Black-Box Behavioral Choice Experiment")

    # Stage 3: White-Box Experiment
    if "whitebox" in stages:
        if args.skip_cached and os.path.exists(wb_npz) and os.path.exists(wb_json):
            print(f"\n[Skipped] White-box artifacts already exist at '{wb_npz}' (--skip-cached).")
            timings["whitebox"] = 0.0
        else:
            wb_cmd = [
                python_bin,
                "whitebox/probe_and_steer.py",
                "--artifacts_npz",
                wb_npz,
                "--summary_json",
                wb_json,
                "--sweep_layers",
            ]
            if args.config:
                wb_cmd.extend(["--config", args.config])
            if args.layer is not None:
                wb_cmd.extend(["--layer", str(args.layer)])
            if args.alpha is not None:
                wb_cmd.extend(["--alpha", str(args.alpha)])
            if args.load_in_4bit:
                wb_cmd.append("--load_in_4bit")
            if args.dry_run:
                wb_cmd.append("--dry-run")
            timings["whitebox"] = run_step(wb_cmd, "3/4 White-Box Probing, Layer Sweep, Steering & Causal Bridge")

    # Stage 4: Publication Figures
    if "plot" in stages:
        plot_cmd = [
            python_bin,
            "visualization/plot_results.py",
            "--input_csv",
            bb_out,
            "--whitebox_npz",
            wb_npz,
            "--whitebox_json",
            wb_json,
            "--output_dir",
            fig_dir,
        ]
        if args.config:
            plot_cmd.extend(["--config", args.config])
        timings["plot"] = run_step(plot_cmd, "4/4 Publication Figures from Empirical Artifacts")

    manifest_path = os.path.join(cfg["experiment"]["results_dir"], "run_manifest.json")
    save_run_manifest(
        manifest_path,
        {
            "mode": "dry_run" if args.dry_run else "live",
            "stages_executed": stages,
            "stage_seconds": {k: round(v, 3) for k, v in timings.items()},
            "total_seconds": round(sum(timings.values()), 3),
            "artifacts": {
                "blackbox_csv": bb_out,
                "blackbox_summary_csv": bb_sum,
                "whitebox_npz": wb_npz,
                "whitebox_json": wb_json,
                "figures_dir": fig_dir,
            },
        },
    )

    print_banner("PIPELINE EXECUTION COMPLETE")
    for stg, secs in timings.items():
        print(f"  - Stage '{stg:<10}': {secs:>6.2f}s")
    print(f"  - Provenance Log   : {manifest_path}")
    print("=" * 84 + "\n")


if __name__ == "__main__":
    main()
