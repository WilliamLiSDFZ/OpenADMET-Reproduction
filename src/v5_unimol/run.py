"""End-to-end v5_unimol runner.

  python -m src.v5_unimol.run                  # full pipeline
  python -m src.v5_unimol.run --skip-train     # rebuild submission from saved ckpts
  python -m src.v5_unimol.run --eval           # opt-in: run official eval after
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd

from . import config as cfg
from .predict import build_submission
from .train import train_all


def run_official_eval(submission_csv: Path) -> Path | None:
    """Optional opt-in step. Mirrors v4_hybridadmet.run.run_official_eval.

    Most servers don't have the eval repo cloned; default behavior is to
    skip this and only write the submission. Set --eval to enable.
    """
    metrics_path = cfg.V5_OUT / "official_eval_v5.csv"
    if not Path(cfg.EVAL_REPO).is_dir():
        print(f"\n  WARN: official-eval repo not at {cfg.EVAL_REPO}.")
        print("  Score it manually with:")
        print(f"    cd <ExpansionRx-Challenge-Eval>")
        print(f"    python -m eval {submission_csv} \\")
        print(f"        --ground-truth {cfg.GROUND_TRUTH_CSV} \\")
        print(f"        --output {metrics_path}")
        return None

    cmd = [sys.executable, "-m", "eval", str(submission_csv),
           "--ground-truth", str(cfg.GROUND_TRUTH_CSV),
           "--output", str(metrics_path)]
    print(f"Running official eval -> {metrics_path}")
    print("  $ " + " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=cfg.EVAL_REPO, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  WARN: eval failed: {e}")
        return None
    res = pd.read_csv(metrics_path)
    print(res[["Endpoint", "mean_RAE", "mean_R2"]].to_string(index=False))
    macro = res[res["Endpoint"].str.contains("Macro", case=False, na=False)]
    if not macro.empty:
        r = macro.iloc[0]
        print(f"\n   *** v5_unimol MA-RAE = {r['mean_RAE']:.3f}  "
              f"R² = {r['mean_R2']:+.3f}")
    return metrics_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse cached per-seed test predictions if present")
    ap.add_argument("--eval", action="store_true",
                    help="run official `python -m eval` after writing the submission. "
                         "OFF by default — most servers don't have the eval repo. "
                         "Set EVAL_REPO=/path/to/ExpansionRx-Challenge-Eval to enable.")
    args = ap.parse_args()

    print(f"v5 output dir: {cfg.V5_OUT}")
    train_df = pd.read_csv(cfg.TRAIN_CSV)
    test_df  = pd.read_csv(cfg.TEST_CSV)
    print(f"  train={len(train_df)}  test={len(test_df)}")

    if not args.skip_train:
        print("\n[1/2] Training all (group × seed) Uni-Mol2-only models")
        results = train_all(train_df, test_df)
        with open(cfg.V5_OUT / "results.pkl", "wb") as f:
            pickle.dump(results, f)
    else:
        with open(cfg.V5_OUT / "results.pkl", "rb") as f:
            results = pickle.load(f)
        print("\n[1/2] Loaded cached training results")

    print("\n[2/2] Aggregating into submission")
    submission = build_submission(results, test_df)
    sub_path = cfg.V5_OUT / "submission_v5.csv"
    submission.to_csv(sub_path, index=False)
    print(f"  ✓ Wrote {sub_path}")
    print()
    print("Done. To score, copy submission_v5.csv to a machine that has the")
    print("official ExpansionRx-Challenge-Eval repo and run:")
    print()
    print("    cd ExpansionRx-Challenge-Eval")
    print("    python -m eval /path/to/submission_v5.csv \\")
    print("        --ground-truth /path/to/test_ground_truth.csv \\")
    print("        --output /path/to/official_eval_v5.csv")

    if args.eval and cfg.GROUND_TRUTH_CSV.exists():
        print("\n[opt-in] --eval requested; running official eval")
        run_official_eval(sub_path)


if __name__ == "__main__":
    main()
