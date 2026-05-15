"""End-to-end v4_hybridadmet runner.

  python -m src.v4_hybridadmet.run                  # full pipeline
  python -m src.v4_hybridadmet.run --skip-train     # rebuild submission from saved ckpts
  python -m src.v4_hybridadmet.run --skip-fp        # don't recompute PaDEL FP (use cache)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from . import config as cfg
from .predict import build_submission
from .train import train_all


def run_official_eval(submission_csv: Path) -> Path | None:
    metrics_path = cfg.V4_OUT / "official_eval_v4.csv"
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
        print(f"\n   *** v4_hybridadmet MA-RAE = {r['mean_RAE']:.3f}  "
              f"R² = {r['mean_R2']:+.3f}")
    return metrics_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse cached per-seed test predictions if present")
    args = ap.parse_args()

    print(f"v4 output dir: {cfg.V4_OUT}")
    train_df = pd.read_csv(cfg.TRAIN_CSV)
    test_df  = pd.read_csv(cfg.TEST_CSV)
    print(f"  train={len(train_df)}  test={len(test_df)}")

    if not args.skip_train:
        print("\n[1/3] Training all (group × seed) HybridADMET models")
        results = train_all(train_df, test_df)
        # Cache results so --skip-train works next time
        import pickle
        with open(cfg.V4_OUT / "results.pkl", "wb") as f:
            pickle.dump(results, f)
    else:
        import pickle
        with open(cfg.V4_OUT / "results.pkl", "rb") as f:
            results = pickle.load(f)
        print("\n[1/3] Loaded cached training results")

    print("\n[2/3] Aggregating into submission")
    submission = build_submission(results, test_df)
    sub_path = cfg.V4_OUT / "submission_v4.csv"
    submission.to_csv(sub_path, index=False)
    print(f"  Wrote {sub_path}")

    if cfg.GROUND_TRUTH_CSV.exists():
        print("\n[3/3] Official eval")
        run_official_eval(sub_path)
    else:
        print("\n[3/3] Skipping official eval (no ground truth file at "
              f"{cfg.GROUND_TRUTH_CSV})")


if __name__ == "__main__":
    main()
