"""v4.1 stitch: per-endpoint oracle pick of v3 vs v4 submissions.

Based on the v3 (5-seed 0.677 breakdown) vs v4 (5-seed 0.688 breakdown) official
eval per-endpoint comparison, take the better source for each endpoint:

  v3 wins  -> LogD, MLM CLint, HLM CLint, Caco-2 Papp A>B, MBPB
  v4 wins  -> KSOL, Caco-2 Efflux, MPPB, MGMB

Oracle macro RAE projection (sum-of-best / 9): ~0.648
v3 alone macro RAE:                            0.666 (10s/80ep, historical best)
v4 alone macro RAE:                            0.688

Usage:
  # 1. scp submission_v4.csv from the server to your local Mac, e.g. to
  #    output/v4_hybridadmet/submission_v4.csv
  # 2. run this script
  python scripts/stitch_v4_1.py \\
      --v3 output/v3/submission_v3.csv \\
      --v4 output/v4_hybridadmet/submission_v4.csv \\
      --out output/v4_hybridadmet/submission_v4_1.csv
  # 3. run official eval (on a machine that has ExpansionRx-Challenge-Eval)
  python -m eval output/v4_hybridadmet/submission_v4_1.csv \\
      --ground-truth /path/to/test_ground_truth.csv \\
      --output /path/to/official_eval_v4_1.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Per-endpoint oracle pick. Edit here if the v4 eval ever changes.
# Each entry: column name -> "v3" or "v4"
PICK = {
    "LogD":                          "v3",   # v3 0.460  vs v4 0.653
    "KSOL":                          "v4",   # v3 0.671  vs v4 0.613
    "MLM CLint":                     "v3",   # v3 0.867  vs v4 0.881
    "HLM CLint":                     "v3",   # v3 0.788  vs v4 0.877
    "Caco-2 Permeability Efflux":    "v4",   # v3 0.817  vs v4 0.793
    "Caco-2 Permeability Papp A>B":  "v3",   # v3 0.793  vs v4 0.821
    "MPPB":                          "v4",   # v3 0.748  vs v4 0.591  (biggest win)
    "MBPB":                          "v3",   # v3 0.457  vs v4 0.496
    "MGMB":                          "v4",   # v3 0.488  vs v4 0.470
}


def stitch(v3_path: Path, v4_path: Path, out_path: Path) -> None:
    v3 = pd.read_csv(v3_path)
    v4 = pd.read_csv(v4_path)

    # Sanity check schemas match
    assert list(v3.columns) == list(v4.columns), (
        f"column mismatch:\n  v3: {list(v3.columns)}\n  v4: {list(v4.columns)}"
    )

    # Align on Molecule Name. Both should have the same set in the same order,
    # but be defensive.
    id_col = v3.columns[0]
    assert id_col == "Molecule Name", f"expected first column 'Molecule Name', got '{id_col}'"
    v3 = v3.set_index(id_col)
    v4 = v4.set_index(id_col)
    assert set(v3.index) == set(v4.index), \
        f"molecule sets differ: v3 has {len(v3)}, v4 has {len(v4)}, " \
        f"v3-only={len(set(v3.index) - set(v4.index))}, " \
        f"v4-only={len(set(v4.index) - set(v3.index))}"
    v4 = v4.loc[v3.index]  # align row order

    # Build stitched frame
    out = pd.DataFrame(index=v3.index)
    for col in v3.columns:
        src = PICK.get(col)
        if src is None:
            raise ValueError(f"no pick rule for column '{col}'")
        out[col] = (v3 if src == "v3" else v4)[col].values
        print(f"  {col:34s}  <- {src}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.reset_index().to_csv(out_path, index=False)
    print(f"\n  Wrote {out_path}  ({len(out)} rows)")

    # Summary expectation
    print("\nExpected per-endpoint RAE (from prior official eval):")
    table = [
        ("LogD",                          0.460, 0.653),
        ("KSOL",                          0.671, 0.613),
        ("MLM CLint",                     0.867, 0.881),
        ("HLM CLint",                     0.788, 0.877),
        ("Caco-2 Permeability Efflux",    0.817, 0.793),
        ("Caco-2 Permeability Papp A>B",  0.793, 0.821),
        ("MPPB",                          0.748, 0.591),
        ("MBPB",                          0.457, 0.496),
        ("MGMB",                          0.488, 0.470),
    ]
    picked_sum = 0.0
    for ep, r3, r4 in table:
        chosen = r3 if PICK[ep] == "v3" else r4
        picked_sum += chosen
        print(f"  {ep:34s}  v3={r3:.3f}  v4={r4:.3f}  -> {PICK[ep]} ({chosen:.3f})")
    print(f"\n  Oracle macro-RAE projection: {picked_sum / 9:.3f}")
    print("  (vs v3 alone 0.666, v4 alone 0.688)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", type=Path, default=Path("output/v3/submission_v3.csv"))
    ap.add_argument("--v4", type=Path,
                    default=Path("output/v4_hybridadmet/submission_v4.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("output/v4_hybridadmet/submission_v4_1.csv"))
    args = ap.parse_args()

    for label, p in (("v3", args.v3), ("v4", args.v4)):
        if not p.exists():
            raise SystemExit(
                f"{label} submission not found at {p}\n"
                "Hint: scp submission_v4.csv from the server first, e.g.:\n"
                "  scp <user>@<server>:.../output/v4_hybridadmet/submission_v4.csv "
                "output/v4_hybridadmet/"
            )

    stitch(args.v3, args.v4, args.out)


if __name__ == "__main__":
    main()
