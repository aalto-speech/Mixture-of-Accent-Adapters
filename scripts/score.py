#!/usr/bin/env python
"""
Corpus-level WER / CER (micro-averaged, jiwer) for one or more prediction CSVs.

Example:
  python scripts/score.py outputs/*/predictions/*_itn_dhf.csv --hyp_col pred_text_dhf
  python scripts/score.py a.csv b.csv --out_csv results/table.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moaa.scoring import score_dataframe  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--hyp_col", default="pred_text", help="Falls back to pred_text if missing in a file.")
    p.add_argument("--ref_col", default="ref_text")
    p.add_argument("--drop_empty_refs", action="store_true")
    p.add_argument("--out_csv", default=None)
    args = p.parse_args()

    rows = []
    for inp in args.inputs:
        df = pd.read_csv(inp)
        if args.ref_col not in df.columns:
            print(f"{inp}: no '{args.ref_col}' column, skipped (use scripts/attach_references.py)")
            continue
        hyp_col = args.hyp_col if args.hyp_col in df.columns else "pred_text"
        s = score_dataframe(df, hyp_col=hyp_col, ref_col=args.ref_col, drop_empty_refs=args.drop_empty_refs)
        rows.append({"file": inp, "hyp_col": hyp_col, "wer_pct": 100 * s["wer"], "cer_pct": 100 * s["cer"], **s})
        print(f"{inp} [{hyp_col}]  WER={100 * s['wer']:.2f}%  CER={100 * s['cer']:.2f}%  "
              f"S/I/D/H={s['S']}/{s['I']}/{s['D']}/{s['H']}  utts={s['n_utts']}")

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"Saved table -> {args.out_csv}")


if __name__ == "__main__":
    main()
