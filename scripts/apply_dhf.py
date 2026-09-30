#!/usr/bin/env python
"""
Apply DHF (Deterministic Hallucination Filter) to prediction CSV(s) and report
WER/CER before and after.

DHF is reference-free: only `pred_text` is used to detect and repair hallucinations.
`ref_text` is used ONLY for reporting scores.

Adds columns:
  dhf_suspicious   bool, hypothesis was flagged by the detector
  pred_text_dhf    hypothesis after DHF (unchanged when not flagged)

Recommended input: the ITN-normalized CSV produced by scripts/itn_normalize.py.

Example:
  python scripts/apply_dhf.py outputs/moaa_linear_proj_A20/predictions/aesrc20h_itn.csv
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moaa.dhf import flag_number_only_text, is_suspicious_hypothesis, pick_best_candidate_reference_free  # noqa: E402
from moaa.scoring import normalize_text, score_dataframe  # noqa: E402


def run_dhf(df: pd.DataFrame, pred_col: str = "pred_text") -> pd.DataFrame:
    df = df.copy()
    preds = df[pred_col].fillna("").astype(str)
    df["dhf_suspicious"] = preds.apply(is_suspicious_hypothesis)
    df["pred_text_dhf"] = preds
    mask = df["dhf_suspicious"]
    df.loc[mask, "pred_text_dhf"] = preds[mask].apply(pick_best_candidate_reference_free)
    return df


def _fmt(tag, s):
    return (f"  {tag:<40} WER={s['wer']:.6f}  CER={s['cer']:.6f}  "
            f"S/I/D/H={s['S']}/{s['I']}/{s['D']}/{s['H']}  (N_ref_words={s['n_ref_words']}, utts={s['n_utts']})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out_dir", default=None, help="Default: same directory as each input.")
    p.add_argument("--suffix", default="_dhf")
    p.add_argument("--pred_col", default="pred_text")
    p.add_argument("--ref_col", default="ref_text")
    p.add_argument("--report_number_only_ref_exclusion", action="store_true",
                   help="ANALYSIS ONLY (uses references): additionally report scores after excluding "
                        "utterances whose reference is purely numeric and whose original hypothesis is wrong.")
    args = p.parse_args()

    for inp in args.inputs:
        inp = Path(inp)
        df = pd.read_csv(inp)
        out = run_dhf(df, pred_col=args.pred_col)
        n_changed = int((out["pred_text_dhf"] != out[args.pred_col].fillna("").astype(str)).sum())

        print(f"\n[DHF] {inp}")
        print(f"  rows={len(out)}  flagged={int(out['dhf_suspicious'].sum())}  modified={n_changed}")
        report = {"input": str(inp), "rows": len(out), "flagged": int(out["dhf_suspicious"].sum()),
                  "modified": n_changed}

        has_refs = args.ref_col in out.columns
        if not has_refs:
            print(f"  no '{args.ref_col}' column: DHF applied, scoring skipped "
                  f"(see scripts/attach_references.py to add references)")
        else:
            before = score_dataframe(out, hyp_col=args.pred_col, ref_col=args.ref_col)
            after = score_dataframe(out, hyp_col="pred_text_dhf", ref_col=args.ref_col)
            print(_fmt("Before DHF", before))
            print(_fmt("After DHF", after))
            report.update({"before": before, "after": after})

        if args.report_number_only_ref_exclusion and has_refs:
            ref_num = out[args.ref_col].fillna("").astype(str).apply(flag_number_only_text) == 1
            wrong = out[args.ref_col].apply(normalize_text) != out[args.pred_col].apply(normalize_text)
            excluded = ref_num & wrong
            excl = score_dataframe(out[~excluded], hyp_col="pred_text_dhf", ref_col=args.ref_col)
            print(_fmt("After DHF + number-only-ref exclusion*", excl))
            print(f"  * analysis only (uses references): excluded {int(excluded.sum())} utterances")
            report["after_number_only_ref_exclusion"] = {"excluded": int(excluded.sum()), **excl}

        out_dir = Path(args.out_dir) if args.out_dir else inp.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{inp.stem}{args.suffix}.csv"
        out.to_csv(out_path, index=False)
        (out_dir / f"{inp.stem}{args.suffix}_report.json").write_text(json.dumps(report, indent=2))
        print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
