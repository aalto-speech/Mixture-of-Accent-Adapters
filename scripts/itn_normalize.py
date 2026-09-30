#!/usr/bin/env python
"""
Text normalization before scoring: lowercase, strip punctuation, then NeMo English
Inverse Text Normalization (ITN) on BOTH `pred_text` and `ref_text`
(e.g. "twenty five" -> "25", so number formats do not count as errors).

Input : prediction CSV(s) with columns pred_text, ref_text
Output: <input_stem><suffix>.csv next to the input (or in --out_dir)

Example:
  python scripts/itn_normalize.py outputs/moaa_linear_proj_A20/predictions/aesrc20h.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd

# Keep letters, digits, spaces, apostrophe, dash. Remove everything else (.,?!,: etc.)
_KEEP = re.compile(r"[^a-z0-9\s'\-]+")


def prep_for_itn(s) -> str:
    s = "" if pd.isna(s) else str(s).lower()
    s = _KEEP.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="Prediction CSV file(s).")
    p.add_argument("--out_dir", default=None, help="Default: same directory as each input.")
    p.add_argument("--suffix", default="_itn")
    p.add_argument("--pred_col", default="pred_text")
    p.add_argument("--ref_col", default="ref_text")
    args = p.parse_args()

    from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer

    inverse_normalizer = InverseNormalizer(lang="en", input_case="lower_cased")

    for inp in args.inputs:
        inp = Path(inp)
        df = pd.read_csv(inp)
        pred_in = [prep_for_itn(x) for x in df[args.pred_col]]
        ref_in = [prep_for_itn(x) for x in df[args.ref_col]]

        df[args.pred_col] = inverse_normalizer.inverse_normalize_list(pred_in, verbose=False)
        df[args.ref_col] = inverse_normalizer.inverse_normalize_list(ref_in, verbose=False)

        out_dir = Path(args.out_dir) if args.out_dir else inp.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{inp.stem}{args.suffix}.csv"
        df.to_csv(out_path, index=False)
        print(f"[ITN] {inp} -> {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
