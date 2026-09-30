#!/usr/bin/env python
"""
Re-attach reference transcripts (`ref_text`) to released prediction CSVs.

The prediction CSVs in `results/` are published WITHOUT reference transcripts, because
the test-set transcripts belong to the datasets (e.g. AESRC is not freely
redistributable). If you have the datasets locally, this script rebuilds `ref_text`
exactly as the evaluation code produced it: lower-cased transcript, tokenized and
decoded with the Whisper tokenizer, and for EdAcc with the same >448-token filter. Rows
are matched by position and every row's `utt_id` is checked.

Examples:
  source configs/paths.sh
  # raw predictions
  python scripts/attach_references.py results/dhf_reference/method_linear_proj_aesrc_adapters1.csv \
      --test_set aesrc20h --out outputs/dhf_check/aesrc_adapters1_with_refs.csv
  # then normalise and apply DHF exactly as in the paper pipeline
  python scripts/itn_normalize.py outputs/dhf_check/aesrc_adapters1_with_refs.csv
  python scripts/apply_dhf.py outputs/dhf_check/aesrc_adapters1_with_refs_itn.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moaa.data import EXTERNAL_TEST_SETS  # noqa: E402
from moaa.utils import env_default  # noqa: E402

ENV_VARS = {
    "aesrc20h": "MOAA_AESRC_TEST",
    "openslr": "MOAA_OPENSLR_TEST",
    "edacc": "MOAA_EDACC_TEST",
    "globe": "MOAA_GLOBE_TEST",
}


def build_references(test_set: str, path: str, model_name: str, language: str = "en", task: str = "transcribe"):
    from datasets import load_from_disk
    from transformers import WhisperTokenizer

    spec = EXTERNAL_TEST_SETS[test_set]
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)

    # Text columns only (no audio decoding)
    ds = load_from_disk(path).select_columns([spec["id_col"], spec["text_col"]])
    ids = [str(x) for x in ds[spec["id_col"]]]
    texts = [(t or "").lower() for t in ds[spec["text_col"]]]

    token_ids = tokenizer(texts).input_ids
    if spec["filter_long_labels"]:
        from transformers import WhisperConfig
        max_len = WhisperConfig.from_pretrained(model_name).max_target_positions
        keep = [i for i, t in enumerate(token_ids) if len(t) <= max_len]
        ids = [ids[i] for i in keep]
        token_ids = [token_ids[i] for i in keep]

    refs = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    return ids, refs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="Released prediction CSV (must contain utt_id).")
    p.add_argument("--test_set", required=True, choices=sorted(EXTERNAL_TEST_SETS))
    p.add_argument("--dataset_path", default=None, help="Default: from configs/paths.sh")
    p.add_argument("--model_name", default=env_default("MOAA_BASE_MODEL", "openai/whisper-small"))
    p.add_argument("--out", default=None, help="Default: <csv_stem>_with_refs.csv next to the input")
    args = p.parse_args()

    path = args.dataset_path or env_default(ENV_VARS[args.test_set])
    if not path:
        raise SystemExit(f"No dataset path for {args.test_set}; pass --dataset_path or source configs/paths.sh")

    df = pd.read_csv(args.csv, dtype=str, keep_default_na=False)
    if "utt_id" not in df.columns:
        raise SystemExit("CSV has no utt_id column. Internal-split CSVs cannot be re-attached; "
                         "regenerate them with scripts/run_eval.py --tests internal.")

    ids, refs = build_references(args.test_set, path, args.model_name)
    if len(ids) != len(df):
        raise SystemExit(f"Row count mismatch: CSV has {len(df)} rows, dataset gives {len(ids)}.")
    mismatch = sum(a != b for a, b in zip(df["utt_id"].tolist(), ids))
    if mismatch:
        raise SystemExit(f"{mismatch} rows have a different utt_id than the dataset order; wrong test set/version?")

    cols = list(df.columns)
    insert_at = cols.index("pred_text") + 1 if "pred_text" in cols else len(cols)
    df.insert(insert_at, "ref_text", refs)

    out = Path(args.out) if args.out else Path(args.csv).with_name(Path(args.csv).stem + "_with_refs.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Attached {len(refs)} references ({args.test_set}) -> {out}")


if __name__ == "__main__":
    main()
