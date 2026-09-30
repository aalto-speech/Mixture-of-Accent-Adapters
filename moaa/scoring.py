"""Corpus-level WER/CER with jiwer (micro-averaged over all utterances)."""

import re

import pandas as pd


def corpus_scores(refs, hyps):
    import jiwer

    w = jiwer.process_words(refs, hyps)
    c = jiwer.process_characters(refs, hyps)
    return {
        "wer": w.wer,
        "cer": c.cer,
        "S": w.substitutions, "I": w.insertions, "D": w.deletions, "H": w.hits,
        "n_ref_words": w.hits + w.substitutions + w.deletions,
        "char_S": c.substitutions, "char_I": c.insertions, "char_D": c.deletions, "char_H": c.hits,
        "n_ref_chars": c.hits + c.substitutions + c.deletions,
    }


def score_dataframe(df: pd.DataFrame, hyp_col: str = "pred_text", ref_col: str = "ref_text",
                    drop_empty_refs: bool = False):
    """WER/CER as fractions (multiply by 100 for %). Empty refs are kept unless drop_empty_refs."""
    refs = df[ref_col].fillna("").astype(str)
    hyps = df[hyp_col].fillna("").astype(str)
    n_empty = int((refs.str.strip() == "").sum())
    if drop_empty_refs and n_empty:
        keep = refs.str.strip() != ""
        refs, hyps = refs[keep], hyps[keep]
    out = corpus_scores(refs.tolist(), hyps.tolist())
    out["n_utts"] = len(refs)
    out["n_empty_refs"] = n_empty
    out["empty_refs_dropped"] = bool(drop_empty_refs and n_empty)
    return out


def normalize_text(s) -> str:
    """Lowercase, replace non [a-z0-9] with space, collapse whitespace (used for per-row analysis)."""
    s = "" if s is None or (isinstance(s, float) and pd.isna(s)) else str(s)
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()
