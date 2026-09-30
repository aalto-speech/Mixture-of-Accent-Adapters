#!/usr/bin/env python
"""
Per-utterance WER distribution (two-panel violin plot: all utterances / utterances with WER > 0).

Example:
  python scripts/plot_wer_violin.py outputs/moaa_linear_proj_A1/predictions/aesrc20h_itn_dhf.csv \
      --hyp_col pred_text_dhf --out results/figures/wer_violin.pdf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def norm_text(s) -> str:
    import re
    s = "" if s is None or (isinstance(s, float) and pd.isna(s)) else str(s).lower()
    s = s.replace("'", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def utterance_wer(ref, hyp) -> float:
    from jiwer import process_words

    ref_n, hyp_n = norm_text(ref), norm_text(hyp)
    if ref_n == "":
        return 0.0 if hyp_n == "" else 1.0
    return float(process_words(ref_n, hyp_n).wer)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--hyp_col", default="pred_text")
    p.add_argument("--ref_col", default="ref_text")
    p.add_argument("--out", required=True, help="Output figure path (.pdf/.png)")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    df = pd.read_csv(args.csv)
    wers_all = np.array([utterance_wer(r, h) for r, h in zip(df[args.ref_col], df[args.hyp_col])], dtype=float)
    wers_err = wers_all[wers_all > 0]

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "legend.fontsize": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 4.6), sharex=True)

    def panel(ax, data, label, fill, edge):
        mean_v = float(np.mean(data)) if len(data) else np.nan
        med_v = float(np.median(data)) if len(data) else np.nan
        if len(data):
            vp = ax.violinplot([data], positions=[1], vert=False, widths=0.8,
                               showextrema=True, showmeans=False, showmedians=False)
            body = vp["bodies"][0]
            body.set_facecolor(fill)
            body.set_edgecolor("white")
            body.set_alpha(0.70)
            for k in ["cmins", "cmaxes", "cbars"]:
                vp[k].set_color(edge)
                vp[k].set_linewidth(1.2)
        ax.axvline(mean_v, linestyle="--", linewidth=1.6, color="red")
        ax.axvline(med_v, linestyle="-", linewidth=1.6, color="limegreen")
        ax.set_ylabel(label, rotation=90, va="center", labelpad=25)
        ax.set_yticks([])
        ax.set_ylim(0.5, 1.5)
        ax.text(0.98, 0.88, f"N={len(data)}", transform=ax.transAxes, ha="right", va="top")
        ax.legend(handles=[
            Line2D([0], [0], color="red", linestyle="--", linewidth=1.6, label=f"mean={mean_v:.3f}"),
            Line2D([0], [0], color="limegreen", linestyle="-", linewidth=1.6, label=f"median={med_v:.3f}"),
        ], loc="lower right", frameon=True)

    panel(axes[0], wers_all, "All\nutterances", "#76B7B2", "#3A8F88")
    panel(axes[1], wers_err, "Utterances\nwith WER>0", "#B07AA1", "#7A4E70")
    axes[1].set_xlabel("Sentence WER")
    xmax = wers_all.max() if len(wers_all) else 1.0
    axes[1].set_xlim(0, xmax * 1.05)
    fig.tight_layout()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
