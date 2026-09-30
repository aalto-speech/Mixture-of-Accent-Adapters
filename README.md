# Mixture-of-Accent-Adapters for Robust ASR: Injecting Accent Cues into Pretrained Whisper 

Official code, checkpoints and outputs for **MoAA** (Mixture of Accent Adapters) and
**DHF** (Deterministic Hallucination Filter).

* **MoAA** keeps Whisper frozen and adds lightweight bottleneck adapters on top of the
  encoder. An utterance-level router mixes the adapters. Multi-task heads condition
  the mixture: *accented vs. non-accented*, *accent ID*, and an adversarial *gender*
  head behind a gradient-reversal layer.
* **DHF** is a deterministic, reference-free post-processing step. It detects and
  repairs typical autoregressive decoding artifacts (looping n-grams, runaway
  repetitions, number-word spam, filler tokens) without any model or reference
  transcript.

> **TL;DR (reproduce the paper numbers)**
> 1. `conda env create -f environment.yml && conda activate moaa`
> 2. Edit `configs/paths.sh` (data locations) and `configs/cluster_env.sh` (modules / env)
> 3. `sbatch slurm/evaluate.slrm checkpoints/moaa_linear_proj_A20/checkpoint-101835 outputs/eval_A20`
> 4. `sbatch slurm/postprocess.slrm outputs/eval_A20/predictions`  → ITN → DHF → WER/CER tables

---

## Contents

1. [Repository structure](#1-repository-structure)
2. [Installation](#2-installation)
3. [Configure paths](#3-configure-paths)
4. [Data](#4-data)
5. [Step-by-step reproduction](#5-step-by-step-reproduction)
6. [Method details](#6-method-details)
7. [Results](#7-results)
8. [Output file formats](#8-output-file-formats)
9. [Reproducibility notes](#9-reproducibility-notes)
10. [Citation / License](#10-citation--license)

---

## 1. Repository structure

```
MoAA/
├── README.md
├── LICENSE                    # MIT (code)
├── environment.yml            # conda env (exact versions used for the paper runs)
├── requirements.txt
├── configs/
│   ├── paths.sh               # <-- ALL file-system paths (datasets, outputs, caches)
│   └── cluster_env.sh         # <-- module loads + conda activation for Slurm jobs
├── moaa/                      # Python package
│   ├── model.py               #   MoAA model (adapters, router, heads, GRL), save/load
│   ├── data.py                #   dataset loading, feature extraction, collators
│   ├── trainer.py             #   multi-task Seq2SeqTrainer (classifier outputs in eval)
│   ├── metrics.py             #   WER/CER + classifier metrics during training
│   ├── evaluation.py          #   test-set decoding and prediction CSV export
│   ├── dhf.py                 #   DHF: Deterministic Hallucination Filter
│   ├── scoring.py             #   corpus WER/CER (jiwer)
│   └── utils.py
├── scripts/
│   ├── train.py               # train + evaluate a MoAA model
│   ├── run_eval.py            # evaluate a trained checkpoint (no training)
│   ├── itn_normalize.py       # NeMo inverse text normalization of pred/ref
│   ├── apply_dhf.py           # apply DHF, report WER/CER before/after
│   ├── score.py               # WER/CER table for any prediction CSVs
│   ├── attach_references.py   # re-attach ref_text to released CSVs from your local datasets
│   └── plot_wer_violin.py     # per-utterance WER distribution figure
├── slurm/
│   ├── train.slrm             # sbatch slurm/train.slrm <variant> <num_adapters>
│   ├── evaluate.slrm          # sbatch slurm/evaluate.slrm <ckpt> <out_dir>
│   └── postprocess.slrm       # sbatch slurm/postprocess.slrm <predictions_dir>
├── checkpoints/               # released models (~16 GB, not in git; see checkpoints/README.md)
├── results/
│   ├── original_runs/         # predictions (no reference transcripts) + logs of the paper runs
│   ├── postprocessed/         # ITN + DHF applied to those CSVs, and score tables
│   ├── dhf_reference/         # the exact CSVs used for the DHF result + DHF output
│   └── figures/               # paper figures (PDF)
├── logs/slurm/                # Slurm stdout/stderr
└── archive/                   # (local only, git-ignored) original research scripts/notebooks
```

No script contains hard-coded paths. Every path comes from `configs/paths.sh` (exported
as `MOAA_*` environment variables) and can be overridden with a command-line flag.

---

## 2. Installation

Tested with Python 3.11, PyTorch 2.5.0 (CUDA 12.4), transformers 4.44.2, datasets 3.0.1
on NVIDIA V100 32 GB.

```bash
git clone <this-repo> MoAA && cd MoAA

# Option A: conda (recommended; pynini for ITN comes from conda-forge)
conda env create -f environment.yml -p /path/to/conda_envs/moaa
conda activate /path/to/conda_envs/moaa

# Option B: pip
pip install torch==2.5.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
conda install -c conda-forge pynini=2.1.6    # needed only for ITN
pip install -r requirements.txt
```

> ⚠️ Please use **transformers 4.44.x**. The code passes `tokenizer=` to the Trainer and
> relies on the Whisper `generate()` behaviour of that release. transformers ≥ 5 is not
> supported.

**On Aalto Triton:**

```bash
module load mamba
mamba env create -f environment.yml -p /scratch/work/$USER/conda_envs/moaa
```

Then set `MOAA_CONDA_ENV` in `configs/cluster_env.sh` to that prefix.

The first run downloads `openai/whisper-small` and the `wer`/`cer` metrics from the
HuggingFace Hub. If your compute nodes have no internet access, run them once on a
login node (e.g. `python -c "import evaluate; evaluate.load('wer'); evaluate.load('cer')"`).

---

## 3. Configure paths

Edit **two files**; nothing else needs to change.

**`configs/paths.sh`**: data, model, output and cache locations:

| Variable | Meaning |
|---|---|
| `MOAA_AESRC_TRAIN_ROOTS` | colon-separated dirs holding one `load_from_disk` dir per AESRC accent |
| `MOAA_LIBRISPEECH100` | LibriSpeech `train-clean-100` (HF format) |
| `MOAA_AESRC_TEST` | AESRC 2020 test set |
| `MOAA_OPENSLR_TEST` | OpenSLR-83 British Isles accents, our test split |
| `MOAA_EDACC_TEST` | EdAcc test |
| `MOAA_GLOBE_TEST` | GLOBE test |
| `MOAA_BASE_MODEL` | `openai/whisper-small` (Hub id or local dir) |
| `MOAA_OUTPUT_ROOT` | where new runs are written (default `./outputs`) |
| `MOAA_CHECKPOINT_ROOT` | released checkpoints (default `./checkpoints`) |
| `HF_HOME` | HuggingFace cache (keep it off a small `$HOME`) |

Set any test-set variable to `""` to skip that test set.

**`configs/cluster_env.sh`**: `module load ...` lines and the conda environment used
inside Slurm jobs.

For interactive use:

```bash
source configs/paths.sh
python scripts/train.py --help
```

---

## 4. Data

All datasets are stored in HuggingFace `datasets` **`save_to_disk`** format and read
with `load_from_disk`. Audio is resampled to 16 kHz on the fly. Transcripts are
lower-cased by the loaders.

| Dataset | Use | Source | Required columns |
|---|---|---|---|
| **AESRC 2020** (Accented English Speech Recognition Challenge; 10 accents × ~20 h) | train (all 10 accents) | request from the challenge organisers / Datatang | `audio`, `transcript`, `SEX` (`Male`/`Female`), `accent` (see below) |
| **LibriSpeech train-clean-100** | train (non-accented, accent = `unknown`) | [openslr.org/12](https://www.openslr.org/12) | `id`, `audio`, `text`, `speaker_id`, `chapter_id`, `sex` (`male`/`female`, from `SPEAKERS.TXT`) |
| **AESRC 2020 test** ("AESRC20H") | test | AESRC | `utt_id`, `audio`, `transcript` |
| **OpenSLR-83** British Isles English dialects | test | [openslr.org/83](https://www.openslr.org/83) | `line_id`, `audio`, `text` |
| **EdAcc** | test | HF `edinburghcstr/edacc` (test split) | `speaker`, `audio`, `text` |
| **GLOBE** | test | HF `MushanW/GLOBE` (test split) | `speaker_id`, `audio`, `transcript` |

**AESRC accent folders / labels.** Each AESRC training root holds one sub-directory per
accent. The `accent` column must equal the folder name:

```
American English Speech Data            British English Speech Data
Canadian English Speech Data            Chinese Speaking English Speech Data
Indian English Speech Data              Japanese Speaking English Speech Data
Korean Speaking English Speech Data     Portuguese Speaking English Speech Data
Russian Speaking English Speech Data    Spanish Speaking English Speech Data
```

The paper models use all 10 accents: 172,607 AESRC utterances + 28,539 LibriSpeech
utterances = **201,146** training-pool utterances. If your copy splits the accents over
several folders (ours has 8 in `train/` and Canadian + Spanish in `test_unseen/`), list
all of them in `MOAA_AESRC_TRAIN_ROOTS`.

**Building the HF-format datasets (sketch).**

```python
from datasets import Dataset, Audio
ds = Dataset.from_dict({"audio": wav_paths, "text": transcripts, ...}).cast_column("audio", Audio(sampling_rate=16000))
ds.save_to_disk("/path/to/dataset")
```

* *LibriSpeech*: one row per `.flac` with transcripts from `*.trans.txt`. Add `sex`
  from `SPEAKERS.TXT` (`M` → `male`, `F` → `female`).
* *OpenSLR-83*: load each dialect/gender folder, add `accent` = folder name and `sex`
  = folder suffix, concatenate. Then split with
  `train_test_split(test_size=0.2, stratify_by_column="accent", seed=42)` (after
  casting `accent` to `ClassLabel`) and save the `test` part.
* *EdAcc / GLOBE*: `load_dataset(...)["test"].save_to_disk(...)`.

**Splits used during training** (built in `moaa/data.py::build_multitask_splits`):

* `train`: 90% of the training pool (stratified by accent label, seed 42)
* `test` ("internal test"): the remaining 10%
* `devel`: 1,000 random utterances from `test`, used for per-epoch WER and best-checkpoint selection

---

## 5. Step-by-step reproduction

All commands are run **from the repository root**.

### Step 0: set up

```bash
conda activate moaa            # or: source configs/cluster_env.sh
source configs/paths.sh
mkdir -p logs/slurm
```

### Step 1: sanity check DHF (CPU, no GPU needed)

`results/dhf_reference/` ships the exact predictions used for the DHF numbers: MoAA
`linear_proj` with 1 adapter, AESRC 2020 test set, 18,194 utterances. It comes from an
earlier run of that model; see `results/README.md`.

> **Reference transcripts are not included** in any released CSV. They belong to the
> datasets, and AESRC is not freely redistributable. Released CSVs contain `utt_id` and
> predictions only. `scripts/attach_references.py` rebuilds `ref_text` exactly from your
> local copy of the dataset.

**1a. DHF only (~15 s, no dataset needed).** Applies DHF to the ITN-normalized
predictions. The output must equal `results/dhf_reference/*_cleaned_dhf.csv`:

```bash
python scripts/apply_dhf.py results/dhf_reference/method_linear_proj_aesrc_adapters1_cleaned.csv \
    --out_dir outputs/dhf_check
# rows=18194  flagged=254  modified=254
```

**1b. Full WER/CER check (needs the AESRC test set; ITN takes ~1 h on CPU).**

```bash
python scripts/attach_references.py results/dhf_reference/method_linear_proj_aesrc_adapters1.csv \
    --test_set aesrc20h --out outputs/dhf_check/aesrc_A1.csv
python scripts/itn_normalize.py outputs/dhf_check/aesrc_A1.csv            # -> aesrc_A1_itn.csv
python scripts/apply_dhf.py outputs/dhf_check/aesrc_A1_itn.csv --report_number_only_ref_exclusion
```

Expected output:

```
rows=18194  flagged=254  modified=254
Before DHF   WER=0.118167  CER=0.080565  S/I/D/H=9422/8252/1140/148653
After DHF    WER=0.080708  CER=0.042233  S/I/D/H=9497/1531/1822/147896
After DHF + number-only-ref exclusion*   WER=0.079241  CER=0.040710   (* analysis only)
```

DHF cuts insertions from 8,252 to 1,531 (−81%) and WER from 11.82% to 8.07%.

### Step 2: evaluate the released checkpoints (GPU)

Always point to the `checkpoint-<step>/` directory; see `checkpoints/README.md`.

```bash
sbatch slurm/evaluate.slrm checkpoints/moaa_linear_proj_A1/checkpoint-33945   outputs/eval_moaa_linear_proj_A1
sbatch slurm/evaluate.slrm checkpoints/moaa_linear_proj_A5/checkpoint-33945   outputs/eval_moaa_linear_proj_A5
sbatch slurm/evaluate.slrm checkpoints/moaa_linear_proj_A10/checkpoint-79205  outputs/eval_moaa_linear_proj_A10
sbatch slurm/evaluate.slrm checkpoints/moaa_linear_proj_A20/checkpoint-101835 outputs/eval_moaa_linear_proj_A20
sbatch slurm/evaluate.slrm checkpoints/moaa_unfreeze_last2_A1/checkpoint-113150  outputs/eval_moaa_unfreeze_last2_A1
sbatch slurm/evaluate.slrm checkpoints/moaa_unfreeze_last2_A10/checkpoint-113150 outputs/eval_moaa_unfreeze_last2_A10
```

By default this decodes `aesrc20h openslr edacc globe`. Add `--tests internal` to also
rebuild and decode the internal test split (needs the training data). Interactive
equivalent:

```bash
python scripts/run_eval.py --checkpoint checkpoints/moaa_linear_proj_A20/checkpoint-101835 \
    --out_dir outputs/eval_moaa_linear_proj_A20 --tests aesrc20h openslr edacc globe
```

Each test set produces `predictions/<test>.csv` and `predictions/<test>_metrics.json`.
The JSON holds WER/CER on raw lower-cased text, the same numbers as in the training logs.

### Step 3: post-process (ITN → DHF → score)

```bash
sbatch slurm/postprocess.slrm outputs/eval_moaa_linear_proj_A20/predictions
```

or step by step:

```bash
P=outputs/eval_moaa_linear_proj_A20/predictions
python scripts/score.py $P/aesrc20h.csv                                   # raw WER/CER
python scripts/itn_normalize.py $P/aesrc20h.csv                           # -> aesrc20h_itn.csv
python scripts/apply_dhf.py $P/aesrc20h_itn.csv                           # -> aesrc20h_itn_dhf.csv (+ report json)
python scripts/score.py $P/*_itn_dhf.csv --hyp_col pred_text_dhf --out_csv $P/scores_itn_dhf.csv
```

* **ITN** (`itn_normalize.py`): lower-cases, strips punctuation (keeps `'` and `-`), then
  applies NeMo English inverse text normalization to both hypothesis and reference
  (e.g. "twenty five" → "25"), so formatting differences are not counted as errors.
* **DHF** (`apply_dhf.py`) rewrites only flagged hypotheses and never looks at the
  reference; see §6.2.

### Step 4: train from scratch (GPU, ~2.5–4 days per model on one V100)

```bash
# MoAA, frozen Whisper + linear projection (main variant), 1/5/10/20 adapters
sbatch slurm/train.slrm linear_proj 1
sbatch slurm/train.slrm linear_proj 5
sbatch slurm/train.slrm linear_proj 10
sbatch slurm/train.slrm linear_proj 20

# Ablation: last 2 Whisper encoder+decoder blocks unfrozen, no projection
sbatch slurm/train.slrm unfreeze_last2 1
sbatch slurm/train.slrm unfreeze_last2 10

# Ablation: fully frozen Whisper, no projection
sbatch slurm/train.slrm frozen 1
sbatch slurm/train.slrm frozen 10
```

Every flag of `scripts/train.py` can be appended, e.g.
`sbatch slurm/train.slrm linear_proj 20 --run_name my_run --batch_size 8`.

A run writes to `$MOAA_OUTPUT_ROOT/moaa_<variant>_A<N>/`:

```
checkpoint-<step>/   last_checkpoint/   runs/ (TensorBoard)   train.log   run_config.json
predictions/internal_test.csv  aesrc20h.csv  openslr.csv  edacc.csv  globe.csv  (+ *_metrics.json)
```

Then post-process as in Step 3. For a quick end-to-end smoke test (minutes), use:

```bash
python scripts/train.py --variant linear_proj --num_adapters 2 --run_name smoke \
    --epochs 1 --max_train_samples 64 --max_test_samples 8 --skip_tests edacc globe
```

### Step 5: figures

```bash
python scripts/plot_wer_violin.py outputs/eval_moaa_linear_proj_A1/predictions/aesrc20h_itn_dhf.csv \
    --hyp_col pred_text_dhf --out results/figures/wer_violin_aesrc20h.pdf
```

The PDFs in `results/figures/` are the original figures from the paper runs.

---

## 6. Method details

### 6.1 MoAA architecture (`moaa/model.py`)

For log-Mel input *x*:

1. **Encoder.** Frozen Whisper encoder gives *H* ∈ ℝ^{T×D} (D = 768 for whisper-small).
2. **Pooling.** *h̄* = mean_t *H*. The `linear_proj` variant uses *h̄* ← *W h̄ + b*.
3. **Heads** on *h̄*:
   * accented-or-not (2 classes): LibriSpeech = 0, AESRC = 1
   * accent ID (10 AESRC accents), loss ignored for LibriSpeech
   * gender (2 classes) behind a **gradient reversal layer** (λ = 1), which pushes the
     representation to be gender-invariant
4. **Soft accent gate.** *p* = P(accented | x).
5. **Accent conditioning.** *e* = *p* · (softmax(accent logits) · *E*_accent) + (1 − *p*) · *e*_neutral,
   then *H*_c = CondProj(*H* + *e*).
6. **Mixture of accent adapters.** *N* bottleneck adapters (D → 192 → D, ReLU, residual).
   A router (linear + softmax on mean-pooled *H*_c) gives utterance-level weights *w*,
   and *A* = Σᵢ *wᵢ* · Adapterᵢ(*H*_c).
7. **Gated residual.** Encoder output = *p* · *A* + (1 − *p*) · *H*. Non-accented speech
   keeps (close to) the original Whisper representation.
8. **Decoder.** Frozen Whisper decoder produces the transcript.

**Loss:** L = L_ASR + L_accented + 2·L_accent + L_gender (gender gradient reversed).

| Variant (`--variant`) | Whisper params trained | `linear_proj` | LR |
|---|---|---|---|
| `linear_proj` | output projection `proj_out` only (encoder/decoder frozen) | ✓ | 5e-4 |
| `frozen` | output projection `proj_out` only (encoder/decoder frozen) | ✗ | 5e-4 |
| `unfreeze_last2` | `proj_out` + last 2 encoder + last 2 decoder blocks + final LayerNorms | ✗ | 5e-5 |

See §9 for exact trainable-parameter counts.

**Common training setup:** whisper-small, AdamW (HF default), weight decay 0.01, warm-up
5%, linear decay, batch size 16, 10 epochs, fp16, max grad-norm 1.0, greedy decoding
(max 225 tokens), best checkpoint by devel WER, seed 42.

### 6.2 DHF: Deterministic Hallucination Filter (`moaa/dhf.py`)

DHF is applied per hypothesis *h* and never uses the reference:

1. **Detect.** *h* is flagged if any of these hold:
   * a filler/noise token appears (`sil, sp, uh, um, erm, eh, ah, mm, hmm, noise, background, static`) or glued syllable spam (`silalalala…`)
   * ≥ 4 single-letter tokens
   * the same token repeats ≥ 4 times in a row
   * repeated 3-gram coverage ≥ 6 tokens
   * ≥ 25 tokens with ≥ 70% number words/digits
   * ≥ 20 tokens with type/token ratio < 0.35
2. **Propose candidates.** Clean *h* first: drop apostrophe garbage, noise and
   single-letter tokens, and cap token runs at 2. Then build:
   * **A**, the cleaned text
   * **B**, A with the number-word spam tail cut
   * **C**, A with the dominant repeated block (length 2–16, ≥ 3 repeats) compressed to one copy
   * **D**, A cut at the second occurrence of any repeated 3–10-gram
   * the empty string, when *h* is number spam
3. **Select.** Score each candidate and the original with a reference-free quality score:

   score = 2·TTR − 0.03·max(0, L−40) − 2·#spam − 1.5·#single-char − 2·max(0, run−2)
   − 0.25·rep3 − 0.35·rep4 − 8·max(0, num_ratio−0.5) − (20 if L≤1, 8 if L≤2)

   The best candidate replaces *h* only if its score is at least **1.0** higher than
   the original's (conservative replacement).

DHF has no learned parameters and is fully deterministic. The same input always gives
the same output.

---


## 8. Output file formats

Files you generate with `train.py` / `run_eval.py` contain `ref_text`. The released CSVs
in `results/` do not: they have the same columns minus `ref_text` (use
`scripts/attach_references.py` to add it). Do not publish CSVs that contain `ref_text`.

| File | Columns |
|---|---|
| `predictions/<external>.csv` | `utt_id, pred_text, ref_text, pred_accented_or_not, pred_gender, pred_accent` |
| `predictions/internal_test.csv` | `pred_text, ref_text, accented_or_not_clf, gender_clf, accent_clf` (gold labels) |
| `*_itn.csv` | same, with `pred_text`/`ref_text` ITN-normalized |
| `*_dhf.csv` | + `dhf_suspicious` (bool), `pred_text_dhf` |
| `*_dhf_report.json` | flagged/modified counts, WER/CER and S/I/D/H before and after DHF |
| `scores_*.csv` | one row per file: `wer`, `cer` (fractions), `wer_pct`, `cer_pct`, S/I/D/H counts |

`pred_accent` ∈ 0..9 indexes the accents in `moaa/data.py::ACCENT2ID` order minus one
(0 = American, 1 = British, 2 = Canadian, 3 = Chinese, 4 = Indian, 5 = Japanese,
6 = Korean, 7 = Portuguese, 8 = Russian, 9 = Spanish). `pred_gender`: 0 = female, 1 = male.

---

## 9. Reproducibility notes

* **Faithful refactor.** `moaa/` and `scripts/` refactor the original research scripts
  (kept locally in `archive/original_scripts/`). Model structure, parameter
  initialisation order, data processing, losses and training arguments are unchanged.
  This was verified against the original outputs:
  * `run_eval.py` on `moaa_linear_proj_A1/checkpoint-33945` reproduces the original
    AESRC20H transcripts **and** the accent/gender/accented classifier predictions exactly
    (checked on a sample of utterances)
  * `itn_normalize.py` reproduces the original ITN file **byte-for-byte**
  * `apply_dhf.py` reproduces the original DHF notebook to the last digit (Step 1)
* **Trainable parameters include Whisper's output projection.** When Whisper is wrapped,
  `proj_out` (the 51,865 × 768 vocabulary projection, 39.8M parameters) is untied from
  the decoder embeddings and is **not frozen**, so it is trained together with the MoAA
  modules. This was also the case for the released checkpoints (Whisper-small has 241.7M
  parameters in total):

  | Model | MoAA modules | `proj_out` | Whisper blocks | Total trainable |
  |---|---|---|---|---|
  | `linear_proj`, 1 adapter    | 1.50M | 39.83M | – | 41.33M |
  | `linear_proj`, 5 adapters   | 2.68M | 39.83M | – | 42.52M |
  | `linear_proj`, 10 adapters  | 4.17M | 39.83M | – | 44.00M |
  | `linear_proj`, 20 adapters  | 7.13M | 39.83M | – | 46.97M |
  | `frozen`, 1 / 10 adapters   | 0.91M / 3.58M | 39.83M | – | 40.74M / 43.41M |
  | `unfreeze_last2`, 1 / 10 adapters | 0.91M / 3.58M | 39.83M | 33.08M | 73.82M / 76.49M |
* **`linear_proj` and decoding.** In the released `linear_proj` models the projection
  feeds the classification heads during training (`forward`) and in `predict_clf`.
  Inside `generate()` the gate and accent conditioning use the *un-projected* pooled
  features. All reported numbers were produced this way, so this is the default.
  `--proj_in_generate` switches it for experiments.
* **Checkpoint selection.** `devel` (1,000 utterances) is a subset of the internal test
  split. Internal-test numbers are therefore slightly optimistic. The external test sets
  (AESRC20H, OpenSLR-83, EdAcc, GLOBE) are fully held out.
* **Data order.** Accent sub-folders are loaded in sorted order. The original script used
  `os.listdir` order, so the exact random 90/10 split can differ slightly on another
  file system.
* **Numerical nondeterminism.** fp16 kernels on different GPUs or driver versions can
  change individual greedy tokens, so re-training reproduces the numbers only
  approximately. Evaluating the released checkpoints is the most faithful check.
* **Legacy `last_checkpoint/`** of the released `linear_proj` models lacks the
  `linear_proj` weights. Use `checkpoint-<step>/` (see `checkpoints/README.md`).

---

## 10. Citation / License

```bibtex
@inproceedings{moaa,
  title     = {TODO},
  author    = {TODO},
  booktitle = {TODO},
  year      = {TODO}
}
```

**License.** The source code in this repository is released under the [MIT License](LICENSE).

The license covers the code only. The datasets (AESRC, LibriSpeech, OpenSLR-83, EdAcc,
GLOBE) are not included and remain under their own licenses and terms of use. Reference
transcripts are therefore not redistributed. The trained checkpoints build on
`openai/whisper-small` (MIT) and were trained on these datasets, so their use is also
subject to the datasets' terms, including AESRC's.

Computations were performed on the Aalto University Science-IT **Triton** cluster.
