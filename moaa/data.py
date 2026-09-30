"""
Dataset loading, feature preparation and collators.

All datasets are expected in HuggingFace `save_to_disk` format (loadable with
`datasets.load_from_disk`). See README.md -> "Data" for the expected schemas.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from datasets import Audio, DatasetDict, concatenate_datasets, load_from_disk

SAMPLING_RATE = 16_000

# Accent label ids used for the accent-ID head (0 = no accent label / LibriSpeech).
ACCENT2ID = {
    "unknown": 0,
    "American English Speech Data": 1,
    "British English Speech Data": 2,
    "Canadian English Speech Data": 3,
    "Chinese Speaking English Speech Data": 4,
    "Indian English Speech Data": 5,
    "Japanese Speaking English Speech Data": 6,
    "Korean Speaking English Speech Data": 7,
    "Portuguese Speaking English Speech Data": 8,
    "Russian Speaking English Speech Data": 9,
    "Spanish Speaking English Speech Data": 10,
}


# =============================================================================
# Training data (AESRC + LibriSpeech-100)
# =============================================================================
def normalize_sex(example):
    example["sex"] = example["sex"].strip().lower()
    return example


def load_aesrc_train(roots: Sequence[str], logger=None):
    """
    Load and concatenate all per-accent AESRC subsets.

    Each root must contain one `load_from_disk` directory per accent, e.g.
      <root>/American English Speech Data/
      <root>/British English Speech Data/
      ...
    The paper models were trained on all 10 accents. If your copy keeps two accents in a
    separate folder (e.g. `test_unseen/`), pass both roots.
    """
    splits = []
    cols_to_keep = ["audio", "transcript", "SEX", "accent"]
    for root in roots:
        for split_name in sorted(os.listdir(root)):
            path = os.path.join(root, split_name)
            if not os.path.isdir(path):
                continue
            ds_split = load_from_disk(path)
            ds_split = ds_split.select_columns(cols_to_keep)
            ds_split = ds_split.rename_columns({"SEX": "sex", "transcript": "text"})
            if logger is not None:
                logger.info("AESRC train subset: %s (%d rows)", path, len(ds_split))
            splits.append(ds_split)

    if not splits:
        raise FileNotFoundError(f"No AESRC accent subsets found under: {list(roots)}")

    aesrc_train = concatenate_datasets(splits)
    aesrc_train = aesrc_train.map(normalize_sex)
    return aesrc_train


def load_librispeech(path: str):
    """LibriSpeech train-clean-100 with columns: id, audio, text, speaker_id, chapter_id, sex."""
    ds = load_from_disk(path)
    ds = ds.add_column("accent", ["unknown"] * len(ds))
    ds = ds.remove_columns(["id", "speaker_id", "chapter_id"])
    return ds


def build_multitask_splits(
    aesrc_train_roots: Sequence[str],
    librispeech_path: str,
    eval_split_ratio: float = 0.1,
    seed: int = 42,
    devel_size: int = 1000,
    logger=None,
) -> DatasetDict:
    """
    Build the multi-task train / devel / internal-test splits.

    - train : 90% of (AESRC-train + LibriSpeech-100), stratified by accent label
    - test  : remaining 10% ("internal test")
    - devel : random `devel_size` utterances drawn from `test` (used for checkpoint
              selection by WER during training, as in the paper runs)
    """
    aesrc_train = load_aesrc_train(aesrc_train_roots, logger=logger)
    librispeech = load_librispeech(librispeech_path)

    dataset_train = concatenate_datasets([aesrc_train, librispeech])
    dataset_train = dataset_train.map(lambda x: {"accented_or_not_clf": 0 if x["accent"] == "unknown" else 1})
    dataset_train = dataset_train.map(lambda x: {"gender_clf": 0 if x["sex"] == "female" else 1})
    dataset_train = dataset_train.map(lambda x: {"accent_clf": ACCENT2ID[x["accent"]]})
    dataset_train = dataset_train.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    if logger is not None:
        logger.info("Train rows: %d | cols=%s", len(dataset_train), list(dataset_train.features.keys()))

    if eval_split_ratio <= 0.0:
        raise ValueError("eval_split_ratio must be > 0")

    try:
        split = dataset_train.train_test_split(
            test_size=eval_split_ratio,
            seed=seed,
            stratify_by_column="accent_clf",
        )
    except Exception:
        split = dataset_train.train_test_split(test_size=eval_split_ratio, seed=seed)

    devel_split = split["test"].shuffle(seed=seed)
    devel_split = devel_split.shuffle(seed=seed).select(range(min(devel_size, len(devel_split))))

    dataset = DatasetDict({
        "train": split["train"],
        "devel": devel_split,
        "test": split["test"],
    })
    dataset = dataset.map(lambda b: {"text": (b["text"] or "").lower()})
    return dataset


# =============================================================================
# Feature preparation
# =============================================================================
def prepare_dataset(batch, feature_extractor, tokenizer):
    """
    Multi-task example preparation.

    Input columns: audio, text, accented_or_not_clf (0/1), gender_clf (0/1),
                   accent_clf (0..10, 0 = unknown).
    accent_clf is shifted to 0..9 for the accent head; 0 (unknown) -> -100 (ignored).
    """
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    batch["labels"] = tokenizer(batch["text"]).input_ids

    batch["accented_or_not_labels"] = int(batch["accented_or_not_clf"])
    batch["gender_labels"] = int(batch["gender_clf"])

    raw_accent = int(batch["accent_clf"])
    batch["accent_labels"] = -100 if raw_accent == 0 else raw_accent - 1

    return batch


def prepare_dataset_asr_only(batch, feature_extractor, tokenizer):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


def featurize_multitask_splits(dataset: DatasetDict, feature_extractor, tokenizer) -> DatasetDict:
    keep_after_map = {"input_features", "labels", "accented_or_not_labels", "gender_labels", "accent_labels"}
    out = DatasetDict()
    for split_name in dataset.keys():
        cols = dataset[split_name].column_names
        ds = dataset[split_name].map(
            lambda b: prepare_dataset(b, feature_extractor, tokenizer),
            remove_columns=[c for c in cols if c not in {"audio", "text", "accented_or_not_clf", "gender_clf", "accent_clf"}],
        )
        drop2 = [c for c in ds.column_names if c not in keep_after_map]
        if drop2:
            ds = ds.remove_columns(drop2)
        out[split_name] = ds
    return out


# =============================================================================
# External (ASR-only) test sets
# =============================================================================
# name -> how to map the raw columns to (utt_id, text)
EXTERNAL_TEST_SETS = {
    # AESRC 2020 official test set ("AESRC20H"); columns: utt_id, audio, transcript
    "aesrc20h": {"id_col": "utt_id", "text_col": "transcript", "filter_long_labels": False},
    # OpenSLR-83 British Isles accents (our 20% held-out split); columns: line_id, audio, text, ...
    "openslr": {"id_col": "line_id", "text_col": "text", "filter_long_labels": False},
    # EdAcc test; columns: speaker, text, audio, ...  (some transcripts exceed Whisper's 448-token limit)
    "edacc": {"id_col": "speaker", "text_col": "text", "filter_long_labels": True},
    # GLOBE test; columns: audio, speaker_id, transcript, ...
    "globe": {"id_col": "speaker_id", "text_col": "transcript", "filter_long_labels": False},
}


def load_external_test_set(name: str, path: str, feature_extractor, tokenizer, max_label_len: int,
                           max_samples: Optional[int] = None, seed: int = 42, logger=None):
    """
    Returns (raw_ds, feat_ds):
      raw_ds  : columns utt_id, audio, text  (kept for writing CSVs)
      feat_ds : columns input_features, labels
    """
    spec = EXTERNAL_TEST_SETS[name]
    raw = load_from_disk(path)
    raw = raw.select_columns([spec["id_col"], "audio", spec["text_col"]])

    renames = {}
    if spec["id_col"] != "utt_id":
        renames[spec["id_col"]] = "utt_id"
    if spec["text_col"] != "text":
        renames[spec["text_col"]] = "text"
    if renames:
        raw = raw.rename_columns(renames)

    if max_samples is not None:
        raw = raw.shuffle(seed=seed).select(range(min(max_samples, len(raw))))

    raw = raw.map(lambda b: {"text": (b["text"] or "").lower()})

    if spec["filter_long_labels"]:
        raw = raw.map(lambda b: {"text_len": len(tokenizer(b["text"]).input_ids)})
        n_before = len(raw)
        raw = raw.filter(lambda x: x["text_len"] <= max_label_len)
        if logger is not None:
            logger.info("[%s] Filtered %d overlong samples (kept %d / %d) using max_label_len=%d",
                        name, n_before - len(raw), len(raw), n_before, max_label_len)

    raw = raw.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    feat = raw.map(lambda b: prepare_dataset_asr_only(b, feature_extractor, tokenizer))
    feat = feat.remove_columns([c for c in feat.column_names if c not in {"input_features", "labels"}])
    return raw, feat


# =============================================================================
# Data collators
# =============================================================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPaddingAndClf:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels

        batch["accented_or_not_labels"] = torch.tensor([f["accented_or_not_labels"] for f in features], dtype=torch.long)
        batch["gender_labels"] = torch.tensor([f["gender_labels"] for f in features], dtype=torch.long)
        batch["accent_labels"] = torch.tensor([f["accent_labels"] for f in features], dtype=torch.long)
        return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPaddingASROnly:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
