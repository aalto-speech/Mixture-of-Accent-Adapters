"""Prediction export + evaluation routines shared by scripts/train.py and scripts/run_eval.py."""

import os
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import Seq2SeqTrainer

from .data import DataCollatorSpeechSeq2SeqWithPaddingASROnly, load_external_test_set
from .metrics import compute_metrics_asr_only
from .utils import write_json


def _decode(pred_output, tokenizer):
    pred_ids = pred_output.predictions
    if isinstance(pred_ids, (tuple, list)):
        pred_ids = pred_ids[0]
    pred_ids = np.asarray(pred_ids)

    label_ids = np.asarray(pred_output.label_ids).copy()
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    ref_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return pred_str, ref_str


def save_predictions_csv(pred_output, raw_ds, tokenizer, csv_path: str):
    """Internal multi-task test: pred_text, ref_text + gold classifier labels."""
    pred_str, ref_str = _decode(pred_output, tokenizer)

    rows = []
    for i in range(len(raw_ds)):
        rows.append({
            "pred_text": pred_str[i],
            "ref_text": ref_str[i],
            "accented_or_not_clf": int(raw_ds[i]["accented_or_not_clf"]),
            "gender_clf": int(raw_ds[i]["gender_clf"]),
            "accent_clf": int(raw_ds[i]["accent_clf"]),
        })

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["pred_text", "ref_text", "accented_or_not_clf", "gender_clf", "accent_clf"]).to_csv(
        csv_path, index=False, encoding="utf-8"
    )


def save_predictions_csv_external_with_clf(pred_output, trainer: Seq2SeqTrainer, dataset, raw_ds, tokenizer,
                                           csv_path: str, id_key: str = "utt_id"):
    """External ASR-only test: utt_id, pred_text, ref_text + predicted classifier labels."""
    pred_text, ref_text = _decode(pred_output, tokenizer)

    model = trainer.model
    model.eval()
    pred_accbin, pred_gender, pred_accent = [], [], []

    with torch.no_grad():
        for batch in trainer.get_test_dataloader(dataset):
            input_features = batch["input_features"].to(trainer.args.device)
            clf_out = model.predict_clf(input_features)
            pred_accbin.extend(clf_out["logits_accbin"].argmax(dim=-1).cpu().tolist())
            pred_gender.extend(clf_out["logits_gender"].argmax(dim=-1).cpu().tolist())
            pred_accent.extend(clf_out["logits_accent"].argmax(dim=-1).cpu().tolist())

    assert len(raw_ds) == len(pred_text) == len(pred_accbin) == len(pred_gender) == len(pred_accent)

    rows = []
    for i in range(len(raw_ds)):
        rows.append({
            id_key: raw_ds[i].get(id_key, i),
            "pred_text": pred_text[i],
            "ref_text": ref_text[i],
            "pred_accented_or_not": int(pred_accbin[i]),
            "pred_gender": int(pred_gender[i]),
            "pred_accent": int(pred_accent[i]),
        })

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")


def evaluate_external_test_set(name, path, model, processor, tokenizer, feature_extractor, training_args,
                               metric_wer, metric_cer, pred_dir, logger, max_samples=None, seed=42):
    """Decode one external test set, log WER/CER, write `<pred_dir>/<name>.csv` and `<name>_metrics.json`."""
    logger.info("*" * 60)
    logger.info("Loading %s test set from: %s", name, path)

    raw_ds, feat_ds = load_external_test_set(
        name, path, feature_extractor, tokenizer,
        max_label_len=model.whisper.config.max_target_positions,
        max_samples=max_samples, seed=seed, logger=logger,
    )

    collator = DataCollatorSpeechSeq2SeqWithPaddingASROnly(
        processor=processor,
        decoder_start_token_id=model.whisper.config.decoder_start_token_id,
    )
    ext_trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        data_collator=collator,
        compute_metrics=partial(compute_metrics_asr_only, tokenizer=tokenizer,
                                metric_wer=metric_wer, metric_cer=metric_cer),
        tokenizer=processor.tokenizer,
    )

    logger.info("Evaluation on %s (ASR-only)", name)
    results = ext_trainer.predict(feat_ds)
    logger.info("%s WER: %.4f", name, results.metrics.get("test_wer", float("nan")))
    logger.info("%s CER: %.4f", name, results.metrics.get("test_cer", float("nan")))
    logger.info("*" * 60)

    csv_path = os.path.join(pred_dir, f"{name}.csv")
    save_predictions_csv_external_with_clf(
        pred_output=results, trainer=ext_trainer, dataset=feat_ds, raw_ds=raw_ds,
        tokenizer=tokenizer, csv_path=csv_path, id_key="utt_id",
    )
    write_json({"test_set": name, "n": len(raw_ds), **results.metrics},
               os.path.join(pred_dir, f"{name}_metrics.json"))
    logger.info("Saved predictions to %s", csv_path)
    return results.metrics
