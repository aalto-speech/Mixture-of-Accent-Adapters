"""Metrics used inside the HF Trainer (ASR WER/CER + classifier accuracies)."""

import numpy as np


def compute_metrics_asr_and_clf(pred, tokenizer, metric_wer, metric_cer, accent_ignore_index: int = -100):
    """
    pred.predictions is the tuple produced by MultiTaskSeq2SeqTrainer.prediction_step:
      (generated_ids,
       logits_acc, logits_gender, logits_accent,
       labels_acc, labels_gender, labels_accent,
       lossvec_acc, lossvec_gender, lossvec_accent)
    pred.label_ids are the ASR labels.
    """
    predictions = pred.predictions
    gen_ids = predictions[0] if isinstance(predictions, (tuple, list)) else predictions

    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
    cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)

    if not isinstance(predictions, (tuple, list)) or len(predictions) < 10:
        return {"wer": wer, "cer": cer}

    _, logits_acc, logits_gender, logits_accent, lab_acc, lab_gender, lab_accent, lvec_acc, lvec_gender, lvec_accent = predictions

    logits_acc = np.asarray(logits_acc)
    logits_gender = np.asarray(logits_gender)
    logits_accent = np.asarray(logits_accent)

    lab_acc = np.asarray(lab_acc).astype(np.int64).reshape(-1)
    lab_gender = np.asarray(lab_gender).astype(np.int64).reshape(-1)
    lab_accent = np.asarray(lab_accent).astype(np.int64).reshape(-1)

    lvec_acc = np.asarray(lvec_acc).astype(np.float32).reshape(-1)
    lvec_gender = np.asarray(lvec_gender).astype(np.float32).reshape(-1)
    lvec_accent = np.asarray(lvec_accent).astype(np.float32).reshape(-1)

    # Accented-or-not
    acc_acc = float((logits_acc.argmax(axis=-1) == lab_acc).mean())
    loss_acc = float(lvec_acc.mean())

    # Gender (adversarial head) on ALL samples
    acc_gender = float((logits_gender.argmax(axis=-1) == lab_gender).mean())
    loss_gender = float(lvec_gender.mean())

    # Accent only on samples with a known accent
    mask_known = (lab_accent != int(accent_ignore_index))
    n_known = int(mask_known.sum())
    pred_accent = logits_accent.argmax(axis=-1)

    if n_known > 0:
        acc_accent = float((pred_accent[mask_known] == lab_accent[mask_known]).mean())
        loss_accent = float(lvec_accent[mask_known].mean())
    else:
        acc_accent = float("nan")
        loss_accent = float("nan")

    return {
        "wer": wer,
        "cer": cer,
        "acc_accented_or_not": acc_acc,
        "loss_accented_or_not": loss_acc,
        "acc_gender": acc_gender,
        "loss_gender": loss_gender,
        "acc_accent": acc_accent,
        "loss_accent": loss_accent,
        "n_known_accent": int(n_known),
    }


def compute_metrics_asr_only(pred, tokenizer, metric_wer, metric_cer):
    predictions = pred.predictions
    gen_ids = predictions[0] if isinstance(predictions, (tuple, list)) else predictions

    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
    cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}
