"""Seq2SeqTrainer that also returns classifier logits / per-example losses during evaluation."""

from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer


class MultiTaskSeq2SeqTrainer(Seq2SeqTrainer):
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)

        has_labels = "labels" in inputs and inputs["labels"] is not None
        labels = inputs.get("labels", None)

        if prediction_loss_only:
            with torch.no_grad():
                outputs = model(**inputs)
            loss = outputs["loss"].detach() if has_labels else None
            return (loss, None, labels)

        # ---------- Generate ----------
        generated_tokens = None
        if self.args.predict_with_generate:
            gen_kwargs = {}
            if hasattr(self, "_gen_kwargs") and self._gen_kwargs is not None:
                gen_kwargs.update(self._gen_kwargs)

            if "max_length" not in gen_kwargs or gen_kwargs["max_length"] is None:
                gen_kwargs["max_length"] = self.args.generation_max_length
            if "num_beams" not in gen_kwargs or gen_kwargs["num_beams"] is None:
                gen_kwargs["num_beams"] = self.args.generation_num_beams

            with torch.no_grad():
                generated_tokens = model.generate(
                    input_features=inputs["input_features"],
                    attention_mask=inputs.get("attention_mask", None),
                    **gen_kwargs,
                )

            if generated_tokens is not None and hasattr(self, "_pad_tensors_to_max_len"):
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        # ---------- Forward for classifier logits + per-example losses ----------
        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs["loss"].detach() if has_labels else None

        logits_acc = outputs["logits_accented_or_not"]   # [B,2]
        logits_gender = outputs["logits_gender"]         # [B,2]
        logits_accent = outputs["logits_accent"]         # [B,num_accents]

        lab_acc = inputs.get("accented_or_not_labels", None)
        lab_gender = inputs.get("gender_labels", None)
        lab_accent = inputs.get("accent_labels", None)   # -100 or 0..num_accents-1

        lossvec_acc = None
        lossvec_gender = None
        lossvec_accent = None

        if lab_acc is not None:
            lossvec_acc = F.cross_entropy(logits_acc, lab_acc, reduction="none")

        if lab_gender is not None:
            lossvec_gender = F.cross_entropy(logits_gender, lab_gender, reduction="none")

        if lab_accent is not None:
            lossvec_accent = torch.zeros_like(lab_accent, dtype=torch.float32)
            mask_known = (lab_accent != int(model.accent_ignore_index))
            if mask_known.any():
                lossvec_accent[mask_known] = F.cross_entropy(
                    logits_accent[mask_known],
                    lab_accent[mask_known],
                    reduction="none",
                ).to(torch.float32)

        def make_2d(x):
            return x.unsqueeze(1) if (x is not None and x.dim() == 1) else x

        preds = (
            generated_tokens,
            logits_acc,
            logits_gender,
            logits_accent,
            make_2d(lab_acc),
            make_2d(lab_gender),
            make_2d(lab_accent),
            make_2d(lossvec_acc),
            make_2d(lossvec_gender),
            make_2d(lossvec_accent),
        )

        return (loss, preds, labels)
