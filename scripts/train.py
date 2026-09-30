#!/usr/bin/env python
"""
Train MoAA (Mixture of Accent Adapters) on AESRC + LibriSpeech-100 and evaluate it.

Outputs (under <output_root>/<run_name>/):
  checkpoint-<step>/          best-dev-WER checkpoint (HF Trainer format; full weights)
  last_checkpoint/            final model via save_pretrained (+ tokenizer/processor)
  runs/                       TensorBoard logs
  train.log                   log file
  predictions/internal_test.csv            (+ _metrics.json)
  predictions/{aesrc20h,openslr,edacc,globe}.csv   (+ _metrics.json)

All paths default to environment variables exported by `configs/paths.sh`
and can be overridden on the command line.

Example:
  source configs/paths.sh
  python scripts/train.py --variant linear_proj --num_adapters 20
"""

import argparse
import logging
import os
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate  # noqa: E402
from datasets import disable_caching  # noqa: E402
from transformers import (  # noqa: E402
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperTokenizer,
)

from moaa.data import (  # noqa: E402
    EXTERNAL_TEST_SETS,
    DataCollatorSpeechSeq2SeqWithPaddingAndClf,
    build_multitask_splits,
    featurize_multitask_splits,
)
from moaa.evaluation import evaluate_external_test_set, save_predictions_csv  # noqa: E402
from moaa.metrics import compute_metrics_asr_and_clf  # noqa: E402
from moaa.model import build_moaa_model  # noqa: E402
from moaa.trainer import MultiTaskSeq2SeqTrainer  # noqa: E402
from moaa.utils import (  # noqa: E402
    check_gpu,
    env_default,
    env_list,
    log_clf_metrics,
    set_seed,
    setup_logging,
    write_json,
)

disable_caching()

# Variant presets (exactly the settings of the released runs)
VARIANTS = {
    # Frozen Whisper; linear projection on pooled encoder features before the heads.
    "linear_proj": {"use_linear_proj": True, "unfreeze_last_n": 0, "lr": 5e-4},
    # Frozen Whisper; no projection.
    "frozen": {"use_linear_proj": False, "unfreeze_last_n": 0, "lr": 5e-4},
    # Whisper frozen except the last 2 encoder + last 2 decoder blocks; no projection.
    "unfreeze_last2": {"use_linear_proj": False, "unfreeze_last_n": 2, "lr": 5e-5},
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("data paths (defaults come from configs/paths.sh)")
    g.add_argument("--aesrc_train_roots", nargs="+", default=env_list("MOAA_AESRC_TRAIN_ROOTS"),
                   help="One or more dirs, each containing per-accent AESRC load_from_disk subdirs.")
    g.add_argument("--librispeech100", default=env_default("MOAA_LIBRISPEECH100"))
    g.add_argument("--aesrc_test", default=env_default("MOAA_AESRC_TEST"))
    g.add_argument("--openslr_test", default=env_default("MOAA_OPENSLR_TEST"))
    g.add_argument("--edacc_test", default=env_default("MOAA_EDACC_TEST"))
    g.add_argument("--globe_test", default=env_default("MOAA_GLOBE_TEST"))
    g.add_argument("--output_root", default=env_default("MOAA_OUTPUT_ROOT", "./outputs"))
    g.add_argument("--run_name", default=None, help="Default: moaa_<variant>_A<num_adapters>")

    m = p.add_argument_group("model")
    m.add_argument("--variant", choices=sorted(VARIANTS), default="linear_proj")
    m.add_argument("--model_name", default=env_default("MOAA_BASE_MODEL", "openai/whisper-small"))
    m.add_argument("--num_adapters", type=int, default=10)
    m.add_argument("--bottleneck_size", type=int, default=192)
    m.add_argument("--num_accents", type=int, default=10)
    m.add_argument("--accent_ignore_index", type=int, default=-100)
    m.add_argument("--grl_lambda", type=float, default=1.0)
    m.add_argument("--loss_weights", type=float, nargs=4, default=[1.0, 1.0, 2.0, 1.0],
                   metavar=("W_ASR", "W_ACCENTED", "W_ACCENT", "W_GENDER"))

    t = p.add_argument_group("training")
    t.add_argument("--language", default="en")
    t.add_argument("--task", default="transcribe")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--batch_size", type=int, default=16)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--lr", type=float, default=None, help="Default: variant preset")
    t.add_argument("--eval_split_ratio", type=float, default=0.1)
    t.add_argument("--devel_size", type=int, default=1000)
    t.add_argument("--eval_before_training", action="store_true",
                   help="Also evaluate the untrained model on the internal test set (slow; not needed).")
    t.add_argument("--skip_tests", nargs="*", default=[], choices=sorted(EXTERNAL_TEST_SETS),
                   help="External test sets to skip after training.")
    t.add_argument("--max_test_samples", type=int, default=None, help="Debug: subsample external test sets.")
    t.add_argument("--max_train_samples", type=int, default=None,
                   help="Debug: subsample train/devel/internal-test splits (smoke tests only).")
    t.add_argument("--no_fp16", action="store_true", help="Disable fp16 (e.g. CPU debugging).")
    return p.parse_args()


def main():
    args = parse_args()
    preset = VARIANTS[args.variant]
    lr = args.lr if args.lr is not None else preset["lr"]

    if not args.aesrc_train_roots or not args.librispeech100:
        raise SystemExit("AESRC train roots and LibriSpeech path are required (source configs/paths.sh or pass flags).")

    run_name = args.run_name or f"moaa_{args.variant}_A{args.num_adapters}"
    run_dir = Path(args.output_root) / run_name
    pred_dir = run_dir / "predictions"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(level=logging.INFO, log_file=str(run_dir / "train.log"))
    check_gpu(logger)
    logger.info("Run: %s | variant=%s | lr=%g | args=%s", run_name, args.variant, lr, vars(args))
    write_json({**vars(args), "lr_effective": lr, **preset}, run_dir / "run_config.json")

    set_seed(args.seed)

    # ---------------- Data ----------------
    dataset = build_multitask_splits(
        aesrc_train_roots=args.aesrc_train_roots,
        librispeech_path=args.librispeech100,
        eval_split_ratio=args.eval_split_ratio,
        seed=args.seed,
        devel_size=args.devel_size,
        logger=logger,
    )
    if args.max_train_samples:
        n = args.max_train_samples
        for split_name in list(dataset.keys()):
            k = n if split_name == "train" else max(1, n // 4)
            dataset[split_name] = dataset[split_name].select(range(min(k, len(dataset[split_name]))))
        logger.warning("DEBUG: subsampled splits to %s", {k: len(v) for k, v in dataset.items()})
    internal_test_raw = dataset["test"]

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.language, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task=args.task)

    dataset = featurize_multitask_splits(dataset, feature_extractor, tokenizer)
    for split_name in dataset.keys():
        logger.info("%s: %d rows | cols=%s", split_name, len(dataset[split_name]), dataset[split_name].column_names)

    # ---------------- Model ----------------
    model = build_moaa_model(
        model_name=args.model_name,
        num_adapters=args.num_adapters,
        bottleneck_size=args.bottleneck_size,
        num_accents=args.num_accents,
        accent_ignore_index=args.accent_ignore_index,
        grl_lambda=args.grl_lambda,
        loss_weights=tuple(args.loss_weights),
        use_linear_proj=preset["use_linear_proj"],
        unfreeze_last_n=preset["unfreeze_last_n"],
        logger=logger,
    )
    model.whisper.generation_config.language = args.language
    model.whisper.generation_config.task = args.task
    model.whisper.generation_config.forced_decoder_ids = None
    model.whisper.config.use_cache = False
    model.whisper.config.pad_token_id = tokenizer.pad_token_id

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: %d / %d (%.2f%%)", n_trainable, n_total, 100 * n_trainable / n_total)

    data_collator = DataCollatorSpeechSeq2SeqWithPaddingAndClf(
        processor=processor,
        decoder_start_token_id=model.whisper.config.decoder_start_token_id,
    )

    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
        learning_rate=lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=not args.no_fp16,
        gradient_checkpointing=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=1,
        report_to=["tensorboard"],
        seed=args.seed,
    )

    trainer = MultiTaskSeq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["devel"],
        data_collator=data_collator,
        compute_metrics=partial(
            compute_metrics_asr_and_clf,
            tokenizer=tokenizer,
            metric_wer=metric_wer,
            metric_cer=metric_cer,
            accent_ignore_index=args.accent_ignore_index,
        ),
        tokenizer=processor.tokenizer,
    )

    if args.eval_before_training:
        logger.info("Evaluation on INTERNAL TEST (BEFORE TRAINING)")
        log_clf_metrics(logger, trainer.predict(dataset["test"]).metrics)

    # ---------------- Train ----------------
    trainer.train()
    for item in trainer.state.log_history:
        logger.info("%s", item)
    logger.info("Training complete. Best checkpoint: %s", trainer.state.best_model_checkpoint)

    # ---------------- Save final model (best weights are loaded at end) ----------------
    last_dir = run_dir / "last_checkpoint"
    model.save_pretrained(str(last_dir))
    processor.save_pretrained(str(last_dir))
    tokenizer.save_pretrained(str(last_dir))
    feature_extractor.save_pretrained(str(last_dir))
    (last_dir / "training_args.json").write_text(training_args.to_json_string())
    trainer.state.save_to_json(str(last_dir / "trainer_state.json"))
    logger.info("Saved final model to: %s", last_dir)

    # ---------------- Internal test ----------------
    logger.info("Evaluation on INTERNAL TEST (AFTER TRAINING)")
    results = trainer.predict(dataset["test"])
    log_clf_metrics(logger, results.metrics)
    save_predictions_csv(results, internal_test_raw, tokenizer, str(pred_dir / "internal_test.csv"))
    write_json({"test_set": "internal_test", "n": len(internal_test_raw), **results.metrics},
               pred_dir / "internal_test_metrics.json")

    # ---------------- External test sets ----------------
    test_paths = {
        "aesrc20h": args.aesrc_test,
        "openslr": args.openslr_test,
        "edacc": args.edacc_test,
        "globe": args.globe_test,
    }
    for name, path in test_paths.items():
        if name in args.skip_tests or not path:
            logger.info("Skipping %s test set.", name)
            continue
        evaluate_external_test_set(
            name, path, model, processor, tokenizer, feature_extractor, training_args,
            metric_wer, metric_cer, str(pred_dir), logger,
            max_samples=args.max_test_samples, seed=args.seed,
        )

    logger.info("Done. Outputs in %s", run_dir)


if __name__ == "__main__":
    main()
