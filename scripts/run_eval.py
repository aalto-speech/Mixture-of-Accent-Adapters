#!/usr/bin/env python
"""
Evaluate a trained MoAA checkpoint (no training) on the external test sets and,
optionally, on the internal AESRC+LibriSpeech test split.

Writes <out_dir>/predictions/<test_set>.csv and <test_set>_metrics.json.

Example:
  source configs/paths.sh
  python scripts/run_eval.py \
      --checkpoint "$MOAA_CHECKPOINT_ROOT/moaa_linear_proj_A20/checkpoint-101835" \
      --out_dir outputs/eval_moaa_linear_proj_A20 \
      --tests aesrc20h openslr edacc globe
"""

import argparse
import logging
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
from moaa.model import load_moaa_checkpoint  # noqa: E402
from moaa.trainer import MultiTaskSeq2SeqTrainer  # noqa: E402
from moaa.utils import check_gpu, env_default, env_list, log_clf_metrics, set_seed, setup_logging, write_json  # noqa: E402

disable_caching()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True,
                   help="checkpoint-<step>/ dir (recommended) or last_checkpoint/ dir.")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tests", nargs="+", default=["aesrc20h", "openslr", "edacc", "globe"],
                   choices=sorted(EXTERNAL_TEST_SETS) + ["internal"])

    p.add_argument("--aesrc_train_roots", nargs="+", default=env_list("MOAA_AESRC_TRAIN_ROOTS"),
                   help="Only needed for --tests internal (to rebuild the split).")
    p.add_argument("--librispeech100", default=env_default("MOAA_LIBRISPEECH100"))
    p.add_argument("--aesrc_test", default=env_default("MOAA_AESRC_TEST"))
    p.add_argument("--openslr_test", default=env_default("MOAA_OPENSLR_TEST"))
    p.add_argument("--edacc_test", default=env_default("MOAA_EDACC_TEST"))
    p.add_argument("--globe_test", default=env_default("MOAA_GLOBE_TEST"))

    p.add_argument("--model_name", default=env_default("MOAA_BASE_MODEL", "openai/whisper-small"),
                   help="Base Whisper model (architecture, tokenizer, feature extractor).")
    p.add_argument("--language", default="en")
    p.add_argument("--task", default="transcribe")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_split_ratio", type=float, default=0.1)
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--proj_in_generate", action="store_true",
                   help="Apply linear_proj inside generate(). Off for the released results.")
    p.add_argument("--max_test_samples", type=int, default=None, help="Debug: subsample test sets.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(level=logging.INFO, log_file=str(out_dir / "evaluate.log"))
    check_gpu(logger)
    logger.info("args=%s", vars(args))
    write_json(vars(args), out_dir / "eval_config.json")
    set_seed(args.seed)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.language, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task=args.task)

    model = load_moaa_checkpoint(args.checkpoint, base_model_name=args.model_name,
                                 proj_in_generate=args.proj_in_generate, logger=logger)
    model.whisper.generation_config.language = args.language
    model.whisper.generation_config.task = args.task
    model.whisper.generation_config.forced_decoder_ids = None
    model.whisper.config.use_cache = False
    model.whisper.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")

    eval_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "trainer_tmp"),
        per_device_eval_batch_size=args.batch_size,
        fp16=not args.no_fp16,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=[],
        seed=args.seed,
    )

    if "internal" in args.tests:
        if not args.aesrc_train_roots or not args.librispeech100:
            raise SystemExit("--tests internal needs --aesrc_train_roots and --librispeech100")
        dataset = build_multitask_splits(args.aesrc_train_roots, args.librispeech100,
                                         eval_split_ratio=args.eval_split_ratio, seed=args.seed, logger=logger)
        test_raw = dataset["test"]
        if args.max_test_samples:
            test_raw = test_raw.select(range(min(args.max_test_samples, len(test_raw))))
        test_feat = featurize_multitask_splits({"test": test_raw}, feature_extractor, tokenizer)["test"]
        trainer = MultiTaskSeq2SeqTrainer(
            args=eval_args,
            model=model,
            data_collator=DataCollatorSpeechSeq2SeqWithPaddingAndClf(
                processor=processor, decoder_start_token_id=model.whisper.config.decoder_start_token_id),
            compute_metrics=partial(compute_metrics_asr_and_clf, tokenizer=tokenizer,
                                    metric_wer=metric_wer, metric_cer=metric_cer),
            tokenizer=processor.tokenizer,
        )
        logger.info("Evaluation on INTERNAL TEST")
        results = trainer.predict(test_feat)
        log_clf_metrics(logger, results.metrics)
        save_predictions_csv(results, test_raw, tokenizer, str(pred_dir / "internal_test.csv"))
        write_json({"test_set": "internal_test", "n": len(test_raw), **results.metrics},
                   pred_dir / "internal_test_metrics.json")

    test_paths = {
        "aesrc20h": args.aesrc_test,
        "openslr": args.openslr_test,
        "edacc": args.edacc_test,
        "globe": args.globe_test,
    }
    for name in args.tests:
        if name == "internal":
            continue
        if not test_paths[name]:
            logger.warning("No path configured for %s; skipping.", name)
            continue
        evaluate_external_test_set(
            name, test_paths[name], model, processor, tokenizer, feature_extractor, eval_args,
            metric_wer, metric_cer, str(pred_dir), logger,
            max_samples=args.max_test_samples, seed=args.seed,
        )

    logger.info("Done. Predictions in %s", pred_dir)


if __name__ == "__main__":
    main()
