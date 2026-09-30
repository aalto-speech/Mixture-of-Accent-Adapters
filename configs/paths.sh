#!/usr/bin/env bash
# =============================================================================
# MoAA path configuration  --  EDIT THIS FILE for your machine.
#
# This is the ONLY place where file-system paths are defined. Every script reads
# these variables as defaults (and every value can still be overridden with a
# command-line flag). Usage:
#
#     source configs/paths.sh
#
# The values below are the ones used on Aalto Triton.
# =============================================================================

# Repository root (auto-detected; no need to edit)
export MOAA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# -----------------------------------------------------------------------------
# Datasets (HuggingFace `save_to_disk` format; see README.md -> Data)
# -----------------------------------------------------------------------------
DATA_ROOT="/scratch/elec/puhe/c"

# AESRC training data: colon-separated list of directories, each holding one
# load_from_disk sub-directory per accent. All 10 accents were used for training;
# in our copy 8 accents live in train/ and 2 (Canadian, Spanish) in test_unseen/.
export MOAA_AESRC_TRAIN_ROOTS="${DATA_ROOT}/AESRC/data_hf_format/train:${DATA_ROOT}/AESRC/data_hf_format/test_unseen"

# LibriSpeech train-clean-100 (non-accented speech; accent label "unknown")
export MOAA_LIBRISPEECH100="${DATA_ROOT}/librispeech/LibriSpeech/data_hf_format/train-clean-100"

# Test sets (set a variable to "" to skip that test set)
export MOAA_AESRC_TEST="${DATA_ROOT}/AESRC/data_hf_format/test"
export MOAA_OPENSLR_TEST="${DATA_ROOT}/OpenSLR_British-Accents/data_hf_format/test"
export MOAA_EDACC_TEST="${DATA_ROOT}/EDACC/data_hf_format/edacc/test"
export MOAA_GLOBE_TEST="${DATA_ROOT}/GLOBE/test"

# -----------------------------------------------------------------------------
# Models / outputs
# -----------------------------------------------------------------------------
# Base Whisper model (HF Hub id or local directory)
export MOAA_BASE_MODEL="openai/whisper-small"

# Where new training / evaluation runs are written
export MOAA_OUTPUT_ROOT="${MOAA_ROOT}/outputs"

# Released MoAA checkpoints (see checkpoints/README.md)
export MOAA_CHECKPOINT_ROOT="${MOAA_ROOT}/checkpoints"

# -----------------------------------------------------------------------------
# Caches (keep them off your small $HOME quota on clusters)
# -----------------------------------------------------------------------------
export HF_HOME="/scratch/elec/puhe/p/bijoym1/hf_home"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TMPDIR="${TMPDIR:-/tmp}"
