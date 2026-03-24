#!/bin/bash
# Run this first in the terminal:
# conda activate SpecFun

DATA="desi-sv1" # desi-sv1, provabgs-v2
MODS=("sp" "im" "ph" "im+ph" "sp+im" "sp+ph" "sp+im+ph")
TASK="classification" # predict_labels, direct_z
LABELS=("type")
# MASKS=("spec_snr_mean_cut1" "spec_snr_mean_cut2" "spec_snr_mean_cut3" "spec_snr_mean_cut4")
# MASKS=("spec_snr_median_cut1" "spec_snr_median_cut2" "spec_snr_median_cut3" "spec_snr_median_cut4")
# MASKS=("zq_cut1" "zq_cut2" "zq_cut3" "zq_cut4")

# CUDA_VISIBLE_DEVICES=3 python classification.py --tasks $TASK --data $DATA --mods "${MODS[@]}" --labels "${LABELS[@]}" --masks "${MASKS[@]}"
CUDA_VISIBLE_DEVICES=0 python classification.py --tasks $TASK --data $DATA --mods "${MODS[@]}" --labels "${LABELS[@]}"