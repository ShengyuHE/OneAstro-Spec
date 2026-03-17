#!/bin/bash
# Run this first in the terminal:
# conda activate SpecFun

# Classification task for DESI-SV1 dataset
# Modalities to use
MODS=("sp" "im" "ph" "sp+im" "sp+ph" "im+ph" "sp+im+ph")

# Run feature extraction and classification
python classification.py --task predict --data desi-sv1 --mods "${MODS[@]}" --labels type
