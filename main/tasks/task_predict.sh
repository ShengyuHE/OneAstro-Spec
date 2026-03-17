#!/bin/bash
# Run this first in the terminal:
# conda activate SpecFun
# tasks: predict_labels, direct_z

MODS=("sp" "im" "ph" "im+ph" "sp+im" "sp+ph" "sp+im+ph")

# python prediction.py --task direct_z --data desi-sv1 --mod "${MODS[@]}" --labels z
# python prediction.py --task direct_z --data provabgs-v2 --mod "${MODS[@]}" --labels z

# MODS=("ph" "im" "sp+im" "sp+im+ph")
LABELS=("z" "m_star" "z_mw" "t_age" "sfr")
for lab in "${LABELS[@]}"; do
    python prediction.py --data provabgs-v2 --mod "${MODS[@]}" --labels "$lab"
done