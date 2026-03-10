#!/bin/bash
conda actiavete SpecFun

echo "prediction task"
MODS=("sp" "im" "ph" "im+ph" "sp+im" "sp+im+ph")
for mod in "${MODS[@]}" 
do
    echo "Running modality: $mod"
    python prediction.py --mod "$mod" --overwrite
done