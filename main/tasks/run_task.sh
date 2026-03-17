#!/bin/bash
##  to use: e.g. bash run_tasks.sh extract

activate_env() {
    case $1 in
        predict|extract|mask|class)
            source ~/.bashrc
            conda activate SpecFun
            ;;
        *)
            echo "Unknown mode for environment activation: $1"
            exit 1
            ;;
    esac
}

DATA="desi-sv1" # desi-sv1, provabgs-v2
MODS=("sp" "im" "ph" "im+ph" "sp+im" "sp+ph" "sp+im+ph")

run_task() {
    case $1 in
        extract)
            python extract_features.py --task extract_feature --data $DATA --mod "${MODS[@]}" --overwrite
            ;;
        mask)
            python quality_ids.py --tasks snr_quality --data $DATA --overwrite
            ;;
        predict)
            LABELS=("z")
            # LABELS=("z" "m_star" "z_mw" "t_age" "sfr")
            python prediction.py --task predict_labels --data $DATA --mod "${MODS[@]}" --labels "${LABELS[@]}"
            # for lab in "${LABELS[@]}"; do
            #     python prediction.py --data provabgs-v2 --mod "${MODS[@]}" --labels "$lab"
            # done    
            ;;
        class)
            LABEL="type"
            python classification.py --task classfication --data $DATA --mod "${MODS[@]}" --labels $LABEL 
            ;;            
    esac
}

if [ -z "$1" ]; then
    echo "Usage: ./run_specfun.sh [extract|mask]"
    exit 1
fi

# Run command
activate_env $1
run_task $1

