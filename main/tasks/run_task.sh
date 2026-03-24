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
            # --overwrite
            CUDA_VISIBLE_DEVICES=0 python extract_features.py --tasks extract_feature --data "$DATA" --mods "${MODS[@]}"
            ;;
        mask)
            # MASK="snr_quality"
            MASKS=("R_cut") # spec_snr, mag_cut, zq_cut, R_cut
            CUDA_VISIBLE_DEVICES=0 python quality_ids.py  --masks "${MASKS[@]}" --data $DATA --overwrite
            ;;
        predict)
            TASK="predict_labels" # predict_labels, direct_z
            LABELS=("z")
            # LABELS=("z" "m_star" "z_mw" "t_age" "sfr")
            MASKS=("spec_snr_mean_cut1" "spec_snr_mean_cut2" "spec_snr_mean_cut3" "spec_snr_mean_cut4")
            CUDA_VISIBLE_DEVICES=0 python prediction.py --tasks $TASK --data $DATA --mods "${MODS[@]}" --labels "${LABELS[@]}" --masks "${MASKS[@]}"
            # for lab in "${LABELS[@]}"; do
                # CUDA_VISIBLE_DEVICES=0 python prediction.py --data provabgs-v2 --mod "${MODS[@]}" --labels "$lab"
            # done
            ;;
        class)
            TASK="classification" # predict_labels, direct_z
            LABELS=("type")
            CUDA_VISIBLE_DEVICES=3 python classification.py --tasks $TASK --data $DATA --mods "${MODS[@]}" --labels "${LABELS[@]}"
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

