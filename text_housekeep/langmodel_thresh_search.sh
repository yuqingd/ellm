#! /usr/bin/env bash

./generate_configs.sh

# configs=(cos_eor/configs/local/nav_sp/{ihlen_1_int,merom_1_int,beechwood_1_int,benevolence_1_int,ihlen_0_int,merom_0_int}.yaml)
# configs=(cos_eor/configs/local/nav_sp/{ihlen_1_int,merom_1_int}.yaml)
configs=(cos_eor/configs/local/nav_sp/ihlen_1_int.yaml)
# configs=(cos_eor/configs/local/nav_sp/ihlen_1_int.yaml)

eval_config=(cos_eor/configs/local/nav_sp/merom_1_int.yaml)
ckpt_file=lm_mlm_scores_thresh_agg

python cos_eor/scripts/langmodel_thresh_search.py \
    --config-files ${configs[@]} \
    --num-jobs 60 \
    --eval-config-files ${eval_config} \
    --scores-dump-file ./${ckpt_file} \
    --beta 1 \
    --aggregate-threshold
    # --scores-checkpoint-file ./${ckpt_file}.npy \
