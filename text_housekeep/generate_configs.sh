#!/usr/bin/env bash

exp_id=$1
explore_type=$2
rank_type=$3
echo $exp_id
out_dir=
if [[ -z "$exp_id" ]] ; then
    out_dir=./cos_eor/configs/local/nav_sp/
else
    out_dir=./logs/$exp_id/configs/
fi

python cos_eor/scripts/generate_configs.py --config-file ./cos_eor/configs/local/igib_v2_nav_sp.yaml --out-dir $out_dir \
--explore-type $2 --rank-type $3
