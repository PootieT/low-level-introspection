module load cuda/11.8   # 11.2
module load miniconda/23.11.0
conda activate /projectnb/llamagrp/peter/low-level-introspection/introspection_env

export HF_HOME=/projectnb/ivc-ml/SharedHuggingFaceCache

for LAYER in 0 3 6 9 12 15 18 21 24; do
  python select_sae_features.py \
    --model google/gemma-2-2b \
    --sae-release gemma-scope-2b-pt-res-canonical \
    --layer $LAYER \
    --width 16k \
    --n-tokens 1000000 \
    --max-seq-length 128 \
    --fixed-position -1 \
    --cache-size 20000 \
    --out data/sae_stats/gemma2-2b_res9_16k.parquet

done

# for layer 9
# [select] wrote 16384 rows -> data/sae_stats/gemma2-2b_res9_16k.parquet  (tokens=1000083, seqs=8800, selected=7897)
  #
  #=== TOP 40 SELECTED FEATURES (density in [0.001,0.01]) ===
  #  pick one with proj_pos_at_fixed_pos comfortably > a few hundred, then
  #  confirm its concept on Neuronpedia before caching.
  # feature  density  pos_density  proj_pos_at_fixed_pos  active_mean  active_max
  #    6543 0.006995       0.1301                   2602        4.761       20.69
  #     855 0.009481       0.0758                   1516        5.088       18.82
  #    8512 0.008986      0.07568                   1514         4.53       15.95
  #    5546  0.00585      0.07034                   1407        4.367       15.37
  #    4471 0.008608      0.06352                   1270        4.673       18.88
  #   12408 0.008833      0.05943                   1189        4.703       21.88
  #   12076 0.009929      0.05352                   1070        4.843       24.44
  #    4796 0.006209      0.04761                    952        5.225       19.56
  #   13157 0.002916       0.0458                    916        4.827       27.88
  #    5444 0.008721      0.04477                    895        4.244       15.73
  #     817 0.007059      0.04466                    893        4.066       14.28
  #    9021 0.004639      0.04273                    855         3.96       14.56
  #   12357 0.006517      0.04136                    827        5.288       24.74
  #    3390 0.008744      0.03898                    780         4.33       15.38
  #    2569 0.003614      0.03636                    727        26.59       42.52
  #    6965 0.007977      0.03602                    720        4.463       15.07
  #   12065 0.006203      0.03477                    695        4.327       20.83
  #    3422 0.006783      0.03239                    648        5.073       19.35

# 12408 first non-punctuation feature: expression of positive feedback or appreciation