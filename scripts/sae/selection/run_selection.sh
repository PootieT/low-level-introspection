#!/bin/bash
#SBATCH --job-name=sae_feature_selection
#SBATCH --partition=rtx-batch,rtx-devel,b200-batch,b200-devel
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=log/%x-%j.out

# ============================================================================
# Deterministic SAE feature selection, end to end. All parameters: scripts/sae/selection/config.json
#   1. candidates      top <depth> per layer, density band, frozen descriptions -> results/.../candidates.csv
#   2. classify        Qwen3.8-27B, description only, written rubric          -> results/.../concept_labels.csv
#   3. steer           seeded steering samples for every semantic candidate   -> data/sae_selection/steering/
#   4. judge           Qwen3.8-27B, one greedy request per prompt             -> data/sae_selection/judge/
#   5. score + select  pass rule, top-n per layer                             -> results/.../{steering_scores,selection_record}.csv, sweep_features.txt
# Before step 1, a preflight checks that the Neuronpedia cache holds every description up to
# max_rank (fill it with prefetch_neuronpedia.sh on a node with network).
# Steps 1-5 repeat, deepening short layers 200 ranks at a time (see below). Finished work is
# kept, so a resubmit continues where a timed-out job stopped; FORCE=1 starts over. Prerequisites:
# for every layer, data/sae_stats/gemma2-2b_res{L}_pos-1_16k.annotated.parquet (Steps 1-1.5 of the
# README) and gemma2-2b_res{L}_pos{P}_16k.parquet at each rank position (precompute_sae_feature_stats.sh
# with POSITIONS="8 16 24 31"); introspection_env, judge_env, and the Qwen weights in .hf_cache.
#     sbatch scripts/sae/selection/run_selection.sh
# ============================================================================
echo "Start: $(date) on $(hostname)"; nvidia-smi -L || true
set -e
ROOT=/work/neu/p2026_0061_neu/low-level-introspection; cd $ROOT
module load miniforge3; source "$(conda info --base)/etc/profile.d/conda.sh"
module load ${CUDA_MODULE:-cuda/13.1.1}   # vLLM/FlashInfer JIT needs nvcc; nodes have no /usr/local/cuda
export CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
export HF_HOME=${HF_HOME:-$ROOT/.hf_cache} HF_HUB_OFFLINE=1 WANDB_MODE=offline
CFG=scripts/sae/selection/config.json
RES=results/sae_feature_selection; DAT=data/sae_selection
mkdir -p $RES $DAT/steering $DAT/judge
cfg() { python -c "import json,sys; c=json.load(open('$CFG')); print(eval(sys.argv[1], {}, {'c': c}))" "$1"; }
ENV_I=${INTRO_ENV:-$ROOT/introspection_env}; ENV_J=${JUDGE_ENV:-$ROOT/judge_env}

# Search down the ranking one block (config "block", 200) at a time. After each block every
# layer that still has fewer than n_select passing features is deepened by one block, up to
# "max_rank". The picks are the highest-ranked passing features, so stopping once a layer is
# full gives the same picks as searching to max_rank everywhere. Steering samples are seeded
# per (layer, feature), so splitting the work into blocks does not change any sample. (The
# Qwen steps are greedy but vLLM is not batch-invariant, so a rerun with a different batch
# makeup can, rarely, flip a label.) Step 1 runs --offline: descriptions for ranks up to
# max_rank must be prefetched into data/neuronpedia_cache first, and the steering prompt pool
# frozen to data/sae_selection/prompt_pool.json (data_prep/generate_sae_steering_samples.py
# --write-prompt-file ... --prompt-pool 200 --prompt-words 8), on a node with network:
#     bash scripts/sae/selection/prefetch_neuronpedia.sh   # = data_prep/sae_select_candidates.py --depth-all <max_rank> \
#         --out data/sae_selection/prefetch/candidates_1000.csv --annotated-out data/sae_selection/prefetch/annotated
BLOCK=$(cfg "c['candidates']['block']"); MAXR=$(cfg "c['candidates']['max_rank']")
DEPTH=$DAT/depth.json

# Preflight: every description up to max_rank must already be in the Neuronpedia cache (this job
# runs offline). Fails in about a minute, naming what is missing, instead of hours into the search.
conda activate $ENV_I
python data_prep/sae_select_candidates.py --depth-all $MAXR --offline \
    --out $DAT/preflight/candidates.csv --annotated-out $DAT/preflight/annotated > $DAT/preflight.log 2>&1 \
  || { tail -3 $DAT/preflight.log; echo "Neuronpedia cache incomplete for max_rank=$MAXR; on a node with network run:"
       echo "    bash scripts/sae/selection/prefetch_neuronpedia.sh"; exit 1; }
echo "preflight: descriptions cached for ranks 1-$MAXR in every layer"
[ -n "$FORCE" ] && rm -rf $DEPTH $RES/concept_labels.csv $RES/concept_labels.raw.jsonl $DAT/steering $DAT/judge
[ -e $DEPTH ] || python -c "
import json; c = json.load(open('$CFG')); json.dump({str(L): c['candidates']['block'] for L in c['layers']}, open('$DEPTH', 'w'))"

while true; do
  echo "=== depth per layer: $(cat $DEPTH)"
  conda activate $ENV_I
  echo "=== 1. candidates"
  python data_prep/sae_select_candidates.py --out $RES/candidates.csv --annotated-out $DAT/annotated --depth-file $DEPTH --offline

  conda activate $ENV_J
  echo "=== 2. classify (new candidates only)"
  python data_prep/sae_classify_concepts.py --candidates $RES/candidates.csv --out $RES/concept_labels.csv

  conda activate $ENV_I
  echo "=== 3. steering samples (semantic candidates, one folder per block)"
  for L in $(cfg "' '.join(map(str, c['layers']))"); do
    D=$(python -c "import json; print(json.load(open('$DEPTH'))['$L'])")
    for (( HI=BLOCK; HI<=D; HI+=BLOCK )); do
      OUT=$DAT/steering/r$((HI-BLOCK+1))-$HI; STEM=$OUT/gemma-2-2b_res${L}_16k
      FEATS=$(python -c "
import pandas as pd; d = pd.read_csv('$RES/concept_labels.csv')
d = d[(d.layer == $L) & (d.semantic == True) & (d['rank'] > $HI - $BLOCK) & (d['rank'] <= $HI)]
print(' '.join(s.split('.')[1] for s in d.feature_id))")
      SIG="$FEATS | $(python -c "import json; print(json.dumps(json.load(open('$CFG'))['steering'], sort_keys=True))")"
      if [ -z "$FEATS" ]; then echo "layer $L ranks $((HI-BLOCK+1))-$HI: no semantic candidates"; continue; fi
      # reuse a block only if it is complete and was generated for exactly this feature list
      if [ -e $STEM.judge.md ] && [ -e $STEM.key.json ] && [ "$(cat $STEM.features 2>/dev/null)" = "$SIG" ]; then continue; fi
      rm -f $STEM.judge.md $STEM.key.json $STEM.samples.jsonl $STEM.features
      echo "layer $L ranks $((HI-BLOCK+1))-$HI: $(echo $FEATS | wc -w) features"
      python data_prep/generate_sae_steering_samples.py \
          --annotated $DAT/annotated/gemma2-2b_res${L}_pos-1_16k.annotated.parquet \
          --model $(cfg "c['steering']['model']") --sae-release $(cfg "c['steering']['sae_release']") \
          --layer $L --width $(cfg "c['steering']['width']") --site res \
          --n-prompts $(cfg "c['steering']['n_prompts']") --prompt-pool $(cfg "c['steering']['prompt_pool']") \
          --prompt-words $(cfg "c['steering']['prompt_words']") --max-new-tokens $(cfg "c['steering']['max_new_tokens']") \
          --strengths $(cfg "' '.join(map(str, c['steering']['strengths']))") --seed $(cfg "c['steering']['seed']") \
          --magnitude-column $(cfg "c['steering']['magnitude']") --prompt-file $DAT/prompt_pool.json --features $FEATS --out-dir $OUT
      echo "$SIG" > $STEM.features
    done
  done

  conda activate $ENV_J
  echo "=== 4. judge (features without picks only)"
  python scripts/sae/judge_steering_local.py --out-dir $DAT/judge \
      --model $(cfg "c['judge']['model']") --revision $(cfg "c['judge']['revision']") --seed $(cfg "c['judge']['seed']") \
      --judge-md $DAT/steering/r*/gemma-2-2b_res*_16k.judge.md

  conda activate $ENV_I
  echo "=== 5. score + select"
  python -c "
import pandas as pd, json
d = pd.read_csv('$RES/candidates.csv'); json.dump(dict(zip(d.feature_id, d['rank'].astype(int).tolist())), open('$DAT/ranks.json', 'w'))"
  python scripts/sae/score_steering_judge.py --picks $DAT/judge/picks --key-dir $DAT/steering/r* \
      --rank-file $DAT/ranks.json --hit-rule $(cfg "c['pass_rule']['hit_rule']") --min-hit $(cfg "c['pass_rule']['min_hit']") \
      --min-judged $(cfg "c['pass_rule']['min_judged']") \
      --csv $RES/steering_scores.csv > $RES/steering_scores.txt
  python data_prep/sae_select_features.py --config $CFG --dir $RES --depth-file $DEPTH

  # deepen every short layer by one block; stop when none is short or all short ones hit max_rank
  MORE=$(python -c "
import json, pandas as pd
c = json.load(open('$CFG')); dep = json.load(open('$DEPTH')); r = pd.read_csv('$RES/selection_record.csv')
more = False
for L in c['layers']:
    k = int(c['n_select'].get(str(L), c['n_select']['default']))
    if int(r[(r.layer == L) & r.selected].shape[0]) < k and dep[str(L)] < c['candidates']['max_rank']:
        dep[str(L)] += c['candidates']['block']; more = True
json.dump(dep, open('$DEPTH', 'w')); print(int(more))")
  [ "$MORE" = 1 ] || break
done
cp $DEPTH $RES/search_depth.json
cp $DAT/judge/run_info.json $RES/judge_run_info.json
echo "Done: $(date)"
