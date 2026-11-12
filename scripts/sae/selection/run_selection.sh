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

echo "Start: $(date) on $(hostname)"; nvidia-smi -L || true
set -e
ROOT=${ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}}; cd $ROOT
module load miniforge3; source "$(conda info --base)/etc/profile.d/conda.sh"
module load ${CUDA_MODULE:-cuda/13.1.1}
export CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
export HF_HOME=${HF_HOME:-$ROOT/.hf_cache} HF_HUB_OFFLINE=1 WANDB_MODE=offline
CFG=scripts/sae/selection/config.json
RES=results/sae_feature_selection; DAT=data/sae_selection
mkdir -p $RES $DAT/steering $DAT/judge
cfg() { python -c "import json,sys; c=json.load(open('$CFG')); print(eval(sys.argv[1], {}, {'c': c}))" "$1"; }
ENV_I=${INTRO_ENV:-$ROOT/introspection_env}; ENV_J=${JUDGE_ENV:-$ROOT/judge_env}

BLOCK=$(cfg "c['candidates']['block']"); MAXR=$(cfg "c['candidates']['max_rank']")
DEPTH=$DAT/depth.json

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
