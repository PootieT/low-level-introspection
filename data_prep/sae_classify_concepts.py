#!/usr/bin/env python3
"""Step 2: semantic vs structural, from the description alone (no steering results are
seen). One greedy, thinking-off request per candidate to the judge model in config.json,
with the written rubric there. Writes concept_labels.csv (feature_id, semantic, category,
reason) and raw responses next to it. Candidates already in --out keep their label, so
deepening the pool only classifies the new ones.

    python data_prep/sae_classify_concepts.py \
        --candidates results/sae_feature_selection/candidates.csv \
        --out results/sae_feature_selection/concept_labels.csv
"""
import argparse, csv, json, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[1]
CATS = {"semantic", "punctuation_formatting", "numbers", "code_markup", "lexical_grammar",
        "template_web", "exclamation", "vague", "other"}   # "other" = outside the rubric; not selected


def parse(text):
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(d.get("semantic"), bool) or d.get("category") not in CATS:
        return None
    if d["semantic"] != (d["category"] == "semantic"):
        return None
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "scripts/sae/selection/config.json")
    p.add_argument("--candidates", type=Path, default=ROOT / "results/sae_feature_selection/candidates.csv")
    p.add_argument("--out", type=Path, default=ROOT / "results/sae_feature_selection/concept_labels.csv")
    p.add_argument("--max-num-seqs", type=int, default=256)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    jc = cfg["judge"]
    # stdlib csv only: this runs in judge_env (vLLM), which has no pandas
    with open(args.candidates, newline="") as fh:
        cand = [r for r in csv.DictReader(fh) if r["explanation"]]   # no description: not eligible
    old, old_raw = [], []
    if args.out.exists():   # unparsed rows from a failed run are classified again, not kept
        with open(args.out, newline="") as fh:
            old = [r for r in csv.DictReader(fh) if r["category"] != "unparsed"]
        keep = {r["feature_id"] for r in old}
        raw_path = args.out.with_suffix(".raw.jsonl")
        if raw_path.exists():
            old_raw = [r for r in map(json.loads, raw_path.read_text().splitlines()) if r["feature_id"] in keep]
        cand = [r for r in cand if r["feature_id"] not in keep]
        if not cand:
            print(f"all candidates already labelled in {args.out}"); return

    from vllm import LLM, SamplingParams
    llm = LLM(model=jc["model"], revision=jc["revision"], dtype="bfloat16", max_model_len=4096,
              seed=jc["seed"], max_num_seqs=args.max_num_seqs, limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(temperature=jc["temperature"], max_tokens=256, seed=jc["seed"])
    kw = dict(chat_template_kwargs={"enable_thinking": jc["enable_thinking"]})
    conv = [[{"role": "system", "content": cfg["semantic_rubric"]},
             {"role": "user", "content": f"Feature description: {r['explanation']}"}] for r in cand]
    texts = [o.outputs[0].text for o in llm.chat(conv, sp, use_tqdm=True, **kw)]
    bad = [i for i, t in enumerate(texts) if parse(t) is None]
    if bad:   # one stricter retry, then record as unparsed
        retry = [conv[i][:1] + [{"role": "user", "content": conv[i][1]["content"] + "\n\nAnswer with the JSON object only."}] for i in bad]
        for i, o in zip(bad, llm.chat(retry, sp, use_tqdm=False, **kw)):
            texts[i] = o.outputs[0].text
    new = []
    for r, t in zip(cand, texts):
        d = parse(t)
        new.append({"layer": r["layer"], "rank": r["rank"], "feature_id": r["feature_id"],
                    "explanation": r["explanation"],
                    "semantic": "" if d is None else str(d["semantic"]),
                    "category": "unparsed" if d is None else d["category"],
                    "reason": "" if d is None else d.get("reason", "")})
    print(f"classified {len(new)} new candidates")
    out = sorted(old + new, key=lambda r: (int(r["layer"]), int(r["rank"])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["layer", "rank", "feature_id", "explanation", "semantic", "category", "reason"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(out)
    raw = args.out.with_suffix(".raw.jsonl")
    with open(raw, "w") as fh:
        for r in old_raw:
            fh.write(json.dumps(r) + "\n")
        for r, t in zip(cand, texts):
            fh.write(json.dumps({"feature_id": r["feature_id"], "response": t}) + "\n")
    n_bad = sum(r["category"] == "unparsed" for r in out)
    print(f"wrote {len(out)} labels -> {args.out} ({sum(r['semantic'] == 'True' for r in out)} semantic, "
          f"{n_bad} unparsed)")
    counts = Counter((int(r["layer"]), r["category"]) for r in out)
    for L in sorted({k[0] for k in counts}):
        print(f"  layer {L}: " + ", ".join(f"{c} {n}" for (l, c), n in sorted(counts.items()) if l == L))
    if n_bad:
        raise SystemExit(f"{n_bad} descriptions could not be classified; see {raw}")


if __name__ == "__main__":
    main()
