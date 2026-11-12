#!/usr/bin/env python3
import argparse, json, re
from collections import defaultdict
from pathlib import Path

KEY = "{dir}/gemma-2-2b_res{L}_16k.key.json"


def score(picks, key_feature):
    n = len(key_feature["prompts"])
    chose = defaultdict(int)
    steered = hit = hit_all = judged = 0
    for i, kp in enumerate(key_feature["prompts"], 1):
        sel = picks.get(str(i))
        if sel is None:
            continue
        judged += 1
        unknown = [x for x in sel if str(x).strip().upper() not in kp["mapping"]]
        if unknown:
            raise SystemExit(f"prompt {i}: judge picked {unknown}, not one of {sorted(kp['mapping'])}")
        labels = {kp["mapping"][str(x).strip().upper()] for x in sel}
        for lab in labels:
            chose[lab] += 1
        any_steer = any(l.startswith("steer") for l in labels)
        steered += any_steer
        hit += any_steer and "baseline" not in labels
        doses = {v for v in kp["mapping"].values() if v.startswith("steer")}
        hit_all += doses <= labels and "baseline" not in labels
    j = max(judged, 1)
    doses = sorted({v for kp in key_feature["prompts"] for v in kp["mapping"].values() if v.startswith("steer")},
                   key=lambda s: float(s[len("steer_x"):]))
    return dict(n_prompts=n, judged=judged, hit=hit / j, hit_all=hit_all / j, steered=steered / j,
                baseline=chose["baseline"] / j, **{k: chose[k] / j for k in doses})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--picks", type=Path, required=True)
    p.add_argument("--key-dir", nargs="+", default=["data/sae_steering_samples"],
                   help="Folder(s) with the *.key.json the judged samples came from; a feature "
                        "is looked up in whichever folder's key has it.")
    p.add_argument("--rank-file", type=Path,
                   help="Optional JSON {feature_id: rank} overriding the key-file order "
                        "(the deep pass samples features out of rank order).")
    p.add_argument("--min-hit", type=float, default=0.6,
                   help="min fraction of prompts where a steered option is marked and the unsteered one is not")
    p.add_argument("--hit-rule", choices=["any", "all"], default="any",
                   help="any: a prompt is a hit if at least one steered dose shows the concept; "
                        "all: only if every dose does (unsteered never may)")
    p.add_argument("--min-judged", type=int, default=50,
                   help="Prompts that must be judged; the sweep judged all 50 per feature.")
    p.add_argument("--csv", type=Path)
    args = p.parse_args()

    import judge_steering_local as jl
    rows, keys, shas, n_out = [], {}, {}, 0
    ranks = json.loads(args.rank_file.read_text()) if args.rank_file else None
    for f in sorted(args.picks.glob("S*.json")):
        d = json.loads(f.read_text())
        if ranks is not None and d["feature"] not in ranks:
            n_out += 1
            continue
        L, idx = re.match(r"S(\d+)\.(\d+)$", d["feature"]).groups()
        if L not in keys:
            keys[L] = {}
            for kd in args.key_dir:
                kp = Path(KEY.format(dir=kd, L=L))
                if kp.exists():
                    keys[L].update(json.loads(kp.read_text()))
                    md = kp.with_name(kp.name.replace(".key.json", ".judge.md"))
                    shas.update({fe[0]: jl.samples_sha(fe) for fe in jl.parse_judge_md(md)})
        if idx not in keys[L]:
            raise SystemExit(f"{d['feature']} not found in any key under {args.key_dir}")
        if d.get("samples_sha") != shas.get(d["feature"]):
            raise SystemExit(f"{d['feature']}: picks were judged on different samples or rubric "
                             f"than the current ones; re-run the judge")
        k = keys[L][idx]
        r = score(d["picks"], k)
        rank = ranks[d["feature"]] if ranks is not None else list(keys[L]).index(idx) + 1
        r.update(layer=int(L), feature=d["feature"], rank=rank,
                 concept=k["concept"])
        r["passes"] = r["judged"] >= args.min_judged and r["hit" if args.hit_rule == "any" else "hit_all"] >= args.min_hit
        rows.append(r)
    rows.sort(key=lambda r: (r["layer"], r["rank"]))
    if n_out:
        print(f"skipped {n_out} picks of features outside the current candidate pool")
    doses = sorted({k for r in rows for k in r if k.startswith("steer_x")}, key=lambda s: float(s[7:]))
    print(f"rule: hit ({args.hit_rule}) >= {args.min_hit:.0%}, "
          f"judged >= {args.min_judged}")
    for r in rows:
        print(f"L{r['layer']:<2} #{r['rank']:<2} {r['feature']:<10} judged={r['judged']:>2}  hit={r['hit']:.2f}  "
              f"hit_all={r['hit_all']:.2f}  steered={r['steered']:.2f} ("
              + " ".join(f"{d[6:]}={r.get(d, 0):.2f}" for d in doses) + ")  "
              + f"baseline={r['baseline']:.2f}  {'PASS' if r['passes'] else '    '}  {r['concept'][:70]}")
    if args.csv:
        import csv
        cols = ["layer", "rank", "feature", "judged", "hit", "hit_all", "steered", *doses, "baseline",
                "passes", "concept"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
