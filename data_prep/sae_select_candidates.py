#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import sys

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "scripts/sae/selection/config.json")
    p.add_argument("--out", type=Path, default=ROOT / "results/sae_feature_selection/candidates.csv")
    p.add_argument("--annotated-out", type=Path, default=ROOT / "data/sae_selection/annotated",
                   help="Per-layer copy of the stats with the candidates' descriptions filled in; "
                        "the steering step reads these.")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data/neuronpedia_cache")
    p.add_argument("--depth-file", type=Path, default=None)
    p.add_argument("--depth-all", type=int, default=None, help="use this depth for every layer (prefetch)")
    p.add_argument("--offline", action="store_true",
                   help="read descriptions only from --cache-dir; error if one is not cached. The "
                        "SLURM job uses this so a network failure cannot silently drop a candidate.")
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    c = cfg["candidates"]
    lo, hi = c["density_band"]
    depth = json.loads(args.depth_file.read_text()) if args.depth_file and args.depth_file.exists() else {}
    if args.depth_all:
        depth = {str(L): args.depth_all for L in cfg["layers"]}
    from src.introspector.signals.neuronpedia import fetch_feature, source_id_for
    from data_prep.annotate_sae_features_neuronpedia import _row_from_info
    args.annotated_out.mkdir(parents=True, exist_ok=True)
    rows = []
    for L in cfg["layers"]:
        full = pd.read_parquet(ROOT / c["stats"].format(L=L))
        pos = [pd.read_parquet(ROOT / c["pos_stats"].format(L=L, P=P)).set_index("feature").pos_density
               for P in c["rank_positions"]]
        mean = pd.concat(pos, axis=1).mean(axis=1)
        q = pd.read_parquet(ROOT / c["act_quantiles"].format(L=L)).set_index("feature")
        for col in [k for k in q.columns if k.startswith("act_q")]:
            full[col] = full.feature.map(q[col])
        full["pos_density_mean"] = full.feature.map(mean)
        full[c["rank_by"]] = (full.pos_density_mean * c["cache_size"]).round().astype("int64")
        d = full[(full.density >= lo) & (full.density <= hi)]
        d = d.sort_values(["pos_density_mean", "feature"], ascending=[False, True], kind="mergesort")
        k = min(int(depth.get(str(L), c["block"])), c["max_rank"])
        d = d.head(k).copy()
        src = source_id_for(L, site="res", width=cfg["steering"]["width"])
        for i in d.index[d.explanation.isna()]:
            f = int(d.at[i, "feature"])
            info = fetch_feature("gemma-2-2b", src, f, cache_dir=args.cache_dir, offline=args.offline)
            if info is None and not (args.cache_dir / src / f"{f}.404").exists():
                raise SystemExit(f"Neuronpedia request for {src}/{f} failed (not a 404); rerun when "
                                 f"the network is back -- a skipped candidate would change the picks")
            if info is not None and info.has_explanation:
                for col, v in _row_from_info(info).items():
                    full.at[i, col] = v
                d.at[i, "explanation"] = info.explanation
        print(f"layer {L}: {int(d.explanation.isna().sum())} of the top {len(d)} have no explanation (not eligible)")
        full.to_parquet(args.annotated_out / f"gemma2-2b_res{L}_pos-1_16k.annotated.parquet")
        d["layer"], d["rank"] = L, range(1, len(d) + 1)
        d["feature_id"] = [f"S{L}.{int(f)}" for f in d.feature]
        rows.append(d[["layer", "rank", "feature_id", "feature", "density", "pos_density_mean",
                       c["rank_by"], "explanation"]])
    out = pd.concat(rows, ignore_index=True)
    out["explanation"] = out.explanation.str.strip()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} candidates ({', '.join(f'L{L}' for L in cfg['layers'])}) -> {args.out}")


if __name__ == "__main__":
    main()
