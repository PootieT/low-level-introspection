#!/usr/bin/env python3
"""
Does an SAE-feature introspector generalize to features it was never trained on?

The SAE analogue of score_neuron_generalization.py. A run trains on the first N
features of a fixed pool (build_sae_featgen_pool.py) and is scored here on EVERY pool
feature, each labelled trained or held-out afterwards, so one pass per run serves the
whole N sweep and a feature can be followed across it. The readout is binary
detection on the GENERATED token (active / inactive / neither), as in
train_sae_detector_joint.py, scored by balanced accuracy.

    SEEN      the trained features: how does fidelity hold up as N grows?
    UNSEEN    pool features absent from training. Reachable only if the readout
              learned to address features by something shared -- the point of naming
              them by their natural-language label (prompt_sae_label) rather than an
              index, which gives an unseen feature nothing to go on.

WHICH NUMBER TO READ
  interventional BA (unseen)  THE headline. Each text is run twice, the feature steered
                              to its median active value and to 0, and the label is the
                              forced state, so context cannot predict it.
  endogenous BA (unseen)      weak evidence on its own: a label like "references to dogs"
                              lets the model guess from the TEXT whether the feature
                              fires, with no introspection at all.
  distractor matrix           the control for interventional BA. At eval most texts are
                              naturally inactive, so steering "off" changes nothing and
                              "on" is the only perturbation: a readout that answers
                              "was SOMETHING injected here?" scores well on seen AND
                              unseen features without reading the label. The matrix
                              steers feature j while the prompt names feature i. A
                              readout addressed to i says "active" on the diagonal and
                              stays at its "off" rate off it.

SAMPLES (all chosen by subsample_u, so they are identical across runs on one pool and
the comparison across N and arms is paired)
  endogenous      up to --fg-endo-max-pos active rows + --fg-endo-neg inactive rows
                  per feature. BA does not depend on the mix; precision does, so it is
                  reported at the feature's NATURAL base rate (from TPR/FPR), comparable
                  to the single-feature runs' precision on their full test split.
  interventional  the --fg-interv-texts rows with the smallest subsample_u among those
                  below the cache's negative_keep -- a uniform sample of texts, i.e. the
                  natural mix, like the single-feature eval's full test split.
  distractor      the first --fg-distractor-texts of those, for --fg-distractor-features
                  seen and the same number of held-out features (pool order).

'On' magnitudes are each feature's median active value in the POOL's train split,
fitted the same way for trained and held-out features so the arms cannot differ by how
their injection was chosen.

Called from train_sae_detector_joint.py (multi-feature runs) at the end of training,
or on a checkpoint with --feature-gen-only.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from src.introspector.config import ExperimentConfig
from src.introspector.prompts import feature_key
from src.introspector.sae_featgen import ensure_subsample_u
from src.introspector.training.model import SAEInterventionHook


def _load_split(cfg: ExperimentConfig, split: str, cfg_path=None):
    path = cfg.dataset_cache_dir / split
    if not path.exists():
        raise SystemExit(f"feature-gen cache split missing: {path}. Build it first with\n"
                         f"    python cache_signals.py --config {cfg_path or '<pool config>'}")
    return load_from_disk(str(path))


def pool_on_values(pool_cfg: ExperimentConfig, fallback) -> tuple[dict, dict]:
    """feature index -> median active value, from the pool's train (+dev) split, which
    is disjoint from the evaluated test rows. Falls back to the test split's active rows
    for a feature with none there -- only the steering magnitude comes from it, never a
    label, but it is reported."""
    frames = []
    for split in ("train", "dev"):
        path = pool_cfg.dataset_cache_dir / split
        if path.exists():
            ds = load_from_disk(str(path))
            if len(ds):
                frames.append(pd.DataFrame({"f": ds["signal_index"],
                                            "a": np.asarray(ds["activation"], float)}))
    df = pd.concat(frames) if frames else pd.DataFrame({"f": [], "a": []})
    df = df[df["a"] > 0]
    on = df.groupby("f")["a"].median().to_dict()
    n_act = df.groupby("f").size().to_dict()
    fb = pd.DataFrame({"f": fallback["signal_index"], "a": np.asarray(fallback["activation"], float)})
    fb = fb[fb["a"] > 0]
    missing = sorted(set(fb["f"].unique()) - set(on))
    if missing:
        print(f"[featgen] WARNING: {len(missing)} feature(s) have no active rows in the pool "
              f"train split; using the test split's median active value for them")
        for f, v in fb[fb["f"].isin(missing)].groupby("f")["a"].median().items():
            on[f] = v
    return {int(k): float(v) for k, v in on.items()}, {int(k): int(v) for k, v in n_act.items()}


@torch.no_grad()
def readout(model, ds_pred, collator, batch_size, module=None, bundle=None,
            positions=None, feat_idx=None, values=None):
    """Token the model GENERATES at the readout (argmax over the vocabulary) per row.
    With `module`, row r is steered: feature feat_idx[r] clamped to values[r] at its
    token (difference-of-decode, SAEInterventionHook), each row its own feature."""
    model.eval()
    preds, ptr = [], 0
    for batch in DataLoader(ds_pred, batch_size=batch_size, collate_fn=collator, shuffle=False):
        ii = batch["input_ids"].to(model.device)
        am = batch["attention_mask"].to(model.device)
        n = ii.shape[0]
        handle = None
        if module is not None:
            dev = model.device
            abs_pos = torch.as_tensor(positions[ptr:ptr + n], device=dev) + (am == 0).sum(1)
            hook = SAEInterventionHook(
                bundle, abs_pos, torch.as_tensor(feat_idx[ptr:ptr + n], device=dev),
                torch.as_tensor(values[ptr:ptr + n], dtype=torch.float32, device=dev))
            handle = module.register_forward_hook(hook)
        try:
            logits = model(input_ids=ii, attention_mask=am).logits[:, -1, :]
        finally:
            if handle is not None:
                handle.remove()
        preds.append(logits.argmax(dim=-1).cpu())
        ptr += n
    return torch.cat(preds).numpy() if preds else np.zeros(0, dtype=int)


def natural_precision(m: dict, base_rate: float) -> float:
    """Precision at the feature's natural base rate, from TPR and FPR measured on a
    reweighted (negative-subsampled) set: pi*TPR / (pi*TPR + (1-pi)*FPR)."""
    n_neg = sum(m.get(f"ina_pred_{k}", 0) for k in ("active", "inactive", "neither"))
    if "tpr" not in m or n_neg == 0 or not np.isfinite(base_rate):
        return float("nan")
    fpr = m["ina_pred_active"] / n_neg
    den = base_rate * m["tpr"] + (1 - base_rate) * fpr
    return float(base_rate * m["tpr"] / den) if den > 0 else float("nan")


def evaluate_feature_generalization(model, config, tokenizer, specs, bundle, module, aid, iid,
                                    out_dir, pool_config=None, batch_size=128, n_interv=500,
                                    endo_max_pos=500, endo_neg=1000, n_distractor=20,
                                    distractor_texts=50):
    from train_sae_detector_joint import (
        EvalCollator, assert_prompts_fit, drop_retokenization_mismatches,
        generation_metrics, make_preprocess)

    out_dir = Path(out_dir)
    pool_cfg = (ExperimentConfig.model_validate_json(Path(pool_config).read_text())
                if pool_config else config)
    raw = _load_split(pool_cfg, "test", pool_config)
    raw = drop_retokenization_mismatches(raw, tokenizer, "featgen-pool-test")
    raw = ensure_subsample_u(raw, pool_cfg.seed)
    layer = specs[0].layer
    q = pool_cfg.test_dataset.negative_keep
    q = 1.0 if q is None else float(q)

    # pool order = the pool config's signal order; the trained set = this run's features
    order = [s.index for s in pool_cfg.test_signals if s.type == "sae" and s.layer == layer]
    sig = np.asarray(raw["signal_index"])
    present = set(np.unique(sig).tolist())
    order = [f for f in order if f in present] + sorted(present - set(order))
    trained = [s.index for s in specs]
    trained_set = set(trained)
    if not trained_set <= present:
        raise SystemExit(f"pool cache lacks {len(trained_set - present)} trained feature(s), e.g. "
                         f"{sorted(trained_set - present)[:5]}; it was built from another pool")
    seen = [f for f in order if f in trained_set]
    unseen = [f for f in order if f not in trained_set]
    rank = {f: i for i, f in enumerate(order)}
    print(f"[featgen] {len(order)} pool features: {len(seen)} trained, {len(unseen)} held out")
    tc = config.training_config
    labels = tc.feature_labels or {}
    if tc.prompt_type == "prompt_sae_label":
        unlabelled = [f for f in order if feature_key(layer, f) not in labels]
        if unlabelled:
            raise SystemExit(f"{len(unlabelled)} pool features have no label in the run "
                             f"config, e.g. {unlabelled[:5]}")

    on_value, n_act_train = pool_on_values(pool_cfg, raw)
    act = np.asarray(raw["activation"], float)
    u = np.asarray(raw["subsample_u"], float)
    positions_all = np.asarray(raw["token_position"])

    # ---- per-feature row choices (by subsample_u: identical in every run on this pool)
    endo_rows, interv_rows, base_rate = {}, {}, {}
    for f in order:
        rows = np.flatnonzero(sig == f)
        pos, neg = rows[act[rows] > 0], rows[act[rows] <= 0]
        pos, neg = pos[np.argsort(u[pos])], neg[np.argsort(u[neg])]
        endo_rows[f] = np.concatenate([pos[:endo_max_pos], neg[:endo_neg]])
        nat = rows[u[rows] < q]
        interv_rows[f] = nat[np.argsort(u[nat])][:n_interv]
        # cached inactive rows are a q-sample, actives are all there
        base_rate[f] = len(pos) / (len(pos) + len(neg) / q) if len(rows) else float("nan")
    short = [f for f in order if len(interv_rows[f]) < n_interv]
    if short:
        print(f"[featgen] WARNING: {len(short)} feature(s) have < {n_interv} natural rows "
              f"(raise the pool cache's test size or negative_keep)")

    needed = np.unique(np.concatenate([*endo_rows.values(), *interv_rows.values()]))
    where = {int(r): i for i, r in enumerate(needed)}
    sub = raw.select(needed.tolist())
    random.seed(config.seed + 1)   # same templates in every run on this pool
    ds_pred = sub.map(make_preprocess(config, tokenizer, for_train=False),
                      remove_columns=sub.column_names)
    assert_prompts_fit(ds_pred, tc.max_seq_length, "featgen")
    collator = EvalCollator(tokenizer)

    def run_steered(rows, feats, vals):
        rows = np.asarray(rows, int)
        return readout(model, ds_pred.select([where[int(r)] for r in rows]), collator,
                       batch_size, module=module, bundle=bundle, positions=positions_all[rows],
                       feat_idx=np.asarray(feats, int), values=np.asarray(vals, float))

    # ---- endogenous: no steering
    e_rows = np.concatenate([endo_rows[f] for f in order])
    print(f"[featgen] endogenous: {len(e_rows)} forwards")
    e_pred = readout(model, ds_pred.select([where[int(r)] for r in e_rows]), collator, batch_size)

    # ---- interventional: every natural row steered on (median active) and off (0)
    i_rows = np.concatenate([interv_rows[f] for f in order])
    i_feat = sig[i_rows]
    i_on = np.array([on_value[int(f)] for f in i_feat])
    print(f"[featgen] interventional: {2 * len(i_rows)} forwards")
    p_on = run_steered(i_rows, i_feat, i_on)
    p_off = run_steered(i_rows, i_feat, np.zeros(len(i_rows)))

    per, ptr_e, ptr_i = [], 0, 0
    for f in order:
        key = feature_key(layer, f)
        common = {"signal": key, "feature": f, "pool_rank": rank[f], "is_trained": f in trained_set,
                  "label": labels.get(key), "on_value": on_value.get(f, float("nan")),
                  "n_active_pool_train": n_act_train.get(f, 0),
                  "natural_base_rate": base_rate[f]}
        k = len(endo_rows[f])
        y = (act[endo_rows[f]] > 0).astype(int)
        m = generation_metrics(e_pred[ptr_e:ptr_e + k], y, aid, iid)
        m["precision_eval_set"] = m.get("precision")
        m["precision"] = natural_precision(m, base_rate[f])
        per.append({**common, "eval_type": "endogenous", **m})
        ptr_e += k
        k = len(interv_rows[f])
        preds = np.concatenate([p_on[ptr_i:ptr_i + k], p_off[ptr_i:ptr_i + k]])
        m = generation_metrics(preds, np.r_[np.ones(k, int), np.zeros(k, int)], aid, iid)
        per.append({**common, "eval_type": "interventional", **m,
                    "off_pred_active_rate": float(np.mean(p_off[ptr_i:ptr_i + k] == aid)) if k else float("nan")})
        ptr_i += k
    per = pd.DataFrame(per)
    per["n_trained_in_run"] = len(seen)
    per.to_csv(out_dir / "feature_generalization_by_feature.csv", index=False)

    rows = []
    for (kind, is_tr), g in per.groupby(["eval_type", "is_trained"]):
        ba = g["balanced_accuracy"].dropna()
        rows.append({"eval_type": kind, "arm": "seen" if is_tr else "unseen",
                     "n_features": len(g), "n_defined": len(ba),
                     "ba_macro": float(ba.mean()) if len(ba) else float("nan"),
                     "ba_std": float(ba.std(ddof=0)) if len(ba) else float("nan"),
                     "ba_sem": float(ba.std(ddof=1) / np.sqrt(len(ba))) if len(ba) > 1 else float("nan"),
                     "precision_macro": float(g["precision"].dropna().mean()) if g["precision"].notna().any() else float("nan"),
                     "tpr_macro": float(g["tpr"].dropna().mean()) if "tpr" in g and g["tpr"].notna().any() else float("nan"),
                     "tnr_macro": float(g["tnr"].dropna().mean()) if "tnr" in g and g["tnr"].notna().any() else float("nan")})
    summary = pd.DataFrame(rows).sort_values(["eval_type", "arm"]).reset_index(drop=True)
    summary.insert(0, "n_trained_in_run", len(seen))
    summary.to_csv(out_dir / "feature_generalization.csv", index=False)
    print("\n[featgen] summary (balanced accuracy macro-averaged over features, per arm):")
    print(summary.to_string(index=False))

    # ---- distractor: steer j, prompt names i
    if n_distractor > 0:
        D = seen[:n_distractor] + unseen[:n_distractor]
        r_all, f_all, v_all, meta = [], [], [], []
        for i in D:
            texts = interv_rows[i][:distractor_texts]
            for j in D:
                stims = [("diagonal", on_value[i])] if j == i else \
                        [("own", on_value[j]), ("target", on_value[i])]
                for name, v in stims:
                    r_all.append(texts)
                    f_all.append(np.full(len(texts), j))
                    v_all.append(np.full(len(texts), v))
                    meta.append((i, j, name, len(texts)))
        print(f"[featgen] distractor: {len(D)}x{len(D)} matrix, "
              f"{sum(m[3] for m in meta)} forwards")
        d_pred = run_steered(np.concatenate(r_all), np.concatenate(f_all), np.concatenate(v_all))
        drows, ptr = [], 0
        off_rate = per[per["eval_type"] == "interventional"].set_index("feature")["off_pred_active_rate"]
        for i, j, name, k in meta:
            p = d_pred[ptr:ptr + k]
            ptr += k
            drows.append({"ask": feature_key(layer, i), "steer": feature_key(layer, j),
                          "stimulus": name, "n": k,
                          "ask_trained": i in trained_set, "steer_trained": j in trained_set,
                          "p_active": float(np.mean(p == aid)), "p_inactive": float(np.mean(p == iid)),
                          "p_neither": float(np.mean((p != aid) & (p != iid))),
                          "ask_off_p_active": float(off_rate.get(i, float("nan")))})
        dist = pd.DataFrame(drows)
        dist["n_trained_in_run"] = len(seen)
        dist.to_csv(out_dir / "feature_generalization_distractor.csv", index=False)
        dsum = (dist.assign(ask_arm=np.where(dist["ask_trained"], "seen", "unseen"),
                            steer_arm=np.where(dist["steer_trained"], "seen", "unseen"))
                .groupby(["ask_arm", "steer_arm", "stimulus"])
                .agg(p_active=("p_active", "mean"), ask_off_p_active=("ask_off_p_active", "mean"),
                     n_cells=("p_active", "size")).reset_index())
        print("\n[featgen] distractor (mean P(active); a readout addressed to the named feature is "
              "high on 'diagonal' and near ask_off_p_active elsewhere):")
        print(dsum.to_string(index=False))
        dsum.to_csv(out_dir / "feature_generalization_distractor_summary.csv", index=False)

    (out_dir / "feature_generalization.json").write_text(json.dumps({
        "n_trained": len(seen), "n_heldout": len(unseen),
        "trained": [feature_key(layer, f) for f in seen],
        "heldout": [feature_key(layer, f) for f in unseen],
        "pool_config": str(pool_config) if pool_config else None,
        "pool_test_negative_keep": q,
        "sizes": {"interv_texts": n_interv, "endo_max_pos": endo_max_pos, "endo_neg": endo_neg,
                  "distractor_features": n_distractor, "distractor_texts": distractor_texts},
        "summary": summary.to_dict(orient="records")}, indent=2, default=str))
    return summary.to_dict(orient="records")
