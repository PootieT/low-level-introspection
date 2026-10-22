#!/usr/bin/env python3
"""Does an injected attention weight (code P) reach the readout token's last layer?

Stage 3 (attn_weight_joint_weight_sweep) found interventional corr ~0 at every W, and the
fresh ceiling probe ~0 too. Two explanations predict that equally well:

  (a) TRANSMISSION  the injection barely moves the residual stream downstream of the
                    port, or moves it at dst but the readout token never picks it up.
  (b) OPTIMIZATION  the change is there and linearly readable, but training never found
                    it (L_interv sat at ~1.0, i.e. the head predicted a constant).

This script separates them WITHOUT training anything. It runs the exact forward the
trainer runs (IntrospectionRegressionModel.forward -> AttentionWeightPort), once clean and
once with an injection, and measures the difference layer by layer at two positions:

    dst      the introspected token (where the port writes)
    readout  the last prompt token (where the regression head reads, hidden_states[-1])

Part 1  DOSE     fixed targets t; per layer/position, ||dh|| relative to ||h|| and to the
                 natural across-example spread of h; plus the head's prediction change.
                 Also asserts the injection is LOCAL: dh == 0 before the port's layer and
                 at every position before dst -- if not, the port wrote the wrong place.
Part 2  DECODE   t drawn per example from the TRAINING sampler (and gaussian@1); ridge
                 probes predicting logit(t) from dh (the injected component alone) and
                 from h_interv (what a real readout sees, content variance included),
                 cross-validated, per layer, at both positions. No PCA: the ceiling
                 probe's PCA-256 can drop a low-variance injection direction, which is
                 one of the things this is checking.

Models: `base` (adapter disabled on the same checkpoint -- same added-token embeddings)
and the checkpoint itself. Run with several --config to compare W=0 and W=1.

    python mechanistic_analysis/probe_attention_weight_transmission.py \
        --config experiment_configs/attn_weight_joint_weight_sweep/joint_P2.31@-1_POS-1_gateoff_scale4_kl0.1_w1.0_60K.json
"""
from __future__ import annotations

import sys
import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from mechanistic_analysis.attribute_attention_weight_readout import latest_checkpoint, load_everything
from src.introspector.config import ExperimentConfig
from src.introspector.training.model import (
    PROB_SIGNAL_EPS, make_bounded_prob_intervention_fn,
)

DOSE_TARGETS = (1e-4, 0.01, 0.1, 0.5, 0.9, 1 - 1e-4)
RIDGE_ALPHAS = np.logspace(-2, 6, 17)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--n-dose", type=int, default=256)
    ap.add_argument("--n-decode", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--models", default="base,tuned",
                    help="comma list of base (adapter disabled) and tuned")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def _fixed_fn(t):
    """intervention_fn that ignores the cell stats and injects exactly t."""
    def f(layer, idx, **stats):
        return t
    return f


@torch.no_grad()
def forward_pair(model, batch):
    """Clean and injected forwards of one batch, through the trainer's own forward.

    Returns (hs_clean, hs_int, dst_abs, t, pred_clean, pred_int):
      hs_*   [n_layers+1, B, 2, H] fp32 -- hidden_states at (dst, readout)
      dst_abs [B], t [B], pred_* [B]
    """
    am = batch["attention_mask"]                                     # [B, L]
    # The trainer's forward shifts token_position by the left-pad count and asserts left
    # padding; this indexes hidden states with the same shift, so assert it here too.
    assert bool((am[:, -1] == 1).all()), "expected LEFT padding"
    B, L = am.shape
    dst_abs = batch["token_position"] + (am == 0).sum(dim=1)        # [B]
    assert bool(((dst_abs >= 0) & (dst_abs < L - 1)).all()), (
        f"dst outside the text region: {dst_abs.tolist()} with L={L}")
    common = dict(input_ids=batch["input_ids"], attention_mask=am, output_hidden_states=True)
    interv = dict(token_position=batch["token_position"], signal_layer=batch["signal_layer"],
                  signal_index=batch["signal_index"],
                  src_token_position=batch["src_token_position"],
                  signal_src_offset=batch["signal_src_offset"])

    out_c = model(**common, do_intervention=False)
    out_i = model(**common, **interv, do_intervention=True)
    t = out_i.intervention_values.float()                            # [B]
    assert t.shape == (B,), f"intervention_values {tuple(t.shape)} != ({B},)"

    bidx = torch.arange(B, device=am.device)
    def gather(hs):                                                  # tuple of [B, L, H]
        st = torch.stack(hs, dim=0).float()                          # [n+1, B, L, H]
        at_dst = st[:, bidx, dst_abs]                                # [n+1, B, H]
        at_ro = st[:, :, -1]                                         # [n+1, B, H]
        return torch.stack([at_dst, at_ro], dim=2)                   # [n+1, B, 2, H]
    hs_c, hs_i = gather(out_c.hidden_states), gather(out_i.hidden_states)

    # LOCALITY. The injection is at layer `lay`'s attention, so hidden_states[0..lay]
    # (inputs to layers <= lay) must be identical, and so must every position before
    # dst at every layer (causal). A nonzero diff here means the port wrote somewhere
    # other than (lay, dst) -- the position bug class of scripts_old_with_off_by_one_bug/.
    lay = int(batch["signal_layer"][0])
    assert bool((batch["signal_layer"] == lay).all()), "mixed layers in one batch"
    full_c = torch.stack(out_c.hidden_states, 0).float()             # [n+1, B, L, H]
    full_i = torch.stack(out_i.hidden_states, 0).float()
    pre = (full_i[: lay + 1] - full_c[: lay + 1]).abs().max().item()
    pos_idx = torch.arange(L, device=am.device)
    # real text tokens before dst only: a left-pad row's query sees no valid key, and
    # transformers' mask then un-masks that row, so pad positions can legitimately move.
    # Nothing reads them (they are masked as keys), so they are excluded, not tolerated.
    before_dst = (pos_idx[None, :] < dst_abs[:, None]) & (am == 1)   # [B, L]
    diff_pos = (full_i - full_c).abs().amax(dim=(0, 3))              # [B, L]
    early = (diff_pos * before_dst).max().item()
    pad_moved = (diff_pos * (am == 0)).max().item()
    # Tolerance, not exact zero, only so a nondeterministic GEMM cannot trip it; a real
    # misplaced write moves h by O(1) relative, many orders above this.
    tol = 1e-4 * full_c.abs().max().item()
    assert pre <= tol and early <= tol, (
        f"injection leaked: max|dh| before layer {lay} = {pre}, at positions < dst = "
        f"{early} (tol {tol}); pad positions moved by {pad_moved}")

    head = model.regression_head
    def pred(hs):
        h = hs[-1][:, -1, :].to(device=head.weight.device, dtype=head.weight.dtype)
        return head(h).squeeze(-1).float()                           # [B]
    return (hs_c.cpu(), hs_i.cpu(), dst_abs.cpu(), t.cpu(),
            pred(out_c.hidden_states).cpu(), pred(out_i.hidden_states).cpu())


def run_dose(model, dataset, collator, n, bs):
    """Fixed-t sweep. One row per (t, layer, position)."""
    sub = dataset.select(range(min(n, len(dataset))))
    rows = []
    for t in DOSE_TARGETS:
        model.intervention_fn = _fixed_fn(t)
        acc = {k: [] for k in ("hc", "dh", "p_c", "p_i", "p_nat")}
        for batch in DataLoader(sub, batch_size=bs, collate_fn=collator, shuffle=False):
            batch = batch.to(model.device)
            hs_c, hs_i, _, _, p_c, p_i = forward_pair(model, batch)
            acc["hc"].append(hs_c); acc["dh"].append(hs_i - hs_c)
            acc["p_c"].append(p_c); acc["p_i"].append(p_i)
            acc["p_nat"].append(batch["activation"].float().cpu())
        hc = torch.cat(acc["hc"], 1)                                 # [n+1, N, 2, H]
        dh = torch.cat(acc["dh"], 1)                                 # [n+1, N, 2, H]
        p_nat = torch.cat(acc["p_nat"])                              # [N]
        p_c, p_i = torch.cat(acc["p_c"]), torch.cat(acc["p_i"])      # [N]
        # natural spread: RMS distance of h from its across-example mean
        nat = (hc - hc.mean(1, keepdim=True)).norm(dim=-1).pow(2).mean(1).sqrt()  # [n+1, 2]
        dn = dh.norm(dim=-1)                                         # [n+1, N, 2]
        hn = hc.norm(dim=-1)                                         # [n+1, N, 2]
        for li in range(dh.shape[0]):
            for pi, pos in enumerate(("dst", "readout")):
                rows.append({
                    "t": t, "hs_index": li, "position": pos,
                    "dh_over_h": float((dn[li, :, pi] / hn[li, :, pi]).mean()),
                    "dh_over_natural": float(dn[li, :, pi].mean() / nat[li, pi]),
                    "mean_natural_p": float(p_nat.mean()),
                    "pred_clean": float(p_c.mean()), "pred_interv": float(p_i.mean()),
                    "pred_shift_abs": float((p_i - p_c).abs().mean()),
                })
    return pd.DataFrame(rows)


def _cv_corr(X, y, seed):
    """5-fold CV pearson of a standardized RidgeCV. X [N, H], y [N]."""
    pred = np.zeros_like(y)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return float(np.corrcoef(pred, y)[0, 1])


def run_decode(model, dataset, collator, n, bs, seed, dist, scale):
    """Random t per example from `dist`@`scale`; ridge decodability per layer."""
    model.intervention_fn = make_bounded_prob_intervention_fn(model)
    model.intervention_distribution, model.intervention_scale = dist, scale
    model.seed_intervention_rng(seed)
    sub = dataset.select(range(min(n, len(dataset))))
    hc_all, hi_all, t_all, pc_all, pi_all = [], [], [], [], []
    for batch in DataLoader(sub, batch_size=bs, collate_fn=collator, shuffle=False):
        batch = batch.to(model.device)
        hs_c, hs_i, _, t, p_c, p_i = forward_pair(model, batch)
        hc_all.append(hs_c); hi_all.append(hs_i); t_all.append(t)
        pc_all.append(p_c); pi_all.append(p_i)
    hc, hi = torch.cat(hc_all, 1).numpy(), torch.cat(hi_all, 1).numpy()   # [n+1, N, 2, H]
    t = torch.cat(t_all).numpy()                                          # [N]
    p_i = torch.cat(pi_all).numpy()
    eps = PROB_SIGNAL_EPS
    y = np.log(np.clip(t, eps, 1 - eps) / (1 - np.clip(t, eps, 1 - eps)))  # logit(t)
    rows = []
    for li in range(hc.shape[0]):
        for pi, pos in enumerate(("dst", "readout")):
            dh = hi[li, :, pi] - hc[li, :, pi]                            # [N, H]
            rows.append({
                "dist": dist, "scale": scale, "hs_index": li, "position": pos,
                "decode_from_dh": _cv_corr(dh, y, seed) if np.abs(dh).max() > 0 else float("nan"),
                "decode_from_h_interv": _cv_corr(hi[li, :, pi], y, seed),
            })
    head_corr = float(np.corrcoef(p_i, t)[0, 1]) if p_i.std() > 1e-12 else float("nan")
    summary = {"dist": dist, "scale": scale, "n": int(len(t)),
               "t_mean": float(t.mean()), "t_std": float(t.std()),
               "frac_t_below_1e-3": float((t < 1e-3).mean()),
               "frac_t_above_0.999": float((t > 0.999).mean()),
               "trained_head_corr": head_corr}
    return pd.DataFrame(rows), summary


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    config = ExperimentConfig.model_validate_json(args.config.read_text())
    ckpt = latest_checkpoint(config, args.checkpoint)
    out = args.out or (Path(config.output_dir) / config.experiment_name / "transmission")
    out.mkdir(parents=True, exist_ok=True)

    # pin_template=None keeps the per-example random template draw, as in training
    model, tokenizer, test_ds, collator, gate_input = load_everything(config, ckpt, None)
    if gate_input:
        raise SystemExit("gate-on run: the adapter never touches the text, so 'tuned' and "
                         "'base' transmission are the same by construction; use a gate-off run")
    train_scale = config.training_config.intervention_extreme_scale
    train_dist = config.training_config.intervention_signal_distribution
    print(f"[tx] {config.experiment_name}  ckpt={ckpt}  n_test={len(test_ds)}  "
          f"train sampler={train_dist}@{train_scale}")

    dose_all, dec_all, summaries = [], [], []
    for which in args.models.split(","):
        if which not in ("base", "tuned"):
            raise ValueError(f"--models entry {which!r} not in base,tuned")
        ctx = model.disable_adapter() if which == "base" else nullcontext()
        with ctx:
            print(f"\n[tx] === {which} ===")
            d = run_dose(model, test_ds, collator, args.n_dose, args.batch_size)
            d.insert(0, "model", which); dose_all.append(d)
            for dist, scale in ((train_dist, train_scale), ("gaussian", 1.0)):
                r, s = run_decode(model, test_ds, collator, args.n_decode, args.batch_size,
                                  args.seed, dist, scale)
                r.insert(0, "model", which); dec_all.append(r)
                s["model"] = which; summaries.append(s)
                print(f"[tx] {which} {dist}@{scale}: {json.dumps(s)}")

    dose = pd.concat(dose_all); dec = pd.concat(dec_all)
    dose.to_csv(out / "dose.csv", index=False)
    dec.to_csv(out / "decode.csv", index=False)
    (out / "summary.json").write_text(json.dumps(
        {"config": str(args.config), "checkpoint": str(ckpt), "samplers": summaries}, indent=2))

    pd.set_option("display.width", 200)
    print("\n[tx] DOSE at readout, final layer (pred shift = |head(interv) - head(clean)|):")
    last = dose["hs_index"].max()
    print(dose[(dose.position == "readout") & (dose.hs_index == last)]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n[tx] DOSE ||dh||/natural spread by layer (t = 0.9):")
    print(dose[dose.t == 0.9].pivot_table(index="hs_index", columns=["model", "position"],
                                          values="dh_over_natural")
          .to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n[tx] DECODE logit(t), 5-fold ridge corr:")
    print(dec.pivot_table(index="hs_index", columns=["model", "dist", "position"],
                          values=["decode_from_dh", "decode_from_h_interv"])
          .to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\n[tx] wrote {out}/dose.csv, decode.csv, summary.json")


if __name__ == "__main__":
    main()
