#!/usr/bin/env python3
"""
Which neuron, at which token, does the readout actually depend on -- with nothing
injected?

Every causal measurement in this project perturbs the model. This one does not: it
runs the ENDOGENOUS forward the readout was trained on, takes one backward pass from
the prediction, and scores every MLP neuron at every token position by

    attribution(layer, position, neuron) = (d prediction / d a) * a

evaluated at `a`, the neuron's own live activation (the `act_fn` output -- the same
tensor the intervention writes to, so attribution and injection address exactly the
same quantity). The question is whether (L, i, t) -- the locale the prompt names --
comes out on top of all 16 x seq_len x 8192 of them.

WHY ENDOGENOUS, given that the interventional regime is the one everything else uses.
Two reasons, and they are the point of the script rather than a caveat:
  * It is the regime the model is DEPLOYED in. Interventional specificity says the
    readout follows an injected value; it does not say the readout consults the neuron
    when nobody is injecting anything.
  * It is the regime where a context bypass can exist at all. Under injection the
    value is independent of context, so a bypass scores zero by construction and is
    invisible. Endogenously, a readout that predicts the activation from surrounding
    text instead of reading it would show its hand HERE -- as attribution spread over
    other neurons, other positions, or the embedding path, with little on (L, i, t).

WHAT THIS CANNOT TELL YOU. grad x act is a LOCAL linearization. A neuron that the
readout depends on through a saturated or strongly nonlinear path can carry a small
gradient while being causally essential, so a low attribution at the target is not
evidence of no dependence -- score_injection_specificity.py is what settles causality.
What this adds is the other direction: WHICH OTHER units the readout leans on, which
is not something a perturbation of the target can show.

The three summaries are written per run:
  attribution_top.csv        the highest-attribution units, with the target's rank
                             among them. TWO ranks are reported, and the second is the
                             one to quote: the global ranking is dominated by neurons
                             at the READOUT token, whose MLP writes straight into the
                             residual the head reads, in every run regardless of what
                             is being introspected. `rank_text` restricts to units at
                             INPUT-TEXT positions, which is where the introspected
                             token lives and where localizing it means something
  attribution_by_position.csv the target neuron's attribution as a function of offset
                             from t. The readout reads the channel at ONE token; if
                             this is flat, it is reading it everywhere (see the
                             position-generalization question)
  attribution_by_neuron.csv  at (L, t), every neuron's attribution -- the target's
                             percentile, and the other neurons that rival it

Usage
-----
python score_gradient_attribution.py --config <run config>
python score_gradient_attribution.py --group mlp_joint_grid_layer_token \
    --run-glob 'joint_*kl0.1*' --skip-existing
"""
from __future__ import annotations

import json
import argparse
import operator
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.introspector.analysis_loading import (
    load_run_for_analysis, absolute_positions, run_per_locale)
from src.introspector.models.signal_paths import resolve_mlp_signal_path
from src.introspector.run_discovery import add_discovery_args, runs_from_args, resolve_run_action

# Attribution is accumulated in coordinates RELATIVE to the introspected token t,
# because t is what the prompt names and what the readout has to find; absolute
# positions would smear rows with different text lengths together.
DEFAULT_MAX_REL = 48
TOP_K = 40


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--n-test", type=int, default=None,
                   help="Test rows to use (default: the whole split).")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Smaller than the eval default on purpose: this pass keeps the "
                        "act_fn output AND its gradient for every layer "
                        "([B, seq, 8192] x 16 x 2), which is the memory constraint.")
    p.add_argument("--max-rel", type=int, default=DEFAULT_MAX_REL,
                   help="Half-width of the relative-position window kept around t.")
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Single-run override; a sweep always writes to "
                        "<run>/gradient_attribution/.")
    add_discovery_args(p)
    return p.parse_args()


class ActGradCapture:
    """Forward hook that keeps a layer's act_fn output AND asks autograd to retain its
    gradient, so grad x act can be read off after one backward pass."""

    def __init__(self):
        self.act = None

    def __call__(self, module, inputs, outputs):
        if not outputs.requires_grad:
            raise RuntimeError(
                "act_fn output does not require grad, so there is no gradient to "
                "retain. enable_activation_graph() must be in effect for this forward.")
        outputs.retain_grad()
        self.act = outputs
        return None


class enable_activation_graph:
    """Context manager making every activation from layer 0 onwards part of a grad graph.

    A checkpoint loaded for analysis is FROZEN -- PeftModel.from_pretrained defaults to
    is_trainable=False -- so no parameter requires grad, no activation has a grad_fn,
    and retain_grad() has nothing to retain. Rather than unfreeze the adapter (whose
    gradients are large and irrelevant here), this flips ONE small early parameter:
    layer 0's input LayerNorm weight, 2048 numbers that every later activation is
    downstream of. That is enough to build the graph, and the only parameter gradient
    materialized is 2048-long.

    Not `torch.enable_grad()`: grad mode is not the problem -- the absence of any
    graph-participating leaf is.
    """

    def __init__(self, run):
        self.param = run.base.model.layers[0].input_layernorm.weight

    def __enter__(self):
        self.prev = self.param.requires_grad
        self.param.requires_grad_(True)
        return self

    def __exit__(self, *exc):
        self.param.requires_grad_(self.prev)
        self.param.grad = None
        return False


def accumulate(sum_signed, sum_abs, counts, attr, rel_idx, keep):
    """Scatter one batch's [B, S, N] attribution into [n_rel, N] accumulators.

    `rel_idx` is each (row, position)'s index in the relative-position window and
    `keep` masks out padding and anything outside the window. index_add_ over the
    flattened (row, position) axis is what makes this affordable: per-row python
    loops over 8192 neurons would dominate the backward pass.
    """
    flat = attr.reshape(-1, attr.shape[-1])[keep]
    idx = rel_idx.reshape(-1)[keep]
    sum_signed.index_add_(0, idx, flat)
    sum_abs.index_add_(0, idx, flat.abs())
    counts.index_add_(0, idx, torch.ones_like(idx, dtype=counts.dtype))


def compute_attribution(run, args, out_dir: Path):
    device = run.model.device
    dataset = run.dataset if args.n_test is None else run.dataset.select(
        range(min(args.n_test, len(run.dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=run.collator,
                        shuffle=False)
    n_layers = run.base.config.num_hidden_layers
    n_neurons = run.base.config.intermediate_size
    n_rel = 2 * args.max_rel + 1
    head = run.model.regression_head

    # accumulators live on CPU in fp32: [layers, rel, neurons] is ~64MB each at
    # 16 x 97 x 8192, which is cheap on the host and would be a real slice of a GPU
    sum_signed = torch.zeros(n_layers, n_rel, n_neurons)
    sum_abs = torch.zeros(n_layers, n_rel, n_neurons)
    counts = torch.zeros(n_layers, n_rel)
    # how often each relative position falls in the introspection PROMPT rather than
    # the input text (1 = prompt): the two regions are not comparable, see report()
    region_sum = torch.zeros(n_rel)
    region_count = torch.zeros(n_rel)
    preds, acts = [], []

    hooks = []
    handles = []
    for layer in range(n_layers):
        hook = ActGradCapture()
        path = resolve_mlp_signal_path(run.model, layer, path_to_output=False)
        handles.append(operator.attrgetter(path)(run.model).register_forward_hook(hook))
        hooks.append(hook)

    try:
      with enable_activation_graph(run):
        for bi, batch in enumerate(loader):
            batch = batch.to(device)
            act_true = batch.pop("activation")
            attn_mask = batch["attention_mask"]
            bs, seq_len = attn_mask.shape
            abs_pos = absolute_positions(batch["token_position"], attn_mask)

            run.model.zero_grad(set_to_none=True)
            out = run.model(input_ids=batch["input_ids"], attention_mask=attn_mask,
                            lora_token_mask=batch.get("lora_token_mask"),
                            do_intervention=False, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :]
            pred = head(h.to(device=head.weight.device, dtype=head.weight.dtype)).squeeze(-1)
            # sum over the batch: rows are independent, so d(sum)/d(a_row) is that
            # row's own d(pred)/d(a) -- one backward serves the whole batch
            pred.sum().backward()
            preds.append(pred.detach().float().cpu())
            acts.append(act_true.detach().float().cpu())

            # (row, position) -> index in the relative window, dropping padding and
            # anything outside it
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, -1)
            rel = positions - abs_pos.unsqueeze(1)
            in_window = (rel.abs() <= args.max_rel) & (attn_mask == 1)
            rel_idx = (rel + args.max_rel).clamp(0, n_rel - 1)
            keep = in_window.reshape(-1)

            region = batch.get("lora_token_mask", batch.get("anchor_region_mask"))
            if region is not None:
                idx_cpu = rel_idx.reshape(-1)[keep].cpu()
                region_sum.index_add_(0, idx_cpu, region.reshape(-1)[keep].float().cpu())
                region_count.index_add_(0, idx_cpu, torch.ones_like(idx_cpu, dtype=torch.float))

            for layer, hook in enumerate(hooks):
                a = hook.act
                g = a.grad
                if g is None:
                    raise RuntimeError(
                        f"No gradient reached layer {layer}'s act_fn output. The backward "
                        f"pass did not run through it -- check that the model is not in "
                        f"inference/no_grad mode.")
                attr = (a.detach().float() * g.detach().float()).cpu()
                accumulate(sum_signed[layer], sum_abs[layer], counts[layer],
                           attr, rel_idx.cpu(), keep.cpu())
                hook.act = None
            if (bi + 1) % 20 == 0:
                print(f"[grad] {bi + 1} batches")
    finally:
        for h in handles:
            h.remove()
        run.model.zero_grad(set_to_none=True)

    denom = counts.clamp(min=1).unsqueeze(-1)
    mean_signed = sum_signed / denom
    mean_abs = sum_abs / denom
    frac_prompt = (region_sum / region_count.clamp(min=1)) if region_count.sum() > 0 else None
    return mean_signed, mean_abs, counts, frac_prompt, torch.cat(preds), torch.cat(acts)


def _rank_of(flat_scores, target_flat, mask_flat=None):
    """(rank of the target, number of units ranked) over the units `mask_flat` keeps."""
    scores = flat_scores.clone()
    if mask_flat is not None:
        scores[~mask_flat] = -float("inf")
    n = int(mask_flat.sum()) if mask_flat is not None else scores.numel()
    rank = int((scores > scores[target_flat]).sum()) + 1
    return rank, n


def report(run, args, mean_signed, mean_abs, counts, frac_prompt, preds, acts, out_dir: Path):
    n_layers, n_rel, n_neurons = mean_abs.shape
    rel0 = args.max_rel                                   # index of the introspected token
    L, i = run.layer, run.index

    # ---- ranking ----
    # positions never populated (short documents, window edge) must not rank
    valid = counts > 0
    flat = mean_abs.clone()
    flat[~valid] = -1.0
    flat = flat.reshape(-1)
    valid_flat = valid.unsqueeze(-1).expand_as(mean_abs).reshape(-1)
    target_flat = (L * n_rel + rel0) * n_neurons + i
    rank, n_units = _rank_of(flat, target_flat, valid_flat)

    # ...and again over INPUT-TEXT positions only. A unit at the readout token writes
    # directly into the residual the head reads, so those dominate the global list in
    # every run and say nothing about where the introspected signal was picked up.
    is_text = None
    rank_text, n_units_text = float("nan"), 0
    if frac_prompt is not None:
        text_rel = (frac_prompt < 0.5) & (counts.sum(0) > 0)
        is_text = (valid & text_rel.unsqueeze(0)).unsqueeze(-1).expand_as(mean_abs).reshape(-1)
        if bool(is_text[target_flat]):
            rank_text, n_units_text = _rank_of(flat, target_flat, is_text)

    def _rows(order_idx, label):
        out = []
        for r, pos in enumerate(order_idx.tolist(), 1):
            layer, rest = divmod(pos, n_rel * n_neurons)
            rel, neuron = divmod(rest, n_neurons)
            fp = float(frac_prompt[rel]) if frac_prompt is not None else float("nan")
            out.append({"ranking": label, "rank": r, "layer": layer,
                        "rel_position": rel - args.max_rel, "neuron": neuron,
                        "frac_prompt_region": fp,
                        "mean_abs_attr": float(mean_abs[layer, rel, neuron]),
                        "mean_signed_attr": float(mean_signed[layer, rel, neuron]),
                        "is_target": bool(layer == L and rel == rel0 and neuron == i)})
        return out

    rows = _rows(torch.argsort(flat, descending=True)[:args.top_k], "all_positions")
    if is_text is not None:
        text_scores = flat.clone(); text_scores[~is_text] = -float("inf")
        rows += _rows(torch.argsort(text_scores, descending=True)[:args.top_k], "text_positions")
    target_row = {"ranking": "target", "rank": rank, "layer": L, "rel_position": 0,
                  "neuron": i,
                  "frac_prompt_region": float(frac_prompt[rel0]) if frac_prompt is not None else float("nan"),
                  "mean_abs_attr": float(mean_abs[L, rel0, i]),
                  "mean_signed_attr": float(mean_signed[L, rel0, i]), "is_target": True}
    top_df = pd.DataFrame(rows + [target_row])
    top_df.insert(0, "run", run.config.experiment_name)
    top_df.to_csv(out_dir / "attribution_top.csv", index=False)

    # ---- the target neuron across positions ----
    pos_rows = []
    for r in range(n_rel):
        if counts[L, r] == 0:
            continue
        pos_rows.append({"rel_position": r - args.max_rel, "n": int(counts[L, r]),
                         "mean_abs_attr": float(mean_abs[L, r, i]),
                         "mean_signed_attr": float(mean_signed[L, r, i]),
                         # what an ordinary neuron looks like at the same position,
                         # so the target's number has a scale to be read against
                         "layer_median_abs_attr": float(mean_abs[L, r].median())})
    pos_df = pd.DataFrame(pos_rows)
    pos_df.insert(0, "run", run.config.experiment_name)
    pos_df.to_csv(out_dir / "attribution_by_position.csv", index=False)

    # ---- every neuron at (L, t) ----
    at_t = mean_abs[L, rel0]
    pct = float((at_t < at_t[i]).float().mean() * 100)
    order_n = torch.argsort(at_t, descending=True)[:args.top_k]
    neu_df = pd.DataFrame([{"rank": r, "neuron": int(n_), "mean_abs_attr": float(at_t[n_]),
                            "mean_signed_attr": float(mean_signed[L, rel0, n_]),
                            "is_target": bool(int(n_) == i)}
                           for r, n_ in enumerate(order_n.tolist(), 1)])
    neu_df.insert(0, "run", run.config.experiment_name)
    neu_df.to_csv(out_dir / "attribution_by_neuron.csv", index=False)

    endo_corr = float(np.corrcoef(preds.numpy(), acts.numpy())[0, 1]) if len(preds) > 2 else float("nan")
    summary = {
        "run": run.config.experiment_name, "checkpoint": str(run.checkpoint),
        "signal": f"N{L}.{i}",
        "token_position": (run.config.test_token_selector.position
                           if run.config.test_token_selector.type == "fixed" else None),
        "target_rank": rank, "n_units_ranked": n_units,
        "target_rank_percentile": 100.0 * (1 - (rank - 1) / max(n_units - 1, 1)),
        "target_rank_text_positions": rank_text, "n_units_ranked_text": n_units_text,
        "target_mean_abs_attr": float(mean_abs[L, rel0, i]),
        "target_percentile_within_layer_at_t": pct,
        "layer_median_abs_attr_at_t": float(at_t.median()),
        "top1": {"layer": int(top_df.iloc[0]["layer"]), "neuron": int(top_df.iloc[0]["neuron"]),
                 "rel_position": int(top_df.iloc[0]["rel_position"]),
                 "mean_abs_attr": float(top_df.iloc[0]["mean_abs_attr"])},
        "endogenous_corr": endo_corr, "n_rows": int(len(preds)),
    }
    with open(out_dir / "attribution_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[grad] target N{L}.{i} @t: rank {rank} of {n_units} units "
          f"(|attr|={summary['target_mean_abs_attr']:.3e}, "
          f"{pct:.2f}th pct within layer {L} at t)")
    print(f"[grad] rank among INPUT-TEXT units only: {rank_text} of {n_units_text} "
          f"-- quote this one; the global list is dominated by the readout token")
    print(f"[grad] top unit: L{summary['top1']['layer']} n{summary['top1']['neuron']} "
          f"rel{summary['top1']['rel_position']:+d} |attr|={summary['top1']['mean_abs_attr']:.3e}")
    print(top_df.head(10).to_string(index=False))
    # small arrays the plot needs but the CSVs do not hold, cached so --from-cache can
    # redraw without a forward/backward pass (the full [L, rel, neurons] tensor is not saved)
    mass = mean_abs.sum(-1).numpy()
    np.savez(out_dir / "gradient_attribution_arrays.npz",
             mass=mass, counts=counts.numpy(), at_t=at_t.numpy())
    _plot(run, args, mass, counts.numpy(), pos_df, at_t.numpy(), out_dir)
    return summary


def replot_gradient(spec, out_dir: Path, args) -> bool:
    """Redraw the gradient-attribution figure from cache; False if the cache is incomplete."""
    npz = out_dir / "gradient_attribution_arrays.npz"
    pos_csv = out_dir / "attribution_by_position.csv"
    if not (npz.exists() and pos_csv.exists()):
        return False
    d = np.load(npz)
    _plot(spec, args, d["mass"], d["counts"], pd.read_csv(pos_csv), d["at_t"], out_dir)
    return True


def _plot(run, args, mass, counts, pos_df, at_t, out_dir):
    """`mass` [layers, rel] and `at_t` [neurons] are the small plot-only arrays cached
    to npz (the full [layers, rel, neurons] tensor is not); `counts` [layers, rel]."""
    L, i = run.layer, run.index
    rel0 = args.max_rel
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    # 1: attribution mass by (layer, relative position), summed over neurons
    mass = np.asarray(mass, dtype=float).copy()
    mass[np.asarray(counts) == 0] = np.nan
    lo, hi = -args.max_rel, args.max_rel
    im = axes[0].imshow(mass, aspect="auto", origin="lower", extent=[lo, hi, -0.5, mass.shape[0] - 0.5],
                        cmap="magma")
    axes[0].axvline(0, color="w", ls="--", lw=1)
    axes[0].axhline(L, color="cyan", ls=":", lw=1.2)
    axes[0].set_xlabel("position relative to t"); axes[0].set_ylabel("layer")
    axes[0].set_title("total |grad x act| by (layer, position)")
    fig.colorbar(im, ax=axes[0])

    # 2: the target neuron across positions, against its layer's typical neuron
    axes[1].plot(pos_df["rel_position"], pos_df["mean_abs_attr"], "o-", label=f"N{L}.{i}")
    axes[1].plot(pos_df["rel_position"], pos_df["layer_median_abs_attr"], "s--",
                 color="grey", label=f"layer {L} median neuron")
    axes[1].axvline(0, color="k", ls="--", lw=1)
    axes[1].set_yscale("log"); axes[1].set_xlabel("position relative to t")
    axes[1].set_ylabel("mean |grad x act|"); axes[1].legend(fontsize=9)
    axes[1].set_title("target neuron across positions"); axes[1].grid(alpha=0.25)

    # 3: where the target sits in its layer's distribution at t
    v = np.asarray(at_t, dtype=float)
    v = v[v > 0]
    axes[2].hist(np.log10(v), bins=60, color="tab:grey")
    tgt = float(at_t[i])
    if tgt > 0:
        axes[2].axvline(np.log10(tgt), color="tab:red", lw=2, label=f"N{L}.{i}")
    axes[2].set_xlabel("log10 mean |grad x act|"); axes[2].set_ylabel("neurons")
    axes[2].legend(fontsize=9); axes[2].set_title(f"all {len(v)} neurons of layer {L} at t")

    fig.suptitle(f"Endogenous gradient attribution: {run.config.experiment_name}", y=1.02)
    fig.tight_layout()
    path = out_dir / "gradient_attribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[grad] wrote {path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    runs = runs_from_args(args)
    if not runs:
        raise SystemExit("No runs with checkpoints matched.")

    def analyze(run, spec, out_dir):
        grads = compute_attribution(run, args, out_dir)
        report(run, args, *grads, out_dir)

    run_per_locale(args, runs, "gradient_attribution", "attribution_summary.json",
                   analyze=analyze, replot=replot_gradient, log="[grad]")


if __name__ == "__main__":
    main()
