#!/usr/bin/env python3
"""
Is the readout causally tied to the LOCALE the prompt names, or to something broader?

The interventional metric already reported in every run injects at the run's own
(layer L, index i, token position t) and correlates the readout with the injected
value. A high number there says the model reads SOMETHING that the injection moves.
It does not say the model reads THAT neuron: an adapter that hoovered up a whole
layer, a band of neighbouring indices, or "whatever changed anywhere in the text"
would score exactly the same.

This script injects into WRONG locales and asks the same question. One coordinate is
varied at a time, so each category isolates one claim:

    vary_index      (L, i', t)   i' = neighbours i+/-1 and K random indices of layer L
                                 -> is it this neuron, or a band / the whole layer?
    vary_layer      (L', i, t)   every other layer, same index
                                 -> is it this layer, or index i anywhere?
    vary_position   (L, i, t')   same neuron, other token positions
                                 -> is it this token, or the neuron anywhere in the text?

TWO STIMULUS TYPES (--stimuli, both by default), because "no response at the wrong
locale" has a boring reading that has to be excluded. Values are drawn per row from:

    target_dist   (--stimuli target) the TARGET neuron's own distribution -- the same
                  stimulus the true locale receives, delivered to the wrong wire. This
                  is the like-for-like control: identical numbers, different address.
    own_dist      (--stimuli locale) the PROBED locale's own empirical distribution,
                  measured on this same corpus. A neuron with a tiny dynamic range
                  would look unresponsive under target_dist merely because the injected
                  values are off its natural scale; under own_dist it gets a
                  perturbation sized to itself, so a null cannot be blamed on scale.

    For the true locale the two coincide by definition, so it is scored once.

The draws are PAIRED: one standard-normal z per row is drawn once and reused for
every locale and stimulus, rescaled by that locale's mean/std. Differences between
categories are then differences in the model, not in the sampling noise.

WHAT TO EXPECT, and what each outcome means
    true locale high, every wrong locale ~0      the readout is addressed to (L, i, t)
    vary_index high at i+/-1 only                a band, not a neuron: the wire is the
                                                 down_proj column's neighbourhood
    vary_layer high at L'                        the index, not the locale -- would mean
                                                 the prompt's layer field is inert
    vary_position high at t'                     the channel is read wherever it appears,
                                                 i.e. no token addressing at all (which
                                                 is the mechanism the position-
                                                 generalization result would need)

READ `corr` TOGETHER WITH `pred_shift_rms`. The latter is RMS(readout under
injection - readout with no injection) and separates two very different nulls at a
wrong locale:
    pred_shift_rms ~ 0   the injection did not perturb the readout at all. Usually
                         uninteresting -- and if the TRUE locale ever shows this, the
                         injection plumbing is broken and every null in the file is
                         vacuous, which is why it is checked explicitly.
    pred_shift_rms > 0, corr ~ 0   the readout WAS perturbed and still does not report
                         the value. That is the strong form of the specificity claim:
                         not "nothing reached it" but "it is not listening to that
                         address".

Note that vary_position's null is the strongest of the three claims and the easiest
to get wrong: a neuron written at every text position (which is what a
position-independent down_proj edit produces) is only reportable at ONE of them if
something downstream selects by position. If the wrong-position corr is ~0, that
selection exists and is the thing to localize next.

Usage
-----
python score_injection_specificity.py --config <run config>
python score_injection_specificity.py --group mlp_joint_grid_layer_token \
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
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from src.introspector.analysis_loading import load_run_for_analysis, absolute_positions
from src.introspector.models.signal_paths import resolve_mlp_signal_path
from src.introspector.run_discovery import add_discovery_args, runs_from_args, resolve_run_action
from src.introspector.training.model import InterventionHook

# Token-position offsets probed by the vary_position category, relative to the
# introspected token t. Log-ish spacing: what matters is the ORDER of magnitude of the
# displacement, and +/-1 is the case a band-limited selector would fail on.
DEFAULT_POSITION_OFFSETS = (-16, -8, -4, -2, -1, 1, 2, 4, 8, 16)
# --stimuli values -> (column value written to the CSV, which locale's stats to draw
# from). Named after WHOSE distribution supplies the numbers, since "native" is
# ambiguous about whether it means the trained neuron or the one being probed.
STIMULUS_CHOICES = {"target": "target_dist", "locale": "own_dist"}
# Below this, the true locale is not reporting the injected value at that scale and the
# wrong-locale comparison has nothing to stand against (see _summarize).
WEAK_TRUE_CORR = 0.3

# A locale is only scored when this many rows can actually receive the injection: a
# position offset can fall outside a short document, and a correlation over a handful
# of surviving rows is not comparable with one over the whole split. Override with
# --min-valid-rows (a small smoke-test split legitimately needs a lower floor).
MIN_VALID_ROWS = 50


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--n-test", type=int, default=None,
                   help="Test rows to use (default: the whole split).")
    p.add_argument("--n-random-indices", type=int, default=10,
                   help="Random indices of the signal layer probed by vary_index.")
    p.add_argument("--position-offsets", type=str, default=None,
                   help=f"Comma-separated offsets from t (default: {DEFAULT_POSITION_OFFSETS}).")
    p.add_argument("--scales", type=str, default="1",
                   help="Injection scales (x the locale's std) to run the whole grid at. "
                        "1 is the natural scale -- the size of perturbation the neuron "
                        "actually undergoes. CAVEAT: a wrong-locale null only means "
                        "something if the TRUE locale responds at the same scale, and a "
                        "run trained with a 10x injection can sit near its noise floor "
                        "at 1 (its own scale sweep is the reference). The true locale's "
                        "corr is printed per scale and warned about when it is weak; "
                        "pass e.g. --scales 1,10 to bracket it.")
    p.add_argument("--stimuli", type=str, default="target",
                   help="Which value distributions to inject, comma-separated (default: "
                        "both). 'target' draws from the TRAINED neuron's own range and "
                        "delivers it to the wrong wire; 'locale' draws from the probed "
                        "(e.g. random) neuron's own range, so a null cannot be blamed on "
                        "the values being off its natural scale. They coincide at the "
                        "true locale, which is therefore scored once.") # empirically found target vs. locale doesn't change much
    p.add_argument("--batch-size", type=int, default=None,
                   help="Default: the run's own eval_batch_size.")
    p.add_argument("--min-valid-rows", type=int, default=MIN_VALID_ROWS,
                   help="Skip a locale scored on fewer rows than this (see MIN_VALID_ROWS).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Single-run override; a sweep always writes to "
                        "<run>/injection_specificity/.")
    add_discovery_args(p)
    return p.parse_args()


# ======================================================================
# Capture: the natural activation stats of every locale we will inject into
# ======================================================================
class MultiNeuronCapture:
    """Forward hook recording several neurons of one layer at every position.

    Captures [B, S, n_idx] rather than one (row, position, index) triple, because the
    same pass has to serve locales at several positions (vary_position) and several
    indices (vary_index). Returns None, so the module output passes through untouched.
    """

    def __init__(self, indices: torch.Tensor):
        self.indices = indices
        self.out = None

    def __call__(self, module, inputs, outputs):
        self.out = outputs[:, :, self.indices].detach().float().cpu()
        return None


@torch.no_grad()
def capture_locale_stats(run, loader, layer_indices, layer_offsets, device):
    """
    {(layer, index, position_offset): {mean, std, n}} measured on this corpus with the
    model as it is (adapter included).

    `layer_indices`: {layer: set of indices to capture}.
    `layer_offsets`: {layer: set of position offsets to summarize at}.
    Everything is captured in ONE forward pass per batch: stats are needed for ~40
    locales, and a pass per locale would dominate the runtime of the script.

    Stats are taken at the position each locale is INJECTED at, not pooled over the
    sequence -- a neuron's activation distribution at the last token is not its
    distribution in mid-document, and the own_dist stimulus is only honest if it is
    sized to the place it is delivered to. Rows whose t+offset falls outside the real
    sequence are excluded, so a stat is never computed over padding.
    """
    sums = {}
    for batch in loader:
        batch = batch.to(device)
        attn_mask = batch["attention_mask"]
        abs_pos = absolute_positions(batch["token_position"], attn_mask)
        hooks, handles = {}, []
        try:
            for layer, idxs in layer_indices.items():
                hook = MultiNeuronCapture(torch.as_tensor(sorted(idxs), device=device))
                path = resolve_mlp_signal_path(run.model, layer, path_to_output=False)
                handles.append(operator.attrgetter(path)(run.model).register_forward_hook(hook))
                hooks[layer] = hook
            run.model(input_ids=batch["input_ids"], attention_mask=attn_mask,
                      lora_token_mask=batch.get("lora_token_mask"), do_intervention=False)
        finally:
            for h in handles:
                h.remove()
        seq_len = attn_mask.shape[1]
        num_left_pad = (attn_mask == 0).sum(dim=1)
        rows = torch.arange(attn_mask.shape[0])
        for layer, hook in hooks.items():
            idxs = sorted(layer_indices[layer])
            for off in sorted(layer_offsets.get(layer, {0})):
                pos = abs_pos + off
                valid = ((pos >= num_left_pad) & (pos <= seq_len - 1)).cpu()
                vals = hook.out[rows, pos.clamp(0, seq_len - 1).cpu(), :]    # [B, n_idx]
                for j, idx in enumerate(idxs):
                    v = vals[valid, j]
                    s = sums.setdefault((layer, idx, off), [0.0, 0.0, 0])
                    s[0] += float(v.sum()); s[1] += float((v ** 2).sum()); s[2] += int(v.numel())
    stats = {}
    for key, (s1, s2, n) in sums.items():
        if n < 2:
            continue
        mean = s1 / n
        var = max(s2 / n - mean ** 2, 0.0)
        stats[key] = {"mean": mean, "std": float(np.sqrt(var)), "n": n}
    return stats


# ======================================================================
# Injection at an arbitrary locale
# ======================================================================
@torch.no_grad()
def inject_and_read(run, loader, layer, index, offset, values_fn, device):
    """
    Inject into (layer, index, t + offset) on every row and read the regression head.

    The model's own intervention path is NOT used: it samples from the run's cached
    signal_ranges, which only knows the trained neuron. Registering InterventionHook
    directly -- the same hook training used, so the injection site is identical -- is
    what lets any (layer, index, position) be addressed with any value.

    `values_fn(rows, stats_key)` returns the value for each row; see paired_values.
    Rows whose t+offset falls outside the real sequence are excluded from the
    correlation rather than clamped onto a neighbouring token.
    """
    head = run.model.regression_head
    path = resolve_mlp_signal_path(run.model, layer, path_to_output=False)
    module = operator.attrgetter(path)(run.model)
    preds, injected, valids, row0 = [], [], [], 0
    for batch in loader:
        batch = batch.to(device)
        attn_mask = batch["attention_mask"]
        bs, seq_len = attn_mask.shape
        abs_pos = absolute_positions(batch["token_position"], attn_mask) + offset
        num_left_pad = (attn_mask == 0).sum(dim=1)
        valid = (abs_pos >= num_left_pad) & (abs_pos <= seq_len - 1)
        pos = abs_pos.clamp(0, seq_len - 1)
        vals = values_fn(np.arange(row0, row0 + bs)).to(device)
        hook = InterventionHook(
            row_indices=torch.arange(bs, device=device), token_positions=pos,
            signal_indices=torch.full((bs,), index, device=device, dtype=torch.long),
            intervention_values=vals)
        handle = module.register_forward_hook(hook)
        try:
            out = run.model(input_ids=batch["input_ids"], attention_mask=attn_mask,
                            lora_token_mask=batch.get("lora_token_mask"),
                            do_intervention=False, output_hidden_states=True)
        finally:
            handle.remove()
        h = out.hidden_states[-1][:, -1, :]
        preds.append(head(h.to(device=head.weight.device, dtype=head.weight.dtype))
                     .squeeze(-1).float().cpu())
        injected.append(vals.float().cpu())
        valids.append(valid.cpu())
        row0 += bs
    return torch.cat(preds).numpy(), torch.cat(injected).numpy(), torch.cat(valids).numpy()


@torch.no_grad()
def endogenous_reference(run, loader, device):
    """Readout with NO injection, against the cached activation: reproduces the run's
    own endogenous corr, which is the check that this script loaded the right model
    the right way before any of its numbers are believed."""
    head = run.model.regression_head
    preds, acts = [], []
    for batch in loader:
        batch = batch.to(device)
        act = batch.pop("activation")
        out = run.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        lora_token_mask=batch.get("lora_token_mask"),
                        do_intervention=False, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :]
        preds.append(head(h.to(device=head.weight.device, dtype=head.weight.dtype))
                     .squeeze(-1).float().cpu())
        acts.append(act.float().cpu())
    return torch.cat(preds).numpy(), torch.cat(acts).numpy()


def corr(a, b, mask=None):
    if mask is not None:
        a, b = a[mask], b[mask]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(pearsonr(a, b)[0])


# ======================================================================
# The locale grid
# ======================================================================
def build_locales(run, args, rng):
    """Every (category, layer, index, offset) this run is probed at, true locale first."""
    n_layers = run.base.config.num_hidden_layers
    n_neurons = run.base.config.intermediate_size
    L, i = run.layer, run.index
    offsets = ([int(x) for x in args.position_offsets.split(",")] if args.position_offsets
               else list(DEFAULT_POSITION_OFFSETS))

    locales = [dict(category="true", layer=L, index=i, offset=0)]
    # neighbours first: a band-limited wire shows up here and nowhere else
    for d in (-1, 1):
        j = i + d
        if 0 <= j < n_neurons:
            locales.append(dict(category="vary_index", layer=L, index=j, offset=0,
                                detail=f"i{d:+d}"))
    pool = np.delete(np.arange(n_neurons), [x for x in (i - 1, i, i + 1) if 0 <= x < n_neurons])
    for j in sorted(rng.choice(pool, size=min(args.n_random_indices, len(pool)), replace=False)):
        locales.append(dict(category="vary_index", layer=L, index=int(j), offset=0,
                            detail="random"))
    for L2 in range(n_layers):
        if L2 != L:
            locales.append(dict(category="vary_layer", layer=L2, index=i, offset=0))
    for off in offsets:
        locales.append(dict(category="vary_position", layer=L, index=i, offset=int(off)))
    return locales


def parse_stimuli(spec: str):
    """'target,locale' -> [(csv value, which stats to use), ...], order preserved."""
    out = []
    for name in (x.strip().lower() for x in spec.split(",")):
        if not name:
            continue
        if name not in STIMULUS_CHOICES:
            raise SystemExit(f"Unknown --stimuli value {name!r}; "
                             f"choose from {sorted(STIMULUS_CHOICES)}.")
        pair = (STIMULUS_CHOICES[name], name)
        if pair not in out:
            out.append(pair)
    if not out:
        raise SystemExit("--stimuli selected nothing.")
    return out


def paired_values(z, stats, scale):
    """mean + scale*std*z, with ONE z per row shared across every locale and stimulus.

    Pairing is what makes the categories comparable: an unpaired draw would let two
    locales differ by sampling noise, and with ~40 locales something would look
    responsive eventually.
    """
    return torch.from_numpy(stats["mean"] + scale * stats["std"] * z).float()


def score_run(run, args, out_dir: Path):
    device = run.model.device
    batch_size = args.batch_size or run.config.training_config.eval_batch_size
    dataset = run.dataset if args.n_test is None else run.dataset.select(
        range(min(args.n_test, len(run.dataset))))
    n_rows = len(dataset)
    loader = lambda: DataLoader(dataset, batch_size=batch_size, collate_fn=run.collator,
                                shuffle=False)
    rng = np.random.RandomState(args.seed + 1000 * run.layer + run.index)
    z = rng.randn(n_rows)
    scales = [float(s) for s in args.scales.split(",")]
    stimuli = parse_stimuli(args.stimuli)

    locales = build_locales(run, args, rng)
    print(f"[spec] {len(locales)} locales x {len(scales)} scale(s) x "
          f"{len(stimuli)} stimulus type(s) {[c for c, _ in stimuli]} on {n_rows} rows")

    # ---- natural stats for every locale (one pass) ----
    layer_indices: dict = {}
    layer_offsets: dict = {}
    for loc in locales:
        layer_indices.setdefault(loc["layer"], set()).add(loc["index"])
        layer_offsets.setdefault(loc["layer"], {0}).add(loc["offset"])
    print(f"[spec] capturing natural stats for "
          f"{sum(len(v) for v in layer_indices.values())} (layer, index) pairs ...")
    stats = capture_locale_stats(run, loader(), layer_indices, layer_offsets, device)
    target_stats = stats[(run.layer, run.index, 0)]
    print(f"[spec] target N{run.layer}.{run.index} @t: mean={target_stats['mean']:.4f} "
          f"std={target_stats['std']:.4f}")

    # ---- endogenous reference ----
    # also the baseline every locale's pred_shift_rms is measured against, so the
    # "the injection never happened" failure mode is visible instead of silent
    en_pred, en_act = endogenous_reference(run, loader(), device)
    en_corr = corr(en_pred, en_act)
    print(f"[spec] endogenous corr (no injection) = {en_corr:.4f}  "
          f"-- compare with this run's reported endogenous_test_metrics")

    rows = []
    for scale in scales:
        for k, loc in enumerate(locales, 1):
            key = (loc["layer"], loc["index"], loc["offset"])
            own = stats.get(key)
            if own is None or own["n"] < args.min_valid_rows:
                print(f"[spec] skip (L{key[0]} i{key[1]} off{key[2]}): "
                      f"{own['n'] if own else 0} valid rows < --min-valid-rows "
                      f"{args.min_valid_rows} (of {n_rows} rows)")
                continue
            for stimulus, which in stimuli:
                # At the true locale the probed neuron IS the target, so the two
                # distributions are the same object and it is scored ONCE -- under
                # whichever stimulus was requested first. Keyed off the request rather
                # than hardcoding "skip own_dist", which dropped the true locale
                # altogether under `--stimuli locale` and left the wrong-locale nulls
                # with no reference and the positive control with nothing to check.
                if loc["category"] == "true" and stimulus != stimuli[0][0]:
                    continue
                st = target_stats if which == "target" else own
                vals = paired_values(z, st, scale)
                pred, inj, valid = inject_and_read(
                    run, loader(), loc["layer"], loc["index"], loc["offset"],
                    lambda idx, v=vals: v[idx], device)
                c = corr(pred, inj, valid)
                shift = float(np.sqrt(np.mean((pred[valid] - en_pred[valid]) ** 2)))
                rows.append({
                    "run": run.config.experiment_name,
                    "signal_layer": run.layer, "signal_index": run.index,
                    "token_position": (run.config.test_token_selector.position
                                       if run.config.test_token_selector.type == "fixed" else None),
                    "category": loc["category"], "detail": loc.get("detail", ""),
                    "inject_layer": loc["layer"], "inject_index": loc["index"],
                    "position_offset": loc["offset"], "stimulus": stimulus, "scale": scale,
                    "corr": c, "n_valid": int(valid.sum()), "pred_shift_rms": shift,
                    "inj_std": float(np.std(inj[valid])), "pred_std": float(np.std(pred[valid])),
                    "locale_mean": own["mean"], "locale_std": own["std"],
                    "endogenous_corr": en_corr,
                })
            if k % 10 == 0 or k == len(locales):
                print(f"[spec] scale={scale}: {k}/{len(locales)} locales")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "injection_specificity.csv", index=False)

    # POSITIVE CONTROL. If injecting at the run's own locale does not even move the
    # readout, the hook never fired (wrong module path, wrong positions, adapters
    # disabled) and every null in this file is an artefact rather than a result.
    true_rows = df[df["category"] == "true"]
    if len(true_rows) and true_rows["pred_shift_rms"].max() < 1e-6:
        print("[spec] *** POSITIVE CONTROL FAILED: injecting at the true locale left "
              "the readout unchanged. The injection did not take effect -- do NOT read "
              "the wrong-locale nulls as specificity. ***")
    _summarize(df, out_dir)
    _plot(df, run, out_dir)
    with open(out_dir / "injection_specificity_meta.json", "w") as f:
        json.dump({"run": run.config.experiment_name, "checkpoint": str(run.checkpoint),
                   "signal": f"N{run.layer}.{run.index}", "n_rows": n_rows,
                   "scales": scales, "stimuli": [c for c, _ in stimuli],
                   "endogenous_corr": en_corr,
                   "target_stats": target_stats}, f, indent=2)
    print(f"[spec] wrote {out_dir}")
    return df


def _summarize(df, out_dir):
    if df.empty:
        return
    s = (df.groupby(["scale", "category", "stimulus"])["corr"]
         .agg(["mean", "std", "max", "count"]).reset_index())
    s.to_csv(out_dir / "injection_specificity_summary.csv", index=False)
    print("\n[spec] corr(readout, injected value) by category:")
    print(s.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    true = df[df["category"] == "true"]
    for scale, g in df.groupby("scale"):
        t = true[true["scale"] == scale]["corr"]
        wrong = g[g["category"] != "true"]["corr"]
        if len(t) and len(wrong):
            print(f"[spec] scale={scale}: true={t.iloc[0]:.4f} | wrong locales: "
                  f"max={wrong.max():.4f} mean={wrong.mean():.4f} "
                  f"(n={len(wrong)}) -- the gap IS the specificity claim")
            # A null at a wrong locale is only evidence of addressing if the RIGHT
            # locale answers at the same scale. Runs trained with a large injection can
            # sit near their noise floor at scale 1, where every locale reads ~0 and the
            # file looks like perfect specificity while measuring nothing.
            if abs(t.iloc[0]) < WEAK_TRUE_CORR:
                print(f"[spec] *** WARNING: the TRUE locale only reads "
                      f"{t.iloc[0]:.4f} at scale {scale}. The wrong-locale nulls below "
                      f"it are NOT evidence of specificity -- nothing is being reported "
                      f"at this scale. Re-run with a larger --scales (the run's own "
                      f"interventional_scale_sweep.csv says where it responds). ***")


# x axis of the summary figure: the three ways a locale can be wrong.
PLOT_CONDITIONS = [("vary_index", "wrong index\n(same layer, same token)"),
                   ("vary_layer", "wrong layer\n(same index, same token)"),
                   ("vary_position", "wrong token\n(same neuron)")]


def _plot(df, run, out_dir):
    """One small panel per injection scale: the wrong locales as a distribution, the
    true locale as the reference they are read against.

    A box rather than corr-vs-coordinate scatters, because the coordinate carries no
    information here -- neuron 5676 is not "between" 24 and 8191 in any sense the model
    respects, and plotting against it invites reading a trend into what is a set of
    independent nulls. What matters is the SPREAD of the wrong locales against the one
    number that is supposed to stand apart from them.

    Only the target-distribution stimulus is drawn (own_dist tracks it closely enough
    that two series per condition was clutter); the CSV keeps both.
    """
    if df.empty:
        return
    if (df["stimulus"] == "target_dist").any():
        df = df[df["stimulus"] == "target_dist"]

    for scale, g in df.groupby("scale"):
        true_corr = g[g["category"] == "true"]["corr"]
        true_corr = float(true_corr.iloc[0]) if len(true_corr) else float("nan")

        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        rng = np.random.RandomState(0)
        data, labels, positions = [], [], []
        for k, (cat, label) in enumerate(PLOT_CONDITIONS, 1):
            vals = g[g["category"] == cat]["corr"].dropna().values
            labels.append(f"{label}\nn={len(vals)}")
            positions.append(k)
            data.append(vals)
            if len(vals):
                # the individual locales, so a box drawn over 3 points is not mistaken
                # for a distribution over many
                ax.scatter(k + rng.uniform(-0.09, 0.09, len(vals)), vals, s=14,
                           color="tab:blue", alpha=0.55, zorder=3, linewidths=0)
        drawn = [(p, d) for p, d in zip(positions, data) if len(d)]
        if drawn:
            bp = ax.boxplot([d for _, d in drawn], positions=[p for p, _ in drawn],
                            widths=0.5, patch_artist=True, showfliers=False, zorder=2)
            for box in bp["boxes"]:
                box.set(facecolor="tab:blue", alpha=0.22, edgecolor="tab:blue")
            for part in ("whiskers", "caps", "medians"):
                for artist in bp[part]:
                    artist.set(color="tab:blue")

        ax.axhline(0, color="k", lw=0.8, zorder=1)
        if np.isfinite(true_corr):
            ax.axhline(true_corr, color="tab:red", ls="--", lw=1.4, zorder=4,
                       label=f"target locale N{run.layer}.{run.index} @t ({true_corr:.2f})")
            ax.legend(fontsize=8, loc="best", framealpha=0.9)
        ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("corr(readout, injected v)", fontsize=9)
        lo = min(-0.08, float(np.nanmin([v.min() for v in data if len(v)], initial=0)) - 0.05)
        ax.set_ylim(lo, max(1.02, true_corr + 0.08 if np.isfinite(true_corr) else 1.02))
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(f"{run.config.experiment_name}\ninjection scale {scale:g}x",
                     fontsize=8)
        fig.tight_layout()
        path = out_dir / f"injection_specificity_scale{scale:g}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"[spec] wrote {path}")


def replot_specificity(spec, out_dir: Path, args) -> bool:
    """Redraw the specificity figures from injection_specificity.csv; False if absent."""
    csv = out_dir / "injection_specificity.csv"
    if not csv.exists():
        return False
    _plot(pd.read_csv(csv), spec, out_dir)
    return True


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    runs = runs_from_args(args)
    if not runs:
        raise SystemExit("No runs with checkpoints matched.")
    sweeping = len(runs) > 1 or args.group is not None
    for k, spec in enumerate(runs, 1):
        out_dir = ((args.output_dir or (spec.run_dir / "injection_specificity")) if not sweeping
                   else spec.run_dir / "injection_specificity")
        action = resolve_run_action(args, out_dir / "injection_specificity.csv")
        if action == "skip":
            print(f"[spec] [{k}/{len(runs)}] {spec.name}: results exist, skipping")
            continue
        if action == "replot":
            if replot_specificity(spec, out_dir, args):
                print(f"[spec] [{k}/{len(runs)}] {spec.name}: re-plotted from cache")
                continue
            print(f"[spec] [{k}/{len(runs)}] {spec.name}: cache incomplete, recomputing")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[spec] ===== [{k}/{len(runs)}] {spec.name} ({spec.cell}) =====")
        run = load_run_for_analysis(spec.config, checkpoint=args.checkpoint or spec.checkpoint())
        try:
            score_run(run, args, out_dir)
        finally:
            del run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
