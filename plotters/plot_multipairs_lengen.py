#!/usr/bin/env python3
"""
Paper experiment 3-1: multi (dst, src) pairs + length generalization, endogenous.

    python plotters/plot_multipairs_lengen.py \
        outputs/runs/attn_weight_paper31_multipairs_lengen_endo_logistic_ndev_300K

Reads every run `endo_P<L>.<H>_pairs_n<N>_...` of the group (3 heads x 4 training
lengths n) and its by-pair files (written by src/introspector/training/pair_metrics.py):

    endogenous_test_by_pair_endogenous.csv         m = n (the run's own test split)
    endogenous_test_by_pair_len<m>.csv             m > n (score_length_generalization)
    interventional_test_by_pair_<dist>.csv         m = n
    interventional_test_by_pair_len<m>[_<dist>].csv  m > n (gaussian has no _<dist>)

POSITIONS ARE ABSOLUTE (0 = BOS). The headline per pair is the WITHIN-PAIR corr (over the
test documents of that one pair), averaged over the 3 heads. Pooled corrs are not used:
between-pair variance dominates them (see pair_metrics.py).

Outputs (under outputs/plots/<group>/):
    multipairs_by_pair.csv        one row per (head, n, m, eval, dst, src)
    triangle_<eval>.png           FIGURE A: rows n, columns m >= n. m <= 32: (src, dst)
                                  on linear axes, every pair. m >= 64: dst (log grid) x
                                  delta = dst - src (log grid) + a separate BOS column.
                                  Dashed line: dst = n, above it pairs no training document
                                  ever asked for. Dot: target std < LOW_STD in that cell.
    paired_in_domain_<eval>.png   FIGURE B: pairs with dst < n, evaluated at every m >= n,
                                  by pair class. Their TARGET is the same number at every m
                                  (causal mask), so a fall is the cost of the extra context.
    paired_pred_shift.png         FIGURE B': mean |pred(m) - pred(n)| on the same
                                  (document, dst, src), endogenous.
    by_dst_<eval>.png             FIGURE C: within-pair corr against dst, one line per m,
                                  one panel per n, vertical line at dst = n.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RUN_RE = re.compile(r"^(?:endo|joint)_P(?P<layer>\d+)\.(?P<head>\d+)_pairs_n(?P<n>\d+)_")
LEN_RE = re.compile(r"_len(?P<m>\d+)(?:_(?P<dist>[a-z_]+))?\.csv$")
EVAL_LENGTHS = (16, 32, 64, 128, 256, 512)
DENSE_MAX_LEN = 32          # signals/pair_grid.py: every pair up to here, log grid above
LOW_STD = 1e-3              # a cell whose target barely moves: its corr is mostly noise

SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
SEQ_CMAP.set_bad("#f0efec")
INK, MUTED = "#1a1a19", "#8a8a85"
CLASS_COLORS = {"self (s=d)": "#3987e5", "previous (s=d-1)": "#e5813a",
                "BOS (s=0)": "#5a9e4b", "other": "#8a6bbf"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("group_dir", type=Path)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def _eval_name(kind, dist):
    return "endogenous" if kind == "endogenous" else f"interventional_{dist}"


def collect(group_dir: Path) -> pd.DataFrame:
    frames = []
    for run in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        mo = RUN_RE.match(run.name)
        if mo is None:
            continue
        layer, head, n = int(mo["layer"]), int(mo["head"]), int(mo["n"])
        for f in sorted(run.glob("*_test_by_pair*.csv")):
            kind = "endogenous" if f.name.startswith("endogenous") else "interventional"
            ml = LEN_RE.search(f.name)
            if ml:
                m, dist = int(ml["m"]), ml["dist"] or "gaussian"
            else:
                m = n
                dist = f.stem.split("_test_by_pair_", 1)[1]   # endogenous / <dist>
            if kind == "interventional" and dist.startswith("ood"):
                continue
            if f.name.startswith("endogenous_test_by_pair_ood"):
                continue
            # the by-pair "n" is the document count of the pair; renamed first, or the
            # training length assigned below would overwrite it without any error
            df = pd.read_csv(f).rename(columns={"n": "n_docs"})
            df["layer"], df["head"], df["n"], df["m"] = layer, head, n, m
            df["eval"] = _eval_name(kind, dist)
            frames.append(df)
    if not frames:
        raise SystemExit(f"no by-pair files under {group_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"token_position": "dst", "src_token_position": "src"})
    assert (df["src"] <= df["dst"]).all() and (df["src"] >= 0).all(), "pair outside causal range"
    assert not ((df["dst"] == 0) & (df["src"] == 0)).any(), "(0, 0) should never be evaluated"
    return df


def pair_class(d, s):
    if s == 0:
        return "BOS (s=0)"
    if s == d:
        return "self (s=d)"
    if s == d - 1:
        return "previous (s=d-1)"
    return "other"


def head_mean(df: pd.DataFrame, metric="corr") -> pd.DataFrame:
    """Mean over heads per (n, m, eval, dst, src); low_std if ANY head's target is flat."""
    g = df.groupby(["n", "m", "eval", "dst", "src"])
    out = g[metric].mean().rename("value").to_frame()
    out["n_heads"] = g[metric].count()
    out["low_std"] = g["target_std"].min() < LOW_STD
    return out.reset_index()


def _delta_axis(m):
    deltas, d = [0], 1
    while d < m:
        deltas.append(d)
        d *= 2
    return deltas


def _panel_linear(ax, cell, m, n, norm):
    mat = np.full((m, m), np.nan)                              # [m, m], rows dst, cols src
    for d, s, v in zip(cell["dst"], cell["src"], cell["value"]):
        mat[d, s] = v
    ax.imshow(np.ma.masked_invalid(mat), cmap=SEQ_CMAP, norm=norm, origin="lower",
              aspect="equal", interpolation="nearest")
    flat = cell[cell["low_std"]]
    ax.scatter(flat["src"], flat["dst"], s=2, c=MUTED, marker=".")
    if n < m:
        ax.axhline(n - 0.5, color=INK, lw=0.8, ls="--")
    ax.set_xlabel("src", fontsize=7)
    ax.set_ylabel("dst", fontsize=7)


def _panel_log(ax, cell, m, n, norm):
    dsts = sorted(cell["dst"].unique())
    deltas = _delta_axis(m)
    cols = [f"{x}" for x in deltas] + ["BOS"]
    mat = np.full((len(dsts), len(cols)), np.nan)               # [n_dst, n_delta + 1]
    flat = []
    for d, s, v, lo in zip(cell["dst"], cell["src"], cell["value"], cell["low_std"]):
        r = dsts.index(d)
        c = len(cols) - 1 if s == 0 else deltas.index(d - s)
        mat[r, c] = v
        if lo:
            flat.append((c, r))
    ax.imshow(np.ma.masked_invalid(mat), cmap=SEQ_CMAP, norm=norm, origin="lower",
              aspect="auto", interpolation="nearest")
    if flat:
        ax.scatter(*zip(*flat), s=4, c=MUTED, marker=".")
    ax.axvline(len(cols) - 1.5, color="white", lw=2)           # BOS column apart
    above = [i for i, d in enumerate(dsts) if d >= n]
    if above and above[0] > 0:
        ax.axhline(above[0] - 0.5, color=INK, lw=0.8, ls="--")
    ax.set_yticks(range(len(dsts)), [str(d) for d in dsts], fontsize=5)
    ax.set_xticks(range(len(cols)), cols, fontsize=5, rotation=90)
    ax.set_xlabel("delta = dst - src", fontsize=7)
    ax.set_ylabel("dst", fontsize=7)


def plot_triangles(hm: pd.DataFrame, ev: str, out: Path):
    sub = hm[hm["eval"] == ev]
    if sub.empty:
        return
    ns, ms = sorted(sub["n"].unique()), [m for m in EVAL_LENGTHS if m in set(sub["m"])]
    norm = plt.Normalize(vmin=min(0.0, float(np.nanmin(sub["value"]))), vmax=1.0)
    fig, axes = plt.subplots(len(ns), len(ms), figsize=(2.6 * len(ms), 2.5 * len(ns)),
                             squeeze=False)
    for i, n in enumerate(ns):
        for j, m in enumerate(ms):
            ax = axes[i][j]
            cell = sub[(sub["n"] == n) & (sub["m"] == m)]
            if m < n or cell.empty:
                ax.axis("off")
                continue
            (_panel_linear if m <= DENSE_MAX_LEN else _panel_log)(ax, cell, m, n, norm)
            ax.set_title(f"train n={n}, eval m={m}", fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=SEQ_CMAP, norm=norm)
    fig.colorbar(sm, ax=axes, shrink=0.6).set_label("within-pair corr (mean over heads)")
    fig.suptitle(f"{ev}: within-pair corr per (dst, src); dashed = dst = n", fontsize=10)
    fig.savefig(out / f"triangle_{ev}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[3-1] wrote {out / f'triangle_{ev}.png'}")


def plot_paired(hm: pd.DataFrame, ev: str, out: Path):
    sub = hm[(hm["eval"] == ev) & (hm["dst"] < hm["n"])].copy()
    if sub.empty:
        return
    sub["cls"] = [pair_class(d, s) for d, s in zip(sub["dst"], sub["src"])]
    ns = sorted(sub["n"].unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(3.4 * len(ns), 3.0), squeeze=False)
    for ax, n in zip(axes[0], ns):
        g = sub[sub["n"] == n]
        ms = sorted(g["m"].unique())
        # only pairs present at EVERY m, so each line follows the same pairs
        per_m = [set(zip(g[g["m"] == m]["dst"], g[g["m"] == m]["src"])) for m in ms]
        common = set.intersection(*per_m)
        g = g[[p in common for p in zip(g["dst"], g["src"])]]
        for cls, color in CLASS_COLORS.items():
            c = g[g["cls"] == cls].groupby("m")["value"].mean()
            if len(c):
                ax.plot(c.index, c.values, "o-", color=color, ms=3, lw=1.2, label=cls)
        ax.set_xscale("log", base=2)
        ax.set_title(f"train n={n} ({len(common)} pairs, dst < n)", fontsize=8)
        ax.set_xlabel("eval length m", fontsize=7)
        ax.set_ylim(min(0, ax.get_ylim()[0]), 1)
    axes[0][0].set_ylabel("within-pair corr", fontsize=7)
    axes[0][-1].legend(fontsize=6)
    fig.suptitle(f"{ev}: same (document, dst, src), target unchanged; only the context after "
                 f"dst grows", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"paired_in_domain_{ev}.png", dpi=200)
    plt.close(fig)
    print(f"[3-1] wrote {out / f'paired_in_domain_{ev}.png'}")


def plot_pred_shift(group_dir: Path, out: Path):
    """|pred(m) - pred(n)| on the same (document, dst, src), endogenous predictions."""
    rows = []
    for run in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        mo = RUN_RE.match(run.name)
        base = run / "endogenous_test_predictions_endogenous.csv"
        if mo is None or not base.exists():
            continue
        n = int(mo["n"])
        keys = ["doc_index", "token_position", "src_token_position"]
        ref = pd.read_csv(base, usecols=keys + ["predictions"])
        if ref["doc_index"].isna().any():
            continue
        for f in run.glob("endogenous_test_predictions_len*.csv"):
            m = int(LEN_RE.search(f.name)["m"])
            cur = pd.read_csv(f, usecols=keys + ["predictions"])
            j = ref.merge(cur, on=keys, suffixes=("_n", "_m"))
            j = j[j["token_position"] < n]
            if j.empty:
                continue
            for cls, g in j.groupby([pair_class(d, s) for d, s in
                                     zip(j["token_position"], j["src_token_position"])]):
                rows.append({"head": int(mo["head"]), "n": n, "m": m, "cls": cls,
                             "shift": float((g["predictions_m"] - g["predictions_n"]).abs().mean())})
    if not rows:
        return
    df = pd.DataFrame(rows).groupby(["n", "m", "cls"])["shift"].mean().reset_index()
    ns = sorted(df["n"].unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(3.4 * len(ns), 3.0), squeeze=False)
    for ax, n in zip(axes[0], ns):
        for cls, color in CLASS_COLORS.items():
            c = df[(df["n"] == n) & (df["cls"] == cls)].sort_values("m")
            if len(c):
                ax.plot(c["m"], c["shift"], "o-", color=color, ms=3, lw=1.2, label=cls)
        ax.set_xscale("log", base=2)
        ax.set_title(f"train n={n}", fontsize=8)
        ax.set_xlabel("eval length m", fontsize=7)
    axes[0][0].set_ylabel("mean |pred(m) - pred(n)|", fontsize=7)
    axes[0][-1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "paired_pred_shift.png", dpi=200)
    plt.close(fig)
    print(f"[3-1] wrote {out / 'paired_pred_shift.png'}")


def plot_by_dst(hm: pd.DataFrame, ev: str, out: Path):
    sub = hm[hm["eval"] == ev]
    if sub.empty:
        return
    ns = sorted(sub["n"].unique())
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, len(ns), figsize=(3.4 * len(ns), 3.0), squeeze=False)
    for ax, n in zip(axes[0], ns):
        g = sub[sub["n"] == n]
        ms = sorted(g["m"].unique())
        for k, m in enumerate(ms):
            c = g[g["m"] == m].groupby("dst")["value"].mean()
            ax.plot(c.index, c.values, "-", color=cmap(k / max(len(ms) - 1, 1)), lw=1,
                    label=f"m={m}")
        ax.axvline(n, color=INK, lw=0.8, ls="--")
        ax.set_xscale("symlog", base=2, linthresh=1)
        ax.set_title(f"train n={n}", fontsize=8)
        ax.set_xlabel("dst (absolute)", fontsize=7)
    axes[0][0].set_ylabel("within-pair corr, mean over src", fontsize=7)
    axes[0][-1].legend(fontsize=6)
    fig.suptitle(f"{ev}: by destination; right of the dashed line = dst never trained",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"by_dst_{ev}.png", dpi=200)
    plt.close(fig)
    print(f"[3-1] wrote {out / f'by_dst_{ev}.png'}")


def main():
    args = parse_args()
    out = args.out or Path("outputs/plots") / args.group_dir.name
    out.mkdir(parents=True, exist_ok=True)
    df = collect(args.group_dir)
    df.to_csv(out / "multipairs_by_pair.csv", index=False)
    print(f"[3-1] {len(df)} (head, n, m, eval, pair) rows from {args.group_dir}")
    hm = head_mean(df)
    for ev in sorted(hm["eval"].unique()):
        plot_triangles(hm, ev, out)
        plot_paired(hm, ev, out)
        plot_by_dst(hm, ev, out)
    plot_pred_shift(args.group_dir, out)


if __name__ == "__main__":
    main()
