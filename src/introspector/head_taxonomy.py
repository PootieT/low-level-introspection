"""
Classify attention heads by how they take part in moving the introspected value.

WHY THIS EXISTS
---------------
Ranking heads by their dose-response slope conflates heads that do very different
things. Two properties are measured separately by mechanistic_analysis/trace_mover_head.py, and a head's
role is the combination:

  attention sensitivity   does the head's attention to the introspected token t MOVE
                          as the injected value moves? (`attn_t_delta`, measured in the
                          largest-scale global regime, where a switch is visible if it
                          exists at all). A head that jumps from ~0 to ~1 is detecting an
                          out-of-distribution activation, not routing a value.
  sourced from t          of the head's v-dependence, how much arrives through source
                          position t (`frac_slope_from_t`) -- i.e. does it fetch the
                          value from the origin, or re-read it from wherever an earlier
                          head already deposited it?

                          | sourced from t     | sourced elsewhere
      ------------------- | ------------------ | ------------------
      attention switches  | threshold_grabber  | threshold_other
      attention partial   | graded_transporter |   (re_reader)
      attention static    | linear_transporter | re_reader

plus `inert` for heads that neither switch nor carry a material share of the signal.

WHY THREE ATTENTION STATES AND NOT TWO. Measured on joint_N0.1377_POS-12, attn_t_delta
at x10 falls as 1.00, 1.00, 1.00, 1.00, 1.00, 0.86, 0.84, 0.67, 0.61, 0.55, 0.53 |
0.27, 0.24, 0.22, 0.19, ... -- a clear cluster of switches, a factor-of-two GAP, then a
continuous tail. A binary split at the gap calls the tail "static", which it is not:
those heads' attention moves by ~0.2 over the sweep. So `static` means genuinely ~0
(below static_tau) and the tail gets its own name. Notably the tail sits at the same
depth as the switches (rel_depth 1 there), which reads as a weaker form of the same
outlier response rather than a separate mechanism -- worth checking across the grid.

The linear_transporter cell is the one that matters most and is easy to miss: static
attention, value fetched from t through the OV circuit. It is the only kind of head
that can carry the signal at natural magnitudes, where a threshold never fires.

This module is the ONE place those definitions live. The Stage B aggregator
(plotters/plot_head_taxonomy.py) and the Stage C ablation grouping
(mechanistic_analysis/score_head_ablation.py) both call classify_heads, so a head cannot be a "transporter"
in one figure and a "re_reader" in the other.

THRESHOLDS are arguments, not constants buried in code, and every one is reported
alongside the class so a borderline head is visible as borderline:
  att_tau         attn_t_delta at or above which attention counts as SWITCHING (placed
                  in the gap above; see DEFAULT_ATT_TAU).
  static_tau      attn_t_delta below which attention counts as STATIC. Between the two
                  is the partial tail.
  frac_t_min      frac_slope_from_t at or above which a head is "sourced from t". This
                  is a statement about WHERE a head's v-dependence comes from, and is
                  kept separate from how LARGE it is: an earlier version gated sourcing
                  on size, and six layer-1 detectors with frac_slope_from_t of 0.99-1.9
                  were labelled "sourced elsewhere" merely for being small.
  t_share_min     a head carries material TRANSPORT if its |slope at t| is at least this
                  share of the regime's total |slope at t| mass.
  total_share_min a head carries material SIGNAL overall if its |slope| is at least
                  this share of the regime's total |slope| mass.

  Shares of the regime's total, not fractions of the top head: when one head dominates
  (L1h31 holds 34% of the at-t mass at x10) a fraction-of-max cut is set by that one
  head and moves with it. Measured on joint_N0.1377_POS-12, the two quantities have
  very different shapes -- at-t transport is CONCENTRATED (3 heads cover half the mass
  at x10, 8 cover 80%) while the total contribution is DIFFUSE (45-56 heads for half,
  139-170 for 80%) -- and the meaningful at-t heads sit at >=3% share (the one static
  transporter, L3h1, holds 5% locally) while the noise tail sits near 1%. That contrast
  is a result in its own right; report the effective head count alongside these
  classes rather than letting the cut hide it.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

HEAD_CLASSES = ("threshold_grabber", "threshold_other", "graded_transporter",
                "linear_transporter", "re_reader", "inert")
# the gap measured on joint_N0.1377_POS-12 runs from 0.27 to 0.53; 0.3 sits inside it
DEFAULT_ATT_TAU = 0.3
# below this, attention genuinely does not move over the sweep
DEFAULT_STATIC_TAU = 0.05
DEFAULT_FRAC_T_MIN = 0.5
DEFAULT_T_SHARE_MIN = 0.03
DEFAULT_TOTAL_SHARE_MIN = 0.01
_GLOBAL_RE = re.compile(r"^x(?P<scale>[0-9.]+)$")


def threshold_regime(heads_df: pd.DataFrame) -> Optional[str]:
    """The regime attention sensitivity is judged in: the global grid with the
    LARGEST scale, since a switch whose knee sits outside the natural range is only
    visible when the sweep reaches past it. Falls back to any regime if none is
    global (a local-only run cannot show a switch, and says so via a NaN delta)."""
    regimes = list(dict.fromkeys(heads_df["regime"].dropna()))
    scaled = [(float(m.group("scale")), r) for r in regimes
              if (m := _GLOBAL_RE.match(str(r)))]
    if scaled:
        return max(scaled)[1]
    return regimes[0] if regimes else None


def static_targets(profile_all: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Per head: where it looks with nothing injected, EXCLUDING BOS.

    BOS is Llama's attention sink and wins the raw argmax for ~31 of 32 heads at
    every depth, which makes the unfiltered label nearly constant. The sink share is
    kept as its own column so a head that is 99% sink stays visible rather than being
    silently relabelled by its 1% remainder.
    """
    cols = ["layer", "head", "static_target", "static_target_share", "attn_to_bos"]
    if profile_all is None or profile_all.empty:
        return pd.DataFrame(columns=cols)
    p = profile_all
    if "condition" in p:
        p = p[p["condition"] == "endogenous"]
    if "adapter" in p:
        p = p[p["adapter"] == "on"]
    bos = (p[p["token_bin"] == "BOS"].set_index(["layer", "head"])["mean_attention"]
           .rename("attn_to_bos"))
    nb = p[p["token_bin"] != "BOS"]
    idx = nb.groupby(["layer", "head"])["mean_attention"].idxmax()
    top = nb.loc[idx, ["layer", "head", "token_bin", "mean_attention"]].rename(
        columns={"token_bin": "static_target", "mean_attention": "static_target_share"})
    return top.merge(bos.reset_index(), on=["layer", "head"], how="left")[cols]


def classify_heads(heads_df: pd.DataFrame,
                   profile_all: Optional[pd.DataFrame] = None,
                   att_tau: float = DEFAULT_ATT_TAU,
                   static_tau: float = DEFAULT_STATIC_TAU,
                   frac_t_min: float = DEFAULT_FRAC_T_MIN,
                   t_share_min: float = DEFAULT_T_SHARE_MIN,
                   total_share_min: float = DEFAULT_TOTAL_SHARE_MIN) -> pd.DataFrame:
    """
    One row per (layer, head) with its class and every quantity the class was
    decided from.

    `heads_df` is a run's mover_heads.csv (one row per head per regime).
    `profile_all` is its mover_attention_profile_all_heads.csv (optional; supplies the
    static target).
    """
    h = heads_df.copy()
    thr_regime = threshold_regime(h)

    # ---- attention sensitivity: a property of the head's attention function ----
    if thr_regime is not None and "attn_t_delta" in h:
        att = (h[h["regime"] == thr_regime]
               .set_index(["layer", "head"])[["attn_t_delta", "attn_t_ratio",
                                              "attn_t_at_centre", "attn_t_max"]])
    else:
        att = pd.DataFrame(columns=["attn_t_delta", "attn_t_ratio",
                                    "attn_t_at_centre", "attn_t_max"])

    # ---- materiality and sourcing, per regime; a head qualifies if it does in ANY ----
    # regime -- the circuit differs by regime, and a head that carries the signal only
    # at the operating point is exactly the one that must not be called inert
    per_regime = []
    for rg, g in h.groupby("regime"):
        g = g.copy()
        sum_t, sum_tot = g["mean_abs_slope_at_t"].sum(), g["mean_abs_slope"].sum()
        g["share_t"] = g["mean_abs_slope_at_t"] / sum_t if sum_t > 0 else 0.0
        g["share_total"] = g["mean_abs_slope"] / sum_tot if sum_tot > 0 else 0.0
        g["sourced_t"] = g["frac_slope_from_t"] >= frac_t_min          # WHERE from
        g["material_t"] = g["share_t"] >= t_share_min                    # HOW MUCH
        g["material_total"] = g["share_total"] >= total_share_min
        g["transports"] = g["sourced_t"] & g["material_t"]
        per_regime.append(g[["layer", "head", "regime", "sourced_t", "material_t",
                             "material_total", "transports", "share_t", "share_total",
                             "rank", "rank_at_t", "frac_slope_from_t"]])
    pr = pd.concat(per_regime, ignore_index=True)
    agg = pr.groupby(["layer", "head"]).agg(
        transports=("transports", "any"), material_total=("material_total", "any"),
        best_rank=("rank", "min"), best_rank_at_t=("rank_at_t", "min"),
        max_frac_slope_from_t=("frac_slope_from_t", "max"),
        max_share_t=("share_t", "max"), max_share_total=("share_total", "max"),
    )
    # which regimes a head transports in -- a transporter only at the operating point
    # is a different finding from one only at 10x
    agg["regimes_transporting"] = (pr[pr["transports"]].groupby(["layer", "head"])["regime"]
                                   .agg(lambda s: ",".join(sorted(s.unique())))
                                   .reindex(agg.index).fillna(""))
    # sourcing in the THRESHOLD regime specifically: a switching head is classified by
    # where its v-dependence comes from in the regime where it actually switches
    if thr_regime is not None:
        agg["sourced_t_in_threshold_regime"] = (
            pr[pr["regime"] == thr_regime].set_index(["layer", "head"])["sourced_t"]
            .reindex(agg.index).fillna(False))
    else:
        agg["sourced_t_in_threshold_regime"] = False

    out = agg.join(att, how="left").reset_index()
    rel = h.drop_duplicates(["layer", "head"]).set_index(["layer", "head"])
    if "rel_depth" in rel:
        out = out.merge(rel[["rel_depth"]].reset_index(), on=["layer", "head"], how="left")

    delta = out["attn_t_delta"].fillna(0.0)
    switching = delta >= att_tau
    static = delta < static_tau
    partial = ~switching & ~static
    src_thr = out["sourced_t_in_threshold_regime"].astype(bool)
    tr = out["transports"].astype(bool)
    mat = out["material_total"].astype(bool)
    cls = np.select(
        # a switch is notable at any size, so switching heads are split by SOURCING
        # alone; everything else must first carry a material share to be classified
        [switching & src_thr, switching & ~src_thr,
         partial & tr, static & tr,
         ~switching & ~tr & mat],
        ["threshold_grabber", "threshold_other", "graded_transporter",
         "linear_transporter", "re_reader"],
        default="inert")
    out["head_class"] = cls
    out["attention_state"] = np.select([switching, static], ["switching", "static"],
                                       default="partial")
    out["threshold_regime"] = thr_regime
    out["att_tau"], out["static_tau"] = att_tau, static_tau
    out["frac_t_min"], out["t_share_min"] = frac_t_min, t_share_min
    out["total_share_min"] = total_share_min

    st = static_targets(profile_all)
    if not st.empty:
        out = out.merge(st, on=["layer", "head"], how="left")
    return out.sort_values(["head_class", "best_rank"]).reset_index(drop=True)


def class_members(classified: pd.DataFrame, classes) -> list[tuple[int, int]]:
    """(layer, head) pairs belonging to any of `classes`."""
    if isinstance(classes, str):
        classes = [classes]
    sub = classified[classified["head_class"].isin(classes)]
    return [(int(L), int(h)) for L, h in zip(sub["layer"], sub["head"])]


# ======================================================================
# Gating shape of an attention-vs-injection curve
# ======================================================================
GATING_SHAPES = ("not gating", "1-way", "2-way")


def gating_shape(v, attn, tau: float = DEFAULT_STATIC_TAU) -> dict:
    """
    Classify how a head's attention to t responds to the injected value, from its
    (v, attention) curve over one regime's grid (normally x10, where the injection is
    anomalous).

    Measured as RISES relative to the grid point nearest the curve's centre (the
    neuron's own mean for a global grid), separately on each side:
        rise_lo = max attention at v < centre  - attention at centre
        rise_hi = max attention at v > centre  - attention at centre
    and a side GATES when its rise is >= tau:
        not gating   neither side      (flat: attention does not respond)
        1-way        one side          (L-shaped / sigmoid: switches on for one sign)
        2-way        both sides        (U-shaped: switches on for any large |v|)

    Rises only, as in attn_t_delta: a monotonic sigmoid sits partway up at the centre,
    so it DROPS on one side and rises on the other -- counting any change would call
    it 2-way. The default tau is the taxonomy's static threshold, i.e. "not gating"
    means the same thing as "static" there.
    """
    v, attn = np.asarray(v, float), np.asarray(attn, float)
    order = np.argsort(v)
    v, attn = v[order], attn[order]
    ci = int(np.argmin(np.abs(v - np.median(v))))
    a_c = attn[ci]
    rise_lo = float((attn[:ci] - a_c).max()) if ci > 0 else 0.0
    rise_hi = float((attn[ci + 1:] - a_c).max()) if ci < len(v) - 1 else 0.0
    n_sides = int(rise_lo >= tau) + int(rise_hi >= tau)
    return {"attn_at_centre": float(a_c), "rise_lo": rise_lo, "rise_hi": rise_hi,
            "shape": GATING_SHAPES[n_sides]}
