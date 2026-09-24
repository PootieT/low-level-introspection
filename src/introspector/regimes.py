"""
The injection regimes a circuit measurement can be taken in.

WHY THIS EXISTS
---------------
Attribution in this project is a dose-response: vary the value at the introspected
neuron, regress each component's contribution on it, and call the slope the
component's role. That answer depends on WHICH VALUES you sweep, and not weakly --
on joint_N0.1377_POS-12 the top mover head at 10x injection is rank 68 at 1x, and the
1x transporter falls to rank 455 at 10x. A measurement is therefore only meaningful
next to the regime it was taken in, so the regime is a first-class object that every
script carries into its outputs rather than an implicit property of a --scale flag.

THE REGIMES

  x<scale>   A GLOBAL grid: every example's neuron is overwritten with the SAME value,
             swept over mean +/- grid_std_mult * std * scale. `x1` matches the natural
             standard deviation and `x10` (or whatever the run trained at) is where the
             signal is well clear of the bf16 floor.

             What it measures is a CHORD: the average slope across the whole swept
             range. Note that "natural scale" here means matching the std, not staying
             where the data is -- at +/-3 sigma the endpoints sit in the extreme tails
             and can fall outside the observed support entirely (measured on N0.1377:
             the bottom x1 grid point is -0.2342 against an observed minimum of
             -0.2256, i.e. a value the neuron never takes, and the two grid points
             carrying nearly all of the attention response cover under 0.73% of
             examples). It also replaces each example's own value with a common one,
             so for an example in the upper tail a "1x" injection can still be a
             several-sigma swing.

  local      A PER-EXAMPLE grid: each example's neuron is swept over its OWN current
             value plus offsets delta in [-delta_max, +delta_max]. This measures the
             TANGENT at the operating point the model is actually in, and it never
             leaves the neighbourhood of a state that really occurs.

             Still causal: the slope uses only the within-example delta variation,
             which is independent of context by construction; the context-correlated
             part (the centre) lands in the intercept and does not enter the slope.

             Two properties fall out of centring on the live value:
               * delta = 0 writes the neuron's own value back, so the prediction must
                 reproduce the unperturbed forward EXACTLY. That is a stronger
                 alignment check than any offset scan (see identity_check).
               * the slope is regressed on delta, not on the absolute value. Slope is
                 invariant to shifting x per example, so this is the same estimator as
                 the global regimes, and downstream code needs no change.

  Global and local coincide only if the response is linear across the natural range.
  It is not -- attention to the introspected token is a saturating switch -- so their
  disagreement is itself the measurement of that nonlinearity. Keep both.

NOT INCLUDED: a no-injection "endogenous" regime. Attribution there cannot be a
dose-response at all (nothing is varied), and needs a different estimator -- either
correlational against the natural activation, which the context bypass confounds, or
a gradient, which score_gradient_attribution.py already provides. `local` is the
endogenous-OPERATING-POINT measurement that stays causal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

# delta_max for `local`, in units of the neuron's natural standard deviation. 1 sigma
# keeps essentially every example inside the observed support while staying as far
# above the bf16 floor as a local measurement can (see the R^2 guard in power_warning).
DEFAULT_LOCAL_DELTA_SIGMA = 1.0
DEFAULT_REGIMES = "x1,x10,local"
_X_RE = re.compile(r"^x(?P<scale>[0-9.]+)$")
# Below this, a dose-response is not a null result -- it is an underpowered one.
MIN_TRUSTWORTHY_R2 = 0.5


@dataclass
class Regime:
    """One injection regime, resolved against a particular neuron's statistics."""
    name: str
    kind: str                      # "global" | "local"
    grid: np.ndarray               # values (global) or offsets from the centre (local)
    scale: Optional[float] = None  # global only
    delta_sigma: Optional[float] = None  # local only

    @property
    def is_local(self) -> bool:
        return self.kind == "local"

    @property
    def regressor(self) -> np.ndarray:
        """The x the contributions are regressed on.

        For `local` this is the OFFSET, not the absolute value: the slope is identical
        either way (shifting x per example cannot change it), and the offset is shared
        across examples, so every downstream consumer -- the per-example line fits, the
        decodability probe's target -- keeps working unchanged.
        """
        return self.grid

    def values_for(self, centres: Optional[torch.Tensor], point: int) -> torch.Tensor | float:
        """The value(s) to write at grid `point`.

        Global: one scalar for the whole batch. Local: `centres` (this batch's live
        activations at the introspected locale) plus the offset, one value per row.
        """
        if not self.is_local:
            return float(self.grid[point])
        if centres is None:
            raise ValueError("the `local` regime needs per-example centres")
        return centres.float() + float(self.grid[point])

    def describe(self, stats: dict) -> str:
        if self.is_local:
            return (f"{self.name}: per-example centre +/- {self.grid[-1]:.4f} "
                    f"({self.delta_sigma:g} sigma), {len(self.grid)} offsets")
        return (f"{self.name}: global grid [{self.grid[0]:.4f}, {self.grid[-1]:.4f}] "
                f"({self.scale:g}x, neuron mean={stats['mean']:.4f} std={stats['std']:.4f})")


def parse_regimes(spec: str, stats: dict, grid_points: int, grid_std_mult: float,
                  delta_sigma: float = DEFAULT_LOCAL_DELTA_SIGMA,
                  train_scale: Optional[float] = None) -> list[Regime]:
    """
    'x1,x10,local' -> [Regime, ...], resolved against this neuron's stats.

    Accepts `xN` for any N (`xtrain` resolves to the run's own training scale) and
    `local` (optionally `local@0.5` to override delta in sigma units).
    """
    out: list[Regime] = []
    for tok in (t.strip().lower() for t in spec.split(",")):
        if not tok:
            continue
        if tok.startswith("local"):
            d = delta_sigma
            if "@" in tok:
                d = float(tok.split("@", 1)[1])
            half = d * stats["std"]
            out.append(Regime(name=f"local@{d:g}s", kind="local", delta_sigma=d,
                              grid=np.linspace(-half, half, grid_points)))
            continue
        if tok == "xtrain":
            if train_scale is None:
                raise SystemExit("'xtrain' needs the run's training scale")
            tok = f"x{train_scale:g}"
        m = _X_RE.match(tok)
        if not m:
            raise SystemExit(f"Unknown regime {tok!r}; expected 'xN', 'xtrain' or 'local[@sigma]'.")
        sc = float(m.group("scale"))
        half = grid_std_mult * stats["std"] * sc
        out.append(Regime(name=f"x{sc:g}", kind="global", scale=sc,
                          grid=np.linspace(stats["mean"] - half, stats["mean"] + half,
                                           grid_points)))
    if not out:
        raise SystemExit("--regimes selected nothing.")
    # de-duplicate by name, preserving order
    seen, uniq = set(), []
    for r in out:
        if r.name not in seen:
            seen.add(r.name)
            uniq.append(r)
    return uniq


def zero_offset_index(regime: Regime) -> Optional[int]:
    """Index of the delta==0 point of a local grid, if the grid contains one."""
    if not regime.is_local:
        return None
    k = int(np.argmin(np.abs(regime.grid)))
    return k if abs(float(regime.grid[k])) < 1e-12 else None


def identity_check(regime: Regime, preds_at_zero: np.ndarray, preds_unperturbed: np.ndarray,
                   label: str = "") -> Optional[float]:
    """
    For `local`: writing the neuron's own value back must be a no-op.

    Returns the max absolute discrepancy, or None when the regime has no delta==0
    point. A non-zero result means the captured centres are not the values the model
    actually computes -- wrong row order, wrong position, or centres taken from the
    cache while the live neuron has drifted -- and every slope in this regime is then
    measured around the wrong point.
    """
    if not regime.is_local:
        return None
    err = float(np.abs(np.asarray(preds_at_zero) - np.asarray(preds_unperturbed)).max())
    tag = f" [{label}]" if label else ""
    if err > 1e-3:
        print(f"[regime] *** {regime.name}{tag}: delta=0 changed the prediction by up to "
              f"{err:.3e}. Re-injecting the neuron's own value must be a no-op, so the "
              f"per-example centres do NOT match the live activations. Slopes in this "
              f"regime are measured around the wrong point. ***")
    else:
        print(f"[regime] {regime.name}{tag}: delta=0 identity check passed "
              f"(max |dpred| = {err:.2e})")
    return err


def power_warning(regime: Regime, mean_r2: float, label: str = "") -> bool:
    """
    Flag a dose-response that is at the numerical floor rather than genuinely flat.

    A deterministic linear response gives R^2 ~ 1. When the swept range is small enough
    that the readout moves by less than bf16 can represent, the "response" is
    quantization noise and R^2 collapses -- which looks exactly like a null result but
    is not one. Natural-scale signals in this project really do sit near that floor
    (it is why the runs train at 10x), so the distinction has to be made explicitly.
    """
    if not np.isfinite(mean_r2) or mean_r2 >= MIN_TRUSTWORTHY_R2:
        return False
    tag = f" [{label}]" if label else ""
    print(f"[regime] WARNING {regime.name}{tag}: mean R^2 = {mean_r2:.3f} over the "
          f"dose-response. The slope is not resolved above the numerical floor -- read "
          f"this as UNDERPOWERED, not as 'no effect'. Widen the sweep or run in fp32.")
    return True
