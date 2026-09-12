"""Composite 3D strategy: the ``m10_3d`` wallbreaker, then the no-damage windowed engine on
its residual. The #122 head-to-head measured ``m10_3d`` certifying 9 of 12 artefacts at
3-11x less wall than the windowed engine and usually closer to the input, leaving 1-3
residual folds where it failed; the windowed engine certified 12/12 from raw. Chaining
them gives the certificate at roughly the wallbreaker's cost (CHANGELOG, "3D composite").
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Optional

from dvfopt.constraints import SimplexConstraint3D
from dvfopt.metrics import fold_stats
from dvfopt.objectives import L1Objective, L2Objective, NoneObjective
from dvfopt.strategies.base import Strategy, register_strategy
from dvfopt.strategies.wallbreakers import HarmonicALMBarrier3DStrategy
from dvfopt.strategies.windowed import ISQPWindowedStrategy


@register_strategy('harmonic_alm_barrier_windowed_3d')
@register_strategy('m10_windowed_3d')  # short alias, the head-to-head's column name
@dataclass
class HarmonicALMBarrierWindowed3DStrategy(Strategy):
    """``m10_3d`` (harmonic seed + PHR-ALM + barrier polish) then ``ISQPWindowedStrategy``
    on its output. Two stages, each a registered strategy with its own knobs:
    ``bulk`` (the wallbreaker) and ``windowed`` (the no-damage engine, whose ``damage``
    is measured against the wallbreaker's output, not the input). Apart from
    ``time_budget_s`` the composite has no knobs of its own — tune the stages.

    Both stages always run with ``record_history=True`` (the merged
    :class:`~dvfopt.solver.SolveInfo` is cheap and the ``bulk_*`` / ``damage``
    extras are the point of the composite), whatever the caller passed.

    Parameters
    ----------
    bulk, windowed : Strategy
        The two stages. Replace either with a differently-configured instance
        of the same class to tune it.
    time_budget_s : float, optional
        Wall-clock budget forwarded to the **windowed stage only** —
        ``HarmonicALMBarrier3DStrategy`` has no budget knob, so the bulk stage
        runs to its own termination whatever this is set to. Present so the
        GUI toolbar's budget (``SolverWorker._apply_time_budget``, which only
        sets strategies that expose the field) reaches the stage that can
        honour it.

    Extras
    ------
    On top of the windowed stage's extras (``damage``, ``n_windows``,
    ``l2_move``, ...) the merged :class:`~dvfopt.solver.SolveInfo` carries the
    residual the bulk stage handed over:

    ``bulk_n_neg_after``
        Rows at or below zero after the bulk stage — true folds, the
        package-wide ``n_neg`` definition.
    ``bulk_n_below_after``
        Rows below ``threshold`` after the bulk stage (strict-feasibility
        misses; ``>= bulk_n_neg_after``).
    ``bulk_min_after``
        Smallest row value after the bulk stage.
    ``bulk_wall_s``
        Wall-clock seconds the bulk stage took.
    ``bulk_extras``
        The bulk stage's own ``SolveInfo.extras``.
    """

    bulk: HarmonicALMBarrier3DStrategy = field(default_factory=HarmonicALMBarrier3DStrategy)
    windowed: ISQPWindowedStrategy = field(default_factory=ISQPWindowedStrategy)
    time_budget_s: Optional[float] = None

    accepts_constraints = (SimplexConstraint3D,)
    accepts_objectives = (L1Objective, L2Objective, NoneObjective)
    supports_3d = True

    def solve(
        self,
        phi_in,
        *,
        constraint,
        objective,
        threshold,
        verbose=0,
        record_history=False,
        step_callback=None,
        **_,
    ):
        from dvfopt.solver import SolveInfo

        self._check_constraint(constraint)
        t0 = time.perf_counter()
        phi1, info1 = self.bulk.solve(
            phi_in,
            constraint=constraint,
            objective=objective,
            threshold=threshold,
            verbose=verbose,
            record_history=True,
            step_callback=step_callback,
        )
        bulk_wall = time.perf_counter() - t0
        bulk_stats = fold_stats(constraint.values(constraint.flatten(phi1)), threshold)
        windowed = (
            self.windowed
            if self.time_budget_s is None
            else replace(self.windowed, time_budget_s=self.time_budget_s)
        )
        phi2, info2 = windowed.solve(
            phi1,
            constraint=constraint,
            objective=objective,
            threshold=threshold,
            verbose=verbose,
            record_history=True,
            step_callback=step_callback,
        )
        phases = [_prefixed(p, 'bulk:') for p in info1.phases] + [
            _prefixed(p, 'windowed:') for p in info2.phases
        ]
        # "The index where n_neg FIRST hit 0" — so a bulk stage that already
        # cleared the field wins over the windowed stage's (later) index.
        n_bulk = len(info1.phases)
        if info1.feasible_after_phase >= 0:
            feasible_after = info1.feasible_after_phase
        elif info2.feasible_after_phase >= 0:
            feasible_after = info2.feasible_after_phase + n_bulk
        else:
            feasible_after = -1
        extras = dict(info2.extras)
        extras.update(
            bulk_n_neg_after=bulk_stats.n_neg,
            bulk_n_below_after=bulk_stats.n_below,
            bulk_min_after=bulk_stats.min_val,
            bulk_wall_s=bulk_wall,
            bulk_extras=dict(info1.extras),
        )
        info = SolveInfo(
            strategy_name='m10_windowed_3d',
            phases=phases,
            total_iter=int(info1.total_iter + info2.total_iter),
            feasible_after_phase=feasible_after,
            extras=extras,
        )
        return phi2, info


def _prefixed(p, prefix: str):
    """``p`` with ``prefix`` prepended to its name (a new :class:`PhaseInfo`).

    ``dataclasses.replace`` is shallow, so ``extras`` is copied explicitly —
    the merged info must not alias the stage info's dicts.
    """
    return replace(p, name=prefix + p.name, extras=dict(p.extras))


M10WindowedTetStrategy = HarmonicALMBarrierWindowed3DStrategy

__all__ = ['HarmonicALMBarrierWindowed3DStrategy', 'M10WindowedTetStrategy']
