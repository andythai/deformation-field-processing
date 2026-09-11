"""Composite 3D strategy: the ``m10_3d`` wallbreaker, then the no-damage windowed engine on
its residual. The #122 head-to-head measured ``m10_3d`` certifying 9 of 12 artefacts at
3-11x less wall than the windowed engine and usually closer to the input, leaving 1-3
residual folds where it failed; the windowed engine certified 12/12 from raw. Chaining
them gives the certificate at roughly the wallbreaker's cost (CHANGELOG, "3D composite").
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

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
    is measured against the wallbreaker's output, not the input). The composite has no
    knobs of its own — tune the stages.

    Both stages always run with ``record_history=True`` (the merged
    :class:`~dvfopt.solver.SolveInfo` is cheap and the ``bulk_*`` / ``damage``
    extras are the point of the composite), whatever the caller passed.
    """

    bulk: HarmonicALMBarrier3DStrategy = field(default_factory=HarmonicALMBarrier3DStrategy)
    windowed: ISQPWindowedStrategy = field(default_factory=ISQPWindowedStrategy)

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
        phi2, info2 = self.windowed.solve(
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
        n_bulk = len(info1.phases)
        feasible_after = (
            info2.feasible_after_phase + n_bulk
            if info2.feasible_after_phase >= 0
            else info1.feasible_after_phase
        )
        extras = dict(info2.extras)
        extras.update(
            bulk_n_neg_after=bulk_stats.n_below,
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
    """``p`` with ``prefix`` prepended to its name (a new :class:`PhaseInfo`)."""
    return replace(p, name=prefix + p.name)


M10WindowedTetStrategy = HarmonicALMBarrierWindowed3DStrategy

__all__ = ['HarmonicALMBarrierWindowed3DStrategy', 'M10WindowedTetStrategy']
