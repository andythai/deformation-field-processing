"""The composite 3D strategy: m10_3d (harmonic + ALM + barrier polish) then the no-damage
windowed engine on its residual — the fast certificate the #122 head-to-head pointed at."""

from __future__ import annotations

import numpy as np
import pytest

from dvfopt import SimplexConstraint3D, Solver
from dvfopt.core.primitives import isqp as isqp_mod
from dvfopt.objectives import L2Objective
from dvfopt.strategies.base import _STRATEGY_REGISTRY, make_strategy
from tests.conftest import planted_fold_3d

needs_osqp = pytest.mark.skipif(not isqp_mod.HAS_OSQP, reason='osqp not installed')
THR = 0.01


def test_registered_under_both_labels_and_exported():
    import dvfopt
    from dvfopt.strategies.composite3d import HarmonicALMBarrierWindowed3DStrategy

    assert (
        _STRATEGY_REGISTRY['harmonic_alm_barrier_windowed_3d']
        is HarmonicALMBarrierWindowed3DStrategy
    )
    assert _STRATEGY_REGISTRY['m10_windowed_3d'] is HarmonicALMBarrierWindowed3DStrategy
    assert dvfopt.HarmonicALMBarrierWindowed3DStrategy is HarmonicALMBarrierWindowed3DStrategy
    assert dvfopt.M10WindowedTetStrategy is HarmonicALMBarrierWindowed3DStrategy
    s = make_strategy('m10_windowed_3d')
    assert isinstance(s, HarmonicALMBarrierWindowed3DStrategy)
    assert s.supports_3d and s.accepts_constraints == (SimplexConstraint3D,)


@needs_osqp
def test_composite_certifies_a_planted_fold_and_reports_both_stages():
    phi = planted_fold_3d(6, 10, 10, depth=1.4)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    res = Solver(
        constraint=c,
        objective=L2Objective(),
        strategy=make_strategy('m10_windowed_3d'),
        threshold=THR,
    ).fit(phi)
    assert res.feasible
    info = res.info
    names = [p.name for p in info.phases]
    assert any(n.startswith('bulk:') for n in names) and any(
        n.startswith('windowed:') for n in names
    )
    assert names.index(next(n for n in names if n.startswith('windowed:'))) > names.index(
        next(n for n in names if n.startswith('bulk:'))
    )
    assert info.total_iter == sum(p.n_iter for p in info.phases)
    assert 'bulk_n_neg_after' in info.extras
    assert 'bulk_wall_s' in info.extras
    assert 'damage' in info.extras


@needs_osqp
def test_windowed_stage_receives_the_bulk_output(monkeypatch):
    import dvfopt.strategies.composite3d as mod

    seen = {}
    real = mod.ISQPWindowedStrategy.solve

    def spy(self, phi_in, **kw):
        seen['phi_in'] = np.asarray(phi_in).copy()
        return real(self, phi_in, **kw)

    monkeypatch.setattr(mod.ISQPWindowedStrategy, 'solve', spy)
    phi = planted_fold_3d(6, 10, 10, depth=1.4)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    strat = make_strategy('m10_windowed_3d')
    bulk_out, _ = strat.bulk.solve(
        phi.copy(), constraint=c, objective=L2Objective(), threshold=THR, verbose=0
    )
    Solver(constraint=c, objective=L2Objective(), strategy=strat, threshold=THR).fit(phi)
    assert np.allclose(seen['phi_in'], bulk_out)  # the windowed stage starts from m10_3d's output


def test_solver_rejects_a_2d_constraint():
    from dvfopt import SimplexConstraint2D
    from dvfopt.exceptions import IncompatibleConstraintError

    with pytest.raises(IncompatibleConstraintError):
        Solver(
            constraint=SimplexConstraint2D(shape=(8, 8)),
            objective=L2Objective(),
            strategy=make_strategy('m10_windowed_3d'),
        )
