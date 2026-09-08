"""Phase 2 of the 3D port: the engine's stages (coarse warm start, giant tiler, mop,
harmonic re-seed, re-anchor, polish) on (3, D, H, W) fields. Additive: the 2D stages
are covered by their own suites and by benchmarks/windowed_2d_identity.py.
"""

import numpy as np
import pytest

from dvfopt.constraints import SimplexConstraint3D
from dvfopt.core.primitives import isqp as isqp_mod
from dvfopt.core.windowed import _common as engine
from dvfopt.core.windowed import min_field, pixel_fold_mask, windowed_correct
from dvfopt.objectives import L2Objective, NoneObjective

needs_osqp = pytest.mark.skipif(not isqp_mod.HAS_OSQP, reason="osqp not installed")
THR = 0.01


def _blob(shape, centre, size, amp, seed=0):
    """Identity field with one random blob (a (3, *size) normal patch) at ``centre``."""
    rng = np.random.default_rng(seed)
    phi = np.zeros((3, *shape))
    sl = tuple(slice(c - s // 2, c - s // 2 + s) for c, s in zip(centre, size))
    phi[(slice(None), *sl)] = rng.normal(0, amp, (3, *size))
    return phi


def test_restrict_3d_rescales_to_coarse_units_and_drops_partial_blocks():
    phi = np.zeros((3, 8, 9, 10))
    phi[0], phi[1], phi[2] = 4.0, -2.0, 1.0
    coarse = engine._restrict(phi, 2)
    assert coarse.shape == (3, 4, 4, 5)
    assert (
        np.allclose(coarse[0], 2.0) and np.allclose(coarse[1], -1.0) and np.allclose(coarse[2], 0.5)
    )


def test_restrict_2d_is_unchanged():
    phi = np.zeros((2, 8, 8))
    phi[0], phi[1] = 4.0, -2.0
    coarse = engine._restrict(phi, 2)
    assert (
        coarse.shape == (2, 4, 4) and np.allclose(coarse[0], 2.0) and np.allclose(coarse[1], -1.0)
    )
    assert engine._restrict(np.ones((2, 7, 9)), 2).shape == (2, 3, 4)


def test_prolongate_3d_rescales_back_and_zero_pads_odd_dims():
    delta = engine._prolongate(np.ones((3, 3, 4, 2)), (7, 9, 5), 2)
    assert delta.shape == (3, 7, 9, 5)
    assert np.allclose(delta[:, :6, :8, :4], 2.0)
    assert (
        np.all(delta[:, 6] == 0.0)
        and np.all(delta[:, :, 8] == 0.0)
        and np.all(delta[:, :, :, 4] == 0.0)
    )


def test_restrict_prolongate_round_trip_3d_is_identity_on_a_constant_field():
    phi = np.full((3, 8, 8, 8), 3.0)
    assert np.allclose(engine._prolongate(engine._restrict(phi, 2), (8, 8, 8), 2), phi)


def test_inner_opts_resolve_the_3d_tile():
    from dvfopt.core.windowed._common import _InnerOpts

    o = _InnerOpts()
    assert (
        o.giant_tile == 64 and o.giant_tile_3d == 16
    )  # 2D default untouched; 16^3 ~= 64^2 by count


@needs_osqp
def test_coarse_warm_start_runs_on_3d_and_keeps_healthy_area_byte_identical():
    # min(shape) >= 4 * giant_tile_3d (= 64) -> the stage fires; coarse_factor 4 -> a 16^3 coarse problem
    phi = _blob((64, 64, 64), (32, 32, 32), (4, 6, 6), amp=0.5)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    assert (min_field(c, phi) < THR).any()
    out, rep = windowed_correct(
        phi.copy(), "isqp", constraint=c, objective=NoneObjective(), threshold=THR, verbose=0
    )
    assert rep.coarse_folds_before >= 0, "the coarse stage must have run"
    assert rep.folds_after == 0 and rep.damage == 0
    assert np.array_equal(out[:, :16], phi[:, :16]) and np.array_equal(
        out[:, :, :, 48:], phi[:, :, :, 48:]
    )


@needs_osqp
def test_strategy_forwards_the_3d_knobs(monkeypatch):
    from dvfopt import ISQPWindowedStrategy
    from dvfopt.strategies import windowed as strat

    seen = {}

    def fake(phi, inner, **kw):
        seen.update(kw)
        return np.asarray(phi, float), engine.SliceReport()

    monkeypatch.setattr(strat, "windowed_correct", fake)
    s = ISQPWindowedStrategy(giant_tile_3d=12, mop_margin_3d=4)
    s.solve(
        np.zeros((3, 6, 8, 8)),
        constraint=SimplexConstraint3D(shape=(6, 8, 8)),
        objective=NoneObjective(),
        threshold=THR,
    )
    assert seen["giant_tile_3d"] == 12 and seen["mop_margin_3d"] == 4
    assert seen["giant_tile"] == 64 and seen["mop_margin"] == 25  # the 2D knobs are not repurposed
