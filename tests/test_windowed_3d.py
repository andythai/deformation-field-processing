"""3D (SimplexConstraint3D) path of the windowed engine — phase 1 of the 3D port.

Every test here is additive: the 2D families are covered by the existing
windowed suites and by benchmarks/windowed_2d_identity.py (byte-identity).
"""

import numpy as np
import pytest

from dvfopt.constraints import SimplexConstraint3D
from dvfopt.core.primitives import isqp as isqp_mod
from dvfopt.core.windowed import (
    LOCALITY,
    build_subproblem,
    find_windows,
    min_field,
    windowed_correct,
)
from dvfopt.objectives import L2Objective, NoneObjective

needs_osqp = pytest.mark.skipif(not isqp_mod.HAS_OSQP, reason="osqp not installed")

THR = 0.01


def test_box_helpers_are_dimension_agnostic():
    from dvfopt.core.windowed._common import _box_size, _box_slices, _pad_box

    assert _box_slices((1, 4, 2, 6)) == (slice(1, 4), slice(2, 6))
    assert _box_slices((0, 2, 1, 4, 2, 6)) == (slice(0, 2), slice(1, 4), slice(2, 6))
    assert _box_size((1, 4, 2, 6)) == 12
    assert _box_size((0, 2, 1, 4, 2, 6)) == 24
    assert _pad_box((1, 4, 2, 6), (5, 8), 2) == (0, 5, 0, 8)
    assert _pad_box((2, 4, 3, 5, 4, 6), (10, 10, 10), 1) == (1, 5, 2, 6, 3, 7)


def test_find_windows_3d_boxes_and_border_rule():
    mask = np.zeros((10, 12, 14), bool)
    mask[4, 5, 6] = True  # interior fold
    mask[0, 1, 1] = True  # fold on the z=0 face (and near the y/x=0 faces)
    boxes = sorted(find_windows(mask, margin=2, ring=1))
    # dilation by margin+ring=3 is an L1 ball; the bbox is inset by ring except on a
    # side that reached the volume border, where the fold must stay free
    assert boxes == [(0, 3, 0, 4, 0, 4), (2, 7, 3, 8, 4, 9)]


def test_find_windows_2d_is_unchanged():
    mask = np.zeros((20, 20), bool)
    mask[10, 10] = True
    assert find_windows(mask, margin=3, ring=1) == [(7, 14, 7, 14)]


def test_locality_registers_tet3d_with_ring_one():
    assert LOCALITY[SimplexConstraint3D].ring == 1


def test_min_field_tet3d_matches_values_and_pads_inf():
    rng = np.random.default_rng(0)
    phi = rng.normal(0, 0.3, (3, 5, 6, 7))
    c = SimplexConstraint3D(shape=phi.shape[1:])
    m = min_field(c, phi)
    assert m.shape == (5, 6, 7)
    ref = np.asarray(c.values(c.flatten(phi))).reshape(6, 4, 5, 6).min(0)
    np.testing.assert_allclose(m[:4, :5, :6], ref, atol=1e-12)
    assert np.isinf(m[4]).all() and np.isinf(m[:, 5]).all() and np.isinf(m[:, :, 6]).all()


def test_influenced_tet3d_is_the_eight_corner_rule():
    c = SimplexConstraint3D(shape=(5, 5, 5))
    fm = np.zeros((5, 5, 5), bool)
    fm[2, 2, 2] = True
    idx, _jac_of = LOCALITY[SimplexConstraint3D].influenced(c, fm, 5, 5, 5, (False,) * 6)
    m = 4 * 4 * 4
    expect = sorted((z * 4 + y) * 4 + x for z in (1, 2) for y in (1, 2) for x in (1, 2))
    assert sorted(set(int(i) % m for i in idx)) == expect  # the 8 cubes cornered by the voxel
    assert sorted(idx.tolist()) == sorted(b * m + cf for b in range(6) for cf in expect)  # x 6 tets


def test_tet3d_patch_jacobian_matches_finite_differences():
    rng = np.random.default_rng(1)
    phi = rng.normal(0, 0.2, (3, 4, 4, 5))
    c = SimplexConstraint3D(shape=phi.shape[1:])
    f0 = np.asarray(c.flatten(phi), dtype=np.float64)
    fm = np.ones(phi.shape[1:], bool)
    _idx, jac_of = LOCALITY[SimplexConstraint3D].influenced(c, fm, 4, 4, 5, (True,) * 6)
    J = jac_of(f0)
    assert J.shape == (c.n_constraints, c.n_variables)
    Jd = J.toarray()
    eps = 1e-6
    for k in rng.choice(f0.size, 12, replace=False):
        e = np.zeros_like(f0)
        e[k] = eps
        fd = (np.asarray(c.values(f0 + e)) - np.asarray(c.values(f0 - e))) / (2 * eps)
        np.testing.assert_allclose(Jd[:, k], fd, atol=1e-6)
