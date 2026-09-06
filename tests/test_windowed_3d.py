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


def test_build_subproblem_3d_geometry_and_patch_identity():
    from dvfopt.jacobian.tetrahedron_sign import six_tet_volumes_3d

    rng = np.random.default_rng(2)
    phi = rng.normal(0, 0.25, (3, 9, 10, 11))
    c = SimplexConstraint3D(shape=phi.shape[1:])
    box = (2, 5, 3, 7, 4, 8)
    sub = build_subproblem(c, phi, box, THR, NoneObjective(), 1e-3)
    assert sub.patch_box == (1, 6, 2, 8, 3, 9)  # ring 1 on every side (interior box)
    assert sub.free_mask.shape == (5, 6, 6)
    assert sub.free_mask.sum() == 3 * 4 * 4 and sub.free_mask[1:4, 1:5, 1:5].all()
    n = sub.free_mask.size
    assert sub.free_idx.size == 3 * 48  # every free voxel in each of the dx / dy / dz blocks
    assert set((sub.free_idx // n).tolist()) == {0, 1, 2}
    # every cube of this patch has a free corner -> all 6 * cells rows are enforced, and
    # their values at flat0 are the GLOBAL tet volumes (exact cells: no interior-cut mismatch)
    m = 4 * 5 * 5
    assert sub.n_enforced == 6 * m
    vals = (sub.cons(sub.flat0) + (THR + 1e-3)).reshape(6, 4, 5, 5)
    np.testing.assert_allclose(vals, six_tet_volumes_3d(phi)[:, 1:5, 2:7, 3:8], atol=1e-12)
    J = sub.cons_jac(sub.flat0)
    assert J.shape == (sub.n_enforced, sub.flat0.size)


def test_build_subproblem_3d_free_extra_and_volume_border():
    phi = np.zeros((3, 9, 10, 11))
    c = SimplexConstraint3D(shape=phi.shape[1:])
    fe = np.zeros(phi.shape[1:], bool)
    fe[3, 5, 6] = True
    sub = build_subproblem(c, phi, (2, 5, 3, 7, 4, 8), THR, None, 1e-3, free_extra=fe)
    assert sub.free_idx.size == 3 and sub.n_enforced == 6 * 8  # one voxel: its 8 cubes
    sub = build_subproblem(c, phi, (0, 3, 3, 7, 4, 8), THR, None, 1e-3)
    assert sub.patch_box == (0, 4, 2, 8, 3, 9)  # no ring past the z=0 face
    assert sub.free_mask[0:3, 1:5, 1:5].all() and sub.free_mask.sum() == 3 * 4 * 4


def test_rows_3d_identity_equals_one_minus_delta_and_counts_every_edge():
    from dvfopt.core.windowed._common import _orientation_rows_3d

    c = SimplexConstraint3D(shape=(4, 4, 4))
    a, b = _orientation_rows_3d(c, np.ones((4, 4, 4), bool), 0.01)
    assert a.shape == (3 * 48, 3 * 64)  # 48 edges per axis on a 4^3 grid
    np.testing.assert_allclose(a @ np.zeros(3 * 64) + b, 0.99)


def test_rows_3d_keep_free_to_frozen_edges_and_drop_frozen_frozen():
    from dvfopt.core.slsqp_windowed.constraints3d import _injectivity_linear_constraint_3d
    from dvfopt.core.windowed._common import _orientation_rows_3d

    c = SimplexConstraint3D(shape=(3, 3, 3))
    fm = np.zeros((3, 3, 3), bool)
    fm[1, 1, 1] = True
    a, _b = _orientation_rows_3d(c, fm, 0.01)
    assert a.shape[0] == 6  # the centre voxel's six axial edges, each free-to-frozen
    n, centre = 27, 13
    for r in range(6):
        assert {centre, n + centre, 2 * n + centre} & set(a[r].indices.tolist())
    # the 3D injectivity helper's own filter (BOTH endpoints free) would keep none of them
    assert _injectivity_linear_constraint_3d((3, 3, 3), 0.01, freeze_mask=~fm) is None


def test_rows_3d_rotated_cube_violates_and_compressed_cube_does_not():
    from dvfopt.core.windowed._common import _orientation_rows_3d

    # SimplexConstraint3D(shape=(2, 2, 2)).flatten() would hit validate_dvf's
    # dimension-agnostic min_spatial_size=3 floor (unrelated to orientation rows,
    # and shared with the 2D simplex family — see dvfopt/validation.py) before
    # ever reaching the pack, so the DX_FIRST [dx | dy | dz] vector is built
    # directly here, matching SimplexConstraint3D.flatten()'s own body byte for byte.
    def pack(phi):
        return np.concatenate([phi[2].ravel(), phi[1].ravel(), phi[0].ravel()])

    c = SimplexConstraint3D(shape=(2, 2, 2))
    a, b = _orientation_rows_3d(c, np.ones((2, 2, 2), bool), 0.01)
    phi = np.zeros((3, 2, 2, 2))
    phi[2, :, :, 0], phi[2, :, :, 1] = 1.0, -1.0  # x-edges flipped: deformed x goes 1 -> 0
    assert (a @ pack(phi) + b).min() < 0
    phi = np.zeros((3, 2, 2, 2))
    phi[2, :, :, 1] = phi[1, :, 1, :] = phi[0, 1] = -0.5  # 50 % compression on every axis
    assert (a @ pack(phi) + b).min() > 0


def test_subproblem_3d_with_rows_jacobian_matches_finite_differences():
    rng = np.random.default_rng(3)
    phi = rng.normal(0, 0.2, (3, 5, 5, 6))
    c = SimplexConstraint3D(shape=phi.shape[1:])
    sub = build_subproblem(
        c,
        phi,
        (1, 4, 1, 4, 1, 5),
        THR,
        NoneObjective(),
        1e-3,
        orientation_delta=0.01,
        orientation_rows='edges',
    )
    assert sub.n_enforced > 6 * 4 * 4 * 5  # tet rows plus the edge rows
    J = sub.cons_jac(sub.flat0).toarray()
    assert J.shape == (sub.n_enforced, sub.flat0.size)
    for k in rng.choice(sub.free_idx, 10, replace=False):
        e = np.zeros_like(sub.flat0)
        e[k] = 1e-6
        fd = (sub.cons(sub.flat0 + e) - sub.cons(sub.flat0 - e)) / 2e-6
        np.testing.assert_allclose(J[:, k], fd, atol=1e-6)


def test_subproblem_3d_rejects_the_full_rows_kind():
    c = SimplexConstraint3D(shape=(4, 4, 4))
    with pytest.raises(ValueError, match='edges'):
        build_subproblem(
            c,
            np.zeros((3, 4, 4, 4)),
            (1, 3, 1, 3, 1, 3),
            THR,
            None,
            1e-3,
            orientation_delta=0.01,
            orientation_rows='full',
        )
