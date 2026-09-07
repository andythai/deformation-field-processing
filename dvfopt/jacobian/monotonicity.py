"""Monotonicity (global injectivity) helpers for deformation fields."""

import numpy as np
import scipy.sparse

from dvfopt._defaults import _unpack_size, _unpack_size_3d


def axial_gap_matrix(shape, free=None, endpoints='both'):
    """Sparse ``A`` (csr) whose rows are the axial neighbour gaps of a ``(D, H, W)`` grid in
    the DX_FIRST flat pack ``[dx | dy | dz]``: one row per neighbouring voxel pair along
    each axis (x-gaps read the dx block, y-gaps dy, z-gaps dz), with ``-1`` at the previous
    voxel and ``+1`` at the next, so ``1 + A @ phi`` is the deformed axial gap. Rows are
    ordered x-gaps, then y-gaps, then z-gaps, each in C order.

    ``free`` (optional bool array of ``shape``) filters the pairs: ``endpoints='both'``
    keeps a pair only when BOTH voxels are free (the SLSQP sub-problem's semantics —
    frozen pairs the input already violates would make it structurally infeasible);
    ``endpoints='any'`` keeps a pair when AT LEAST ONE voxel is free (the windowed
    engine's prevention rows — the free-to-frozen-ring edges are the ones that stop a
    free voxel rotating against its pinned neighbour). Returns ``None`` when no row
    survives.
    """
    if endpoints not in ('both', 'any'):
        raise ValueError(f"unknown endpoints {endpoints!r}; valid: 'both', 'any'")
    sz, sy, sx = _unpack_size_3d(shape)
    voxels = sz * sy * sx
    lin = np.arange(voxels).reshape(sz, sy, sx)
    free = None if free is None else np.asarray(free, bool)

    def _keep(a, b):
        if free is None:
            return None
        return (a & b) if endpoints == 'both' else (a | b)

    rows_prev, rows_next, block = [], [], []
    # (prev-slice, next-slice, channel block index) per axis:
    # x-gaps -> dx block 0, y-gaps -> dy block 1, z-gaps -> dz block 2.
    specs = [
        (
            lin[:, :, :-1],
            lin[:, :, 1:],
            0,
            None if free is None else _keep(free[:, :, :-1], free[:, :, 1:]),
        ),
        (
            lin[:, :-1, :],
            lin[:, 1:, :],
            1,
            None if free is None else _keep(free[:, :-1, :], free[:, 1:, :]),
        ),
        (
            lin[:-1, :, :],
            lin[1:, :, :],
            2,
            None if free is None else _keep(free[:-1, :, :], free[1:, :, :]),
        ),
    ]
    for prev, nxt, blk, keep in specs:
        p = prev.ravel()
        n = nxt.ravel()
        if keep is not None:
            k = keep.ravel()
            p, n = p[k], n[k]
        rows_prev.append(p + blk * voxels)
        rows_next.append(n + blk * voxels)
        block.append(len(p))
    prev_cols = np.concatenate(rows_prev)
    next_cols = np.concatenate(rows_next)
    n_rows = prev_cols.size
    if n_rows == 0:
        return None
    row_idx = np.repeat(np.arange(n_rows), 2)
    col_idx = np.stack([prev_cols, next_cols], axis=1).ravel()
    data = np.tile(np.array([-1.0, 1.0]), n_rows)
    return scipy.sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n_rows, 3 * voxels))


def _monotonicity_diffs_2d(dy, dx):
    """Forward-difference monotonicity metrics for deformed coordinates.

    Returns ``(h_mono, v_mono)`` with shapes ``(H, W-1)`` and ``(H-1, W)``.
    """
    h_mono = 1.0 + np.diff(dx, axis=1)  # (H, W-1)
    v_mono = 1.0 + np.diff(dy, axis=0)  # (H-1, W)
    return h_mono, v_mono


def _diagonal_monotonicity_diffs_2d(dy, dx):
    """Anti-diagonal monotonicity: ensures each deformed quad cell is convex.

    For cell (r, c) with corners TL/TR/BR/BL:
        d1[r,c] = 1 + dx[r, c+1] - dx[r+1, c]   (TR.x > BL.x)
        d2[r,c] = 1 + dy[r+1, c] - dy[r, c+1]   (BL.y > TR.y)

    Together with h/v monotonicity these 4 conditions guarantee each quad
    cell is convex with positive orientation, preventing cross-row
    pinch-point self-intersections.

    Returns ``(d1, d2)`` each with shape ``(H-1, W-1)``.
    """
    d1 = 1.0 + dx[:-1, 1:] - dx[1:, :-1]  # (H-1, W-1)
    d2 = 1.0 + dy[1:, :-1] - dy[:-1, 1:]  # (H-1, W-1)
    return d1, d2


def _monotonicity_diffs_3d(dz, dy, dx):
    """Forward-difference monotonicity of deformed coordinates in 3D.

    The deformed coordinate along each axis is ``axis_index + displacement``,
    so the gap between neighbours is ``1 + diff(displacement)``.

    Returns ``(z_mono, y_mono, x_mono)`` with shapes ``(D-1, H, W)``,
    ``(D, H-1, W)`` and ``(D, H, W-1)``.
    """
    z_mono = 1.0 + np.diff(dz, axis=0)
    y_mono = 1.0 + np.diff(dy, axis=1)
    x_mono = 1.0 + np.diff(dx, axis=2)
    return z_mono, y_mono, x_mono


def injectivity_quality_2d(phi):
    """Per-pixel minimum axial monotonicity gap (2D).

    2D sibling of :func:`injectivity_quality_3d` — each h/v deformed-
    coordinate gap is spread to both endpoint pixels and the element-wise
    minimum taken. Unit gaps (1.0) everywhere on the identity field;
    negative where deformed columns/rows cross. Axial-only, same caveat
    as the 3D version.

    Parameters
    ----------
    phi : ndarray, shape ``(2, H, W)`` with channels ``[dy, dx]``.
    """
    dy, dx = phi[0], phi[1]
    h_mono, v_mono = _monotonicity_diffs_2d(dy, dx)
    q = np.full(dy.shape, np.inf)
    q[:, :-1] = np.minimum(q[:, :-1], h_mono)
    q[:, 1:] = np.minimum(q[:, 1:], h_mono)
    q[:-1] = np.minimum(q[:-1], v_mono)
    q[1:] = np.minimum(q[1:], v_mono)
    return q


def injectivity_quality_3d(phi):
    """Per-voxel minimum axial monotonicity gap, spread to both endpoints.

    3D analogue of the 2D injectivity quality spread: each axial gap value
    is assigned to both voxels it separates and the element-wise minimum
    is taken, giving a ``(D, H, W)`` map whose low entries mark voxels
    involved in a (near-)crossing.

    .. note::
        Deliberately **axial-only** — the 2D version adds anti-diagonal
        terms that make each quad cell provably convex; the corresponding
        3D closure would need the face- and space-diagonal families.
        These axial gaps are necessary separation conditions (deformed
        coordinate ordering along each axis), not a full 3D injectivity
        certificate.

    Parameters
    ----------
    phi : ndarray, shape ``(3, D, H, W)`` with channels ``[dz, dy, dx]``.
    """
    dz, dy, dx = phi[0], phi[1], phi[2]
    z_mono, y_mono, x_mono = _monotonicity_diffs_3d(dz, dy, dx)
    q = np.full(dz.shape, np.inf)
    q[:-1] = np.minimum(q[:-1], z_mono)
    q[1:] = np.minimum(q[1:], z_mono)
    q[:, :-1] = np.minimum(q[:, :-1], y_mono)
    q[:, 1:] = np.minimum(q[:, 1:], y_mono)
    q[:, :, :-1] = np.minimum(q[:, :, :-1], x_mono)
    q[:, :, 1:] = np.minimum(q[:, :, 1:], x_mono)
    return q


def injectivity_constraint(phi_xy, submatrix_size, exclude_boundaries=True):
    """Return flattened monotonicity diffs for the SLSQP injectivity constraint.

    Concatenates h_mono, v_mono, d1, and d2 (diagonal) diffs.  All four must
    be positive for the deformed grid to be globally injective and convex.

    When *exclude_boundaries* is ``True``, h/v use the standard ``[1:-1,1:-1]``
    interior slice.  Diagonal constraints are extended to all cells where at
    least one vertex is free (i.e. not on the frozen sub-window boundary).
    Only the two corners whose *both* vertices are frozen are excluded:
    cell (0, 0) and cell (sy-2, sx-2).
    """
    sy, sx = _unpack_size(submatrix_size)
    pixels = sy * sx
    dx = phi_xy[:pixels].reshape((sy, sx))
    dy = phi_xy[pixels:].reshape((sy, sx))
    h_mono, v_mono = _monotonicity_diffs_2d(dy, dx)
    d1, d2 = _diagonal_monotonicity_diffs_2d(dy, dx)
    if exclude_boundaries:
        h_vals = h_mono[1:-1, 1:-1].flatten()
        v_vals = v_mono[1:-1, 1:-1].flatten()
        # Include all diagonal cells except the two all-frozen corners.
        # (0,0):        d1 involves dx[0,1] and dx[1,0]  — both boundary
        # (sy-2,sx-2):  d1 involves dx[sy-2,sx-1] and dx[sy-1,sx-2] — both boundary
        n_diag = (sy - 1) * (sx - 1)
        keep = np.ones(n_diag, dtype=bool)
        keep[0] = False
        if n_diag > 1:
            keep[(sy - 2) * (sx - 1) + (sx - 2)] = False
        d1_vals = d1.reshape(-1)[keep]
        d2_vals = d2.reshape(-1)[keep]
    else:
        h_vals = h_mono.flatten()
        v_vals = v_mono.flatten()
        d1_vals = d1.flatten()
        d2_vals = d2.flatten()
    return np.concatenate([h_vals, v_vals, d1_vals, d2_vals])
