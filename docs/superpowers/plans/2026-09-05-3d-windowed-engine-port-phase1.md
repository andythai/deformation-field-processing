# 3D Windowed Engine Port — Phase 1 (family plumbing + certificate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `windowed_correct` / `ISQPWindowedStrategy` run on `(3, D, H, W)` fields with `SimplexConstraint3D` (round loop + window ladder, 3D edge rows, `'tr'` step rule, the fixed-6-tet + best-diagonal certificate) while every 2D path stays byte-identical, and measure the two unknowns that decide the rest of the port: per-SQP-iteration cost vs window volume, and whether the ratio test alone converges on real 3D windows.

**Architecture:** The engine dispatches per constraint family through the `LOCALITY` adapter (`dvfopt/core/windowed/_locality.py`); registering `SimplexConstraint3D` there plus making the box arithmetic n-D (three helpers: `_box_slices` / `_box_size` / `_pad_box`) is the whole port of the core loop. The 2D-shaped stages (giant tiler, coarse warm start, mop, re-seed, re-anchor, polish) are left untouched and skipped or refused on 3D fields — phase 2 ports them. Everything in `dvfopt/core/primitives/isqp.py` is dimension-agnostic and is not modified.

**Tech Stack:** numpy, scipy (`ndimage`, `sparse`), numba kernels in `dvfopt/jacobian/tetrahedron_sign.py`, osqp (+ clarabel for the hybrid backend), pytest, ruff, asv.

**Spec:** `docs/superpowers/specs/2026-09-05-3d-windowed-engine-port-design.md` (the revised version merged from PR #118 — read it first; the "Review correction" paragraphs are binding).

## Global Constraints

- **Never change 2D behaviour.** Every existing test stays green unchanged, and Task 7's main-vs-branch A/B must report every 2D output `np.array_equal` and every report scalar identical. Refactors of 2D code paths are allowed only when they are byte-identical (the box helpers); the 2D-only stages (`_mop_pass`, `_reseed_stage`, `_solve_giant_schwarz`, `_reanchor_pass`, `_coarse_warm_start`, `_ras_*`) are NOT edited in this phase.
- **Conventions:** 3D fields are `(3, D, H, W)` with channels `[dz, dy, dx]`; `SimplexConstraint3D.pack == PhiPack.DX_FIRST` (flat `[dx | dy | dz]`); constraint row layout is `values()`'s: row `k * n_cells + cell` for tet `k` of the C-ordered `(D-1, H-1, W-1)` cube grid (`tet_volumes_flat` and `build_tet_sparse_jac` share it). Boxes are per-axis `(lo, hi)` pairs in axis order: `(fy0, fy1, fx0, fx1)` in 2D, `(fz0, fz1, fy0, fy1, fx0, fx1)` in 3D. Fold threshold `0.01`; `margin_delta 1e-3`.
- **Phase-1 3D behaviour (from the spec):** `step_rule='exact_ls'` degrades to `'tr'` on a 3D field (debug log); `orientation_rows='full'`, `reanchor != 'none'` and `polish` on a 3D field raise `ValueError`; `coarse_to_fine`, the mop and the re-seed are skipped on 3D (report fields keep their did-not-run values); `max_window_area` is advisory on 3D (over-cap regions are solved whole, warned once per call, counted in `giant_regions`).
- **Where to work:** the worktree `C:/Users/Andy/Documents/GitHub/UCI-iGravi/dvfopt-3d-p1` on branch `feat/3d-windowed-phase1`. Run Python as `PYTHONPATH=C:/Users/Andy/Documents/GitHub/UCI-iGravi/dvfopt-3d-p1 C:/Users/Andy/Documents/GitHub/UCI-iGravi/deformation-field-processing/.venv/Scripts/python.exe ...` from the worktree root (the editable install points at the main checkout; PYTHONPATH wins — never `pip install -e .`). Before any test run, verify `python -c "import dvfopt; print(dvfopt.__file__)"` prints the worktree path. Run tests in the FOREGROUND with a generous timeout; do not arm background watchers.
- **Toolchain:** `ruff check dvfopt dvfopt_gui tests benchmarks` and `ruff format --check dvfopt dvfopt_gui tests benchmarks` must pass (ruff 0.16.3 pinned; target py39 — no `match`, no PEP 604 unions at runtime outside `from __future__ import annotations` files; `_common.py` already uses `int | None` in dataclass annotations, which is fine on Python ≥ 3.10). No new dependencies. Data payloads under `data/` and `research/.../output/` are gitignored — unit tests use synthetic fields only.
- **Threads:** benchmark scripts set `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` / `NUMEXPR_NUM_THREADS` to `1` at the top (before numpy) — the existing `benchmarks/make_hard_crops.py` idiom.
- **Commits:** one per task, message style `3D windowed, phase 1: <what>`; end every commit message with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

## File map

| File | Responsibility in this phase |
|---|---|
| `dvfopt/core/windowed/_common.py` | n-D box helpers (`_box_slices`, `_box_size`, `_pad_box`); n-D `find_windows`, `build_subproblem`, round loop, `_solve_window`, damage accounting; `_orientation_rows_3d`; 3D gating of the unported stages; `SliceReport` / `WindowRec` fields |
| `dvfopt/core/windowed/_locality.py` | `LOCALITY[SimplexConstraint3D]`: `_min_field_tet3d`, `_influenced_tet3d`, `_cached_tet_jac` |
| `dvfopt/core/windowed/_inners.py` | docstring only (`WindowSub.free_mask` / `patch_box` are n-D) |
| `dvfopt/strategies/windowed.py` | `supports_3d = True` + docstring |
| `tests/test_windowed_3d.py` | all new 3D unit tests (Tasks 1–6) |
| `tests/test_windowed_strategy.py` | invert the 3D-rejection test; add the Jdet3D rejection |
| `benchmarks/windowed_2d_identity.py` | the main-vs-branch byte-identity A/B (Task 7) |
| `benchmarks/windowed_3d_gate.py` | the 3D gate + the two measurements (Task 8) |
| `asv_bench/benchmarks/bench_solvers.py` | `WindowedEngine3D` (Task 9) |
| `CHANGELOG.md`, `CLAUDE.md` | the measured tables and the 3D support statement (Task 9) |

---

### Task 1: n-D box helpers and n-D `find_windows`

**Files:**
- Modify: `dvfopt/core/windowed/_common.py` (add three helpers above `find_windows` at line 318; replace `find_windows`)
- Test: `tests/test_windowed_3d.py` (create)

**Interfaces:**
- Produces: `_box_slices(box) -> tuple[slice, ...]`, `_box_size(box) -> int`, `_pad_box(box, shape, pad) -> tuple[int, ...]` (private, `_common.py`); `find_windows(mask, margin, ring) -> list[tuple[int, ...]]` now returns `2 * mask.ndim`-tuples.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_windowed_3d.py`:

```python
"""3D (SimplexConstraint3D) path of the windowed engine — phase 1 of the 3D port.

Every test here is additive: the 2D families are covered by the existing
windowed suites and by benchmarks/windowed_2d_identity.py (byte-identity).
"""

import numpy as np
import pytest

from dvfopt.constraints import SimplexConstraint3D
from dvfopt.core.primitives import isqp as isqp_mod
from dvfopt.core.windowed import LOCALITY, build_subproblem, find_windows, min_field, windowed_correct
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
```

- [ ] **Step 2: Run them to verify they fail**

Run (from the worktree root, with the PYTHONPATH prefix from Global Constraints):
`python -m pytest tests/test_windowed_3d.py -v`
Expected: `test_box_helpers_are_dimension_agnostic` FAILS with `ImportError` (no `_box_size`); `test_find_windows_3d_boxes_and_border_rule` FAILS with `ValueError: too many values to unpack` (the 2D `sy, sx` loop); the 2D test passes.

- [ ] **Step 3: Implement the helpers and the n-D `find_windows`**

In `dvfopt/core/windowed/_common.py`, insert directly above `def find_windows`:

```python
def _box_slices(box):
    """``(a0, a1, b0, b1, ...)`` -> ``(slice(a0, a1), slice(b0, b1), ...)`` — one
    slice per axis, so ``arr[(slice(None), *_box_slices(box))]`` crops a
    ``(C, *shape)`` field to the box in any dimension."""
    return tuple(slice(int(box[i]), int(box[i + 1])) for i in range(0, len(box), 2))


def _box_size(box):
    """Grid points inside ``box`` (area in 2D, volume in 3D)."""
    n = 1
    for i in range(0, len(box), 2):
        n *= max(0, int(box[i + 1]) - int(box[i]))
    return n


def _pad_box(box, shape, pad):
    """Grow every side of ``box`` by ``pad`` grid points, clipped to ``shape``."""
    out = []
    for ax, n in enumerate(shape):
        out.append(max(0, int(box[2 * ax]) - pad))
        out.append(min(int(n), int(box[2 * ax + 1]) + pad))
    return tuple(out)
```

Replace the body of `find_windows` (keep its docstring; add one sentence: "Works in any dimension: boxes are per-axis ``(lo, hi)`` pairs in axis order."):

```python
    grow = margin + ring
    dil = ndimage.binary_dilation(mask, iterations=grow)
    lbl, n = ndimage.label(dil)
    boxes = []
    for sl in ndimage.find_objects(lbl):
        box = []
        for s, n_ax in zip(sl, mask.shape):
            box.append(s.start + ring if s.start > 0 else 0)  # keep image-border folds free
            box.append(s.stop - ring if s.stop < n_ax else n_ax)
        boxes.append(tuple(box))
    return boxes
```

- [ ] **Step 4: Run the new tests and the existing windowed suites**

Run: `python -m pytest tests/test_windowed_3d.py tests/test_windowed_isqp.py tests/test_invariants_windowed.py -v`
Expected: all PASS (the 2D suites exercise `find_windows` on every family).

- [ ] **Step 5: Lint and commit**

```bash
ruff check dvfopt tests && ruff format dvfopt/core/windowed/_common.py tests/test_windowed_3d.py
git add dvfopt/core/windowed/_common.py tests/test_windowed_3d.py
git commit -m "3D windowed, phase 1: n-D box helpers + n-D find_windows (2D byte-identical)"
```

---

### Task 2: `LOCALITY[SimplexConstraint3D]` — fold map, influenced rows, cached tet Jacobian

**Files:**
- Modify: `dvfopt/core/windowed/_locality.py` (imports at lines 28-43; new functions; the `LOCALITY` dict at lines 211-230; module docstring lines 11-21; `min_field` docstring lines 245-252)
- Test: `tests/test_windowed_3d.py`

**Interfaces:**
- Consumes: `dvfopt.jacobian.tetrahedron_sign.six_tet_min_volume_3d(phi: (3,D,H,W)) -> (D-1,H-1,W-1)`, `build_tet_sparse_jac(D, H, W) -> jac(phi_flat) -> csr (6*cells, 3*D*H*W)`.
- Produces: `LOCALITY[SimplexConstraint3D] = WindowLocality(ring=1, min_field=_min_field_tet3d, influenced=_influenced_tet3d)` with `_influenced_tet3d(c, free_mask, pd, ph, pw, borders) -> (enforced_idx, jac_of)`; `_cached_tet_jac(pd, ph, pw)` (LRU, module-level). Task 3 calls `loc.influenced(c, free_mask, *free_mask.shape, borders)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_windowed_3d.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_windowed_3d.py -v -k "tet3d or locality"`
Expected: `KeyError: SimplexConstraint3D` on the registry lookups; `IncompatibleConstraintError` from `min_field`.

- [ ] **Step 3: Implement the 3D locality**

In `dvfopt/core/windowed/_locality.py`:

Add to the imports (keep alphabetical groups):

```python
import functools

from dvfopt.constraints import (
    FiniteJdetConstraint2D,
    JdetConstraint2D,
    SimplexConstraint2D,
    SimplexConstraint2DBilinear,
    SimplexConstraint3D,
)
from dvfopt.jacobian.tetrahedron_sign import build_tet_sparse_jac, six_tet_min_volume_3d
```

Add to the module docstring's ring list, after the `finite` line:

```
  tet3d: 1 — the 6-tet family (SimplexConstraint3D): tet volumes are EXACT and a
           free voxel's influenced cubes are its <= 8 corner cubes, all in-patch
           once it is 1 in (the 2tri rule with one more axis).
```

Add after `_min_field_cells`:

```python
def _min_field_tet3d(phi):
    """Fold map of the 6-tet family: each cube's min tet volume at its (z, y, x)
    corner voxel; the last voxel plane / row / column has no cube (+inf)."""
    D, H, W = phi.shape[1:]
    out = np.full((D, H, W), np.inf)
    out[: D - 1, : H - 1, : W - 1] = six_tet_min_volume_3d(np.asarray(phi, dtype=np.float64))
    return out
```

Add after `_influenced_finite`:

```python
@functools.lru_cache(maxsize=64)  # bounded: a volume's windows recur in a few dozen shapes
def _cached_tet_jac(pd, ph, pw):
    """Native sparse tet-Jacobian builder for one patch shape (72 nnz per cube, one
    vectorised pass per call). Cached per SHAPE here because
    ``Constraint._cached_jac_builder`` memoises on the instance and
    :func:`~dvfopt.core.windowed._common.build_subproblem` makes a fresh
    constraint per window."""
    return build_tet_sparse_jac(pd, ph, pw)


def _influenced_tet3d(c, free_mask, pd, ph, pw, borders):
    # cube (k,i,j) is influenced iff any of its 8 corner voxels is free; tet volumes
    # are exact so every cube evaluates correctly (no volume-border special case).
    fm = free_mask
    cell = np.zeros((pd - 1, ph - 1, pw - 1), bool)
    for oz in (0, 1):
        for oy in (0, 1):
            for ox in (0, 1):
                cell |= fm[oz : oz + pd - 1, oy : oy + ph - 1, ox : ox + pw - 1]
    cell_flat = np.nonzero(cell.ravel())[0]
    m = cell.size
    assert 6 * m == c.n_constraints, 'six tet rows per cube'
    enforced_idx = np.concatenate([b * m + cell_flat for b in range(6)])
    jac = _cached_tet_jac(pd, ph, pw)

    def jac_of(f):
        return jac(f)  # csr (6m, 3*pd*ph*pw); the caller slices the enforced rows

    return enforced_idx, jac_of
```

Add to `LOCALITY`:

```python
    SimplexConstraint3D: WindowLocality(
        ring=1, min_field=_min_field_tet3d, influenced=_influenced_tet3d
    ),
```

In `min_field`'s docstring add: "- tet3d: each cube's min tet volume at its corner voxel on the ``(D, H, W)`` voxel grid; the last plane/row/column is ``+inf``." and change "``(H, W)`` pixel grid" to "``(H, W)`` pixel grid (``(D, H, W)`` for the 3D family)". Update the `WindowLocality` docstring's `min_field` line to "``(C, *shape)`` field -> ``shape`` per-location constraint value" and the `influenced` line to "``(constraint, free_mask, *patch_shape, borders)``".

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_windowed_3d.py tests/test_windowed_strategy.py -v`
Expected: the new tests PASS. `tests/test_windowed_strategy.py::test_rejects_6tet_constraint_at_construction` now FAILS (the constraint is in `LOCALITY`, so `accepts_constraints` includes it, but `supports_3d` is still False → it still raises `IncompatibleConstraintError`... verify: if it still passes, fine; if it fails, leave it — Task 6 rewrites that test). Every other test PASSES.

- [ ] **Step 5: Lint and commit**

```bash
ruff check dvfopt tests && ruff format dvfopt/core/windowed/_locality.py tests/test_windowed_3d.py
git add dvfopt/core/windowed/_locality.py tests/test_windowed_3d.py
git commit -m "3D windowed, phase 1: LOCALITY[SimplexConstraint3D] — min field, eight-corner influenced rows, shape-cached tet Jacobian"
```

---

### Task 3: n-D `build_subproblem`

**Files:**
- Modify: `dvfopt/core/windowed/_common.py:169-266` (`build_subproblem`), `dvfopt/core/windowed/_inners.py:23-37` (`WindowSub` docstring)
- Test: `tests/test_windowed_3d.py`

**Interfaces:**
- Consumes: Task 1 helpers; Task 2 `loc.influenced(c, free_mask, *pshape, borders)`.
- Produces: `build_subproblem(...)` accepting a `2*ndim` box on a `(C, *shape)` field; `WindowSub.patch_box` is the per-axis `(lo, hi)` tuple, `WindowSub.free_mask` has the patch shape. Task 4 adds the rows dispatch inside this function; Task 5 pastes back through `_box_slices(sub.patch_box)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_windowed_3d.py`:

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_windowed_3d.py -v -k build_subproblem`
Expected: FAIL with `ValueError: too many values to unpack` at `H, W = phi_dydx.shape[1:]`.

- [ ] **Step 3: Make `build_subproblem` n-D**

Replace the body of `build_subproblem` from `H, W = phi_dydx.shape[1:]` through the `WindowSub(...)` return with:

```python
    shape = phi_dydx.shape[1:]
    loc = _locality_of(constraint)
    ring = loc.ring
    patch_box = _pad_box(free_box, shape, ring)
    patch = np.ascontiguousarray(phi_dydx[(slice(None), *_box_slices(patch_box))])
    pshape = patch.shape[1:]
    c = type(constraint)(shape=pshape)
    flat0 = np.asarray(c.flatten(patch), dtype=np.float64)

    # free pixels (patch-local): the free box, clipped into the patch
    local = tuple(int(free_box[i]) - int(patch_box[i - i % 2]) for i in range(len(free_box)))
    free_mask = np.zeros(pshape, bool)
    free_mask[_box_slices(local)] = True
    if free_extra is not None:
        free_mask &= free_extra[_box_slices(patch_box)]

    borders = tuple(
        b
        for ax, n in enumerate(shape)
        for b in (patch_box[2 * ax] == 0, patch_box[2 * ax + 1] == n)
    )
    enforced_idx, jac_of = loc.influenced(c, free_mask, *pshape, borders)

    # free variable indices in the constraint's own pack (never hand-packed)
    free_phi = np.stack([free_mask] * phi_dydx.shape[0]).astype(float)
    free_idx = np.nonzero(np.asarray(c.flatten(free_phi)) > 0.5)[0]

    target = threshold + margin_delta

    def cons(f):
        return (np.asarray(c.values(f)) - target)[enforced_idx]

    def cons_jac(f):  # sparse (n_enforced, n_vars) — enforced rows only
        return jac_of(f)[enforced_idx]

    n_rows = enforced_idx.size
    if orientation_delta is not None:
        a_or, b_or = _orientation_rows(
            c, free_mask, float(orientation_delta), kind=orientation_rows
        )
        base_cons, base_jac = cons, cons_jac

        def cons(f, _a=a_or, _b=b_or):
            return np.concatenate([base_cons(f), _a @ np.asarray(f) + _b])

        def cons_jac(f, _a=a_or):
            return sparse.vstack([sparse.csr_matrix(base_jac(f)), _a], format='csr')

        n_rows += a_or.shape[0]

    obj, grad, hess = _objective_fns(flat0, L2Objective() if objective is None else objective)
    return WindowSub(
        c,
        flat0,
        cons,
        cons_jac,
        obj,
        grad,
        hess,
        free_idx,
        free_mask,
        patch_box,
        n_rows,
    )
```

Update the docstring: "free box ``(fy0, fy1, fx0, fx1)`` (global)" → "free box of per-axis ``(lo, hi)`` pairs (global; ``(fy0, fy1, fx0, fx1)`` in 2D, ``(fz0, fz1, fy0, fy1, fx0, fx1)`` in 3D)", and "``free_extra`` (optional global ``(H, W)`` bool mask)" → "``free_extra`` (optional global bool mask of the field's spatial shape)".

In `_inners.py` change the two `WindowSub` field comments to `free_mask: np.ndarray  # patch-shaped bool: which patch pixels/voxels are free (for paste-back)` and `patch_box: tuple  # per-axis (lo, hi) pairs in global coords`.

- [ ] **Step 4: Run the tests and the 2D suites**

Run: `python -m pytest tests/test_windowed_3d.py tests/test_windowed_isqp.py tests/test_windowed_orientation_rows.py tests/test_windowed_reanchor.py tests/test_windowed_polish.py tests/test_invariants_windowed.py -v`
Expected: all PASS (2D behaviour is byte-identical: same patch box, same free mask, same border tuple order).

- [ ] **Step 5: Lint and commit**

```bash
ruff check dvfopt tests && ruff format dvfopt/core/windowed/_common.py dvfopt/core/windowed/_inners.py tests/test_windowed_3d.py
git add dvfopt/core/windowed/_common.py dvfopt/core/windowed/_inners.py tests/test_windowed_3d.py
git commit -m "3D windowed, phase 1: n-D build_subproblem (per-axis boxes, n-D masks; 2D byte-identical)"
```

---

### Task 4: 3D edge rows (`_orientation_rows_3d`) and the rows dispatch

**Files:**
- Modify: `dvfopt/core/windowed/_common.py` (new `_orientation_rows_3d` after `_orientation_rows`; the rows branch inside `build_subproblem`; the `orientation_delta` gate in `windowed_correct` at lines ~786-793)
- Test: `tests/test_windowed_3d.py`

**Interfaces:**
- Consumes: `dvfopt.core.slsqp_windowed.constraints3d._injectivity_linear_constraint_3d(subvolume_size: tuple[int,int,int], inj_lb: float, freeze_mask=None) -> scipy.optimize.LinearConstraint | None` (its `.A` is the sparse row matrix over the DX_FIRST `[dx|dy|dz]` flat pack, `A @ phi >= inj_lb - 1`; returns `None` when there are no rows).
- Produces: `_orientation_rows_3d(c, free_mask, delta) -> (A: csr (n_rows, 3*n), b: ndarray)` with `A @ x + b >= 0`; `build_subproblem(..., orientation_delta=δ, orientation_rows='edges')` works on 3D patches and raises `ValueError` for `'full'`; `windowed_correct` keeps `orientation_delta` for 3D simplex fields.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_windowed_3d.py`:

```python
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

    c = SimplexConstraint3D(shape=(2, 2, 2))
    a, b = _orientation_rows_3d(c, np.ones((2, 2, 2), bool), 0.01)
    phi = np.zeros((3, 2, 2, 2))
    phi[2, :, :, 0], phi[2, :, :, 1] = 1.0, -1.0  # x-edges flipped: deformed x goes 1 -> 0
    assert (a @ np.asarray(c.flatten(phi)) + b).min() < 0
    phi = np.zeros((3, 2, 2, 2))
    phi[2, :, :, 1] = phi[1, :, 1, :] = phi[0, 1] = -0.5  # 50 % compression on every axis
    assert (a @ np.asarray(c.flatten(phi)) + b).min() > 0


def test_subproblem_3d_with_rows_jacobian_matches_finite_differences():
    rng = np.random.default_rng(3)
    phi = rng.normal(0, 0.2, (3, 5, 5, 6))
    c = SimplexConstraint3D(shape=phi.shape[1:])
    sub = build_subproblem(
        c, phi, (1, 4, 1, 4, 1, 5), THR, NoneObjective(), 1e-3,
        orientation_delta=0.01, orientation_rows='edges',
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
            c, np.zeros((3, 4, 4, 4)), (1, 3, 1, 3, 1, 3), THR, None, 1e-3,
            orientation_delta=0.01, orientation_rows='full',
        )
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_windowed_3d.py -v -k rows`
Expected: `ImportError` (no `_orientation_rows_3d`); the subproblem tests fail inside `_orientation_rows` with "orientation rows need a 2D DY_FIRST".

- [ ] **Step 3: Implement the rows and the dispatch**

Add after `_orientation_rows` in `_common.py`:

```python
def _orientation_rows_3d(c, free_mask, delta):
    """Sparse ``(A, b)`` with ``A @ x + b >= 0`` the axial edge rows of a 3D patch.

    Every grid edge with at least one FREE endpoint keeps a positive projection of at
    least ``delta`` on its own axis: ``1 + dx[k,i,j+1] - dx[k,i,j] >= delta`` for the
    x-edges (dx block), likewise y-edges on dy and z-edges on dz — the ``'edges'`` kind
    of :func:`_orientation_rows` in the ``DX_FIRST`` pack ``[dx | dy | dz]``. The row
    matrix is the 3D injectivity helper's (``core.slsqp_windowed.constraints3d``) built
    WITHOUT its freeze filter: that filter keeps only rows whose BOTH endpoints are
    free, but the free-to-frozen-ring edges are exactly the rows that stop a free voxel
    rotating against its pinned neighbour, so here a row is kept when it touches ANY
    free column (the ``core.marching._mono_rows.mono_block`` predicate). Frozen-frozen
    rows are dropped: a violated constant row would sit in the elastic slack for the
    whole solve and distort the merit.
    """
    from dvfopt.constraints import PhiPack
    from dvfopt.core.slsqp_windowed.constraints3d import _injectivity_linear_constraint_3d

    if getattr(c, 'pack', None) != PhiPack.DX_FIRST or free_mask.ndim != 3:
        raise ValueError('3D orientation rows need a 3D DX_FIRST (simplex-family) constraint')
    n = free_mask.size
    lc = _injectivity_linear_constraint_3d(tuple(int(v) for v in free_mask.shape), float(delta))
    if lc is None:
        return sparse.csr_matrix((0, 3 * n)), np.zeros(0)
    a = sparse.csr_matrix(lc.A)
    free_cols = np.nonzero(np.tile(free_mask.ravel(), 3))[0]
    touch = np.diff(a[:, free_cols].indptr) > 0  # rows with at least one free endpoint
    a = a[touch]
    return a, np.full(a.shape[0], 1.0 - float(delta))
```

In `build_subproblem`, replace the `a_or, b_or = _orientation_rows(...)` call with:

```python
        if len(pshape) == 3:
            if orientation_rows != 'edges':
                raise ValueError(
                    "3D orientation rows: only kind='edges' exists (no convexity rows in 3D)"
                )
            a_or, b_or = _orientation_rows_3d(c, free_mask, float(orientation_delta))
        else:
            a_or, b_or = _orientation_rows(
                c, free_mask, float(orientation_delta), kind=orientation_rows
            )
```

and extend its docstring's rows paragraph: "On a 3D patch the rows are :func:`_orientation_rows_3d` (axial edges on all three axes, ``'edges'`` only)."

In `windowed_correct` replace the gate

```python
        if getattr(constraint, "pack", None) != PhiPack.DY_FIRST:
            ...
            orientation_delta = None
```

with

```python
        family_rows = getattr(constraint, "pack", None) == PhiPack.DY_FIRST or (
            getattr(constraint, "dim", 2) == 3
        )
        if not family_rows:
            # The edge-monotonicity rows are a simplex-family formulation (DY_FIRST in
            # 2D, the registered 6-tet family in 3D); the Jdet / finite families keep
            # the plain rows (explicit requests on a sub-problem still raise in
            # build_subproblem).
            orientation_delta = None
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_windowed_3d.py tests/test_windowed_orientation_rows.py tests/test_windowed_isqp.py -v`
Expected: all PASS (`test_engine_default_rows_skip_non_dy_first_families` still passes: Jdet2D has `dim == 2`).

- [ ] **Step 5: Lint and commit**

```bash
ruff check dvfopt tests && ruff format dvfopt/core/windowed/_common.py tests/test_windowed_3d.py
git add dvfopt/core/windowed/_common.py tests/test_windowed_3d.py
git commit -m "3D windowed, phase 1: 3D axial edge rows (any-free-endpoint predicate) + rows dispatch"
```

---

### Task 5: n-D round loop, `_solve_window`, certificate fields, 3D stage gating

**Files:**
- Modify: `dvfopt/core/windowed/_common.py` — `WindowRec` (line 347), `SliceReport` (line 367), `windowed_correct` (entry gates ~776-793, setup ~817-826, coarse block ~860-895, round loop ~899-963, accounting ~1058-1088), `_solve_window` (line 1571-1832)
- Test: `tests/test_windowed_3d.py`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: `windowed_correct(phi3d, 'isqp', constraint=SimplexConstraint3D(...), ...) -> (phi_out, SliceReport)` with `SliceReport.folds_after_zero`, `.best_diag_floor_after`, `.best_diag_floor_after_zero` (−1 on 2D) and `WindowRec.patch_box` (per-axis pairs). Task 6 relies on this call working end-to-end.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_windowed_3d.py`:

```python
def _planted_3d(shape=(10, 24, 24), seed=0, amp=1.4, size=(4, 6, 6)):
    """Identity field with one random blob of folds in the interior."""
    rng = np.random.default_rng(seed)
    phi = np.zeros((3, *shape))
    z, y, x = (s // 2 for s in shape)
    dz, dy, dx = size
    phi[:, z - dz // 2 : z + dz // 2, y - dy // 2 : y + dy // 2, x - dx // 2 : x + dx // 2] = (
        rng.normal(0, amp, (3, *size))
    )
    return phi


@needs_osqp
@pytest.mark.parametrize("objective", [NoneObjective, L2Objective])
def test_3d_planted_folds_no_damage_and_untouched_voxels_bit_identical(objective):
    phi = _planted_3d()
    c = SimplexConstraint3D(shape=phi.shape[1:])
    assert (min_field(c, phi) < THR).any(), "fixture must contain folds"
    out, rep = windowed_correct(
        phi.copy(), "isqp", constraint=c, objective=objective(), threshold=THR, verbose=0
    )
    assert rep.damage == 0 and np.isfinite(out).all()
    assert rep.folds_after < rep.folds_before
    assert rep.n_windows >= 1 and len(rep.windows[0].patch_box) == 6
    # the blob's window (free box = blob +- margin 3, ring 1) never reaches y < 5
    assert np.array_equal(out[:, :, :5], phi[:, :, :5])
    # 3D certificate fields are filled; the 2D ones stay at their defaults
    assert rep.best_diag_floor_after >= 0 and rep.best_diag_floor_after_zero >= 0
    assert rep.folds_after_zero >= 0 and rep.best_diag_floor_after <= rep.folds_after
    assert rep.coarse_folds_before == -1 and rep.mop_windows == 0 and rep.reseed_rounds_run == 0


@needs_osqp
def test_3d_mild_blob_reaches_zero_folds_on_tr():
    phi = _planted_3d(amp=0.5)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    assert (min_field(c, phi) < THR).any(), "fixture must contain folds"
    out, rep = windowed_correct(
        phi.copy(), "isqp", constraint=c, objective=NoneObjective(), threshold=THR, verbose=0
    )
    assert rep.folds_after == 0 and rep.damage == 0
    assert (min_field(c, out) >= THR).all()
    assert rep.folds_after_zero == 0 and rep.best_diag_floor_after == 0


@needs_osqp
def test_3d_fold_free_input_is_returned_byte_identical():
    phi = np.zeros((3, 6, 8, 8))
    phi[2] = 0.1
    c = SimplexConstraint3D(shape=phi.shape[1:])
    out, rep = windowed_correct(phi.copy(), "isqp", constraint=c, threshold=THR, verbose=0)
    assert np.array_equal(out, phi) and rep.n_windows == 0 and rep.damage == 0
    assert rep.folds_after_zero == 0 and rep.best_diag_floor_after == 0


@needs_osqp
def test_3d_exact_ls_default_degrades_to_tr(monkeypatch):
    import dvfopt.core.windowed._common as cm

    seen = []
    orig = cm.solve_window_inner

    def spy(sub, inner, maxiter, **kw):
        seen.append(kw.get("step_rule"))
        return orig(sub, inner, maxiter, **kw)

    monkeypatch.setattr(cm, "solve_window_inner", spy)
    phi = _planted_3d((6, 10, 10), amp=1.0)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    windowed_correct(phi, "isqp", constraint=c, threshold=THR, verbose=0, maxiter=3)  # default 'exact_ls'
    assert seen and set(seen) == {"tr"}


def test_3d_refuses_the_unported_stages():
    phi = np.zeros((3, 6, 8, 8))
    c = SimplexConstraint3D(shape=phi.shape[1:])
    with pytest.raises(ValueError, match="3D"):
        windowed_correct(phi, "isqp", constraint=c, threshold=THR, reanchor="l2")
    with pytest.raises(ValueError, match="3D"):
        windowed_correct(phi, "isqp", constraint=c, threshold=THR, polish="l2")


@needs_osqp
def test_3d_over_cap_region_is_solved_whole_and_counted():
    phi = _planted_3d(amp=0.5)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    out, rep = windowed_correct(
        phi.copy(), "isqp", constraint=c, objective=NoneObjective(), threshold=THR,
        verbose=0, max_window_area=100,
    )
    assert rep.giant_regions >= 1 and rep.damage == 0
    assert rep.folds_after == 0  # the cap is advisory in phase 1: the region was solved whole


def test_2d_report_keeps_the_3d_fields_at_minus_one():
    from dvfopt.core.windowed import SliceReport

    rep = SliceReport()
    assert (rep.folds_after_zero, rep.best_diag_floor_after, rep.best_diag_floor_after_zero) == (-1, -1, -1)
```

If `test_3d_mild_blob_reaches_zero_folds_on_tr` or the `folds_after == 0` assertion of the over-cap test fails while damage is 0 and the field is finite, do NOT loosen the assertion or add a mechanism: that IS unknown U2's answer. Mark the failing test `@pytest.mark.xfail(strict=True, reason="<residual folds, min value, SQP iterations measured>")` and report the numbers in your summary; Task 8 measures it properly on the real artefacts.

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_windowed_3d.py -v -k "3d_ or 2d_report"`
Expected: FAIL — `ValueError: step_rule='exact_ls' requires a 2D (2, H, W) field` on the default-rule calls, `too many values to unpack` in `windowed_correct` / `_solve_window` on the rest, `AttributeError` on the report fields.

- [ ] **Step 3: Implement**

(a) `WindowRec`: add after `pw: int = 0`:

```python
    patch_box: tuple = ()  # per-axis (lo, hi) pairs of the patch, any dimension
```

(b) `SliceReport`: add after `reseed_folds_after: int = -1`:

```python
    # 3D certificate (SimplexConstraint3D only; -1 on 2D fields): the fixed-diagonal
    # 6-tet fold count at 0 (folds_after is at `threshold`) and the best-of-4-diagonals
    # floor — cubes no main-diagonal split certifies — at `threshold` and at 0
    # (`tetrahedron_sign.n_neg_best_diagonal`, `<=` semantics; the rows are driven
    # to threshold + margin_delta so no cell parks exactly on the threshold).
    folds_after_zero: int = -1
    best_diag_floor_after: int = -1
    best_diag_floor_after_zero: int = -1
```

(c) `windowed_correct` entry: replace the block

```python
    if step_rule == 'exact_ls' and np.asarray(phi_in).ndim != 3:
        ...
        raise ValueError("step_rule='exact_ls' requires a 2D (2, H, W) field")
    loc = _locality_of(constraint)
```

with

```python
    is3d = np.asarray(phi_in).ndim == 4
    if step_rule == 'exact_ls' and is3d:
        # The exact line model needs rows that are BILINEAR in the displacements — true
        # of every 2D family here, false of a 6-tet volume (trilinear, hence cubic along
        # a line). Degrade to the ratio test — the way orientation_delta is dropped on
        # non-DY_FIRST packs below — until phase 3 of the 3D port ships the cubic model.
        logging.getLogger('dvfopt').debug(
            "windowed_correct: step_rule='exact_ls' is 2D-only; using 'tr' on this 3D field"
        )
        step_rule = 'tr'
    elif step_rule == 'exact_ls' and np.asarray(phi_in).ndim != 3:
        raise ValueError("step_rule='exact_ls' requires a 2D (2, H, W) field")
    if is3d and (reanchor != 'none' or polish is not None):
        raise ValueError(
            'reanchor and polish are not yet supported on 3D fields (3D port, phase 2)'
        )
    loc = _locality_of(constraint)
```

and add `import logging` to the module imports.

After `opts = _InnerOpts(...)` / `objective = ...` and `phi = np.array(phi_in, ...)`, replace `H, W = phi.shape[1:]` with

```python
    shape = phi.shape[1:]
    if is3d:
        # Phase 1 of the 3D port: the round loop + window ladder only. The coarse warm
        # start, the terminal mop and the harmonic re-seed are 2D-shaped stages (phase
        # 2); their report fields keep their did-not-run values.
        coarse_to_fine, mop_margin, reseed_rounds = False, 0, 0
```

and `touched = np.zeros((H, W), bool)` with `touched = np.zeros(shape, bool)`.

Coarse block: `min(H, W) >= 4 * max(giant_tile, coarse_factor)` → `min(shape) >= 4 * max(giant_tile, coarse_factor)`; the warm-box marking

```python
        for fy0, fy1, fx0, fx1 in warm_boxes:  # the warm start is a move over these
            touched[max(0, fy0 - ring) : fy1 + ring, max(0, fx0 - ring) : fx1 + ring] = True
```

→

```python
        for wb in warm_boxes:  # the warm start is a move over these
            touched[_box_slices(_pad_box(wb, shape, ring))] = True
```

Round loop: add `giant_warned = False` before `for _rnd in range(max_rounds):`; replace the window body from `fy0, fy1, fx0, fx1 = box` through the `continue` of the giant branch with

```python
            # touched = the ENFORCED footprint (free box dilated by ring), not the
            # bare free box: a free pixel influences constraints up to `ring` beyond
            # the free box, so an infeasible solve could leave a violated row there.
            # Marking it touched makes any such residual count as residual, never
            # damage — so damage=0 is by construction, not merely for feasible solves.
            touched[_box_slices(_pad_box(box, shape, ring))] = True
            if _box_size(box) > max_window_area:
                rep.giant_regions += 1
                rep.giant_boxes.append(box)
                if not is3d:
                    # too big for one QP -> overlapping-tile Schwarz decomposition
                    giant_w0 = len(rep.windows)
                    _solve_giant_schwarz(
                        phi,
                        constraint,
                        box,
                        threshold,
                        objective,
                        maxiter,
                        ring,
                        rep,
                        margin_delta,
                        inner=inner,
                        opts=opts,
                        expired=_expired,
                    )
                    if record_history:
                        rep.history.append(_stage_entry("giant", giant_w0))
                    _fire("giant", phi)
                    continue
                if not giant_warned:
                    # Phase 1 of the 3D port: the voxel cap is advisory — the region is
                    # solved whole (the 3D tiler is phase 2). Warned once per call.
                    giant_warned = True
                    log_warning(
                        f"windowed_correct: 3D region of {_box_size(box)} voxels exceeds "
                        f"max_window_area={max_window_area}; solving it whole (3D tiler pending)"
                    )
```

(the `_solve_window(...)` call that follows is unchanged).

Final accounting: after `rep.residual_in_window = ...` add

```python
    if is3d:
        from dvfopt.jacobian.tetrahedron_sign import n_neg_best_diagonal

        rep.folds_after_zero = int((jf <= 0).sum())
        rep.best_diag_floor_after = int(n_neg_best_diagonal(phi, threshold))
        rep.best_diag_floor_after_zero = int(n_neg_best_diagonal(phi, 0.0))
```

Docstring of `windowed_correct`: change "Correct a full ``(2, H, W)`` slice" to "Correct a full ``(2, H, W)`` slice — or, with :class:`~dvfopt.constraints.SimplexConstraint3D`, a ``(3, D, H, W)`` volume (3D port, phase 1: round loop + window ladder, ``'tr'`` step rule, 3D edge rows; the coarse warm start, mop and re-seed are skipped, the giant cap is advisory, ``reanchor`` / ``polish`` raise)", and list `SimplexConstraint3D` among the accepted constraints.

(d) `_solve_window`: replace `H, W = phi.shape[1:]` with `shape = phi.shape[1:]`; replace the paste-back

```python
    patch_out = np.asarray(sub.constraint.unflatten(x))
    py0, py1, px0, px1 = sub.patch_box
    # paste back ONLY free pixels (frozen ring is unchanged and may be shared)
    fm = sub.free_mask
    dst = phi[:, py0:py1, px0:px1]
    dst[:, fm] = patch_out[:, fm]
```

with

```python
    patch_out = np.asarray(sub.constraint.unflatten(x))
    psl = (slice(None), *_box_slices(sub.patch_box))
    # paste back ONLY free pixels (frozen ring is unchanged and may be shared)
    fm = sub.free_mask
    dst = phi[psl]
    dst[:, fm] = patch_out[:, fm]
```

In the polish block replace `pdst = phi[:, py0:py1, px0:px1]` with `pdst = phi[psl]`; replace `jpatch = min_field(constraint, phi[:, py0:py1, px0:px1])` with `jpatch = min_field(constraint, phi[psl])`; in the `WindowRec(...)` construction replace

```python
        fy0=box[0],
        fx0=box[2],
        ph=py1 - py0,
        pw=px1 - px0,
```

with

```python
        fy0=box[-4],
        fx0=box[-2],
        ph=sub.patch_box[-3] - sub.patch_box[-4],
        pw=sub.patch_box[-1] - sub.patch_box[-2],
        patch_box=tuple(int(v) for v in sub.patch_box),
```

and the grow block

```python
    if allow_grow and opts.ladder and not ok and _grow < 2:
        fy0, fy1, fx0, fx1 = box
        gy0, gy1 = max(0, fy0 - 4), min(H, fy1 + 4)
        gx0, gx1 = max(0, fx0 - 4), min(W, fx1 + 4)
        if (gy0, gy1, gx0, gx1) != box:
            return _solve_window(
                phi,
                constraint,
                (gy0, gy1, gx0, gx1),
```

with

```python
    if allow_grow and opts.ladder and not ok and _grow < 2:
        grown = _pad_box(box, shape, 4)
        if grown != tuple(box):
            return _solve_window(
                phi,
                constraint,
                grown,
```

(the remaining arguments of that call are unchanged).

- [ ] **Step 4: Run the whole windowed suite**

Run: `python -m pytest tests/test_windowed_3d.py tests/test_windowed_isqp.py tests/test_windowed_strategy.py tests/test_windowed_coarse_to_fine.py tests/test_windowed_escape.py tests/test_windowed_fast_robust.py tests/test_windowed_orientation_rows.py tests/test_windowed_polish.py tests/test_windowed_ras.py tests/test_windowed_reanchor.py tests/test_windowed_reseed.py tests/test_invariants_windowed.py -v`
Expected: all PASS except possibly the two U2-sensitive assertions handled as described in Step 1 (report them), and `test_windowed_strategy.py::test_rejects_6tet_constraint_at_construction` (Task 6 rewrites it — it may pass or fail here depending on `supports_3d`, both are fine).

- [ ] **Step 5: Lint and commit**

```bash
ruff check dvfopt tests && ruff format dvfopt/core/windowed/_common.py tests/test_windowed_3d.py
git add dvfopt/core/windowed/_common.py tests/test_windowed_3d.py
git commit -m "3D windowed, phase 1: n-D round loop + window ladder, 3D certificate fields, unported stages gated"
```

---

### Task 6: strategy `supports_3d` and the Solver composition

**Files:**
- Modify: `dvfopt/strategies/windowed.py:34-58` (class docstring), `:218-220` (`supports_3d`)
- Modify: `tests/test_windowed_strategy.py:103-110`
- Test: `tests/test_windowed_3d.py`

**Interfaces:**
- Consumes: Task 5's `windowed_correct` on 3D.
- Produces: `Solver(constraint=SimplexConstraint3D(...), objective=..., strategy=ISQPWindowedStrategy()).fit(phi3d)` and `correct_dvf(phi3d, constraint='simplex_3d', strategy='isqp_windowed', objective='none')` work; `JdetConstraint3D` still raises `IncompatibleConstraintError`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_windowed_strategy.py` replace `test_rejects_6tet_constraint_at_construction` with:

```python
def test_accepts_simplex3d_constraint_at_construction():
    s = Solver(
        constraint=SimplexConstraint3D(shape=(4, 6, 6)),
        objective=L2Objective(),
        strategy=ISQPWindowedStrategy(),
    )
    assert s.strategy.supports_3d is True


def test_rejects_jdet3d_constraint_at_construction():
    from dvfopt.constraints import JdetConstraint3D

    with pytest.raises(IncompatibleConstraintError):
        Solver(
            constraint=JdetConstraint3D(shape=(4, 6, 6)),
            objective=L2Objective(),
            strategy=ISQPWindowedStrategy(),
        )
```

Append to `tests/test_windowed_3d.py`:

```python
@needs_osqp
def test_3d_solver_composition_and_string_recipe():
    from dvfopt import ISQPWindowedStrategy, Solver, correct_dvf

    phi = _planted_3d((8, 14, 14), amp=0.5)
    res = Solver(
        constraint=SimplexConstraint3D(shape=phi.shape[1:]),
        objective=NoneObjective(),
        strategy=ISQPWindowedStrategy(),
    ).fit(phi)
    assert res.corrected.shape == phi.shape and np.isfinite(res.corrected).all()
    res2 = correct_dvf(phi, constraint='simplex_3d', strategy='isqp_windowed', objective='none')
    assert np.array_equal(res2.corrected, res.corrected)  # same deterministic solve
```

(Check `SolveResult`'s attribute name for the corrected field in `tests/test_windowed_strategy.py::test_solver_composition_reaches_zero_folds` and use exactly that name.)

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest tests/test_windowed_strategy.py tests/test_windowed_3d.py -v -k "simplex3d or jdet3d or composition"`
Expected: the accept test FAILS with `IncompatibleConstraintError` (`supports_3d` is False); the Jdet3D test passes.

- [ ] **Step 3: Implement**

In `dvfopt/strategies/windowed.py` change `supports_3d = False` to

```python
    supports_3d = True  # SimplexConstraint3D via LOCALITY (3D port, phase 1)
```

and add to the class docstring, after the first paragraph: "Since the 3D port's phase 1 the engine also accepts :class:`~dvfopt.constraints.SimplexConstraint3D` on ``(3, D, H, W)`` fields: the round loop + window ladder with the 3D axial edge rows and the ``'tr'`` step rule (``'exact_ls'`` degrades); the coarse warm start, mop and re-seed are skipped, the giant cap is advisory, ``reanchor`` / ``polish`` raise. Jdet3D stays with ``SLSQPWindowedStrategy``."

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_windowed_strategy.py tests/test_windowed_3d.py tests/test_gui_strategy_parity.py tests/test_gui_logic.py -v`
Expected: all PASS (the GUI tests are untouched by `supports_3d`: no menu row is added and `strategy_params` excludes the attribute).

- [ ] **Step 5: Lint and commit**

```bash
ruff check dvfopt tests && ruff format dvfopt/strategies/windowed.py tests/test_windowed_strategy.py tests/test_windowed_3d.py
git add dvfopt/strategies/windowed.py tests/test_windowed_strategy.py tests/test_windowed_3d.py
git commit -m "3D windowed, phase 1: ISQPWindowedStrategy accepts SimplexConstraint3D (supports_3d)"
```

---

### Task 7: the 2D byte-identity A/B (main vs branch)

**Files:**
- Create: `benchmarks/windowed_2d_identity.py`
- Output (gitignored): `benchmarks/output/identity_2d/{main,branch}/` under the MAIN checkout

**Interfaces:**
- Consumes: `windowed_correct` on both trees; `data/dvfs/crops/{z16_twist,z0_cluster,z0_sliver}.npy` (present in the main checkout, gitignored) when available.
- Produces: the gate verdict `IDENTITY PASS` / `FAIL` — the PR body quotes it.

- [ ] **Step 1: Write the script**

```python
"""Byte-identity gate for the 2D windowed engine across a refactor.

Runs ``windowed_correct`` on a fixed 2D case set and either SAVES the outputs
(``--out DIR``) or COMPARES two saved sets (``--compare A B``), asserting every
corrected field is ``np.array_equal`` and every report scalar (times excluded)
identical. Run it once with ``PYTHONPATH`` pointing at ``main`` and once at the
branch, from the main checkout's root (the crops live there) — the script prints
which ``dvfopt`` it imported. The 3D port must leave every 2D path byte-identical.
"""

import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

import dvfopt  # noqa: E402
from dvfopt.constraints import (  # noqa: E402
    FiniteJdetConstraint2D,
    JdetConstraint2D,
    SimplexConstraint2D,
    SimplexConstraint2DBilinear,
)
from dvfopt.core.windowed import windowed_correct  # noqa: E402
from dvfopt.objectives import L2Objective, NoneObjective  # noqa: E402
from dvfopt.testdata import make_random_dvf  # noqa: E402

FAMILIES = {
    "jdet": JdetConstraint2D,
    "2tri": SimplexConstraint2D,
    "finite": FiniteJdetConstraint2D,
    "bilinear": SimplexConstraint2DBilinear,
}
OBJECTIVES = {"l2": L2Objective, "none": NoneObjective}
THR = 0.01


def _cases():
    """(name, phi) for every case; the family x objective matrix is per case."""
    rng = np.random.default_rng(0)
    cases = {}
    for seed in (0, 1):
        cases[f"random48_{seed}"] = (
            np.stack([rng.normal(0, 0.55, (48, 48)), rng.normal(0, 0.55, (48, 48))]),
            list(FAMILIES),
            list(OBJECTIVES),
        )
    patch = np.asarray(make_random_dvf("03a_10x10_random_seed_42"))[1:, 0]
    giant = np.zeros((2, 120, 120))
    for by in range(6):
        for bx in range(6):
            y, x = 25 + by * 10, 25 + bx * 10
            giant[:, y : y + patch.shape[1], x : x + patch.shape[2]] = patch
    cases["giant120"] = (giant, ["bilinear"], ["l2"])  # one region > max_window_area -> tiler + mop
    coarse = np.zeros((2, 260, 260))
    for cy, cx in ((60, 60), (60, 200), (200, 60), (200, 200), (130, 130)):
        coarse[:, cy : cy + patch.shape[1], cx : cx + patch.shape[2]] = patch
    cases["coarse260"] = (coarse, ["bilinear"], ["none"])  # min(H, W) >= 4 * 64 -> coarse warm start
    for name in ("z16_twist", "z0_cluster", "z0_sliver"):
        p = os.path.join("data", "dvfs", "crops", f"{name}.npy")
        if os.path.exists(p):
            cases[name] = (np.load(p).astype(np.float64), ["bilinear"], ["none"])
    return cases


def _scalars(rep):
    """Report scalars that must match exactly (wall times excluded)."""
    out = {
        k: v
        for k, v in rep.__dict__.items()
        if isinstance(v, (int, float, bool)) and not k.endswith("_s") and k != "time_s"
    }
    out["windows"] = [
        [w.fy0, w.fx0, w.ph, w.pw, w.n_free, w.n_enforced, w.inner_iters, w.grows, w.fallback,
         w.backend_fallback, w.patience_fallback, w.min_after, w.feasible]
        for w in rep.windows
    ]
    return out


def run(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"dvfopt from {dvfopt.__file__}", flush=True)
    for name, (phi, fams, objs) in _cases().items():
        for fam in fams:
            for obj in objs:
                key = f"{name}__{fam}__{obj}"
                out, rep = windowed_correct(
                    phi.copy(), "isqp", constraint=FAMILIES[fam](shape=phi.shape[1:]),
                    objective=OBJECTIVES[obj](), threshold=THR, verbose=0,
                )
                np.save(os.path.join(out_dir, f"{key}.npy"), out)
                with open(os.path.join(out_dir, f"{key}.json"), "w") as fh:
                    json.dump(_scalars(rep), fh, sort_keys=True)
                print(f"  {key}: folds {rep.folds_before}->{rep.folds_after} damage {rep.damage} "
                      f"windows {rep.n_windows} {rep.time_s:.1f}s", flush=True)


def compare(a, b):
    keys_a = sorted(f[:-4] for f in os.listdir(a) if f.endswith(".npy"))
    keys_b = sorted(f[:-4] for f in os.listdir(b) if f.endswith(".npy"))
    ok = keys_a == keys_b
    if not ok:
        print(f"key sets differ: {set(keys_a) ^ set(keys_b)}")
    for key in keys_a:
        same_arr = np.array_equal(np.load(os.path.join(a, f"{key}.npy")), np.load(os.path.join(b, f"{key}.npy")))
        with open(os.path.join(a, f"{key}.json")) as fa, open(os.path.join(b, f"{key}.json")) as fb:
            same_rep = json.load(fa) == json.load(fb)
        print(f"  {key}: field {'same' if same_arr else 'DIFFERENT'}, report {'same' if same_rep else 'DIFFERENT'}")
        ok &= same_arr and same_rep
    print("IDENTITY PASS" if ok else "IDENTITY FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.out:
        run(args.out)
    if args.compare:
        sys.exit(0 if compare(*args.compare) else 1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against `main` and against the branch**

Both runs from the MAIN checkout root (the crops are there); only `PYTHONPATH` differs. The script file is the branch's copy.

```bash
MAIN=C:/Users/Andy/Documents/GitHub/UCI-iGravi/deformation-field-processing
WT=C:/Users/Andy/Documents/GitHub/UCI-iGravi/dvfopt-3d-p1
cd $MAIN
PYTHONPATH=$MAIN $MAIN/.venv/Scripts/python.exe $WT/benchmarks/windowed_2d_identity.py --out $MAIN/benchmarks/output/identity_2d/main
PYTHONPATH=$WT   $MAIN/.venv/Scripts/python.exe $WT/benchmarks/windowed_2d_identity.py --out $MAIN/benchmarks/output/identity_2d/branch
PYTHONPATH=$WT   $MAIN/.venv/Scripts/python.exe $WT/benchmarks/windowed_2d_identity.py --compare $MAIN/benchmarks/output/identity_2d/main $MAIN/benchmarks/output/identity_2d/branch
```

Expected: the first run prints `dvfopt from ...deformation-field-processing/dvfopt/__init__.py`, the second `...dvfopt-3d-p1/dvfopt/__init__.py` (if both print the same path, STOP — the PYTHONPATH trick did not take), and the compare prints `same, same` for every key and `IDENTITY PASS`. Each run takes several minutes (the crops are 20-110 s each). Run in the foreground with `timeout` 600000 per command.

If any key differs: that is a 2D behaviour change — find the refactor that caused it (Tasks 1, 3, 5 touch shared code) and fix it; do not proceed with a FAIL.

- [ ] **Step 3: Lint and commit**

```bash
ruff check benchmarks && ruff format benchmarks/windowed_2d_identity.py
git add benchmarks/windowed_2d_identity.py
git commit -m "3D windowed, phase 1: 2D byte-identity A/B script (main vs branch)"
```

Paste the `IDENTITY PASS` block (case list + verdict) into your task summary; the PR body needs it.

---

### Task 8: the 3D gate and the two measured unknowns

**Files:**
- Create: `benchmarks/windowed_3d_gate.py`
- Output (gitignored): `benchmarks/output/windowed_3d/` under the MAIN checkout (`gate_<case>_<cfg>.json`, `gate.md`, `cost.json`, `cost.md`)

**Interfaces:**
- Consumes: the artefacts listed in the spec's table (main checkout paths: `data/dvfs/testcases_3d/*.npy`, `research/strict_feasibility_3d/runners/output/b0039_subvol_{8_easy,16_moderate,24x24x24}.npy`, `data/dvfs/b0039/b0039_laplacian_deformation_field.npy`); `dvfopt.core.primitives.isqp._make_qp` (module-level factory called by name inside `isqp_solve`); `dvfopt.core.windowed._common.solve_window_inner` (looked up in `_common`'s globals by `_solve_window`).
- Produces: the reference tables for the CHANGELOG / PR (Task 9).

- [ ] **Step 1: Write the script**

```python
"""3D windowed engine — the phase-1 gate and the two measured unknowns.

``--gate``  the three ``testcases_3d`` fields + the 16^3 B0039 sub-volume under
            {L2, none} x {edge rows on, off} on the ``'tr'`` step rule: fixed-6-tet
            folds (at threshold and at 0), the best-diagonal floor (both), damage,
            windows, SQP iterations, per-window exit reasons, wall, L1 / L2 move and
            the largest |dz| moved. Asserts 0 folds / damage 0 on the testcases under
            the default config (the plumbing gate) and damage 0 everywhere; the 16^3
            result is reported with its floor.
``--cost``  per-SQP-iteration cost vs window volume: ONE frozen-ring window over the
            interior of 9^3 / 17^3 / 25^3 / 33^3 cubes, 8 SQP iterations, every QP
            solve timed (wall, ADMM iterations, status) through a proxy around
            ``isqp._make_qp``, the Jacobian build and constraint evaluation timed
            separately, and the QP size (free variables + one slack per row).

Both modes write JSON + a markdown table under ``benchmarks/output/windowed_3d/``.
``--case`` / ``--cfg`` restrict ``--gate`` (the 16^3 runs are long: run them one at a
time in the background).
"""

import argparse
import collections
import json
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

import dvfopt  # noqa: E402
from dvfopt.constraints import SimplexConstraint3D  # noqa: E402
from dvfopt.core.primitives import isqp as isqp_mod  # noqa: E402
from dvfopt.core.windowed import _common as _cm  # noqa: E402
from dvfopt.core.windowed import build_subproblem, windowed_correct  # noqa: E402
from dvfopt.core.windowed._inners import solve_window_inner  # noqa: E402
from dvfopt.jacobian.tetrahedron_sign import n_neg_best_diagonal, six_tet_min_volume_3d  # noqa: E402
from dvfopt.objectives import L2Objective, NoneObjective  # noqa: E402

OUT = os.path.join("benchmarks", "output", "windowed_3d")
THR = 0.01
RESEARCH = os.path.join("research", "strict_feasibility_3d", "runners", "output")
RAW = os.path.join("data", "dvfs", "b0039", "b0039_laplacian_deformation_field.npy")
CASES = {
    "slice090": os.path.join("data", "dvfs", "testcases_3d", "slice090_5x10x10.npy"),
    "slice200": os.path.join("data", "dvfs", "testcases_3d", "slice200_5x10x10.npy"),
    "slice350": os.path.join("data", "dvfs", "testcases_3d", "slice350_5x10x10.npy"),
    "subvol16": os.path.join(RESEARCH, "b0039_subvol_16_moderate.npy"),
}
COST_CUBES = {
    9: os.path.join(RESEARCH, "b0039_subvol_8_easy.npy"),
    17: os.path.join(RESEARCH, "b0039_subvol_16_moderate.npy"),
    25: os.path.join(RESEARCH, "b0039_subvol_24x24x24.npy"),
    33: "raw",
}
CFGS = {  # name -> (objective, orientation_delta)
    "l2_rows": (L2Objective, 0.01),
    "none_rows": (NoneObjective, 0.01),
    "l2_norows": (L2Objective, None),
    "none_norows": (NoneObjective, None),
}
GATE_CFG = "l2_rows"  # the engine default: in-solve L2 + edge rows

# ---- instrumentation -------------------------------------------------------
EXITS = []
_orig_inner = _cm.solve_window_inner


def _spy_inner(sub, inner, maxiter, **kw):
    kw.setdefault("trace", {})
    r = _orig_inner(sub, inner, maxiter, **kw)
    EXITS.append(str(kw["trace"].get("exit", "?")))
    return r


_cm.solve_window_inner = _spy_inner  # _solve_window looks the name up in _common's globals

QP_LOG = []  # (wall_s, iters, status)


class _TimedQP:
    """Proxy over the QP object ``isqp._make_qp`` returns: every ``solve()`` is timed."""

    def __init__(self, qp):
        self._qp = qp

    def __getattr__(self, name):
        return getattr(self._qp, name)

    def solve(self):
        t = time.perf_counter()
        res = self._qp.solve()
        QP_LOG.append((
            time.perf_counter() - t,
            int(getattr(res.info, "iter", -1)),
            str(getattr(res.info, "status", "?")).strip(),
        ))
        return res


_orig_make_qp = isqp_mod._make_qp
isqp_mod._make_qp = lambda *a, **k: _TimedQP(_orig_make_qp(*a, **k))


# ---- helpers ---------------------------------------------------------------
def _load(path):
    phi = np.load(path).astype(np.float64)
    assert phi.ndim == 4 and phi.shape[0] == 3, path
    return phi


def cut_raw(L=33):
    """The L^3 box of the raw B0039 field whose fixed-6-tet fold fraction is closest to
    10 % (scanned on a stride-L grid over the central region); returns (phi, offset, frac)."""
    vol = np.load(RAW, mmap_mode="r")
    best = None
    for z0 in range(200, 400, L):
        for y0 in range(60, 260, L):
            for x0 in range(100, 360, L):
                sub = np.asarray(vol[:, z0 : z0 + L, y0 : y0 + L, x0 : x0 + L], dtype=np.float64)
                frac = float((six_tet_min_volume_3d(sub) < THR).mean())
                if best is None or abs(frac - 0.10) < abs(best[2] - 0.10):
                    best = (sub, (z0, y0, x0), frac)
    return best


def gate_case(name, phi, cfg):
    obj_cls, od = CFGS[cfg]
    c = SimplexConstraint3D(shape=phi.shape[1:])
    mv0 = six_tet_min_volume_3d(phi)
    EXITS.clear()
    QP_LOG.clear()
    t = time.perf_counter()
    out, rep = windowed_correct(
        phi.copy(), "isqp", constraint=c, objective=obj_cls(), threshold=THR,
        orientation_delta=od, orientation_rows="edges", verbose=0,
    )
    wall = time.perf_counter() - t
    mv1 = six_tet_min_volume_3d(out)
    move = out - phi
    qp = np.array([w for w, _i, _s in QP_LOG]) if QP_LOG else np.zeros(1)
    rec = dict(
        case=name, cfg=cfg, shape=list(map(int, phi.shape[1:])),
        folds_in=int((mv0 < THR).sum()), folds_in_zero=int((mv0 <= 0).sum()),
        floor_in=int(n_neg_best_diagonal(phi, THR)), floor_in_zero=int(n_neg_best_diagonal(phi, 0.0)),
        folds_out=int(rep.folds_after), folds_out_zero=int(rep.folds_after_zero),
        floor_out=int(rep.best_diag_floor_after), floor_out_zero=int(rep.best_diag_floor_after_zero),
        min_in=float(mv0.min()), min_out=float(mv1.min()), damage=int(rep.damage),
        n_windows=int(rep.n_windows), giant_regions=int(rep.giant_regions), rounds=int(rep.rounds),
        sqp_iters=int(sum(w.inner_iters for w in rep.windows)),
        feasible_windows=int(sum(w.feasible for w in rep.windows)),
        no_tr_fallbacks=int(sum(w.fallback for w in rep.windows)),
        backend_fallbacks=int(rep.backend_fallbacks), grown_windows=int(sum(w.grows > 0 for w in rep.windows)),
        exits=dict(collections.Counter(EXITS)), n_qp=len(QP_LOG),
        qp_s_median=float(np.median(qp)), qp_s_max=float(qp.max()),
        admm_iters_median=float(np.median([i for _w, i, _s in QP_LOG])) if QP_LOG else -1.0,
        wall_s=wall, l2_move=float(np.linalg.norm(move.ravel())), l1_move=float(np.abs(move).sum()),
        max_abs_dz=float(np.abs(move[0]).max()),
    )
    assert rec["damage"] == 0, rec  # the invariant holds on every run
    if name.startswith("slice") and cfg == GATE_CFG:
        assert rec["folds_out"] == 0, rec  # the plumbing gate
    return rec


def cost_case(L, phi):
    c = SimplexConstraint3D(shape=phi.shape[1:])
    box = (1, L - 1, 1, L - 1, 1, L - 1)  # interior free, one-voxel frozen ring = the whole cube
    sub = build_subproblem(
        c, phi, box, THR, NoneObjective(), 1e-3, orientation_delta=0.01, orientation_rows="edges"
    )
    t = time.perf_counter()
    J = sub.cons_jac(sub.flat0)
    jac_s = time.perf_counter() - t
    t = time.perf_counter()
    sub.cons(sub.flat0)
    cons_s = time.perf_counter() - t
    QP_LOG.clear()
    tr = {}
    t = time.perf_counter()
    _x, nit, ok = solve_window_inner(
        sub, "isqp", 8, trace=tr, osqp_max_iter=1000, qp_backend="hybrid", step_rule="tr",
        feas_tol=5e-4, ftol=1e-2,
    )
    wall = time.perf_counter() - t
    return dict(
        L=L, n_free=int(sub.free_idx.size), n_rows=int(sub.n_enforced),
        qp_vars=int(sub.free_idx.size + sub.n_enforced), nnz_jac=int(J.nnz),
        jac_s=jac_s, cons_s=cons_s, sqp_iters=int(nit), feasible=bool(ok), wall_s=wall,
        s_per_sqp_iter=wall / max(1, nit), exit=str(tr.get("exit")),
        qp=[(round(w, 4), i, s) for w, i, s in QP_LOG],
    )


def _md(rows, cols):
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(str(r.get(k, "")) for k in cols) + " |\n" for r in rows)
    return head + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--case", choices=list(CASES))
    ap.add_argument("--cfg", choices=list(CFGS))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print(f"dvfopt from {dvfopt.__file__}", flush=True)
    if args.gate:
        for name, path in CASES.items():
            if args.case and name != args.case:
                continue
            phi = _load(path)
            for cfg in CFGS:
                if args.cfg and cfg != args.cfg:
                    continue
                rec = gate_case(name, phi, cfg)
                with open(os.path.join(OUT, f"gate_{name}_{cfg}.json"), "w") as fh:
                    json.dump(rec, fh, indent=1)
                print(json.dumps(rec), flush=True)
        rows = []
        for f in sorted(os.listdir(OUT)):
            if f.startswith("gate_") and f.endswith(".json"):
                with open(os.path.join(OUT, f)) as fh:
                    rows.append(json.load(fh))
        cols = ["case", "cfg", "folds_in", "folds_out", "folds_out_zero", "floor_in", "floor_out",
                "floor_out_zero", "min_out", "damage", "n_windows", "sqp_iters", "exits", "wall_s",
                "l2_move", "max_abs_dz"]
        with open(os.path.join(OUT, "gate.md"), "w") as fh:
            fh.write(_md(rows, cols))
    if args.cost:
        rows = []
        for L, src in COST_CUBES.items():
            if src == "raw":
                phi, off, frac = cut_raw(L)
                print(f"raw cut at {off}, fold fraction {frac:.3f}", flush=True)
            else:
                phi = _load(src)
            rec = cost_case(L, phi)
            rows.append(rec)
            print(json.dumps(rec), flush=True)
        with open(os.path.join(OUT, "cost.json"), "w") as fh:
            json.dump(rows, fh, indent=1)
        cols = ["L", "n_free", "n_rows", "qp_vars", "nnz_jac", "jac_s", "cons_s", "sqp_iters",
                "s_per_sqp_iter", "exit"]
        for r in rows:
            r["qp_s_median"] = float(np.median([w for w, _i, _s in r["qp"]])) if r["qp"] else -1
            r["admm_median"] = float(np.median([i for _w, i, _s in r["qp"]])) if r["qp"] else -1
        with open(os.path.join(OUT, "cost.md"), "w") as fh:
            fh.write(_md(rows, cols + ["qp_s_median", "admm_median"]))


if __name__ == "__main__":
    main()
```

Adjust `solve_window_inner`'s keyword names to what `dvfopt/core/windowed/_inners.py:40-56` actually declares (they are `trace`, `osqp_max_iter`, `qp_backend`, `step_rule`, `feas_tol`, `ftol` today).

- [ ] **Step 2: Run the fast parts from the MAIN checkout root** (the artefacts live there; the script is the branch's)

```bash
MAIN=C:/Users/Andy/Documents/GitHub/UCI-iGravi/deformation-field-processing
WT=C:/Users/Andy/Documents/GitHub/UCI-iGravi/dvfopt-3d-p1
cd $MAIN
for c in slice090 slice200 slice350; do PYTHONPATH=$WT $MAIN/.venv/Scripts/python.exe $WT/benchmarks/windowed_3d_gate.py --gate --case $c; done
PYTHONPATH=$WT $MAIN/.venv/Scripts/python.exe $WT/benchmarks/windowed_3d_gate.py --cost
```

Expected: `dvfopt from ...dvfopt-3d-p1...`; twelve `gate_slice*` JSON lines with `damage 0`; the three `l2_rows` testcase rows with `folds_out 0` (if a testcase does NOT reach 0 under `l2_rows`, the script's assertion fires — report the record verbatim, do not edit the assertion: that is a phase-1 finding for the orchestrator); `cost.json` with four rows (L = 9, 17, 25, 33 — the 33³ row may take minutes; the QP proxy must show `qp` entries with ADMM iteration counts and statuses).

- [ ] **Step 3: Hand the 16³ runs to the orchestrator**

Do NOT run `--case subvol16` yourself: those four runs are long and the orchestrator runs them one at a time in the background (`--gate --case subvol16 --cfg <cfg>`) after the box is quiet. Report the testcase table (`gate.md`) and the cost table (`cost.md`) verbatim in your summary.

- [ ] **Step 4: Lint and commit**

```bash
ruff check benchmarks && ruff format benchmarks/windowed_3d_gate.py
git add benchmarks/windowed_3d_gate.py
git commit -m "3D windowed, phase 1: gate + cost/convergence measurement script"
```

---

### Task 9: asv bench, CHANGELOG, CLAUDE.md, full verification

**Files:**
- Modify: `asv_bench/benchmarks/bench_solvers.py` (append after `WindowedEngine`)
- Modify: `CHANGELOG.md` (top of `## [Unreleased]`), `CLAUDE.md` (the Strategies paragraph naming `ISQPWindowedStrategy`, and the "Windowed engine knobs" list)

**Interfaces:**
- Consumes: the measured tables from Task 8 (the orchestrator supplies the 16³ rows).

- [ ] **Step 1: Add the asv bench**

Append to `asv_bench/benchmarks/bench_solvers.py`:

```python
class WindowedEngine3D:
    """Phase 1 of the 3D windowed port (SimplexConstraint3D, 'tr' step rule, 3D edge
    rows) on a planted 3D fold blob: wall, SQP iterations, L2 move and folds left."""

    timeout = 600

    def setup(self):
        from dvfopt.constraints import SimplexConstraint3D

        rng = np.random.default_rng(0)
        self.phi = np.zeros((3, 10, 16, 16))
        self.phi[:, 3:7, 5:11, 5:11] = rng.normal(0, 0.8, (3, 4, 6, 6))
        self.constraint = SimplexConstraint3D(shape=self.phi.shape[1:])

    def _run(self):
        from dvfopt.core.windowed import windowed_correct
        from dvfopt.objectives import NoneObjective

        out, rep = windowed_correct(
            self.phi.copy(), "isqp", constraint=self.constraint, objective=NoneObjective(),
            threshold=0.01, verbose=0,
        )
        assert rep.damage == 0
        return out, rep

    def time_window_3d(self):
        self._run()

    def track_sqp_iters_3d(self):
        return float(sum(w.inner_iters for w in self._run()[1].windows))

    def track_l2_move_3d(self):
        out, _rep = self._run()
        return float(np.linalg.norm((out - self.phi).ravel()))

    def track_folds_after_3d(self):
        return float(self._run()[1].folds_after)

    track_sqp_iters_3d.unit = "iterations"
    track_l2_move_3d.unit = "L2"
    track_folds_after_3d.unit = "folds"
```

Smoke it without asv: `python -c "import sys; sys.path.insert(0, 'asv_bench/benchmarks'); import bench_solvers as b; w = b.WindowedEngine3D(); w.setup(); print(w.track_sqp_iters_3d(), w.track_folds_after_3d())"`. Expected: two numbers, no assertion error.

- [ ] **Step 2: CHANGELOG**

Under `## [Unreleased]`, add first:

```markdown
### Added — 3D windowed engine, phase 1: family plumbing + certificate (`SimplexConstraint3D` in `ISQPWindowedStrategy`)

- `windowed_correct` / `ISQPWindowedStrategy` accept `SimplexConstraint3D` on
  `(3, D, H, W)` fields: `LOCALITY[SimplexConstraint3D]` (ring 1, `six_tet_min_volume_3d`
  fold map, eight-corner influenced rule, shape-cached native tet Jacobian), n-D boxes
  through three helpers (`_box_slices` / `_box_size` / `_pad_box`, byte-identical in 2D —
  verified main-vs-branch by `benchmarks/windowed_2d_identity.py`: <N> cases, every field
  `array_equal`, every report identical), 3D axial edge rows (every edge with at least one
  free endpoint, all three axes, DX_FIRST; the 3D injectivity helper's both-endpoints-free
  filter is the wrong predicate for a frozen-ring window), the `'tr'` step rule
  (`'exact_ls'` degrades on 3D), and the certificate fields `folds_after_zero` /
  `best_diag_floor_after` / `best_diag_floor_after_zero`. Not yet ported (phase 2): the
  coarse warm start, mop and re-seed are skipped on 3D, the giant cap is advisory
  (over-cap regions are solved whole), `reanchor` / `polish` raise.
- Measured (`benchmarks/windowed_3d_gate.py`, `'tr'`, threshold 0.01):

  <paste gate.md — the three testcases x four configs, plus the 16^3 rows the orchestrator supplies>

- Per-SQP-iteration cost vs window volume (one frozen-ring window, 8 SQP iterations,
  hybrid backend):

  <paste cost.md>
```

- [ ] **Step 3: CLAUDE.md**

In the Strategies paragraph, after "`WindowedWrapperStrategy(inner=…)` + `ISQPWindowedStrategy` (2D Jdet / standard simplex (2D) / bilinear / finite:" insert "**and `SimplexConstraint3D` on `(3, D, H, W)` volumes since the 3D port's phase 1 — round loop + window ladder, 3D axial edge rows, `'tr'` step rule (`'exact_ls'` degrades); the coarse warm start / mop / re-seed are skipped on 3D, the giant cap is advisory, `reanchor` / `polish` raise; Jdet3D stays with `SLSQPWindowedStrategy`** —". In the "Windowed engine knobs" list add a final bullet:

```markdown
- **3D (`SimplexConstraint3D`, phase 1 of the 3D port)** — the same engine on `(3, D, H, W)` fields: `LOCALITY[SimplexConstraint3D]` (ring 1, exact tet volumes, a free voxel influences its ≤ 8 corner cubes), boxes are per-axis `(lo, hi)` pairs, `orientation_delta` gives the 3D axial edge rows (any-free-endpoint predicate; `'edges'` only), `step_rule` is `'tr'` (the 6-tet row is cubic along a line), and `SliceReport` carries `folds_after_zero` / `best_diag_floor_after` / `best_diag_floor_after_zero` (the best-of-4-diagonals floor, `n_neg_best_diagonal`). Measured on the phase-1 artefacts: <one line: testcases 0 folds / damage 0 under the default config; the 16³ result + floor; the per-iteration cost curve>. Not yet in 3D: coarse warm start, mop, re-seed (skipped), the giant tiler (cap advisory), `reanchor` / `polish` (raise) — phase 2.
```

- [ ] **Step 4: Full verification**

```bash
ruff check dvfopt dvfopt_gui tests benchmarks && ruff format --check dvfopt dvfopt_gui tests benchmarks
mypy
python -m pytest tests/ -p no:cacheprovider
```

Expected: ruff clean, mypy clean (its `files` scope excludes the windowed package), pytest summary line with 0 failures (xfail(strict) entries from Task 5, if any, show as `xfailed`). Read the summary line yourself — never pipe pytest into `tail`.

- [ ] **Step 5: Commit**

```bash
git add asv_bench/benchmarks/bench_solvers.py CHANGELOG.md CLAUDE.md
git commit -m "3D windowed, phase 1: asv WindowedEngine3D, CHANGELOG + CLAUDE.md measured tables"
```

---

## After the tasks (orchestrator)

1. Kill the two 2.5D mop processes if still running (they contaminate the wall-clock columns), then run `--gate --case subvol16 --cfg <cfg>` for the four configs one at a time in the background from the main checkout root with `PYTHONPATH` = the worktree, and add the rows to `gate.md` / the CHANGELOG / CLAUDE.md line.
2. `/code-review` the branch; fix confirmed findings with regression tests.
3. Rebase on `origin/main`, push, `gh pr create -R UCI-iGravi/dvfopt` with: the spec link, the identity verdict, `gate.md`, `cost.md`, the U1/U2 reading (what the cost curve says about the 3D `max_window_area` / `giant_tile`, and whether `'tr'` converged — exits histogram), and what phase 2 must port. Squash-merge on green, sync local `main`, prune the worktree.
