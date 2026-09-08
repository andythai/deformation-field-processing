# 3D Windowed Engine Port — Phase 2 (tiler, coarse warm start, mop, re-seed, re-anchor in n-D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every stage of `windowed_correct` run on `(3, D, H, W)` fields — the giant-region Schwarz tiler (serial and RAS), the coarse-grid warm start, the terminal mop, the harmonic re-seed, the re-anchor stage and the per-window polish — so the 3D engine runs the whole ladder, with 3D-specific defaults sized from phase 1's cost curve, gated on a four-crop 3D pack cut from B0039, while every 2D path stays byte-identical.

**Architecture:** Phase 1 left the round loop and `_solve_window` n-D and gated the 2D-shaped stages off on 3D. This phase rewrites those stages' box arithmetic through the existing helpers (`_box_slices` / `_box_size` / `_pad_box`, `itertools.product` over per-axis ranges) so each stage is dimension-agnostic and byte-identical in 2D, removes the 3D gates, and adds two dimension-specific defaults — `giant_tile_3d=16` (16³ voxels ≈ the 2D 64² tile by count) and `mop_margin_3d=6` — resolved once at the engine entry so no 2D default changes. Nothing in `dvfopt/core/primitives/` is touched.

**Tech Stack:** numpy, scipy (`ndimage`, `sparse`, `spsolve`), the numba tet kernels, osqp + clarabel, pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-09-05-3d-windowed-engine-port-design.md` — the architecture table rows "giant tiler", "coarse-to-fine", "mop / re-seed", and the "Components and sequencing" item 2. Phase 1's measured facts (the plan argues from them): per SQP iteration 0.29 / 11.1 / 68.6 / 411 s at 9³ / 17³ / 25³ / 33³, so windows must stay near 17³; the edge rows are load-bearing; the 16³ B0039 sub-volume reaches 0 folds / 0 floor under the default config.

## Global Constraints

- **Never change 2D behaviour.** Every existing test stays green unchanged, and Task 7's main-vs-branch A/B (`benchmarks/windowed_2d_identity.py`, 21 cases) must print `IDENTITY PASS`. The 2D defaults `giant_tile=64`, `mop_margin=25`, `reanchor_tile=48`, `coarse_factor=4` and the coarse skip rule `min(shape) >= 4 * max(giant_tile, coarse_factor)` keep their values and meaning on 2D fields. Where a loop order changes form (a per-axis `itertools.product` replacing nested `for ty ... for tx`), the 2D iteration order must be identical (axis 0 outer). Where floating-point accumulation order matters (`_harmonic_fill`'s neighbour sum), the 2D order must be reproduced exactly.
- **Conventions:** 3D fields `(3, D, H, W)` `[dz, dy, dx]`; boxes are per-axis `(lo, hi)` pairs in axis order; `SimplexConstraint3D.pack == PhiPack.DX_FIRST`; threshold 0.01; `margin_delta 1e-3`; `WindowRec.patch_box` is the padded patch's per-axis pairs (phase 1).
- **3D defaults (binding, sized from phase 1):** `giant_tile_3d=16` per axis (`_fit_tile` applies per region), `mop_margin_3d=6`; `max_window_area=3000` is a grid-point count and stays; the 3D re-anchor tile is `giant_tile_3d`. The per-dimension resolution happens ONCE at `windowed_correct`'s entry (`opts.giant_tile`, `mop_margin`, `_ReanchorOpts.tile` carry the resolved values; recursive solves receive both the resolved value and the 3D knob).
- **Where to work:** the worktree `C:/Users/Andy/Documents/GitHub/UCI-iGravi/dvfopt-3d-p2` on branch `feat/3d-windowed-phase2` (from main `6511a86`). Run Python as `PYTHONPATH=C:/Users/Andy/Documents/GitHub/UCI-iGravi/dvfopt-3d-p2 C:/Users/Andy/Documents/GitHub/UCI-iGravi/deformation-field-processing/.venv/Scripts/python.exe ...` from the worktree root and verify `dvfopt.__file__` prints the worktree path before any test. Foreground tests with generous timeouts; no background watchers; never `pip install`.
- **Toolchain:** `ruff check dvfopt dvfopt_gui tests benchmarks asv_bench` and `ruff format --check` (ruff 0.16.3, target py39) must pass; `mypy` clean; no new dependencies. Payloads under `data/` are gitignored — unit tests use synthetic fields only; the crop pack is cut from `data/dvfs/b0039/b0039_laplacian_deformation_field.npy` (`(3, 528, 320, 456)` float32, `mmap_mode="r"`) into `data/dvfs/crops_3d/` (gitignored; the cutter script is tracked).
- **Commits:** one per task, `3D windowed, phase 2: <what>`, ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

## File map

| File | Responsibility in this phase |
|---|---|
| `dvfopt/core/windowed/_common.py` | n-D `_restrict` / `_prolongate` / `_coarse_warm_start`; `_fit_tile_nd`; n-D `_solve_giant_schwarz` / `_ras_cores` / `_ras_tile_task`; n-D `_mop_pass`; n-D `_harmonic_fill` / `_reseed_stage`; n-D `_reanchor_tile` / `_reanchor_pass`; the entry resolves `giant_tile_3d` / `mop_margin_3d`; the 3D gates removed |
| `dvfopt/strategies/windowed.py` | `giant_tile_3d`, `mop_margin_3d` fields + docstring |
| `tests/test_windowed_3d_stages.py` | new: all phase-2 3D tests |
| `tests/test_windowed_3d.py` | invert the phase-1 refusal of `reanchor` / `polish` |
| `benchmarks/make_hard_crops_3d.py` | new: cut + validate the 3D crop pack |
| `CHANGELOG.md`, `CLAUDE.md` | measured table, knob docs |

---

### Task 1: dimension-specific knobs and the n-D coarse warm start

**Files:**
- Modify: `dvfopt/core/windowed/_common.py` — `_InnerOpts` (line 69), `_restrict` / `_prolongate` / `_coarse_warm_start` (503-572), `windowed_correct` signature + entry (575-925), the coarse block (958)
- Modify: `dvfopt/strategies/windowed.py` — fields at ~193-224, the `solve` forwarding at ~264-290, the docstring
- Test: `tests/test_windowed_3d_stages.py` (create)

**Interfaces:**
- Produces: `windowed_correct(..., giant_tile_3d=16, mop_margin_3d=6, ...)`; `_InnerOpts.giant_tile` is the RESOLVED tile (`giant_tile_3d` on 3D, `giant_tile` on 2D) and `_InnerOpts.giant_tile_3d` carries the knob; `_restrict(phi, factor)` / `_prolongate(delta_c, shape, factor)` accept any rank; `_coarse_warm_start` works in n-D. Task 2 reads `opts.giant_tile`; Task 3 reads the resolved `mop_margin`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_windowed_3d_stages.py`:

```python
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
    assert np.allclose(coarse[0], 2.0) and np.allclose(coarse[1], -1.0) and np.allclose(coarse[2], 0.5)


def test_restrict_2d_is_unchanged():
    phi = np.zeros((2, 8, 8))
    phi[0], phi[1] = 4.0, -2.0
    coarse = engine._restrict(phi, 2)
    assert coarse.shape == (2, 4, 4) and np.allclose(coarse[0], 2.0) and np.allclose(coarse[1], -1.0)
    assert engine._restrict(np.ones((2, 7, 9)), 2).shape == (2, 3, 4)


def test_prolongate_3d_rescales_back_and_zero_pads_odd_dims():
    delta = engine._prolongate(np.ones((3, 3, 4, 2)), (7, 9, 5), 2)
    assert delta.shape == (3, 7, 9, 5)
    assert np.allclose(delta[:, :6, :8, :4], 2.0)
    assert np.all(delta[:, 6] == 0.0) and np.all(delta[:, :, 8] == 0.0) and np.all(delta[:, :, :, 4] == 0.0)


def test_restrict_prolongate_round_trip_3d_is_identity_on_a_constant_field():
    phi = np.full((3, 8, 8, 8), 3.0)
    assert np.allclose(engine._prolongate(engine._restrict(phi, 2), (8, 8, 8), 2), phi)


def test_inner_opts_resolve_the_3d_tile():
    from dvfopt.core.windowed._common import _InnerOpts

    o = _InnerOpts()
    assert o.giant_tile == 64 and o.giant_tile_3d == 16  # 2D default untouched; 16^3 ~= 64^2 by count


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
    assert np.array_equal(out[:, :16], phi[:, :16]) and np.array_equal(out[:, :, :, 48:], phi[:, :, :, 48:])


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
    s.solve(np.zeros((3, 6, 8, 8)), constraint=SimplexConstraint3D(shape=(6, 8, 8)), objective=NoneObjective(), threshold=THR)
    assert seen["giant_tile_3d"] == 12 and seen["mop_margin_3d"] == 4
    assert seen["giant_tile"] == 64 and seen["mop_margin"] == 25  # the 2D knobs are not repurposed
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_windowed_3d_stages.py -v`
Expected: `_restrict` on rank 4 fails at `hc, wc = ...` (too many values); `_prolongate` fails at `h, w = shape`; `_InnerOpts` has no `giant_tile_3d`; the strategy rejects the unknown field; the coarse test finds `coarse_folds_before == -1`.

- [ ] **Step 3: Implement**

(a) `_InnerOpts`: append after `orientation_rows: str = 'full'`:

```python
    giant_tile_3d: int = 16  # the 3D tile knob; on a 3D field `giant_tile` is RESOLVED to it at entry
```

(b) Replace `_restrict` and `_prolongate`:

```python
def _restrict(phi, factor):
    """Box-average ``factor``-blocks of a ``(C, *shape)`` field, any rank.

    Displacements are divided by ``factor`` so they stay in COARSE grid units —
    the coarse field is then an ordinary deformation field on its own grid and the
    same constraint/threshold means the same thing there. A trailing partial block
    along any axis is dropped.
    """
    coarse = [d // factor for d in phi.shape[1:]]
    trimmed = phi[(slice(None), *(slice(0, factor * n) for n in coarse))]
    blocked = trimmed.reshape(phi.shape[0], *(v for n in coarse for v in (n, factor)))
    return blocked.mean(axis=tuple(range(2, 2 * len(coarse) + 1, 2))) / factor


def _prolongate(delta_c, shape, factor):
    """Multilinear ``factor`` x upsample of a coarse-grid CORRECTION back to ``shape``.

    Displacements are multiplied by ``factor`` (the inverse of :func:`_restrict`'s
    rescale). Planes/rows/cols the integer factor cannot cover (odd sizes) stay
    zero — the fine solve handles that strip itself.
    """
    out = np.zeros((delta_c.shape[0], *shape))
    for c in range(delta_c.shape[0]):
        up = ndimage.zoom(delta_c[c] * factor, factor, order=1)
        keep = tuple(slice(0, min(n, u)) for n, u in zip(shape, up.shape))
        out[(c, *keep)] = up[keep]
    return out
```

(In 2D this is `reshape(2, hc, f, wc, f).mean(axis=(2, 4)) / factor` and the same per-channel zoom — identical arithmetic.)

(c) In `_coarse_warm_start` replace

```python
    for fy0, fy1, fx0, fx1 in boxes:
        allow[fy0:fy1, fx0:fx1] = True
```

with

```python
    for box in boxes:
        allow[_box_slices(box)] = True
```

and in its docstring change "~1/factor**2 the work" to "~1/factor**ndim the work".

(d) `windowed_correct`: add parameters `giant_tile_3d=16,` after `giant_workers=0,` and `mop_margin_3d=6,` after `mop_margin=25,`. In the `_InnerOpts(...)` construction replace the positional `giant_tile,` (5th argument) by `giant_tile_3d if is3d else giant_tile,` and append `giant_tile_3d=giant_tile_3d,` after `polish_maxiter=polish_maxiter,`. Replace the phase-1 block

```python
    if is3d:
        # Phase 1 of the 3D port: ...
        coarse_to_fine, mop_margin, reseed_rounds = False, 0, 0
```

with

```python
    if is3d:
        mop_margin = mop_margin_3d  # the 3D mop margin (a cluster + 6 per side stays under the cap)
```

Change the coarse rule `min(shape) >= 4 * max(giant_tile, coarse_factor)` to `min(shape) >= 4 * max(opts.giant_tile, coarse_factor)` (2D: `opts.giant_tile == giant_tile`, identical). In the two `sub_kw` dicts (the coarse block's and `_run_reseed`'s) add `mop_margin_3d=mop_margin_3d,` next to `mop_margin=mop_margin,`. Update the docstring: the `giant_tile` paragraph gains "On a 3D field the tile is ``giant_tile_3d`` (default 16 per axis: 16³ voxels is the 2D 64² tile by count; phase 1 measured 11 s per SQP iteration at 17³ and 69 s at 25³, so the tile must stay near 16)"; the mop paragraph gains "``mop_margin_3d`` (default 6) is the 3D margin — a residual cluster plus 6 per side is a ~13-17³ window, under the cap"; the phase-1 sentence about skipped stages in the summary line becomes "3D: every stage runs (phase 2 of the 3D port)".

(e) `dvfopt/strategies/windowed.py`: add fields `giant_tile_3d: int = 16` after `giant_tile_fit: bool = True` and `mop_margin_3d: int = 6` after `mop_margin: int = 25`; forward both in `solve` (`giant_tile_3d=self.giant_tile_3d,` after `giant_tile_fit=...`, `mop_margin_3d=self.mop_margin_3d,` after `mop_margin=...`); add two docstring entries under Parameters: `giant_tile_3d : int — Tile per axis for the 3D tiler (16: the 2D 64² tile by voxel count; phase 1's cost curve says windows must stay near 17³).` and `mop_margin_3d : int — Margin of the terminal mop on 3D fields (6; 25 would be a 55³ window).`; update the class docstring's phase-1 sentence to "Since the 3D port's phase 2 every stage runs on 3D fields with the `giant_tile_3d` / `mop_margin_3d` defaults."

- [ ] **Step 4: Run the tests and the 2D suites**

Run: `python -m pytest tests/test_windowed_3d_stages.py tests/test_windowed_coarse_to_fine.py tests/test_windowed_3d.py tests/test_windowed_strategy.py tests/test_gui_logic.py -v` (15-minute timeout).
Expected: all PASS except `tests/test_windowed_3d.py::test_3d_planted_folds_no_damage_and_untouched_voxels_bit_identical` and `test_3d_hard_blob_no_damage_under_a_short_budget`, which assert `rep.mop_windows == 0` / `reseed_rounds_run == 0` — leave them failing here (Task 3 rewrites those assertions); everything else green. `tests/test_gui_logic.py` covers the strategy-params rendering of the two new int fields.

- [ ] **Step 5: Lint and commit**

```bash
ruff check dvfopt tests && ruff format dvfopt/core/windowed/_common.py dvfopt/strategies/windowed.py tests/test_windowed_3d_stages.py
git add dvfopt/core/windowed/_common.py dvfopt/strategies/windowed.py tests/test_windowed_3d_stages.py
git commit -m "3D windowed, phase 2: giant_tile_3d / mop_margin_3d knobs, n-D coarse warm start"
```

---

### Task 2: the giant tiler in n-D (serial sweeps and RAS)

**Files:**
- Modify: `dvfopt/core/windowed/_common.py` — `_fit_tile` (1504), `_solve_giant_schwarz` (1525), `_ras_cores` (1646), `_ras_tile_task` (1656), the round-loop giant branch (1021-1052)
- Test: `tests/test_windowed_3d_stages.py`

**Interfaces:**
- Consumes: `opts.giant_tile` resolved per dimension (Task 1); `_box_slices` / `_box_size` / `_pad_box`.
- Produces: `_fit_tile_nd(extents, target, lo_frac=0.75, hi_frac=1.5) -> int`; `_solve_giant_schwarz(phi, constraint, giant_box, ...)` for any rank; `_ras_cores(tiles, step, inset)` with per-axis pairs. Task 3's mop routes over-cap boxes here.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_windowed_3d_stages.py`:

```python
def test_fit_tile_nd_matches_the_2d_fit_and_clamps():
    assert engine._fit_tile_nd((125, 152), 64) == engine._fit_tile(125, 152, 64) == 51
    assert engine._fit_tile_nd((24, 24, 24), 16) == 12  # ceil(24/2), clamped to [12, 24]
    assert engine._fit_tile_nd((10, 10, 10), 16) == 12  # a region smaller than the target keeps a usable tile


def test_ras_cores_partition_a_3d_inset_region():
    inset = (0, 10, 0, 10, 0, 10)
    tiles = [
        (z, min(z + 6, 10), y, min(y + 6, 10), x, min(x + 6, 10))
        for z in range(0, 10, 4)
        for y in range(0, 10, 4)
        for x in range(0, 10, 4)
    ]
    cores = engine._ras_cores(tiles, 4, inset)
    cover = np.zeros((10, 10, 10), int)
    for core in cores:
        cover[engine._box_slices(core)] += 1
    assert (cover == 1).all()  # a partition: every voxel in exactly one core


@needs_osqp
def test_3d_giant_region_is_tiled_and_cleared():
    # a 10^3 blob -> one connected free box of ~16^3 = 4096 > max_window_area=800 -> the tiler;
    # _fit_tile_nd((16, 16, 16), 10) == 8, step 4 -> 4 tiles per axis of <= 8^3 voxels each
    phi = _blob((22, 22, 22), (11, 11, 11), (10, 10, 10), amp=0.4, seed=3)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    assert (min_field(c, phi) < THR).sum() > 100
    out, rep = windowed_correct(
        phi.copy(), "isqp", constraint=c, objective=NoneObjective(), threshold=THR,
        verbose=0, max_window_area=800, giant_tile_3d=10, mop_margin_3d=0, reseed_rounds=0,
    )
    assert rep.giant_regions >= 1 and len(rep.giant_boxes[0]) == 6
    assert rep.n_windows > 1  # more than one tile was solved
    assert rep.damage == 0 and rep.folds_after == 0
    assert np.array_equal(out[:, :2], phi[:, :2])  # the inset keeps the far planes untouched


@needs_osqp
def test_3d_giant_workers_ras_reaches_zero_folds_damage_zero():
    phi = _blob((22, 22, 22), (11, 11, 11), (10, 10, 10), amp=0.4, seed=3)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    out, rep = windowed_correct(
        phi.copy(), "isqp", constraint=c, objective=NoneObjective(), threshold=THR,
        verbose=0, max_window_area=800, giant_tile_3d=10, giant_workers=2, mop_margin_3d=0, reseed_rounds=0,
    )
    assert rep.giant_regions >= 1 and rep.damage == 0 and rep.folds_after == 0
```

If `test_3d_giant_region_is_tiled_and_cleared` ends with `folds_after > 0` at `damage == 0`, raise `amp` down to 0.3 once; if it still does not clear, mark it `xfail(strict=True, reason=<residual, min>)` and report — the tiler's convergence on a hard synthetic region is a finding, not a plumbing bug (the crop pack in Task 6 is the measured gate).

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_windowed_3d_stages.py -v -k "fit_tile or ras_cores or giant"`
Expected: `AttributeError` (no `_fit_tile_nd`), `_ras_cores` unpacks 4-tuples, the 3D runs raise inside `_solve_giant_schwarz` (`fy0, fy1, fx0, fx1 = giant_box`).

- [ ] **Step 3: Implement**

(a) Replace `_fit_tile`'s body with a wrapper over the new n-D helper (keep its docstring):

```python
def _fit_tile_nd(extents, target, lo_frac=0.75, hi_frac=1.5):
    """:func:`_fit_tile` over a region's per-axis extents (any rank): the largest tile
    no bigger than ``target`` that covers the LONGEST extent with an integer number of
    near-equal tiles, clamped to ``[lo_frac, hi_frac] * target``."""
    longest = max(int(v) for v in extents)
    n = max(1, -(-longest // target))  # tiles along the longest side
    tile = -(-longest // n)
    return int(min(max(tile, math.ceil(lo_frac * target)), math.ceil(hi_frac * target)))


def _fit_tile(h, w, target, lo_frac=0.75, hi_frac=1.5):
    """<existing docstring, unchanged>"""
    return _fit_tile_nd((h, w), target, lo_frac, hi_frac)
```

(b) Rewrite `_solve_giant_schwarz`'s body (keep the docstring; add "Any rank: the giant box, the inset region, the tiles and the RAS cores are per-axis ``(lo, hi)`` pairs."):

```python
    opts = _InnerOpts() if opts is None else opts
    tile, max_sweeps = opts.giant_tile, opts.giant_max_sweeps
    shape = phi.shape[1:]
    ndim = len(shape)
    extents = [giant_box[2 * a + 1] - giant_box[2 * a] for a in range(ndim)]
    if opts.giant_tile_fit:
        tile = _fit_tile_nd(extents, tile)
    inset = []
    for a, n in enumerate(shape):  # inset interior faces by the ring; keep image borders
        lo, hi = giant_box[2 * a], giant_box[2 * a + 1]
        inset += [lo + (ring if lo > 0 else 0), hi - (ring if hi < n else 0)]
    inset = tuple(inset)
    overlap = 2 * ring + 2  # free regions must overlap so seams are some tile's interior
    step = max(1, tile - overlap)
    axes = [range(inset[2 * a], inset[2 * a + 1], step) for a in range(ndim)]
    tiles = [
        tuple(v for a, t in enumerate(starts) for v in (t, min(t + tile, inset[2 * a + 1])))
        for starts in itertools.product(*axes)
    ]
    gsl = _box_slices(giant_box)

    def _nonempty(b):
        return all(b[2 * a + 1] > b[2 * a] for a in range(ndim))

    prev = None
    ras = int(getattr(opts, 'giant_workers', 0) or 0)
    cores = _ras_cores(tiles, step, inset) if ras > 1 else None
    for _sweep in range(max_sweeps):
        if ras > 1:
            from dvfopt.core._pool import pool_map

            if expired is not None and expired():
                return prev if prev is not None else -1
            snap = phi.copy()
            args = [
                (snap, constraint, tb, core, threshold, objective, maxiter, ring, margin_delta, inner, opts)
                for tb, core in zip(tiles, cores)
                if _nonempty(tb) and _nonempty(core)
            ]
            for core, vals, sub_rep in pool_map(_ras_tile_task, args, ras):
                phi[(slice(None), *_box_slices(core))] = vals
                rep.windows.extend(sub_rep.windows)
                rep.backend_fallbacks += sub_rep.backend_fallbacks
                rep.patience_fallbacks += sub_rep.patience_fallbacks
            nf = int((min_field(constraint, phi)[gsl] < threshold).sum())
            if nf == 0 or (prev is not None and nf >= prev):
                return nf
            prev = nf
            continue
        for tb in tiles:
            if expired is not None and expired():
                return prev if prev is not None else -1
            if _nonempty(tb):
                _solve_window(
                    phi, constraint, tb, threshold, objective, maxiter, ring, rep,
                    margin_delta=margin_delta, allow_grow=False, inner=inner, opts=opts,
                )
        nf = int((min_field(constraint, phi)[gsl] < threshold).sum())
        if nf == 0 or (prev is not None and nf >= prev):
            return nf  # cleared, or no further progress (geometric floor)
        prev = nf
    return prev if prev is not None else 0
```

(keep the RAS / deadline comments from the current body; add `import itertools` to the module imports). `itertools.product` iterates axis 0 outermost — the 2D `for ty ... for tx` order.

(c) `_ras_cores`:

```python
def _ras_cores(tiles, step, inset):
    """Disjoint step-grid cores of the tiles (any rank): a tile starting at ``t`` on an
    axis owns ``[t, min(t + step, inset_hi))`` there — a partition of the inset region
    (tiles overlap, cores do not)."""
    ndim = len(inset) // 2
    return [
        tuple(v for a in range(ndim) for v in (tb[2 * a], min(tb[2 * a] + step, inset[2 * a + 1])))
        for tb in tiles
    ]
```

(d) `_ras_tile_task`: replace the final two lines with

```python
    csl = _box_slices(core)
    return core, phi[(slice(None), *csl)].copy(), rep
```

(e) The round loop: replace the giant branch (from `if _box_size(box) > max_window_area:` through the `if not giant_warned:` block) with the pre-phase-1 shape for BOTH dimensions —

```python
            if _box_size(box) > max_window_area:
                # too big for one QP -> overlapping-tile Schwarz decomposition
                rep.giant_regions += 1
                rep.giant_boxes.append(box)
                giant_w0 = len(rep.windows)
                _solve_giant_schwarz(
                    phi, constraint, box, threshold, objective, maxiter, ring, rep, margin_delta,
                    inner=inner, opts=opts, expired=_expired,
                )
                if record_history:
                    rep.history.append(_stage_entry("giant", giant_w0))
                _fire("giant", phi)
                continue
```

— and delete `giant_warned = False`. (Keep the argument list one-per-line as the file's style.)

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_windowed_3d_stages.py tests/test_windowed_isqp.py tests/test_windowed_ras.py tests/test_windowed_fast_robust.py -v` (15-minute timeout).
Expected: all PASS (the 2D tiler / RAS suites exercise the rewritten functions on 2D boxes).

- [ ] **Step 5: Lint and commit**

```bash
ruff check dvfopt tests && ruff format dvfopt/core/windowed/_common.py tests/test_windowed_3d_stages.py
git add dvfopt/core/windowed/_common.py tests/test_windowed_3d_stages.py
git commit -m "3D windowed, phase 2: n-D giant tiler (serial + RAS), 3D over-cap regions go to the tiler"
```

---

### Task 3: the mop and the harmonic re-seed in n-D

**Files:**
- Modify: `dvfopt/core/windowed/_common.py` — `_mop_pass` (1205), `_harmonic_fill` (1400), `_reseed_stage` (1434)
- Modify: `tests/test_windowed_3d.py` — the two assertions `rep.coarse_folds_before == -1 and rep.mop_windows == 0 and rep.reseed_rounds_run == 0`
- Test: `tests/test_windowed_3d_stages.py`

**Interfaces:**
- Consumes: `mop_margin` resolved at entry (Task 1); the n-D tiler (Task 2).
- Produces: `_mop_pass` / `_harmonic_fill(phi, mask)` / `_reseed_stage` for any rank.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_windowed_3d_stages.py`:

```python
def test_harmonic_fill_3d_is_discrete_harmonic_and_keeps_the_boundary():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(3, 12, 12, 12))
    mask = np.zeros((12, 12, 12), bool)
    mask[3:9, 3:9, 3:9] = True
    ref = phi.copy()
    engine._harmonic_fill(phi, mask)
    assert np.array_equal(phi[:, ~mask], ref[:, ~mask])
    inner = mask.copy()
    inner[[3, 8], :, :] = inner[:, [3, 8], :] = inner[:, :, [3, 8]] = False
    lap = 6 * phi[:, 1:-1, 1:-1, 1:-1]
    for ax in (1, 2, 3):
        lo = [slice(1, -1)] * 3
        hi = [slice(1, -1)] * 3
        lo[ax - 1], hi[ax - 1] = slice(0, -2), slice(2, None)
        lap = lap - phi[(slice(None), *lo)] - phi[(slice(None), *hi)]
    assert np.abs(lap[:, inner[1:-1, 1:-1, 1:-1]]).max() < 1e-9


def test_harmonic_fill_2d_neighbour_order_is_unchanged():
    # the 2D neighbour order (x+1, x-1, y+1, y-1) decides the float accumulation of the
    # boundary sum; pin it against a hand-built matrix so the n-D loop cannot drift
    rng = np.random.default_rng(1)
    phi = rng.normal(size=(2, 6, 6))
    mask = np.zeros((6, 6), bool)
    mask[2:4, 2:4] = True
    a = phi.copy()
    engine._harmonic_fill(a, mask)
    b = phi.copy()
    ys, xs = np.nonzero(mask)
    rhs = np.zeros((2, ys.size))
    for k, (y, x) in enumerate(zip(ys, xs)):
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            if not mask[y + dy, x + dx]:
                rhs[:, k] += phi[:, y + dy, x + dx]
    # 4 interior pixels, each with 2 masked neighbours: solve 4 I - A directly
    A = np.zeros((4, 4))
    for k, (y, x) in enumerate(zip(ys, xs)):
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            if mask[y + dy, x + dx]:
                j = int(np.flatnonzero((ys == y + dy) & (xs == x + dx))[0])
                A[k, j] = -1.0
        A[k, k] = 4.0
    for ch in range(2):
        b[ch, ys, xs] = np.linalg.solve(A, rhs[ch])
    assert np.allclose(a, b)


@needs_osqp
def test_3d_mop_fires_on_a_residual_and_never_damages():
    # a hard blob under a short round-loop budget leaves a residual the mop re-windows;
    # blob y 21..27 -> free box y 17..31 -> two grows reach y 9 -> mop box (margin 6) y >= 11:
    # nothing below y = 8 is ever freed
    phi = _blob((14, 40, 30), (7, 24, 15), (4, 6, 6), amp=1.4, seed=0)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    out, rep = windowed_correct(
        phi.copy(), "isqp", constraint=c, objective=NoneObjective(), threshold=THR,
        verbose=0, maxiter=10, fallback_maxiter=10, max_rounds=1, reseed_rounds=0,
    )
    assert rep.damage == 0 and np.isfinite(out).all()
    assert rep.mop_windows >= 1
    assert rep.mop_cleared >= 0  # the mop only helps
    assert np.array_equal(out[:, :, :6], phi[:, :, :6])


@needs_osqp
def test_3d_reseed_stage_runs_on_a_residual_and_books_touched():
    phi = _blob((14, 40, 30), (7, 24, 15), (4, 6, 6), amp=1.4, seed=0)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    out, rep = windowed_correct(
        phi.copy(), "isqp", constraint=c, objective=NoneObjective(), threshold=THR,
        verbose=0, maxiter=10, fallback_maxiter=10, max_rounds=1, mop_margin_3d=0,
    )
    assert rep.damage == 0
    if rep.reseed_folds_before > 0:
        assert rep.reseed_rounds_run >= 1 and rep.reseed_px > 0
        assert rep.reseed_folds_after <= rep.reseed_folds_before
```

In `tests/test_windowed_3d.py::test_3d_planted_folds_no_damage_and_untouched_voxels_bit_identical` replace the line `assert rep.coarse_folds_before == -1 and rep.mop_windows == 0 and rep.reseed_rounds_run == 0` with `assert rep.coarse_folds_before == -1  # too small for the coarse stage (min(shape) < 64)`.

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_windowed_3d_stages.py -v -k "harmonic or mop or reseed"`
Expected: `_harmonic_fill` fails at `H, W = mask.shape` on 3D; the 3D mop/reseed runs raise inside `_mop_pass` (`H, W = phi.shape[1:]`) / `_reseed_stage` (2D corner slicing).

- [ ] **Step 3: Implement**

(a) `_mop_pass` body (keep the docstring, add "Any rank."):

```python
    shape = phi.shape[1:]
    whole_cap = 4 * max_window_area  # the mop is allowed much larger single QPs
    for _sweep in range(max_sweeps):
        mask = pixel_fold_mask(constraint, phi, threshold)
        n = int(mask.sum())
        if n == 0:
            break
        lbl, _ = ndimage.label(mask)  # raw residual clusters (per connected component)
        for sl in ndimage.find_objects(lbl):
            box = tuple(
                v
                for s, n_ax in zip(sl, shape)
                for v in (max(0, s.start - mop_margin), min(n_ax, s.stop + mop_margin))
            )
            touched[_box_slices(_pad_box(box, shape, ring))] = True
            rep.mop_windows += 1
            if _box_size(box) > whole_cap:
                _solve_giant_schwarz(
                    phi, constraint, box, threshold, objective, maxiter, ring, rep, margin_delta,
                    inner=inner, opts=opts,
                )
            else:
                # <keep the existing comment block about ONE attempt for big windows>
                big = _box_size(box) > max_window_area
                _solve_window(
                    phi, constraint, box, threshold, objective, maxiter, ring, rep,
                    margin_delta=margin_delta, inner=inner,
                    opts=replace(opts, ladder=False) if big else opts,
                )
        if int(pixel_fold_mask(constraint, phi, threshold).sum()) >= n:
            break  # no progress -> genuine local floor
```

(b) `_harmonic_fill` (any rank; the neighbour order must reproduce the 2D order `(0,1), (0,-1), (1,0), (-1,0)` — i.e. the LAST axis first, `+1` before `-1`):

```python
def _harmonic_fill(phi, mask):
    """Replace ``phi[:, mask]`` by the discrete-harmonic (2·ndim-neighbour Laplacian)
    interpolation of ``phi`` on the mask's boundary, in place — any rank. One sparse
    solve per channel over the masked grid points (a few hundred on real residuals).
    The neighbour order (last axis first, ``+1`` before ``-1``) is the 2D engine's
    ``(0, 1), (0, -1), (1, 0), (-1, 0)`` and fixes the boundary sum's float order."""
    shape = mask.shape
    ndim = len(shape)
    pts = np.nonzero(mask)
    n = len(pts[0])
    if n == 0:
        return
    idx = np.full(shape, -1, dtype=np.int64)
    idx[pts] = np.arange(n)
    offsets = [
        tuple((d if a == ax else 0) for a in range(ndim))
        for ax in reversed(range(ndim))
        for d in (1, -1)
    ]
    rows, cols, vals = [], [], []
    rhs = np.zeros((phi.shape[0], n))
    for k, p in enumerate(zip(*pts)):
        deg = 0
        for off in offsets:
            q = tuple(int(pi) + oi for pi, oi in zip(p, off))
            if not all(0 <= qi < ni for qi, ni in zip(q, shape)):
                continue
            deg += 1
            if mask[q]:
                rows.append(k)
                cols.append(int(idx[q]))
                vals.append(-1.0)
            else:
                rhs[:, k] += phi[(slice(None), *q)]
        rows.append(k)
        cols.append(k)
        vals.append(float(deg))
    lap = sparse.csc_matrix((vals, (rows, cols)), shape=(n, n))
    for ch in range(phi.shape[0]):
        phi[(ch, *pts)] = spsolve(lap, rhs[ch])
```

(c) `_reseed_stage`: replace the corner block

```python
        corners = fold.copy()
        corners[1:, :] |= fold[:-1, :]
        corners[:, 1:] |= fold[:, :-1]
        corners[1:, 1:] |= fold[:-1, :-1]
```

with

```python
        corners = fold.copy()  # a cell's 2**ndim corner grid points: OR the +1 shift per axis
        for ax in range(fold.ndim):
            shifted = np.zeros_like(corners)
            dst = [slice(None)] * fold.ndim
            src = [slice(None)] * fold.ndim
            dst[ax], src[ax] = slice(1, None), slice(None, -1)
            shifted[tuple(dst)] = corners[tuple(src)]
            corners |= shifted
```

(in 2D this yields exactly `{fold, fold shifted +y, fold shifted +x, fold shifted +y+x}`), and replace the polish-footprint loop

```python
        for w in rep_in.windows:  # the polish's enforced footprints
            py0, px0 = max(0, w.fy0 - ring), max(0, w.fx0 - ring)
            touched[py0 : py0 + w.ph, px0 : px0 + w.pw] = True
```

with

```python
        for w in rep_in.windows:  # the polish's enforced footprints (= the padded patches)
            touched[_box_slices(w.patch_box)] = True
```

(`patch_box[0] == max(0, fy0 - ring)` and `patch_box[1] == patch_box[0] + ph` — the same slab). Update the docstring ("its +1 row/column" → "its +1 shift along every axis"; "Any rank.").

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_windowed_3d_stages.py tests/test_windowed_3d.py tests/test_windowed_reseed.py tests/test_windowed_isqp.py tests/test_windowed_escape.py -v` (20-minute timeout).
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
ruff check dvfopt tests && ruff format dvfopt/core/windowed/_common.py tests/test_windowed_3d_stages.py tests/test_windowed_3d.py
git add dvfopt/core/windowed/_common.py tests/test_windowed_3d_stages.py tests/test_windowed_3d.py
git commit -m "3D windowed, phase 2: n-D mop and harmonic re-seed"
```

---

### Task 4: re-anchor and polish on 3D

**Files:**
- Modify: `dvfopt/core/windowed/_common.py` — `_reanchor_tile` (1282), `_reanchor_pass` (1325), the entry raise for `reanchor` / `polish` (873-876), the `_ReanchorOpts` construction (1146)
- Modify: `tests/test_windowed_3d.py::test_3d_refuses_the_unported_stages`
- Test: `tests/test_windowed_3d_stages.py`

**Interfaces:**
- Consumes: `giant_tile_3d` (the 3D re-anchor tile), `_box_slices`.
- Produces: `reanchor='l2'|'l1'` and `polish='l2'|'l1'` work on 3D fields.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_windowed_3d_stages.py`:

```python
@needs_osqp
@pytest.mark.parametrize("kind", ["l2", "l1"])
def test_3d_reanchor_reduces_the_move_and_keeps_zero_folds(kind):
    phi = _blob((8, 40, 20), (4, 20, 10), (4, 6, 6), amp=0.5)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    base, rep0 = windowed_correct(phi.copy(), "isqp", constraint=c, objective=NoneObjective(), threshold=THR, verbose=0)
    assert rep0.folds_after == 0
    out, rep = windowed_correct(
        phi.copy(), "isqp", constraint=c, objective=NoneObjective(), threshold=THR, verbose=0,
        reanchor=kind, giant_tile_3d=12,
    )
    assert rep.folds_after == 0 and rep.damage == 0
    assert rep.reanchor_sweeps_run >= 1 and rep.reanchor_tiles >= 1
    assert rep.reanchor_l2_after <= rep.reanchor_l2_before
    moved0 = np.any(base != phi, axis=0)
    assert not np.any(out != phi, axis=0)[~moved0].any()  # only voxels the main solve moved


@needs_osqp
def test_3d_polish_runs_and_keeps_zero_folds():
    phi = _blob((8, 40, 20), (4, 20, 10), (4, 6, 6), amp=0.5)
    c = SimplexConstraint3D(shape=phi.shape[1:])
    out, rep = windowed_correct(
        phi.copy(), "isqp", constraint=c, objective=NoneObjective(), threshold=THR, verbose=0, polish="l2"
    )
    assert rep.folds_after == 0 and rep.damage == 0 and rep.polish_windows >= 1
```

In `tests/test_windowed_3d.py::test_3d_refuses_the_unported_stages` delete the two `reanchor="l2"` / `polish="l2"` `pytest.raises` blocks (keep the `orientation_rows="full"` one) and rename the test `test_3d_refuses_the_full_rows_kind`.

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_windowed_3d_stages.py -v -k "reanchor or polish"`
Expected: `ValueError: reanchor and polish are not yet supported on 3D fields`.

- [ ] **Step 3: Implement**

(a) Delete the entry block `if is3d and (reanchor != 'none' or polish is not None): raise ValueError(...)`.

(b) `_ReanchorOpts` construction: `_ReanchorOpts(reanchor, reanchor_maxiter, reanchor_sweeps, giant_tile_3d if is3d else reanchor_tile)` — with a trailing comment `# the 3D re-anchor tile is the 3D giant tile`.

(c) `_reanchor_tile`: replace `py0, py1, px0, px1 = sub.patch_box` / `phi_ref[:, py0:py1, px0:px1]` / `dst = phi[:, py0:py1, px0:px1]` with `psl = (slice(None), *_box_slices(sub.patch_box))`, `phi_ref[psl]`, `dst = phi[psl]`.

(d) `_reanchor_pass`: replace the bbox / tiling lines

```python
    ys, xs = np.nonzero(moved)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    ...
        for ty in range(y0, y1, step):
            for tx in range(x0, x1, step):
                box = (ty, min(ty + tile, y1), tx, min(tx + tile, x1))
                if not moved[box[0] : box[1], box[2] : box[3]].any():
                    continue
```

with

```python
    pts = np.nonzero(moved)
    lo = [int(p.min()) for p in pts]
    hi = [int(p.max()) + 1 for p in pts]
    ...
        for starts in itertools.product(*(range(a, b, step) for a, b in zip(lo, hi))):
            box = tuple(v for t, b in zip(starts, hi) for v in (t, min(t + tile, b)))
            if not moved[_box_slices(box)].any():
                continue
```

(the loop body's remaining lines are unchanged; dedent them by one level). Update the docstring: "tile the MOVED region (``reanchor_tile`` in 2D, ``giant_tile_3d`` in 3D, overlapping by 8)".

Also update `windowed_correct`'s docstring where it says `reanchor` / `polish` raise on 3D (the phase-1 sentence) — they now run.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_windowed_3d_stages.py tests/test_windowed_3d.py tests/test_windowed_reanchor.py tests/test_windowed_polish.py -v` (20-minute timeout).
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
ruff check dvfopt tests && ruff format dvfopt/core/windowed/_common.py tests/test_windowed_3d_stages.py tests/test_windowed_3d.py
git add dvfopt/core/windowed/_common.py tests/test_windowed_3d_stages.py tests/test_windowed_3d.py
git commit -m "3D windowed, phase 2: re-anchor and polish on 3D fields"
```

---

### Task 5: the 2D byte-identity A/B on the phase-2 head

**Files:**
- None created; runs `benchmarks/windowed_2d_identity.py` (phase 1).

The saved reference set `benchmarks/output/identity_2d/main/` in the main checkout was produced by the pre-phase-1 engine and is byte-identical to today's main (proved in phase 1), so only the branch run is needed.

- [ ] **Step 1: Run the branch `--out` and the compare, from the MAIN checkout root**

```bash
MAIN=C:/Users/Andy/Documents/GitHub/UCI-iGravi/deformation-field-processing
WT=C:/Users/Andy/Documents/GitHub/UCI-iGravi/dvfopt-3d-p2
cd $MAIN
PYTHONPATH=$WT $MAIN/.venv/Scripts/python.exe $WT/benchmarks/windowed_2d_identity.py --out $MAIN/benchmarks/output/identity_2d/p2 
PYTHONPATH=$WT $MAIN/.venv/Scripts/python.exe $WT/benchmarks/windowed_2d_identity.py --compare $MAIN/benchmarks/output/identity_2d/main $MAIN/benchmarks/output/identity_2d/p2
```

Expected: `dvfopt from ...dvfopt-3d-p2...`, `7 cases`, `21 keys`, every key `field same, report same` (one-sided keys = the three certificate fields), `IDENTITY PASS`. Any `DIFFERENT` is a 2D behaviour change in Tasks 1-4 — find it (the changed stage is named by the case: `giant120` → the tiler, `coarse260` → the coarse stage, the crops → mop / re-seed / ladder) and fix it in that task's code; do not proceed with a FAIL.

- [ ] **Step 2: Record the verdict** in the task report (paste the compare output).

---

### Task 6: the 3D crop pack and the phase-2 gate

**Files:**
- Create: `benchmarks/make_hard_crops_3d.py`
- Output (gitignored): `data/dvfs/crops_3d/{twist,cluster,sliver,moderate}.npy` and `benchmarks/output/windowed_3d/crops_*.json` / `crops.md` in the main checkout

**Interfaces:**
- Consumes: the raw field `data/dvfs/b0039/b0039_laplacian_deformation_field.npy` (`(3, 528, 320, 456)` float32, load with `mmap_mode="r"`); `windowed_correct` on the phase-2 engine; `six_tet_min_volume_3d`, `n_neg_best_diagonal`.
- Produces: the reference table for the CHANGELOG (Task 7).

The crop offsets come from a stride-12 scan of 24³ boxes of the raw field (2026-09-07; fixed-6-tet stats at threshold 0.01):

| name | offset (z, y, x) | why |
|---|---|---|
| `twist` | (372, 148, 260) | min tet volume −13.4 in a sparse region (2.8 % negative, 3.3 % below threshold) |
| `cluster` | (336, 160, 116) | 25 % of cubes below threshold, 19 % negative, min −8.3 |
| `sliver` | (204, 124, 260) | 880 cubes in [−0.001, 0.01) with only 1.8 % negative: the sub-threshold pathology |
| `moderate` | (228, 136, 140) | 10 % below threshold, 8.6 % negative, min −7.2 |

- [ ] **Step 1: Write the script**

```python
"""Cut the 3D hard-crop pack from the raw B0039 field and run the phase-2 gate on it.

Four 24^3 crops (offsets from a stride-12 scan of the raw field, 2026-09-07):
twist (min tet volume -13.4, sparse), cluster (25 % of cubes below threshold),
sliver (880 cubes in [-0.001, 0.01), 1.8 % negative), moderate (10 % below).
``--build-only`` cuts them into data/dvfs/crops_3d/ (gitignored). Without it the
script also runs ``windowed_correct`` on every crop under the configs given by
``--cfg`` (default: l2_rows, the engine default) and writes crops_<name>_<cfg>.json +
crops.md to benchmarks/output/windowed_3d/ — the reference table of the 3D port's
phase 2 (0 folds / damage 0 is the gate; wall and L2 move are the reference).
"""

import argparse
import json
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

import dvfopt  # noqa: E402
from dvfopt.constraints import SimplexConstraint3D  # noqa: E402
from dvfopt.core.windowed import windowed_correct  # noqa: E402
from dvfopt.jacobian.tetrahedron_sign import n_neg_best_diagonal, six_tet_min_volume_3d  # noqa: E402
from dvfopt.objectives import L2Objective, NoneObjective  # noqa: E402

RAW = os.path.join("data", "dvfs", "b0039", "b0039_laplacian_deformation_field.npy")
OUT_CROPS = os.path.join("data", "dvfs", "crops_3d")
OUT = os.path.join("benchmarks", "output", "windowed_3d")
THR = 0.01
L = 24
CROPS = {
    "twist": (372, 148, 260),
    "cluster": (336, 160, 116),
    "sliver": (204, 124, 260),
    "moderate": (228, 136, 140),
}
CFGS = {"l2_rows": (L2Objective, 0.01), "none_rows": (NoneObjective, 0.01)}


def build():
    os.makedirs(OUT_CROPS, exist_ok=True)
    vol = np.load(RAW, mmap_mode="r")
    for name, (z0, y0, x0) in CROPS.items():
        phi = np.asarray(vol[:, z0 : z0 + L, y0 : y0 + L, x0 : x0 + L], dtype=np.float64)
        np.save(os.path.join(OUT_CROPS, f"{name}.npy"), phi)
        mv = six_tet_min_volume_3d(phi)
        print(
            f"{name}: offset {(z0, y0, x0)} folds {(mv < THR).sum()} (neg {(mv < 0).sum()}), "
            f"floor {n_neg_best_diagonal(phi, THR)}, min {mv.min():+.3f}",
            flush=True,
        )


def run(name, cfg, giant_tile_3d, mop_margin_3d):
    obj_cls, od = CFGS[cfg]
    phi = np.load(os.path.join(OUT_CROPS, f"{name}.npy"))
    c = SimplexConstraint3D(shape=phi.shape[1:])
    mv0 = six_tet_min_volume_3d(phi)
    t = time.perf_counter()
    out, rep = windowed_correct(
        phi.copy(), "isqp", constraint=c, objective=obj_cls(), threshold=THR, orientation_delta=od,
        orientation_rows="edges", giant_tile_3d=giant_tile_3d, mop_margin_3d=mop_margin_3d, verbose=0,
    )
    wall = time.perf_counter() - t
    move = out - phi
    rec = dict(
        case=name, cfg=cfg, giant_tile_3d=giant_tile_3d, mop_margin_3d=mop_margin_3d,
        folds_in=int((mv0 < THR).sum()), floor_in=int(n_neg_best_diagonal(phi, THR)),
        folds_out=int(rep.folds_after), folds_out_zero=int(rep.folds_after_zero),
        floor_out=int(rep.best_diag_floor_after), floor_out_zero=int(rep.best_diag_floor_after_zero),
        min_out=float(rep.min_after), damage=int(rep.damage), rounds=int(rep.rounds),
        n_windows=int(rep.n_windows), giant_regions=int(rep.giant_regions), mop_windows=int(rep.mop_windows),
        mop_cleared=int(rep.mop_cleared), reseed_rounds_run=int(rep.reseed_rounds_run),
        coarse_folds_before=int(rep.coarse_folds_before),
        sqp_iters=int(sum(w.inner_iters for w in rep.windows)), wall_s=wall,
        l2_move=float(np.linalg.norm(move.ravel())), l1_move=float(np.abs(move).sum()),
    )
    assert rec["damage"] == 0, rec
    with open(os.path.join(OUT, f"crops_{name}_{cfg}_t{giant_tile_3d}_m{mop_margin_3d}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec), flush=True)
    return rec


def table():
    rows = []
    for f in sorted(os.listdir(OUT)):
        if f.startswith("crops_") and f.endswith(".json"):
            with open(os.path.join(OUT, f)) as fh:
                rows.append(json.load(fh))
    cols = ["case", "cfg", "giant_tile_3d", "mop_margin_3d", "folds_in", "floor_in", "folds_out", "floor_out",
            "min_out", "damage", "rounds", "n_windows", "giant_regions", "mop_windows", "reseed_rounds_run",
            "sqp_iters", "wall_s", "l2_move"]
    with open(os.path.join(OUT, "crops.md"), "w") as fh:
        fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(str(r.get(k, "")) for k in cols) + " |\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--case", choices=list(CROPS))
    ap.add_argument("--cfg", choices=list(CFGS), default="l2_rows")
    ap.add_argument("--giant-tile-3d", type=int, default=16)
    ap.add_argument("--mop-margin-3d", type=int, default=6)
    a = ap.parse_args()
    print(f"dvfopt from {dvfopt.__file__}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    if not os.path.isdir(OUT_CROPS) or a.build_only:
        build()
    if a.build_only:
        return
    for name in CROPS:
        if a.case and name != a.case:
            continue
        run(name, a.cfg, a.giant_tile_3d, a.mop_margin_3d)
    table()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Cut the crops and run the fast part**, from the MAIN checkout root with `PYTHONPATH` = the worktree: `--build-only` first (prints the four crops' input stats), then `--case twist` and `--case sliver` under the default config (each in the foreground, 600000 ms timeout; if one does not finish, report the JSON lines printed so far and stop — the controller runs the rest). Do NOT run `cluster` / `moderate` or the A/B configs: the controller runs them in the background.

- [ ] **Step 3: Lint and commit**

```bash
ruff check benchmarks && ruff format benchmarks/make_hard_crops_3d.py
git add benchmarks/make_hard_crops_3d.py
git commit -m "3D windowed, phase 2: 3D hard-crop pack cutter + gate"
```

Report the input stats of the four crops and every JSON record produced.

---

### Task 7: docs and the measured table (after the orchestrator's runs)

**Files:**
- Modify: `CHANGELOG.md` (top of `## [Unreleased]`), `CLAUDE.md` (the 3D bullet of "Windowed engine knobs" and the `ISQPWindowedStrategy` sentence in the Strategies paragraph), `docs/superpowers/specs/2026-09-05-3d-windowed-engine-port-design.md` (mark phase 2's items done in the table? No — the spec stays a design; leave it).

- [ ] **Step 1: CHANGELOG** — add first under `## [Unreleased]`:

```markdown
### Added — 3D windowed engine, phase 2: every stage runs on 3D fields

- The giant-region Schwarz tiler (serial sweeps and `giant_workers` RAS), the coarse-grid warm start, the terminal mop, the harmonic re-seed, the re-anchor stage and the per-window polish are dimension-agnostic (`itertools.product` over per-axis ranges through the phase-1 box helpers; byte-identical in 2D — `benchmarks/windowed_2d_identity.py`: 21 cases, `IDENTITY PASS`). Two 3D defaults, sized from phase 1's cost curve: `giant_tile_3d=16` per axis (16³ voxels is the 2D 64² tile by count) and `mop_margin_3d=6`; the 3D re-anchor tile is `giant_tile_3d`. The phase-1 gates (advisory cap, skipped stages, refused `reanchor` / `polish`) are gone.
- Measured (`benchmarks/make_hard_crops_3d.py`, four 24³ crops of the raw B0039 field, threshold 0.01, `'tr'`):

  <paste crops.md>
```

- [ ] **Step 2: CLAUDE.md** — in the 3D bullet replace the sentence "Not yet in 3D: coarse warm start, mop, re-seed (skipped), the giant tiler (cap advisory), `reanchor` / `polish` (raise) — phase 2." with "Since phase 2 every stage runs on 3D fields: the tiler uses `giant_tile_3d` (16 per axis; `max_window_area` is a voxel count), the mop `mop_margin_3d` (6), the re-anchor tile is `giant_tile_3d`. Measured on the 24³ crop pack: <one line from crops.md: which crops reach 0 folds / 0 floor at damage 0, wall, L2>." In the Strategies paragraph replace "the coarse warm start / mop / re-seed are skipped on 3D, the giant cap is advisory, `reanchor` / `polish` raise" with "every stage runs (`giant_tile_3d` / `mop_margin_3d` defaults)".

- [ ] **Step 3: Verification** — `ruff check dvfopt dvfopt_gui tests benchmarks asv_bench`, `ruff format --check ...`, `mypy`, and `python -m pytest tests/ -n 4 -p no:cacheprovider -q` (read the summary line yourself). Then commit:

```bash
git add CHANGELOG.md CLAUDE.md
git commit -m "3D windowed, phase 2: CHANGELOG + CLAUDE.md measured crop-pack table"
```

---

## After the tasks (orchestrator)

1. Run the long crop-pack measurements in the background from the main checkout root: `cluster` and `moderate` under `l2_rows` (the gate), all four under `none_rows` (the second column), then the tile A/B on `cluster` (`--giant-tile-3d 12` and `20`) and, only if any crop leaves a residual, the mop A/B (`--mop-margin-3d 4` / `8`) on that crop. Regenerate `crops.md`, hand the rows to Task 7.
2. Whole-branch review, `/code-review`, fix waves as in phase 1; rebase on `origin/main`, push, `gh pr create -R UCI-iGravi/dvfopt`, squash-merge on green.
