# 3D Windowed Engine — Auto Routing and GUI Row Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the tuned 3D windowed engine (phases 1–3, PRs #119–#121) on a default path: measure it head-to-head against what `auto_strategy` picks today for `SimplexConstraint3D` (barrier at the mild/moderate tiers, the 3D wallbreakers at the extreme tier) and the packaged `correct_dvf_3d` pipeline on twelve 3D artefacts, and — only where it wins under a pre-registered criterion — add the `auto_strategy` 3D rule, the GUI 3D menu row, the CLI help and docs.

**Architecture:** The engine is reachable today only by explicit label (`correct_dvf(phi, constraint='simplex_3d', strategy='isqp_windowed')` works). `auto_strategy`'s 3D branch (`dvfopt/solver.py` ~lines 636-645) never calls `_isqp_windowed_ok`; the GUI's `_METHOD_SPECS_TET3D` (`dvfopt_gui/_shared.py:255-266`) has no windowed row. The measurement decides the rule; the rule mirrors the 2D table's shape (a windowed row gated on `osqp`), keyed on fold COUNT rather than depth because the windowed engine's cost is per fold region while the barrier's stall condition (depth) is irrelevant to it. No engine code changes; the 2D routing branch is untouched.

**Tech Stack:** numpy / scipy, OSQP + Clarabel, pytest; the cohort loaders in `benchmarks/benchmark_utils.py`; the artefacts under `data/dvfs/crops_3d/` (gitignored) and `data/dvfs/cohort/<brain>/laplacian_exterior/laplacian_deformation_field.npz` (7 brains, `(3, 528, 320, 456)` float32, key `arr`).

**Spec:** the phase-3 close-out ruling (ledger `.superpowers/sdd/2026-09-08-3d-windowed-engine-port-phase3/progress.md`, main checkout, gitignored) and the routing table in `CLAUDE.md` ("3D routing is untouched") — this plan is the measurement that changes that sentence or leaves it.

## Global Constraints

- **The 2D routing branch of `auto_strategy` does not change** (`tests/test_unified.py`, `tests/test_m14_schwarz.py`, `tests/test_bilinear_constraint.py` pin it and must stay green untouched). **The windowed engine itself does not change** (`dvfopt/core/windowed/` untouched; `benchmarks/windowed_2d_identity.py` stays `IDENTITY PASS` trivially).
- **The routing change is gated by the pre-registered win criterion in Task 2** — written before the runs. If the engine does not win, Tasks 3's rule is NOT added; the GUI row (Task 4) and the measurement docs land either way.
- Every measured record carries: fixed-6-tet folds out at threshold 0.01 AND at 0, the best-diagonal floor, folds CREATED (cells fold-free on input, folded on output — the cross-method damage analogue), moved-voxel fraction, wall, L2 and L1 move, and for the windowed engine `report.damage`.
- Python `>=3.10`; ruff 0.16.3 (`ruff check dvfopt dvfopt_gui tests benchmarks asv_bench`, `ruff format --check`), `mypy` clean.
- Commit trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; PR to `UCI-iGravi/dvfopt` (never heemmanshuu).
- Long runs (> 10 min) are the controller's, from the MAIN checkout root with `PYTHONPATH=<detached snapshot worktree>`; subagents never start them; subagent verification runs in the FOREGROUND with explicit timeouts.

---

### Task 1: The head-to-head driver and the cohort crop cutter

**Files:**
- Create: `benchmarks/windowed_3d_vs_auto.py`
- Reuse (import, do not edit): `benchmarks/benchmark_utils.py` (`load_cohort_field(brain, variant='laplacian_exterior')` → `(3, D, H, W)`), `dvfopt.jacobian.tetrahedron_sign` (`six_tet_min_volume_3d`, `n_neg_best_diagonal`), `dvfopt.solver` (`correct_dvf`, `auto_strategy`), `dvfopt.pipeline_3d.correct_dvf_3d`

**Interfaces:**
- Produces: CLI `python benchmarks/windowed_3d_vs_auto.py --cut BRAIN [--L 24] [--band 0.08 0.15]` writing `data/dvfs/crops_3d/<brain>_moderate.npy`; `--case NAME --method METHOD [--objective l2]` writing `benchmarks/output/windowed_3d/h2h_<case>_<method>_<objective>.json`; `--table` rendering `benchmarks/output/windowed_3d/h2h.md`.
- `METHODS = ('isqp_windowed', 'barrier', 'm14_3d', 'm10_3d', 'pipeline3d')`; `CASES` = every `*.npy` under `data/dvfs/crops_3d/` plus `subvol16` (`research/strict_feasibility_3d/runners/output/b0039_subvol_16_moderate.npy`).
- Record keys (Task 2 and the docs read them by name): `case, method, objective, auto_pick, shape, n_voxels, folds_in, folds_in_zero, floor_in, min_in, folds_out, folds_out_zero, floor_out, min_out, new_folds, moved_frac, damage (windowed only, else -1), sqp_iters (windowed only, else -1), rounds (windowed only, else -1), wall_s, l2_move, l1_move, feasible (bool: folds_out == 0 and floor_out == 0)`.

- [ ] **Step 1: Write the driver**

```python
"""Head-to-head on the 3D artefacts: the windowed engine vs what ``auto`` picks today
(``barrier`` at the mild / moderate tiers, the 3D wallbreakers ``m14_3d`` / ``m10_3d`` at the
extreme tier) and the packaged ``correct_dvf_3d`` pipeline. ONE run per call; JSON records +
a markdown table. Run from the repo root.

    python benchmarks/windowed_3d_vs_auto.py --cut B0032            # data/dvfs/crops_3d/B0032_moderate.npy
    python benchmarks/windowed_3d_vs_auto.py --case twist --method barrier
    python benchmarks/windowed_3d_vs_auto.py --table                # h2h.md from every h2h_*.json

``auto_pick`` in every record is what ``auto_strategy`` resolves for that artefact and objective
today, so the table shows the current default beside every method. ``new_folds`` counts cells
fold-free on input and folded on output — the cross-method analogue of the windowed engine's
``damage`` (which is a stronger statement: folds created OUTSIDE its touched set, 0 by construction).
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_utils import load_cohort_field  # noqa: E402

import numpy as np  # noqa: E402

from dvfopt.constraints import SimplexConstraint3D  # noqa: E402
from dvfopt.jacobian.tetrahedron_sign import n_neg_best_diagonal, six_tet_min_volume_3d  # noqa: E402
from dvfopt.solver import auto_strategy, correct_dvf  # noqa: E402

OUT = os.path.join("benchmarks", "output", "windowed_3d")
CROPS = os.path.join("data", "dvfs", "crops_3d")
SUBVOL16 = os.path.join(
    "research", "strict_feasibility_3d", "runners", "output", "b0039_subvol_16_moderate.npy"
)
THR = 0.01
METHODS = ("isqp_windowed", "barrier", "m14_3d", "m10_3d", "pipeline3d")


def cases():
    out = {"subvol16": SUBVOL16}
    if os.path.isdir(CROPS):
        for f in sorted(os.listdir(CROPS)):
            if f.endswith(".npy"):
                out[f[:-4]] = os.path.join(CROPS, f)
    return out


def _load(path):
    phi = np.load(path)
    return np.asarray(phi, dtype=np.float64)


def cut(brain, L=24, stride=12, lo=0.08, hi=0.15):
    """One L^3 crop of ``brain``'s exterior Laplacian field with ``lo``-``hi`` of its cubes
    below threshold, the box closest to 10 % — the 'moderate' archetype of the B0039 pack."""
    vol = load_cohort_field(brain)
    _, D, H, W = vol.shape
    best = None
    for z0 in range(0, D - L, stride):
        for y0 in range(0, H - L, stride):
            for x0 in range(0, W - L, stride):
                phi = np.asarray(vol[:, z0 : z0 + L, y0 : y0 + L, x0 : x0 + L], dtype=np.float64)
                frac = float((six_tet_min_volume_3d(phi) < THR).mean())
                if lo <= frac <= hi and (best is None or abs(frac - 0.1) < abs(best[0] - 0.1)):
                    best = (frac, (z0, y0, x0), phi)
    assert best is not None, f"{brain}: no {L}^3 box in the density band [{lo}, {hi}]"
    frac, off, phi = best
    os.makedirs(CROPS, exist_ok=True)
    np.save(os.path.join(CROPS, f"{brain}_moderate.npy"), phi)
    print(f"{brain}_moderate: offset {off} below-threshold fraction {frac:.3f}", flush=True)


def run(case, method, objective="l2"):
    phi = _load(cases()[case])
    c = SimplexConstraint3D(shape=phi.shape[1:])
    mv0 = six_tet_min_volume_3d(phi)
    fold_in = mv0 < THR
    n_neg_in, min_in = int(fold_in.sum()), float(mv0.min())
    pick = auto_strategy(c, n_neg_in, min_in, objective)
    t = time.perf_counter()
    damage = sqp_iters = rounds = -1
    if method == "pipeline3d":
        from dvfopt.pipeline_3d import correct_dvf_3d

        out, rep3 = correct_dvf_3d(phi.copy(), threshold=THR)
    else:
        res = correct_dvf(
            phi.copy(), constraint="simplex_3d", strategy=method, objective=objective, threshold=THR
        )
        out = np.asarray(res.corrected, dtype=np.float64)
        if method == "isqp_windowed":
            ex = res.info.extras  # the windowed strategy lifts the SliceReport's final stats here
            damage = int(ex.get("damage", -1))
            rounds = int(ex.get("rounds", -1))
            sqp_iters = int(res.info.total_iter)
    wall = time.perf_counter() - t
    mv1 = six_tet_min_volume_3d(out)
    fold_out = mv1 < THR
    move = out - phi
    floor_out = int(n_neg_best_diagonal(out, THR))
    rec = dict(
        case=case,
        method=method,
        objective=objective,
        auto_pick=pick,
        shape=list(map(int, phi.shape[1:])),
        n_voxels=int(np.prod(phi.shape[1:])),
        folds_in=n_neg_in,
        folds_in_zero=int((mv0 <= 0).sum()),
        floor_in=int(n_neg_best_diagonal(phi, THR)),
        min_in=min_in,
        folds_out=int(fold_out.sum()),
        folds_out_zero=int((mv1 <= 0).sum()),
        floor_out=floor_out,
        min_out=float(mv1.min()),
        new_folds=int((fold_out & ~fold_in).sum()),
        moved_frac=float((np.abs(move).max(axis=0) > 1e-9).mean()),
        damage=damage,
        sqp_iters=sqp_iters,
        rounds=rounds,
        wall_s=wall,
        l2_move=float(np.linalg.norm(move.ravel())),
        l1_move=float(np.abs(move).sum()),
        feasible=bool(fold_out.sum() == 0 and floor_out == 0),
    )
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"h2h_{case}_{method}_{objective}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec), flush=True)
    return rec


COLS = [
    "case", "method", "objective", "auto_pick", "folds_in", "folds_out", "floor_out",
    "new_folds", "damage", "moved_frac", "sqp_iters", "wall_s", "l2_move", "l1_move", "feasible",
]


def table():
    rows = []
    for f in sorted(os.listdir(OUT)):
        if f.startswith("h2h_") and f.endswith(".json"):
            with open(os.path.join(OUT, f)) as fh:
                rows.append(json.load(fh))
    rows.sort(key=lambda r: (r["case"], METHODS.index(r["method"]) if r["method"] in METHODS else 9))
    lines = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for r in rows:
        cells = [f"{r.get(k, ''):.4g}" if isinstance(r.get(k), float) else str(r.get(k, "")) for k in COLS]
        lines.append("| " + " | ".join(cells) + " |")
    with open(os.path.join(OUT, "h2h.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cut", metavar="BRAIN")
    ap.add_argument("--L", type=int, default=24)
    ap.add_argument("--band", type=float, nargs=2, default=(0.08, 0.15))
    ap.add_argument("--case")
    ap.add_argument("--method", choices=METHODS)
    ap.add_argument("--objective", default="l2", choices=("l2", "l1", "none"))
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.cut:
        cut(a.cut, L=a.L, lo=a.band[0], hi=a.band[1])
    if a.case and a.method:
        run(a.case, a.method, a.objective)
    if a.table:
        table()


if __name__ == "__main__":
    main()
```

Before relying on it: `SolveResult.corrected` is the field (`dvfopt/solver.py:150`); `dvfopt/strategies/windowed.py` ~lines 326-335 lifts the `SliceReport`'s final stats to `SolveInfo.extras` top-level — read those lines for the exact keys (`damage`, `rounds`, `n_windows`, ...) and use them; if a key is absent record `-1` and say so — do NOT edit the strategy in this task. Read `correct_dvf_3d`'s signature (`dvfopt/pipeline_3d.py:81`) and pass what it needs (`threshold`, `thorough=True`, `n_workers=1`). Check `correct_dvf`'s objective label strings (`'l2'`, `'none'`) and that `strategy='m14_3d'` / `'m10_3d'` / `'barrier'` are registry labels (`dvfopt/strategies/base.py` `_STRATEGY_REGISTRY`).

- [ ] **Step 2: Smoke on the smallest artefact**

Run (foreground, 600000 ms each, from the MAIN checkout root with `PYTHONPATH=<worktree>`): `--case subvol16 --method barrier` and `--case subvol16 --method pipeline3d`, then `--table`. Expected: a JSON line with every record key, `auto_pick` = `'barrier'` for subvol16 under `l2` (721 folds, min −3.7: the moderate tier), no traceback. Report the two records' `folds_out / floor_out / wall_s`. Do NOT run `isqp_windowed` or the wallbreakers on anything (the controller runs the chain).

- [ ] **Step 3: Cut one cohort crop to prove the cutter**

Run (foreground, 600000 ms): `--cut B0032`. Expected: `B0032_moderate: offset (z, y, x) below-threshold fraction 0.0xx`, file `data/dvfs/crops_3d/B0032_moderate.npy` of shape `(3, 24, 24, 24)`. If the scan finds nothing in `[0.08, 0.15]`, widen to `[0.05, 0.2]` and report the band used.

- [ ] **Step 4: Lint + commit**

```bash
ruff check benchmarks/windowed_3d_vs_auto.py && ruff format benchmarks/windowed_3d_vs_auto.py
git add benchmarks/windowed_3d_vs_auto.py
git commit -m "3D windowed routing: head-to-head driver (windowed vs auto's picks vs pipeline3d) + the cohort crop cutter"
```

---

### Task 2 (controller): the measurement and the pre-registered ruling

**Files:** none; results to the ledger.

**Artefacts:** the B0039 pack (`twist`, `sliver`, `moderate`, `cluster`, `sub20`), `subvol16`, and one `<brain>_moderate` 24³ crop from each of the other six cohort brains (`--cut B0032 B0049 B0053 B0200 B0213 B0304`). Twelve artefacts.

**Runs** (serial, alone on the box, from the main checkout root, `PYTHONPATH=<detached snapshot worktree>`, log `benchmarks/output/windowed_3d/chain_h2h.log`): for every artefact, `--method barrier`, `--method m14_3d`, `--method m10_3d`, `--method pipeline3d` (objective `l2`); `--method isqp_windowed` on the six new cohort crops and `sub20` (the B0039 pack + `subvol16` already have their windowed rows from phase 3: twist 468 it / L2 27.0, sliver 699 / 23.3, moderate 672 / 58.7, cluster 2818 / 90.4, subvol16 107 / 55.8 — re-run them through this driver too so every row has the same `new_folds` / `moved_frac` columns, time permitting; the phase-3 records are the fallback).

**Pre-registered win criterion (the windowed engine wins the non-extreme tier if ALL hold over the twelve artefacts):**
1. It is `feasible` (0 fixed-6-tet folds at threshold, 0 best-diagonal floor) with `damage == 0` on every artefact on which ANY method is feasible.
2. On every artefact where the current `auto_pick` is also feasible, the windowed engine's `new_folds` is 0 and its L2 move is not more than 10 % larger than the pick's (ties on fidelity go to the certificate: the windowed engine).
3. Wall is REPORTED, not a criterion at this tier (a certificate the barrier cannot give is the point), but the ledger records the ratio.

**The rule, if it wins:** in `auto_strategy`'s `SimplexConstraint3D` branch, BEFORE the extreme-tier test: `if init_n_neg <= 5000 and _isqp_windowed_ok(constraint): return 'isqp_windowed'` — keyed on fold COUNT (the windowed engine's cost driver; the twist crop at min −13.4 is "extreme" by depth and the engine clears it in 468 iterations), with the count boundary at the existing tier edge (measured coverage reaches 3038 folds; 5000 is the ruled extrapolation, cost if wrong: a 5000-fold volume is ~3-4 h through the engine where the barrier is minutes). Above 5000 folds the wallbreaker rows stay exactly as they are. If it does NOT win: no rule; the docs carry the table and the reason.

---

### Task 3: The `auto_strategy` 3D rule, its tests, the CLI help and the solver docstring (only if Task 2 rules "wins")

**Files:**
- Modify: `dvfopt/solver.py` (`auto_strategy` ~lines 629-645: the rule; the docstring routing prose ~lines 561, 585-590; `_isqp_windowed_ok`'s docstring ~lines 504-514, which says the 3D branch never calls it)
- Modify: `dvfopt/cli.py:449` ("3D routing is unchanged" → the new row)
- Test: `tests/test_tetrahedron_sign.py::test_auto_strategy_tet_tiering` (~line 1030) and append the tests below

**Interfaces:**
- Produces: `auto_strategy(SimplexConstraint3D, n_neg, min, objective)` → `'isqp_windowed'` when `n_neg <= 5000` and `osqp` is importable, for every objective the engine accepts; the existing wallbreaker rows above 5000 unchanged; `'barrier'` when `osqp` is missing (the previous default).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_tetrahedron_sign.py`'s class holding `test_auto_strategy_tet_tiering`, and retarget that test)

```python
    def test_auto_routes_3d_simplex_to_the_windowed_engine_by_fold_count(self, monkeypatch):
        """Phase-3 close-out ruling: at <= 5000 folds the windowed engine (no-damage, 0-fold
        certificate) is the 3D default when osqp is importable; the wallbreaker rows above
        stay. Depth does not gate it (twist: min -13.4, cleared in 468 iterations)."""
        import importlib.util

        from dvfopt import SimplexConstraint3D
        from dvfopt.solver import auto_strategy

        c = SimplexConstraint3D(shape=(6, 8, 8))
        monkeypatch.setattr(importlib.util, 'find_spec', lambda name: object() if name == 'osqp' else None)
        for n_neg, mn, obj in [(1, -0.05, 'l2'), (200, -0.5, 'none'), (3038, -3.0, 'l1'), (403, -13.4, 'l2'), (5000, -20.0, 'l2')]:
            assert auto_strategy(c, init_n_neg=n_neg, init_min=mn, objective_label=obj) == 'isqp_windowed', (n_neg, mn, obj)
        # above the count boundary: the wallbreaker rows, exactly as before
        assert auto_strategy(c, init_n_neg=5001, init_min=-5.0, objective_label='l2') == 'm10_3d'
        assert auto_strategy(c, init_n_neg=5001, init_min=-5.0, objective_label='l1') == 'm14_3d'
        # without osqp: the previous default
        monkeypatch.setattr(importlib.util, 'find_spec', lambda name: None)
        assert auto_strategy(c, init_n_neg=200, init_min=-0.5, objective_label='l2') == 'barrier'
```

Retarget `test_auto_strategy_tet_tiering`: its mild/moderate loop asserts `'barrier'` — make that loop run under a monkeypatched `find_spec` returning `None` (no osqp) so it pins the fallback, and keep the extreme-tier assertions as they are (they are above 5000 folds: unchanged).

- [ ] **Step 2: Run them to verify they fail**

Run (foreground, 600000 ms): `pytest tests/test_tetrahedron_sign.py -k "auto" -v -p no:cacheprovider`
Expected: the new test FAILS (`'barrier' != 'isqp_windowed'`); the retargeted one passes.

- [ ] **Step 3: The rule**

In `auto_strategy`, replace the `SimplexConstraint3D` branch's head:

```python
    if isinstance(constraint, SimplexConstraint3D):
        # The windowed engine (phases 1-3 of the 3D port) is the no-damage, 0-fold
        # certificate at every fold tier the crop pack covers; its cost is per fold
        # region, so the gate is the COUNT, not the depth the barrier stalls on.
        # Measured head-to-head (CHANGELOG, "3D windowed engine: auto routing"): <the
        # Task-2 one-clause summary>. Above the count the wallbreakers keep the tier.
        if init_n_neg <= 5000 and _isqp_windowed_ok(constraint):
            return 'isqp_windowed'
        if init_n_neg > 5000 or init_min < -10.0:
            ...unchanged...
        return 'barrier'
```

Update `_isqp_windowed_ok`'s docstring (the 3D branch now calls it), the `auto_strategy` docstring's routing prose (a 3D row in the table: `simplex_3d | any | isqp_windowed at <= 5000 folds (osqp), else the wallbreakers`), and `dvfopt/cli.py:449`'s "3D routing is unchanged" to the same one clause.

- [ ] **Step 4: Run the routing tests**

Run (foreground, 900000 ms): `pytest tests/test_tetrahedron_sign.py tests/test_unified.py tests/test_m14_schwarz.py tests/test_bilinear_constraint.py tests/test_auto_objective.py -q -p no:cacheprovider`
Expected: all pass (the 2D pins untouched).

- [ ] **Step 5: Lint, mypy, commit**

```bash
ruff check dvfopt tests && ruff format dvfopt/solver.py dvfopt/cli.py tests/test_tetrahedron_sign.py && mypy
git add dvfopt/solver.py dvfopt/cli.py tests/test_tetrahedron_sign.py
git commit -m "auto_strategy: SimplexConstraint3D routes to the windowed engine at <= 5000 folds (measured head-to-head); wallbreaker rows above unchanged"
```

---

### Task 4: The GUI 3D menu row (lands either way)

**Files:**
- Modify: `dvfopt_gui/_shared.py:255-266` (`_METHOD_SPECS_TET3D`), `dvfopt_gui/worker.py:393-415` (`_MID_TO_LABEL`)
- Test: `tests/test_gui_strategy_parity.py` (parametrizes over `_MID_TO_LABEL` — the new id is exercised automatically)

**Interfaces:**
- Produces: method id `isqp_windowed_tet3d` → registry label `isqp_windowed`; menu row `('isqp_windowed', 'I-SQP windowed 3D (no-damage cluster windows; needs osqp)')` inserted FIRST in `_METHOD_SPECS_TET3D` if Task 2 ruled "wins" (it is the new default's explicit form), else right before `('auto', ...)`.

- [ ] **Step 1: Add the row and the id**

`_METHOD_SPECS_TET3D`: insert `('isqp_windowed', 'I-SQP windowed 3D (no-damage cluster windows; needs osqp)')`. `_MID_TO_LABEL`: add `'isqp_windowed_tet3d': 'isqp_windowed',` next to the other `_tet3d` rows. Read `dvfopt_gui/worker.py` around `_MID_TO_LABEL`'s consumers and `_shared.py`'s default-method map (`CONSTRAINT_TET3D: 'm14'` at ~line 281): leave the GUI's pinned default as it is unless Task 2 ruled "wins", in which case make it `'isqp_windowed'` there too (the GUI's `auto_tet3d` row calls `auto_strategy` and follows the library automatically either way).

- [ ] **Step 2: Run the parity test**

Run (foreground, 600000 ms): `pytest tests/test_gui_strategy_parity.py -q -p no:cacheprovider`
Expected: every parametrized id passes including `isqp_windowed_tet3d` (`_build_strategy` constructs `ISQPWindowedStrategy`; the test does not solve). If `PySide6` is not installed in the venv the module skips — say so and run `python -c "from dvfopt_gui.worker import _MID_TO_LABEL; from dvfopt.strategies.base import _STRATEGY_REGISTRY; assert _MID_TO_LABEL['isqp_windowed_tet3d'] in _STRATEGY_REGISTRY"` instead.

- [ ] **Step 3: Lint + commit**

```bash
ruff check dvfopt_gui tests && ruff format dvfopt_gui/_shared.py dvfopt_gui/worker.py
git add dvfopt_gui/_shared.py dvfopt_gui/worker.py
git commit -m "GUI: I-SQP windowed 3D row on the simplex-3D method menu"
```

---

### Task 5 (controller + one docs subagent): docs, gates, PR

**Docs subagent** (one commit `3D windowed routing: docs`): `CHANGELOG.md` entry under `[Unreleased]` above the phase-3 entry: `### Changed — 3D windowed engine: auto routing (SimplexConstraint3D at <= 5000 folds) + the GUI row` (or `### Added — ... the head-to-head measurement` if Task 2 ruled "does not win"), with the full `h2h.md` table (twelve artefacts × five methods, the `auto_pick` column), the win-criterion tally, the rule, and the wall ratios; `CLAUDE.md`: the routing table gains the 3D row, the "3D routing is untouched" sentence (line ~139) becomes the rule, the `ISQPWindowedStrategy` sentence in the strategies paragraph and the 3D bullet gain one clause, the GUI paragraph mentions the new row; `docs/recipe-2d-zero-folds.md` untouched.

**Controller gates:** full suite (`-n 4`; the two `test_viz_theme.py` `_tkinter` xdist flakes are known — confirm clean alone if they appear); `benchmarks/windowed_2d_identity.py` A/B (trivially identical — the engine is untouched — but run it); rebase onto `origin/main`; push; `gh pr create -R UCI-iGravi/dvfopt`; squash-merge on all-green; sync main; remove the worktree (copy the ledger to the main checkout's `.superpowers/sdd/` first); memory.
