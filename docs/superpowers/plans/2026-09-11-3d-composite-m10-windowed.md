# 3D Composite `m10_3d` → Windowed Repair, Driver Hygiene, CLI Pin — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the composite strategy the routing measurement (#122) pointed at — the `m10_3d` wallbreaker followed by the no-damage windowed engine on its residual — measure it on the same twelve artefacts, and make it the `auto` default for simplex-3D at ≤ 5000 folds only if it certifies everything at less wall than the windowed engine alone; fix the head-to-head driver's double-counted iteration column and the sweep driver's `--set` collision; pin the CLI's whole-volume route with a test and a help sentence.

**Architecture:** `m10_3d` (`HarmonicALMBarrier3DStrategy`: harmonic seed + PHR-ALM + barrier polish) certified 9 of 12 artefacts at 3–11x less wall than the windowed engine and was usually closer to the input; where it failed it left 1–3 residual folds. The windowed engine certified 12/12 from raw. A composite runs `m10_3d`, then `ISQPWindowedStrategy` on its output: the windowed stage sees a nearly-feasible field, so its cost is a few small windows. New module `dvfopt/strategies/composite3d.py` (imports both inners; nothing imports it back, so no cycle), registered as `'harmonic_alm_barrier_windowed_3d'` with alias `'m10_windowed_3d'`, phase-stack-explicit class name per the repo convention. No engine change. The 2D routing branch is untouched; the windowed engine is untouched (`benchmarks/windowed_2d_identity.py` stays `IDENTITY PASS` trivially).

**Tech Stack:** numpy / scipy, OSQP + Clarabel, pytest; artefacts under `data/dvfs/crops_3d/` (twelve, incl. the six cohort crops) and `research/strict_feasibility_3d/runners/output/b0039_subvol_16_moderate.npy`; the head-to-head driver `benchmarks/windowed_3d_vs_auto.py` and its records `benchmarks/output/windowed_3d/h2h_*.json` (main checkout, gitignored).

**Spec:** the routing ledger `.superpowers/sdd/2026-09-10-3d-windowed-auto-routing/progress.md` (main checkout, gitignored) — "Recommended follow-up: a composite m10_3d -> windowed residual repair" — and the CHANGELOG entry of #122.

## Global Constraints

- **The windowed engine (`dvfopt/core/windowed/`) and the 2D branch of `auto_strategy` do not change.** `tests/test_unified.py`, `tests/test_m14_schwarz.py`, `tests/test_bilinear_constraint.py` stay green untouched.
- **The routing change (Task 4) is gated by Task 3's pre-registered criterion**, written before the runs. The composite lands as an opt-in label either way.
- Every measured record: 0 fixed-6-tet folds at threshold 0.01 AND at 0, 0 best-diagonal floor, folds created (`new_folds`), moved-voxel fraction, wall, L2/L1 move; for the composite the windowed stage's `damage` (relative to `m10_3d`'s output).
- Python `>=3.10`; ruff 0.16.3 (`ruff check dvfopt dvfopt_gui tests benchmarks asv_bench`, `ruff format --check`), `mypy` clean.
- Commit trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; PR to `UCI-iGravi/dvfopt` (never heemmanshuu).
- Long runs (> 10 min) are the controller's, from the MAIN checkout root with `PYTHONPATH=<detached snapshot worktree>`; subagents never start them; subagent verification runs in the FOREGROUND with explicit timeouts; a pytest whose summary line says `failed` is a failure regardless of the pipeline's exit code.

---

### Task 1: The composite strategy, its export, its GUI row

**Files:**
- Create: `dvfopt/strategies/composite3d.py`
- Modify: `dvfopt/strategies/__init__.py` (import + the module docstring's strategy tree), `dvfopt/__init__.py` (import + `__all__`, next to `HarmonicALMBarrier3DStrategy` ~lines 221 / 287)
- Modify: `dvfopt_gui/_shared.py` (`_METHOD_SPECS_TET3D`: the row; the osqp-gated set), `dvfopt_gui/worker.py` (`_MID_TO_LABEL`), `dvfopt_gui/strategy_params.py` (the tet3d family map), `dvfopt_gui/app.py` (`_repopulate_method_combo` ~line 973: the osqp gate currently does `findData('isqp_windowed')` — generalise to every osqp-gated algo)
- Test: `tests/test_composite3d.py` (create), `tests/test_gui_strategy_parity.py` (picks the new id up automatically)

**Interfaces:**
- Consumes: `HarmonicALMBarrier3DStrategy` (`dvfopt/strategies/wallbreakers.py:522`, fields `margin, ring_pad, max_grow_iters, merge_dilation, rho_init, rho_growth, rho_max, outer_max, alm_inner_maxiter, polish, polish_max_iter, polish_grad_rtol`; `accepts_constraints = (SimplexConstraint3D,)`), `ISQPWindowedStrategy` (`dvfopt/strategies/windowed.py`), `Strategy` / `register_strategy` / `_build_solve_info` (`dvfopt/strategies/base.py`), `SolveInfo` / `PhaseInfo` (`dvfopt/solver.py:59-91`: `PhaseInfo(name, n_iter, wall_s, n_neg, min_T, extras)`, `SolveInfo(strategy_name, phases, total_iter, feasible_after_phase, extras)`).
- Produces: class `HarmonicALMBarrierWindowed3DStrategy` (alias `M10WindowedTetStrategy`), registry labels `'harmonic_alm_barrier_windowed_3d'` and `'m10_windowed_3d'`; `solve()` returns `(phi_out, SolveInfo)` whose `phases` are the bulk stage's phases prefixed `bulk:` followed by the windowed stage's prefixed `windowed:`, `total_iter` the sum, `extras` = the windowed stage's extras (`damage`, `n_windows`, ...) plus `bulk_n_neg_after`, `bulk_wall_s`, `bulk_extras`; GUI method id `m10_windowed_3d_tet3d`.

- [ ] **Step 1: Write the failing tests** (`tests/test_composite3d.py`)

```python
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

needs_osqp = pytest.mark.skipif(not isqp_mod.HAS_OSQP, reason="osqp not installed")
THR = 0.01


def test_registered_under_both_labels_and_exported():
    import dvfopt
    from dvfopt.strategies.composite3d import HarmonicALMBarrierWindowed3DStrategy

    assert _STRATEGY_REGISTRY['harmonic_alm_barrier_windowed_3d'] is HarmonicALMBarrierWindowed3DStrategy
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
    res = Solver(constraint=c, objective=L2Objective(), strategy=make_strategy('m10_windowed_3d'), threshold=THR).fit(phi)
    assert res.feasible
    info = res.info
    names = [p.name for p in info.phases]
    assert any(n.startswith('bulk:') for n in names) and any(n.startswith('windowed:') for n in names)
    assert names.index(next(n for n in names if n.startswith('windowed:'))) > names.index(
        next(n for n in names if n.startswith('bulk:'))
    )
    assert info.total_iter == sum(p.n_iter for p in info.phases)
    assert 'bulk_n_neg_after' in info.extras and 'bulk_wall_s' in info.extras and 'damage' in info.extras


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
    bulk_out, _ = strat.bulk.solve(phi.copy(), constraint=c, objective=L2Objective(), threshold=THR, verbose=0)
    Solver(constraint=c, objective=L2Objective(), strategy=strat, threshold=THR).fit(phi)
    assert np.allclose(seen['phi_in'], bulk_out)  # the windowed stage starts from m10_3d's output


def test_solver_rejects_a_2d_constraint():
    from dvfopt import SimplexConstraint2D
    from dvfopt.exceptions import IncompatibleConstraintError

    with pytest.raises(IncompatibleConstraintError):
        Solver(constraint=SimplexConstraint2D(shape=(8, 8)), objective=L2Objective(), strategy=make_strategy('m10_windowed_3d'))
```

Check `Solver(...)`'s real signature (`threshold` may be a `fit` / `Solver` kwarg — read `dvfopt/solver.py`'s `Solver.__init__` / `fit`) and `res.info` / `res.feasible` attribute names on `SolveResult` (`dvfopt/solver.py:150`), and adapt; the assertions stay. `planted_fold_3d(6, 10, 10, depth=1.4)` has ~a dozen folds; `m10_3d` should clear or nearly clear it in seconds.

- [ ] **Step 2: Run them to verify they fail**

Run (foreground, 600000 ms): `pytest tests/test_composite3d.py -v -p no:cacheprovider`
Expected: `ImportError` / `KeyError: 'm10_windowed_3d'`.

- [ ] **Step 3: Implement the module**

```python
"""Composite 3D strategy: the ``m10_3d`` wallbreaker, then the no-damage windowed engine on
its residual. The #122 head-to-head measured ``m10_3d`` certifying 9 of 12 artefacts at
3-11x less wall than the windowed engine and usually closer to the input, leaving 1-3
residual folds where it failed; the windowed engine certified 12/12 from raw. Chaining
them gives the certificate at roughly the wallbreaker's cost (CHANGELOG, "3D composite").
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from dvfopt.constraints import SimplexConstraint3D
from dvfopt.objectives import L1Objective, L2Objective, NoneObjective
from dvfopt.solver import PhaseInfo, SolveInfo
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
        n_neg_after_bulk = int((np.asarray(constraint.values(constraint.flatten(phi1))) < threshold).sum())
        phi2, info2 = self.windowed.solve(
            phi1,
            constraint=constraint,
            objective=objective,
            threshold=threshold,
            verbose=verbose,
            record_history=True,
            step_callback=step_callback,
        )
        phases = [_prefixed(p, 'bulk:') for p in info1.phases] + [_prefixed(p, 'windowed:') for p in info2.phases]
        n_bulk = len(info1.phases)
        feasible_after = (
            info2.feasible_after_phase + n_bulk if info2.feasible_after_phase >= 0 else info1.feasible_after_phase
        )
        extras = dict(info2.extras)
        extras.update(bulk_n_neg_after=n_neg_after_bulk, bulk_wall_s=bulk_wall, bulk_extras=dict(info1.extras))
        info = SolveInfo(
            strategy_name='m10_windowed_3d',
            phases=phases,
            total_iter=int(info1.total_iter + info2.total_iter),
            feasible_after_phase=feasible_after,
            extras=extras,
        )
        return phi2, info


def _prefixed(p: PhaseInfo, prefix: str) -> PhaseInfo:
    return PhaseInfo(name=prefix + p.name, n_iter=p.n_iter, wall_s=p.wall_s, n_neg=p.n_neg, min_T=p.min_T, extras=dict(p.extras))


M10WindowedTetStrategy = HarmonicALMBarrierWindowed3DStrategy

__all__ = ['HarmonicALMBarrierWindowed3DStrategy', 'M10WindowedTetStrategy']
```

Read before writing: how `Strategy.solve` implementations obtain the count of folds (`constraint.values(constraint.flatten(phi))` vs a `min_field` helper — `dvfopt/metrics.py`'s `constraint_fold_stats` may be the canonical one; use it if so), whether `SolveInfo` / `PhaseInfo` import from `dvfopt.solver` creates a cycle with `dvfopt/strategies/__init__.py` (the wallbreakers module imports them from somewhere — copy that import path), how `HarmonicALMBarrier3DStrategy.solve` handles `record_history=True` (its phases must be non-empty for the test), and `_check_constraint`'s name in `Strategy`. If `record_history=False` is passed to the composite, still run both stages with `record_history=True` internally (the merged info is cheap) — state it in the docstring.

- [ ] **Step 4: Export, GUI row, osqp gate**

`dvfopt/strategies/__init__.py`: import the two names and add a tree line `├── HarmonicALMBarrierWindowed3DStrategy (M10WindowedTetStrategy)  m10_3d then the windowed engine on its residual`; `dvfopt/__init__.py`: import + `__all__` entries next to the 3D wallbreakers (both names). GUI: `_METHOD_SPECS_TET3D` gains `('m10_windowed_3d', 'M10Tet → I-SQP windowed repair (fast certificate; needs osqp)')` right AFTER the `isqp_windowed` row (not the default — Task 4 decides); `_MID_TO_LABEL['m10_windowed_3d_tet3d'] = 'm10_windowed_3d'`; `strategy_params.py`'s tet3d map gains `'m10_windowed_3d@tet3d': dvfopt.HarmonicALMBarrierWindowed3DStrategy`; `app.py`'s osqp gate: replace the single `findData('isqp_windowed')` by a loop over `_OSQP_GATED_ALGOS = ('isqp_windowed', 'm10_windowed_3d')` (define it in `_shared.py` next to `DEFAULT_METHOD_FALLBACK`, import it).

- [ ] **Step 5: Run the tests**

Run (foreground, 900000 ms): `pytest tests/test_composite3d.py tests/test_gui_strategy_parity.py tests/test_gui_app.py tests/test_gui_logic.py -q -p no:cacheprovider` (set `QT_QPA_PLATFORM=offscreen`).
Expected: all pass, including the parity test's new `m10_windowed_3d_tet3d` id.

- [ ] **Step 6: Lint, mypy, commit**

```bash
ruff check dvfopt dvfopt_gui tests && ruff format dvfopt/strategies/composite3d.py dvfopt/strategies/__init__.py dvfopt/__init__.py dvfopt_gui/_shared.py dvfopt_gui/worker.py dvfopt_gui/strategy_params.py dvfopt_gui/app.py tests/test_composite3d.py && mypy
git add dvfopt/strategies/composite3d.py dvfopt/strategies/__init__.py dvfopt/__init__.py dvfopt_gui/_shared.py dvfopt_gui/worker.py dvfopt_gui/strategy_params.py dvfopt_gui/app.py tests/test_composite3d.py
git commit -m "3D composite strategy: m10_3d then the windowed engine on its residual (m10_windowed_3d), exported, GUI row"
```

---

### Task 2: Driver hygiene and the CLI pin

**Files:**
- Modify: `benchmarks/windowed_3d_vs_auto.py` (`METHODS` gains `'m10_windowed_3d'`; `sqp_iters` no longer double-counts; for the composite record `bulk_n_neg_after` / `bulk_wall_s` from `res.info.extras` as new keys `bulk_folds_after`, `bulk_wall_s` (−1 elsewhere) and `damage` from the windowed stage)
- Modify: `benchmarks/windowed_3d_sweep.py` (the `--set` collision: pop `orientation_delta`, `orientation_rows`, `threshold` from the `--set` dict and pass them explicitly, so a `--set orientation_delta=None` works instead of raising `TypeError`)
- Modify: `dvfopt/cli.py` (`--pipeline` help ~line 414: one sentence — `solver` on a `(3, D, H, W)` volume with `--constraint simplex_3d` is the whole-volume Solver route, where `auto` picks the 3D default; `3d` is the packaged `correct_dvf_3d` pipeline)
- Test: `tests/test_cli.py` (append)

**Interfaces:**
- Consumes: Task 1's label `'m10_windowed_3d'` and its `extras` keys.
- Produces: `sqp_iters` = the sum of `n_iter` over phases whose name does not start with `giant` (a `giant` entry is nested inside its `round` entry — `dvfopt/core/windowed/_common.py:1134` vs `:1151` — so summing both counts the tiles twice; sliver 1398 = 2 × 699).

- [ ] **Step 1: The CLI test** (append to `tests/test_cli.py`; read its imports and `main` usage at lines 25-85 first)

```python
def test_correct_solver_pipeline_routes_a_3d_volume(tmp_path):
    """``--pipeline solver`` on a (3, D, H, W) volume with ``--constraint simplex_3d`` is the
    whole-volume Solver route; ``auto`` picks the 3D default (the windowed engine at
    <= 5000 folds when osqp is installed, else barrier)."""
    import json

    from dvfopt.core.primitives import isqp as isqp_mod
    from tests.conftest import planted_fold_3d

    p, out, rep = tmp_path / 'in.npy', tmp_path / 'out.npy', tmp_path / 'rep'
    np.save(p, planted_fold_3d(6, 10, 10, depth=1.4))
    rc = main(['correct', str(p), str(out), '--pipeline', 'solver', '--constraint', 'simplex_3d', '--strategy', 'auto', '--report-dir', str(rep)])
    assert rc == 0
    assert np.load(out).shape == (3, 6, 10, 10)
    summary = json.loads((rep / 'summary.json').read_text())
    expected = 'isqp_windowed' if isqp_mod.HAS_OSQP else 'barrier'
    assert expected in json.dumps(summary)  # the resolved strategy is recorded in the summary
```

Read what `summary.json` records for the resolved strategy (`dvfopt/cli.py` ~lines 290-300: the summary dict has `'strategy': args.strategy` = `'auto'` — if the RESOLVED label is not recorded anywhere, record it: `res.info.strategy_name` into the summary as `'strategy_resolved'` — a one-line CLI change — and assert on that key instead).

- [ ] **Step 2: Run it to verify it fails** (foreground, 600000 ms): `pytest tests/test_cli.py -k solver_pipeline -v -p no:cacheprovider` — fails on the missing key (or passes if the summary already names the resolved strategy; then it is a pin).

- [ ] **Step 3: The driver fixes**

`benchmarks/windowed_3d_vs_auto.py`: `METHODS = ("isqp_windowed", "m10_windowed_3d", "barrier", "m14_3d", "m10_3d", "pipeline3d")`; in `run()`, the windowed-stats block becomes:

```python
        if method in ("isqp_windowed", "m10_windowed_3d"):
            ex = res.info.extras
            damage = int(ex.get("damage", -1))
            # a `giant` history entry is nested inside its `round` entry, so summing every
            # phase counts the tiles twice (sliver 1398 = 2 x 699): skip the giant entries
            sqp_iters = int(sum(p.n_iter for p in res.info.phases if not p.name.split(":")[-1].startswith("giant")))
            rounds = sum(1 for p in res.info.phases if p.name.split(":")[-1].startswith("round"))
            bulk_folds_after = int(ex.get("bulk_n_neg_after", -1))
            bulk_wall_s = float(ex.get("bulk_wall_s", -1.0))
```

(initialise `bulk_folds_after = -1; bulk_wall_s = -1.0` with the other defaults; add both to `rec` and to `COLS` after `damage`). `benchmarks/windowed_3d_sweep.py`: in `run()`, `od = kw.pop("orientation_delta", od); rows = kw.pop("orientation_rows", "edges"); thr = kw.pop("threshold", THR)` and pass those instead of the literals (the `cfg` still sets the defaults).

- [ ] **Step 4: Smoke the driver on the smallest artefact with the composite** (foreground, 600000 ms, from the MAIN checkout root with `PYTHONPATH=<worktree>`): `--case subvol16 --method m10_windowed_3d`. Expected: a record with `feasible` true (m10_3d left 3 residuals there; the windowed stage clears them), `bulk_folds_after` 3, `sqp_iters` small, `damage` 0. Report the record's `folds_out / floor_out / bulk_folds_after / bulk_wall_s / wall_s / l2_move`. This one run doubles as the first Task-3 row — leave its JSON in place.

- [ ] **Step 5: Lint, commit**

```bash
ruff check benchmarks dvfopt tests && ruff format benchmarks/windowed_3d_vs_auto.py benchmarks/windowed_3d_sweep.py dvfopt/cli.py tests/test_cli.py
git add benchmarks/windowed_3d_vs_auto.py benchmarks/windowed_3d_sweep.py dvfopt/cli.py tests/test_cli.py
git commit -m "3D drivers: the composite in the head-to-head, sqp_iters without the nested giant entries, --set passes the orientation keys; CLI: the whole-volume solver route pinned"
```

---

### Task 3 (controller): the composite on the twelve artefacts and the pre-registered ruling

**Runs** (serial, alone, from the main checkout root, `PYTHONPATH=<detached snapshot at Task 2's head>`, log `benchmarks/output/windowed_3d/chain_composite.log`): `--case <each of the 12> --method m10_windowed_3d` (subvol16 already done in Task 2 Step 4). Then `--table`.

**Pre-registered criterion — the composite becomes the ≤ 5000-fold `auto` default if ALL hold:**
1. `feasible` (0 folds at threshold, 0 floor) on all 12, with `new_folds == 0` against the INPUT and the windowed stage's `damage == 0`.
2. Wall below the windowed engine's own row on at least 10 of 12.
3. L2 move not more than 10 % above the windowed engine's on any artefact (expected below: `m10_3d` was usually closer).
Otherwise: the composite ships as an opt-in label; `auto` keeps the windowed engine; the docs carry the table and the reason.

---

### Task 4: The routing change (only if Task 3 rules "wins")

**Files:**
- Modify: `dvfopt/solver.py` (`auto_strategy`'s `SimplexConstraint3D` branch: `'isqp_windowed'` → `'m10_windowed_3d'`; the docstring's 3D row; `_isqp_windowed_ok` still gates — the composite needs osqp for its windowed stage), `dvfopt/cli.py` (the `--strategy` help's 3D clause), `dvfopt_gui/_shared.py` (`DEFAULT_METHOD_BY_CONSTRAINT[CONSTRAINT_TET3D] = 'm10_windowed_3d'`; the row order: the composite first), `tests/test_tetrahedron_sign.py` (the routing test's expected label → `'m10_windowed_3d'` for the ≤ 5000 cases; keep one assertion that `make_strategy('isqp_windowed')` still constructs — the explicit label stays), `tests/test_gui_app.py` (the tet3d-default test's expected default → `'m10_windowed_3d'`), `tests/test_cli.py` (Step 1's `expected`).

- [ ] **Step 1: Retarget the tests, run to see them fail, change the rule, run to see them pass** (foreground, 900000 ms: `pytest tests/test_tetrahedron_sign.py tests/test_gui_app.py tests/test_cli.py tests/test_unified.py tests/test_m14_schwarz.py -q -p no:cacheprovider`, `QT_QPA_PLATFORM=offscreen`).
- [ ] **Step 2: Lint, mypy, commit** — `auto_strategy: SimplexConstraint3D at <= 5000 folds routes to the composite m10_windowed_3d (measured 12/12 at a fraction of the windowed wall); GUI default`.

---

### Task 5 (docs subagent + controller): docs, gates, PR

**Docs subagent** (one commit): `CHANGELOG.md` entry under `[Unreleased]` above the #122 entry — `### Added — 3D composite strategy m10_windowed_3d (m10_3d then the windowed engine on its residual)` + `Changed — auto routes simplex-3D at <= 5000 folds to it` if Task 4 landed — with the twelve-row table (composite vs the windowed engine vs `m10_3d`: certifies / bulk folds after / wall / L2), the criterion tally, the driver fixes (the `giant`-nesting double-count), the CLI route sentence; `CLAUDE.md`: the strategies list (the composite, with its two-stage description and both labels), the routing table's 3D row, the 3D bullet, the GUI paragraph, the CLI paragraph (`--pipeline solver --constraint simplex_3d` = the whole-volume route; `--pipeline 3d` = the packaged pipeline); `ARCHITECTURE.md` if it lists strategies.

**Controller gates:** full suite (`-n 4`, `QT_QPA_PLATFORM=offscreen`; check the summary line for `failed`); `benchmarks/windowed_2d_identity.py` A/B (trivially identical); rebase; push; PR; squash-merge on all-green; sync main; remove the worktrees; ledger to the main checkout; memory.

**Deferred, on the record:** the two ~170 s tiler unit tests (no `slow` marker infrastructure exists; adding one plus a CI split is its own change); `_fit_tile` wrapper / `_blob` duplication.
