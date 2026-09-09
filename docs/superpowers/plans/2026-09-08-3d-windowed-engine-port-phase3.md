# 3D Windowed Engine Port — Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a 3D window solve at the right size and cost: a per-dimension defaults table set by measurement (the 3D `max_window_area`, QP caps, re-anchor tile/overlap) and the exact line search on 3D rows, each kept only if it wins on the crop pack.

**Architecture:** Phase 2 left every stage n-D but with the 2D numbers: `max_window_area=3000` read as a voxel count (14.4³) tiles a 17³ region that one window solves with a 43 % smaller move; the QP caps (`qp_max_iter=1000`, `ip_after_admm_iters=800`) were measured on 2D windows of ~2k variables, not 45k; `'exact_ls'` degrades to `'tr'` because a 6-tet row is cubic along a line. Phase 3 (1) adds a sweep driver over the phase-2 artefacts, (2) measures the cap / QP / re-anchor settings and the cubic line model, (3) replaces the phase-2 twin knobs (`giant_tile_3d`, `mop_margin_3d`) by ONE table `DEFAULTS_BY_DIM` resolved at the engine entry, so a knob left at its 2D default takes the 3D value and an explicit value is honoured in any dimension. The 2D path stays byte-identical (`benchmarks/windowed_2d_identity.py`).

**Tech Stack:** numpy / scipy, OSQP + Clarabel (the isqp inner), pytest; the artefacts under `data/dvfs/crops_3d/` (gitignored) and `research/strict_feasibility_3d/runners/output/b0039_subvol_16_moderate.npy`.

**Spec:** `docs/superpowers/specs/2026-09-05-3d-windowed-engine-port-design.md` (phase-3 line: "cubic exact line-min — only if measured load-bearing"; open question 2: "the voxel cap that replaces `max_window_area` — from U1's curve"); phase-2 ledger `.superpowers/sdd/2026-09-07-3d-windowed-engine-port-phase2/progress.md` (main checkout, gitignored) — the deferred list this plan works through.

## Global Constraints

- **Never change 2D behaviour.** `benchmarks/windowed_2d_identity.py --out DIR` on the branch vs `--compare benchmarks/output/identity_2d/main DIR` (run from the MAIN checkout root with `PYTHONPATH=<worktree>`) must print `IDENTITY PASS` on every head that touches shared code. Every new default in the table's 2D column is the current 2D default, copied verbatim.
- **Every 3D change is gated by measurement:** 0 fixed-6-tet folds at threshold 0.01 AND at 0, 0 best-diagonal floor, damage 0, with wall, SQP iterations and L2 move reported. Fidelity (L2 move) is reported next to every speed claim.
- Python `>=3.10`; `scipy>=1.15,<1.19`; ruff 0.16.3 (`ruff check dvfopt dvfopt_gui tests benchmarks asv_bench`, `ruff format --check`), `mypy` clean; tests under `tests/`.
- Commit trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; PR to `UCI-iGravi/dvfopt` (never heemmanshuu).
- Long runs (> 10 min) are the controller's: they run in the background from the main checkout root with `PYTHONPATH=<worktree> .venv/Scripts/python.exe <worktree>/benchmarks/<script>`; subagents never start them. Subagent verification commands run in the FOREGROUND with explicit timeouts.
- Measurement outputs go to `benchmarks/output/windowed_3d/` (gitignored) as `sweep_<case>_<tag>.json` + `sweep.md`; numbers reach `CHANGELOG.md` / `CLAUDE.md` in Task 6.

---

### Task 1: The sweep driver `benchmarks/windowed_3d_sweep.py`

**Files:**
- Create: `benchmarks/windowed_3d_sweep.py`
- Reuse (import, do not edit): `benchmarks/windowed_3d_gate.py` (its import installs the QP timing spy `_TimedQP` into `isqp._make_qp` and the exit spy `_spy_inner` into `_common.solve_window_inner`; exports `QP_LOG`, `EXITS`, `THR`, `_load`, `CFGS`, `RAW`)
- Reuse: `benchmarks/make_hard_crops_3d.py` (`OUT_CROPS`, `OUT`, `L`, the record layout of `run()`)

**Interfaces:**
- Produces: CLI `python benchmarks/windowed_3d_sweep.py --case <name> --tag <tag> [--cfg l2_rows] [--set k=v ...]` writing `benchmarks/output/windowed_3d/sweep_<case>_<tag>.json`; `--cut-sub20` cutting `data/dvfs/crops_3d/sub20.npy`; `--table` rendering `benchmarks/output/windowed_3d/sweep.md` from every `sweep_*.json`.
- Record keys (Tasks 2-4 and 6 read them): `case, cfg, tag, settings (dict), shape, folds_in, floor_in, folds_out, folds_out_zero, floor_out, floor_out_zero, min_out, damage, rounds, n_windows, giant_regions, mop_windows, reseed_rounds_run, sqp_iters, rejected_steps, exits (dict), no_tr_fallbacks, backend_fallbacks, patience_fallbacks, qp_n, qp_s_total, qp_s_median, admm_median, admm_at_cap, ip_solves, wall_s, l2_move, l1_move, reanchor_tiles, reanchor_accepted, reanchor_l2_before, reanchor_l2_after`.

- [ ] **Step 1: Write the script**

```python
"""Phase-3 sweep driver for the 3D windowed engine: ONE `windowed_correct` run per call
on a phase-2 artefact under explicit knob overrides, with the QP / exit spies of
``windowed_3d_gate.py``; JSON records + a markdown table.

    python benchmarks/windowed_3d_sweep.py --case subvol16 --tag cap5000 --set max_window_area=5000
    python benchmarks/windowed_3d_sweep.py --cut-sub20          # cuts data/dvfs/crops_3d/sub20.npy
    python benchmarks/windowed_3d_sweep.py --table              # sweep.md from every sweep_*.json

``--set k=v`` parses ``None`` / ``True`` / ``False`` / ints / floats / strings;
``reanchor_overlap=N`` patches the module constant ``_common._REANCHOR_OVERLAP`` (not an
engine kwarg). Cases: ``subvol16`` (the phase-1 17^3 B0039 sub-volume), ``sub20`` (a 20^3
cut, 8000 voxels — the artefact between the 17^3 that one window solves and the 24^3 crops
the tiler must split), and the phase-2 crops ``twist`` / ``cluster`` / ``sliver`` /
``moderate``. Run from the repo root.
"""

import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from windowed_3d_gate import CFGS, EXITS, QP_LOG, RAW, THR, _load  # noqa: E402  (installs the spies)

import numpy as np  # noqa: E402

import dvfopt.core.windowed._common as _cm  # noqa: E402
from dvfopt.constraints import SimplexConstraint3D  # noqa: E402
from dvfopt.core.windowed import windowed_correct  # noqa: E402
from dvfopt.jacobian.tetrahedron_sign import n_neg_best_diagonal, six_tet_min_volume_3d  # noqa: E402

OUT = os.path.join("benchmarks", "output", "windowed_3d")
CROPS = os.path.join("data", "dvfs", "crops_3d")
SUBVOL16 = os.path.join(
    "research", "strict_feasibility_3d", "runners", "output", "b0039_subvol_16_moderate.npy"
)
TRACES = []  # per-window isqp traces (the gate's _spy_inner keeps only the exit)

_gate_spy = _cm.solve_window_inner  # the gate's _spy_inner, installed at import


def _trace_spy(sub, inner, maxiter, **kw):
    kw.setdefault("trace", {})
    r = _gate_spy(sub, inner, maxiter, **kw)
    TRACES.append(kw["trace"])
    return r


_cm.solve_window_inner = _trace_spy


def _case_path(name):
    return SUBVOL16 if name == "subvol16" else os.path.join(CROPS, f"{name}.npy")


def _parse(v):
    if v in ("None", "True", "False"):
        return {"None": None, "True": True, "False": False}[v]
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def cut_sub20(L=20, stride=12, lo=0.08, hi=0.15):
    """A 20^3 cut of the raw field with 8-15 % of its cubes below threshold — a fold
    region between the 17^3 (4913 voxels) and the 24^3 crops (13824)."""
    vol = np.load(RAW, mmap_mode="r")
    _, D, H, W = vol.shape
    best = None
    for z0 in range(0, D - L, stride):
        for y0 in range(0, H - L, stride):
            for x0 in range(0, W - L, stride):
                phi = np.asarray(vol[:, z0 : z0 + L, y0 : y0 + L, x0 : x0 + L], dtype=np.float64)
                frac = float((six_tet_min_volume_3d(phi) < THR).mean())
                if lo <= frac <= hi and (best is None or abs(frac - 0.1) < abs(best[0] - 0.1)):
                    best = (frac, (z0, y0, x0), phi)
    assert best is not None, "no 20^3 box in the density band"
    frac, off, phi = best
    os.makedirs(CROPS, exist_ok=True)
    np.save(os.path.join(CROPS, "sub20.npy"), phi)
    print(f"sub20: offset {off} below-threshold fraction {frac:.3f}", flush=True)


def _rejected(traces):
    """Ratio-test rejections: iteration records with rule 'tr' (or no rule) that did not step."""
    n = 0
    for tr in traces:
        for it in tr.get("iters", []):
            if it.get("rule", "tr") == "tr" and it.get("stepped") is False:
                n += 1
    return n


def run(case, cfg, tag, settings):
    obj_cls, od = CFGS[cfg]
    phi = _load(_case_path(case))
    c = SimplexConstraint3D(shape=phi.shape[1:])
    mv0 = six_tet_min_volume_3d(phi)
    kw = dict(settings)
    overlap = kw.pop("reanchor_overlap", None)
    if overlap is not None:
        _cm._REANCHOR_OVERLAP = int(overlap)
    EXITS.clear()
    QP_LOG.clear()
    TRACES.clear()
    t = time.perf_counter()
    out, rep = windowed_correct(
        phi.copy(),
        "isqp",
        constraint=c,
        objective=obj_cls(),
        threshold=THR,
        orientation_delta=od,
        orientation_rows="edges",
        verbose=0,
        **kw,
    )
    wall = time.perf_counter() - t
    move = out - phi
    admm = [(w, i) for w, i, s in QP_LOG if not s.startswith("clarabel")]
    qp_w = np.array([w for w, _i, _s in QP_LOG]) if QP_LOG else np.zeros(1)
    cap = settings.get("qp_max_iter", 1000)
    rec = dict(
        case=case,
        cfg=cfg,
        tag=tag,
        settings=settings,
        shape=list(map(int, phi.shape[1:])),
        folds_in=int((mv0 < THR).sum()),
        floor_in=int(n_neg_best_diagonal(phi, THR)),
        folds_out=int(rep.folds_after),
        folds_out_zero=int(rep.folds_after_zero),
        floor_out=int(rep.best_diag_floor_after),
        floor_out_zero=int(rep.best_diag_floor_after_zero),
        min_out=float(rep.min_after),
        damage=int(rep.damage),
        rounds=int(rep.rounds),
        n_windows=int(rep.n_windows),
        giant_regions=int(rep.giant_regions),
        mop_windows=int(rep.mop_windows),
        reseed_rounds_run=int(rep.reseed_rounds_run),
        sqp_iters=int(sum(w.inner_iters for w in rep.windows)),
        rejected_steps=int(_rejected(TRACES)),
        exits=dict(collections.Counter(EXITS)),
        no_tr_fallbacks=int(sum(w.fallback for w in rep.windows)),
        backend_fallbacks=int(rep.backend_fallbacks),
        patience_fallbacks=int(rep.patience_fallbacks),
        qp_n=int(len(QP_LOG)),
        qp_s_total=float(qp_w.sum()),
        qp_s_median=float(np.median(qp_w)),
        admm_median=float(np.median([i for _w, i in admm])) if admm else -1.0,
        admm_at_cap=float(np.mean([i >= cap for _w, i in admm])) if admm and cap else -1.0,
        ip_solves=int(len(QP_LOG) - len(admm)),
        wall_s=wall,
        l2_move=float(np.linalg.norm(move.ravel())),
        l1_move=float(np.abs(move).sum()),
        reanchor_tiles=int(rep.reanchor_tiles),
        reanchor_accepted=int(rep.reanchor_accepted),
        reanchor_l2_before=float(rep.reanchor_l2_before),
        reanchor_l2_after=float(rep.reanchor_l2_after),
    )
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"sweep_{case}_{tag}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec), flush=True)
    return rec


COLS = [
    "case", "tag", "folds_out", "floor_out", "damage", "rounds", "n_windows", "mop_windows",
    "sqp_iters", "rejected_steps", "qp_n", "qp_s_median", "admm_median", "admm_at_cap",
    "ip_solves", "wall_s", "l2_move",
]


def table():
    rows = []
    for f in sorted(os.listdir(OUT)):
        if f.startswith("sweep_") and f.endswith(".json"):
            with open(os.path.join(OUT, f)) as fh:
                rows.append(json.load(fh))
    lines = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for r in rows:
        cells = []
        for k in COLS:
            v = r.get(k, "")
            cells.append(f"{v:.4g}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    with open(os.path.join(OUT, "sweep.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["subvol16", "sub20", "twist", "cluster", "sliver", "moderate"])
    ap.add_argument("--cfg", default="l2_rows", choices=list(CFGS))
    ap.add_argument("--tag", default="base")
    ap.add_argument("--set", nargs="*", default=[], metavar="k=v")
    ap.add_argument("--cut-sub20", action="store_true")
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.cut_sub20:
        cut_sub20()
    if a.case:
        settings = {k: _parse(v) for k, v in (s.split("=", 1) for s in a.set)}
        run(a.case, a.cfg, a.tag, settings)
    if a.table:
        table()


if __name__ == "__main__":
    main()
```

Before relying on `_rejected`: read `dvfopt/core/primitives/isqp.py`'s `_emit(rec)` (~line 558) for the trace key the per-iteration records are appended under (`trace['iters']` is the expectation) and the record's field names (`rule`, `stepped`), and adapt the two names in `_rejected` if they differ. Also confirm `windowed_3d_gate.py` exposes `_load`, `RAW`, `CFGS`, `EXITS`, `QP_LOG`, `THR` at module level (it does at the time of writing; import exactly what exists).

- [ ] **Step 2: Smoke it on the smallest artefact**

Run (foreground, 600000 ms): `PYTHONPATH=. python benchmarks/windowed_3d_sweep.py --case subvol16 --tag smoke --set max_rounds=1 maxiter=5 fallback_maxiter=5 reseed_rounds=0`
Expected: a JSON line with every key above, `sweep_subvol16_smoke.json` written, no traceback (folds will remain — the smoke checks plumbing only). Then `--table` prints a table with that row. Delete the smoke JSON afterwards (`rm benchmarks/output/windowed_3d/sweep_subvol16_smoke.json`).

- [ ] **Step 3: Cut sub20**

Run (foreground, 600000 ms): `PYTHONPATH=. python benchmarks/windowed_3d_sweep.py --cut-sub20`
Expected: `sub20: offset (z, y, x) below-threshold fraction 0.0xx`, file `data/dvfs/crops_3d/sub20.npy` of shape `(3, 20, 20, 20)`. Report the offset and fraction. If the scan finds nothing in `[0.08, 0.15]`, widen to `[0.05, 0.2]` and report.

- [ ] **Step 4: Lint + commit**

```bash
ruff check benchmarks/windowed_3d_sweep.py && ruff format benchmarks/windowed_3d_sweep.py
git add benchmarks/windowed_3d_sweep.py
git commit -m "3D windowed, phase 3: sweep driver over the phase-2 artefacts (+ the 20^3 cut)"
```

---

### Task 2 (controller): the 3D window cap

**Files:** none (measurement); results to the ledger.

**Runs** (serial, alone on the box, from the main checkout root, `PYTHONPATH=<worktree>`; each in the background with its log under `benchmarks/output/windowed_3d/chain_p3_cap.log`):

```
--case subvol16 --tag cap5000 --set max_window_area=5000     # whole 17^3 window; expect ≈ phase 1 (107 it / 638 s / L2 55.8)
--case sub20    --tag cap3000                                # tiled (fitted tile on a 20-extent region)
--case sub20    --tag cap5000 --set max_window_area=5000     # tiled
--case sub20    --tag cap8000 --set max_window_area=8000     # whole 20^3 window (~25 s per SQP iteration; budget 1-3 h)
```

**Decision rule** (recorded in the ledger as the Task-2 ruling): the 3D `max_window_area` is the largest of {3000, 5000, 8000} such that on the artefact of that size the whole-window solve reaches 0 folds / 0 floor / damage 0, its L2 move is smaller than the tiled solve's, and its wall is at most 2.5x the tiled wall. The phase-2 data already put 5000 ahead of 3000 on the 17^3 (whole 638 s / L2 55.8 vs tiled 267 s / 96.9 — a 2.4x wall for a 43 % smaller move); sub20 decides between 5000 and 8000. If the sub20 whole solve exceeds 3 h or fails, the ruling is 5000.

---

### Task 3 (controller): QP caps and the re-anchor overlap in 3D

**Files:** none (measurement).

**Runs** (serial; log `benchmarks/output/windowed_3d/chain_p3_qp.log`). Baseline for both cases = the Task-2 cap, so `subvol16` is ONE 45k-variable window (the U1 anchor) and `twist` is tiled (12^3 tiles, ~15k-variable QPs):

```
for CASE in subvol16 twist:
  --case $CASE --tag qp_base    --set max_window_area=<cap>
  --case $CASE --tag qp_cap500  --set max_window_area=<cap> qp_max_iter=500 qp_max_iter_fallback=250
  --case $CASE --tag qp_cap2000 --set max_window_area=<cap> qp_max_iter=2000 qp_max_iter_fallback=1000
  --case $CASE --tag qp_ip400   --set max_window_area=<cap> ip_after_admm_iters=400
  --case $CASE --tag qp_ip200   --set max_window_area=<cap> ip_after_admm_iters=200
  --case $CASE --tag qp_nocold  --set max_window_area=<cap> ip_cold=False
  --case $CASE --tag qp_osqp    --set max_window_area=<cap> qp_backend=osqp
--case twist --tag ra_ov8 --set max_window_area=<cap> reanchor=l2 reanchor_sweeps=1
--case twist --tag ra_ov4 --set max_window_area=<cap> reanchor=l2 reanchor_sweeps=1 reanchor_overlap=4
```

**Decision rule:** a QP setting enters the 3D column only if it is faster than `qp_base` on BOTH cases at 0 folds / 0 floor / damage 0 and an L2 move within +5 % (the phase-1 trace showed warm OSQP solves hitting the 1000 cap "solved inaccurate" at 2.3 s each right before a 5 s Clarabel solve — `ip_after_admm_iters` is the expected lever). Otherwise the 2D values stay. The re-anchor overlap for 3D is 4 if `ra_ov4` reaches a `reanchor_l2_after` within 2 % of `ra_ov8` in less wall; otherwise 8. `admm_at_cap` and `ip_solves` go in the ledger next to the ruling.

---

### Task 4: The exact line search on 3D rows (`line_model='cubic'`)

**Files:**
- Modify: `dvfopt/core/primitives/isqp.py` (`_exact_line_min` region ~lines 314-345; the `'exact_ls'` branch ~lines 669-680; `isqp_solve` signature ~line 356; the docstring paragraph "``'exact_ls'`` is **2D only**" ~line 500)
- Modify: `dvfopt/core/windowed/_inners.py` (`solve_window_inner` ~lines 40-116: forward `line_model`)
- Modify: `dvfopt/core/windowed/_common.py` (`_InnerOpts` ~line 70: `line_model: str = 'quadratic'`; the entry block ~lines 877-886: replace the degrade by `line_model='cubic'` on 3D; the `solve_window_inner(...)` calls in `_solve_window` ~lines 1800-1920 and the patience rung: forward `line_model=opts.line_model`; `_engine_kwargs` ~line 1404)
- Test: `tests/test_isqp_exact_ls.py` (append; retarget `test_exact_ls_degrades_to_tr_on_a_3d_field_and_still_refuses_other_ranks` at ~line 106)

**Interfaces:**
- Consumes: `_exact_line_min(c0, g, q, w, fco, a_hi=1.0) -> (a_star, m_star, m_zero)` (unchanged, 2D).
- Produces: `_cubic_line_min(c0, g, q2, q3, w, fco, a_hi=1.0, n_grid=64) -> (a_star, m_star, m_zero)`; `isqp_solve(..., line_model='quadratic')` accepting `'quadratic' | 'cubic'`; `solve_window_inner(..., line_model='quadratic')`; `_InnerOpts.line_model`.

**Why cubic is exact for the rows:** a 6-tet volume row is trilinear in the displacements, hence a cubic polynomial along `x + a d`; with `c0 = cons(x)` and `g = J d` exact, the two remaining coefficients follow exactly from two samples: with `r_h = cons(x + d/2) - c0 - g/2` and `r_1 = cons(x + d) - c0 - g`, `q2 = 8 r_h - r_1` and `q3 = 2 r_1 - 8 r_h` (check: a linear edge row gives `r_h = r_1 = 0`). `cons(x + d)` is the evaluation the ratio test already makes; `cons(x + d/2)` is one extra evaluation (0.0007 s at 17^3 against a 9 s QP). The MINIMISER of the piecewise-cubic merit is not swept over roots (a cubic's breakpoints have no clean vectorised closed form); it is found on a dense grid of `[0, a_hi]` and refined by golden-section in the bracketing cell. The true merit at `a*` is verified before stepping, exactly as the 2D branch does, so the rule can never regress a window.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_isqp_exact_ls.py`)

```python
def _sub3d(seed=0, n=7):
    """A frozen-ring 3D window sub-problem on a folded blob: (sub, x0, d)."""
    from dvfopt.constraints import SimplexConstraint3D
    from dvfopt.core.windowed._common import build_subproblem
    from dvfopt.objectives import NoneObjective

    rng = np.random.default_rng(seed)
    phi = np.zeros((3, n, n, n))
    phi[:, 2:5, 2:5, 2:5] = rng.normal(0.0, 0.9, (3, 3, 3, 3))
    c = SimplexConstraint3D(shape=(n, n, n))
    sub = build_subproblem(c, phi, (1, n - 1, 1, n - 1, 1, n - 1), 0.01, NoneObjective(), 1e-3)
    x0 = np.asarray(sub.x0, float)
    d = rng.normal(0.0, 0.3, x0.size)
    return sub, x0, d


def _cubic_coeffs(sub, x0, d):
    cons, jac = sub.cons, sub.jac
    c0 = np.asarray(cons(x0))
    g = np.asarray(jac(x0) @ d)
    r_h = np.asarray(cons(x0 + 0.5 * d)) - c0 - 0.5 * g
    r_1 = np.asarray(cons(x0 + d)) - c0 - g
    return c0, g, 8.0 * r_h - r_1, 2.0 * r_1 - 8.0 * r_h


def test_cubic_line_model_matches_cons_exactly():
    sub, x0, d = _sub3d()
    c0, g, q2, q3 = _cubic_coeffs(sub, x0, d)
    for a in (0.13, 0.5, 0.71, 1.0):
        model = c0 + g * a + q2 * a * a + q3 * a**3
        assert np.allclose(model, sub.cons(x0 + a * d), atol=1e-9, rtol=0), a


def test_cubic_minimiser_is_within_a_grid_cell_of_a_dense_scan():
    from dvfopt.core.primitives.isqp import _cubic_line_min

    sub, x0, d = _sub3d(seed=1)
    c0, g, q2, q3 = _cubic_coeffs(sub, x0, d)
    w = np.full(c0.size, 10.0)
    fco = (0.0, 0.0, 0.0)

    def merit(a):
        return float(w @ np.maximum(0.0, -np.asarray(sub.cons(x0 + a * d))))

    a_star, m_star, m0 = _cubic_line_min(c0, g, q2, q3, w, fco, 1.0)
    grid = np.linspace(0.0, 1.0, 4001)
    dense = np.array([merit(a) for a in grid])
    assert abs(m0 - merit(0.0)) < 1e-9
    assert m_star <= dense.min() + 1e-6 * max(1.0, abs(dense.min()))
    assert abs(merit(a_star) - m_star) < 1e-7 * max(1.0, abs(m_star))


def test_line_model_reaches_the_driver_and_defaults_to_quadratic(monkeypatch):
    import dvfopt.core.windowed._inners as inners

    seen = {}
    real = inners.isqp_solve

    def spy(*a, **k):
        seen["line_model"] = k.get("line_model", "MISSING")
        return real(*a, **k)

    monkeypatch.setattr(inners, "isqp_solve", spy)
    sub, x0, d = _sub3d()
    inners.solve_window_inner(sub, "isqp", 2, line_model="cubic")
    assert seen["line_model"] == "cubic"
    inners.solve_window_inner(sub, "isqp", 2)
    assert seen["line_model"] == "quadratic"
```

Check `build_subproblem`'s real signature and the sub-problem's attribute names (`sub.cons`, `sub.jac`, `sub.x0` — read `dvfopt/core/windowed/_common.py`'s `build_subproblem` and the object it returns) and adapt the helpers to them; the assertions stay. The `jac(x0) @ d` product must be the SAME `J d` the inner uses (`j @ z[:nf]` in `isqp_solve` is on the free variables; if `sub.jac` is the free-variable Jacobian, `d` must be a free-variable vector — match the inner's convention).

- [ ] **Step 2: Run them to verify they fail**

Run (foreground, 600000 ms): `pytest tests/test_isqp_exact_ls.py -k "cubic or line_model" -v -p no:cacheprovider`
Expected: `test_cubic_line_model_matches_cons_exactly` PASSES already (it pins the algebra, not the code — keep it), the other two FAIL with `ImportError` / `TypeError: unexpected keyword 'line_model'`.

- [ ] **Step 3: Implement `_cubic_line_min`** (in `isqp.py`, right after `_exact_line_min`)

```python
def _cubic_line_min(c0, g, q2, q3, w, fco, a_hi=1.0, n_grid=64):
    """Minimiser on ``[0, a_hi]`` of the model merit ``m(a) = f(a) + sum_i w_i max(0, -c_i(a))``
    with CUBIC rows ``c_i(a) = c0 + g a + q2 a**2 + q3 a**3`` (a 6-tet volume along a line)
    and the quadratic objective coefficients ``fco = (f0, f1, f2)``.

    Unlike :func:`_exact_line_min` the breakpoints are not swept (a cubic's roots have no
    clean vectorised closed form): ``m`` is evaluated on ``n_grid + 1`` equispaced points
    and the best cell is refined by golden section. The caller verifies the TRUE merit at
    ``a*`` before stepping, so an off-by-a-cell minimiser costs a little progress, never
    correctness. Returns ``(a_star, m_star, m_zero)``.
    """
    aa = np.linspace(0.0, a_hi, n_grid + 1)

    def m_of(a):
        a = np.asarray(a, float)
        rows = c0[:, None] + g[:, None] * a + q2[:, None] * a * a + q3[:, None] * a**3
        pen = (w[:, None] * np.maximum(0.0, -rows)).sum(axis=0)
        return fco[0] + fco[1] * a + fco[2] * a * a + pen

    vals = m_of(aa)
    b = int(np.argmin(vals))
    lo, hi = aa[max(b - 1, 0)], aa[min(b + 1, n_grid)]
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f1, f2 = float(m_of([x1])[0]), float(m_of([x2])[0])
    for _ in range(24):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = float(m_of([x1])[0])
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = float(m_of([x2])[0])
    cands = np.array([aa[b], x1, x2])
    cv = m_of(cands)
    k = int(np.argmin(cv))
    return float(cands[k]), float(cv[k]), float(vals[0])
```

- [ ] **Step 4: Thread `line_model` through `isqp_solve`**

In `isqp_solve`'s signature add `line_model='quadratic',` after `exact_ls_fallback_steps=0,`; validate next to the `step_rule` check:

```python
    if line_model not in ('quadratic', 'cubic'):
        raise ValueError(f"unknown line_model {line_model!r}; valid: 'quadratic', 'cubic'")
```

In the `'exact_ls'` branch replace the four lines from `gl = ...` to `a_star, _m_star, _m0 = _exact_line_min(...)` by:

```python
            gl = np.asarray(j @ z[:nf])  # (J d)_i, exact linear term
            c1 = np.asarray(cons(x + d)) - c - gl  # residual past the linear term at a = 1
            fh, f1 = float(obj(x + 0.5 * d)), float(obj(x + d))
            fco = (fx, 4.0 * fh - f1 - 3.0 * fx, 2.0 * f1 + 2.0 * fx - 4.0 * fh)
            if line_model == 'cubic':
                # 3D: a 6-tet row is trilinear -> cubic along the line; one extra
                # evaluation at a = 1/2 pins the two remaining coefficients exactly.
                ch = np.asarray(cons(x + 0.5 * d)) - c - 0.5 * gl
                q2, q3 = 8.0 * ch - c1, 2.0 * c1 - 8.0 * ch
                a_star, _m_star, _m0 = _cubic_line_min(c, gl, q2, q3, rho_vec, fco, 1.0)
            else:
                # 2D: every row family is bilinear -> exactly quadratic along the line
                a_star, _m_star, _m0 = _exact_line_min(c, gl, c1, rho_vec, fco, 1.0)
```

(`c1` IS the old `ql`, computed by the same expression in the same order; the 2D call is byte-identical.) Update the docstring paragraph at ~line 500: `'exact_ls'` is no longer "2D only" — the quadratic model is 2D, `line_model='cubic'` is the 3D model, fitted from one extra constraint evaluation; the minimiser is grid + golden-section, verified against the true merit.

- [ ] **Step 5: Forward it from the engine**

`_inners.py`: add `line_model="quadratic",` to `solve_window_inner`'s parameters (next to `step_rule`) and pass `line_model=line_model,` into the `isqp_solve(...)` call; one sentence in its docstring. `_common.py`: `_InnerOpts` gains `line_model: str = 'quadratic'` (after `exact_ls_fallback_steps`); the entry block's degrade becomes

```python
    # A 6-tet row is cubic along a line: the exact line search uses the cubic model on
    # a 3D field (phase 3); the 2D quadratic model is untouched.
    line_model = 'cubic' if is3d else 'quadratic'
```

with `line_model=line_model` passed into `_InnerOpts(...)` (a keyword, after `giant_tile_3d=`), and the DEBUG-log degrade to `'tr'` deleted. Every `solve_window_inner(...)` call in `_solve_window` (the main solve, the no-TR retry, the patience rung, and the polish / re-anchor calls if they forward `step_rule`) gets `line_model=opts.line_model,` next to `step_rule=opts.step_rule,`. `_engine_kwargs` (`asdict(opts)`) would forward the new field to a recursive `windowed_correct`, which has no such parameter (the dimension decides) — drop it there: `if k not in ("ladder", "line_model")`.

- [ ] **Step 6: Retarget the phase-1 degrade test**

`tests/test_isqp_exact_ls.py::test_exact_ls_degrades_to_tr_on_a_3d_field_and_still_refuses_other_ranks` (~line 106): read it; the rank-refusal half stays; the degrade half becomes an assertion that the inner is called with `step_rule='exact_ls'` AND `line_model='cubic'` on the 3D field (monkeypatch `dvfopt.core.windowed._common.solve_window_inner` to record kwargs — the existing test already spies that way; reuse its pattern). Rename it `test_exact_ls_uses_the_cubic_model_on_a_3d_field_and_still_refuses_other_ranks`.

- [ ] **Step 7: Run the tests**

Run (foreground, 900000 ms): `pytest tests/test_isqp_exact_ls.py tests/test_windowed_3d.py tests/test_windowed_isqp.py -q -p no:cacheprovider`
Expected: all pass. The 2D identity A/B is the controller's (it runs after your commit).

- [ ] **Step 8: Lint, format, commit**

```bash
ruff check dvfopt tests && ruff format dvfopt/core/primitives/isqp.py dvfopt/core/windowed/_inners.py dvfopt/core/windowed/_common.py tests/test_isqp_exact_ls.py && mypy
git add dvfopt/core/primitives/isqp.py dvfopt/core/windowed/_inners.py dvfopt/core/windowed/_common.py tests/test_isqp_exact_ls.py
git commit -m "3D windowed, phase 3: exact line search on 3D rows — the cubic line model (2D quadratic path byte-identical)"
```

**Controller measurement after the commit** (log `benchmarks/output/windowed_3d/chain_p3_ls.log`; phase-2 defaults + the Task-2 cap so the `ls_tr` rows are the baseline):

```
--case twist    --tag ls_tr       --set max_window_area=<cap> step_rule=tr
--case twist    --tag ls_cubic    --set max_window_area=<cap>
--case sliver   --tag ls_tr / ls_cubic   (same)
--case moderate --tag ls_tr / ls_cubic   (same)
--case subvol16 --tag ls_tr / ls_cubic   (same; whole window)
```

**Gate:** `ls_cubic` keeps 0 folds / 0 floor / damage 0 on all four AND has fewer SQP iterations on at least 3 of 4 AND less wall on at least 3 of 4, with L2 move within +10 % on every case. Pass: `'exact_ls'` (cubic) is the 3D default and the `a*`-collapse bail / patience rung become live in 3D (note `patience_fallbacks`). Fail: `git revert` the Task-4 commit (the measurement stays in the ledger and the CHANGELOG as a refuted item, with the numbers) and the degrade to `'tr'` remains.

---

### Task 5: The per-dimension defaults table

**Files:**
- Modify: `dvfopt/core/windowed/_common.py` (`_InnerOpts` ~lines 70-110: delete `giant_tile_3d`; `_REANCHOR_OVERLAP` ~line 129; `windowed_correct` signature ~lines 585-625: delete `mop_margin_3d=6` and `giant_tile_3d=16`; the entry block ~lines 897-927; the two recursive-call kwargs ~lines 988 / 1099 (`mop_margin_3d=mop_margin_3d`); the `_ReanchorOpts(...)` build ~line 1150; `_reanchor_pass` ~line 1363; the docstring paragraphs naming `giant_tile_3d` / `mop_margin_3d` ~lines 659-700, 777)
- Modify: `dvfopt/strategies/windowed.py` (delete the `mop_margin_3d` / `giant_tile_3d` fields ~lines 197 / 207 and their docstring entries ~lines 38, 83, 128, 167, 190)
- Modify: `benchmarks/make_hard_crops_3d.py` (`--giant-tile-3d` / `--mop-margin-3d` → `--giant-tile` / `--mop-margin`, forwarded as `giant_tile=` / `mop_margin=`; the record keys become `giant_tile` / `mop_margin`; the `_t{tile}_m{margin}` filename keeps its meaning; docstring line 9)
- Modify: `tests/test_windowed_3d_stages.py` (every `giant_tile_3d=` / `mop_margin_3d=` kwarg → `giant_tile=` / `mop_margin=`; `test_inner_opts_resolve_the_3d_tile` ~line 66 and the strategy-forwarding test ~line 105 rewritten below)
- Test: `tests/test_windowed_3d.py` (append the table tests below)

**Interfaces:**
- Produces: `DEFAULTS_BY_DIM: dict[str, dict[int, object]]` and `resolve_dim_defaults(dim: int, **knobs) -> dict` in `dvfopt/core/windowed/_common.py` (re-export from `dvfopt/core/windowed/__init__.py` if that module re-exports engine names — read it).
- Values: the 2D column is copied verbatim from the current defaults; the 3D column carries the Task 2-4 rulings. This plan's CANDIDATES, to be REPLACED by the ledger's ruled values before implementing: `max_window_area` 5000, `qp_max_iter` 1000, `ip_after_admm_iters` 800, `reanchor_overlap` 8.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_windowed_3d.py`)

```python
def test_defaults_table_resolves_only_knobs_left_at_their_2d_default():
    from dvfopt.core.windowed._common import DEFAULTS_BY_DIM, resolve_dim_defaults

    two = {k: v[2] for k, v in DEFAULTS_BY_DIM.items()}
    assert resolve_dim_defaults(2, **two) == two  # 2D: the table is the identity
    three = resolve_dim_defaults(3, **two)
    assert three["giant_tile"] == 16 and three["mop_margin"] == 6
    assert three["max_window_area"] == DEFAULTS_BY_DIM["max_window_area"][3]
    # an explicit non-default value is honoured in any dimension; 0 still disables the mop
    assert resolve_dim_defaults(3, **{**two, "giant_tile": 12})["giant_tile"] == 12
    assert resolve_dim_defaults(3, **{**two, "mop_margin": 0})["mop_margin"] == 0
    assert resolve_dim_defaults(2, **{**two, "giant_tile": 12})["giant_tile"] == 12


def test_twin_knobs_are_gone():
    import inspect

    from dvfopt.core.windowed import windowed_correct
    from dvfopt.strategies.windowed import ISQPWindowedStrategy

    params = inspect.signature(windowed_correct).parameters
    assert "giant_tile_3d" not in params and "mop_margin_3d" not in params
    assert not hasattr(ISQPWindowedStrategy(), "giant_tile_3d")


def test_strategy_knobs_reach_the_engine_unrenamed_on_3d(monkeypatch):
    import dvfopt.strategies.windowed as strat
    from dvfopt.core.windowed import _common as engine
    from dvfopt.objectives import NoneObjective
    from dvfopt.strategies.windowed import ISQPWindowedStrategy

    seen = {}

    def fake(phi, inner, **kw):
        seen.update(kw)
        return np.asarray(phi, float), engine.SliceReport()

    monkeypatch.setattr(strat, "windowed_correct", fake)
    ISQPWindowedStrategy(giant_tile=12, mop_margin=4).solve(
        np.zeros((3, 6, 8, 8)),
        constraint=SimplexConstraint3D(shape=(6, 8, 8)),
        objective=NoneObjective(),
        threshold=0.01,
    )
    assert seen["giant_tile"] == 12 and seen["mop_margin"] == 4
```

(the module already imports `np` and `SimplexConstraint3D`; check and add what is missing). Replace `tests/test_windowed_3d_stages.py::test_inner_opts_resolve_the_3d_tile` by

```python
def test_inner_opts_have_no_3d_twin():
    from dvfopt.core.windowed._common import _InnerOpts

    assert _InnerOpts().giant_tile == 64 and not hasattr(_InnerOpts(), "giant_tile_3d")
```

and rewrite the strategy-forwarding test there (~line 105) to assert `seen["giant_tile"] == 12 and seen["mop_margin"] == 4` for `ISQPWindowedStrategy(giant_tile=12, mop_margin=4)` (delete its "2D knobs are not repurposed" assertion — the table IS the repurposing now).

- [ ] **Step 2: Run them to verify they fail**

Run (foreground, 600000 ms): `pytest tests/test_windowed_3d.py tests/test_windowed_3d_stages.py -k "defaults_table or twin_knobs or unrenamed or inner_opts" -v -p no:cacheprovider`
Expected: FAIL (`ImportError: cannot import name 'DEFAULTS_BY_DIM'`, `AttributeError`).

- [ ] **Step 3: Implement the table**

In `_common.py`, replace `_REANCHOR_OVERLAP = 8` by

```python
_REANCHOR_OVERLAP = 8  # 2D re-anchor tile overlap, in px (48 stepped by 40 in the prototype)

DEFAULTS_BY_DIM = {
    # knob -> {dim: default}. The 2D column IS the engine's signature default (never
    # change it here without changing the signature); the 3D column is the phase-3
    # measured table (CHANGELOG "3D windowed engine, phase 3"). A knob a caller leaves at
    # its 2D default takes the column for the field's dimension; any other explicit value
    # is honoured in every dimension — so `giant_tile=12` works on a 3D field, and
    # `mop_margin=0` still disables the mop. (`giant_tile=64` on a 3D field therefore
    # reads as "the 3D default"; pass 65 if you really want a 64-voxel tile.)
    'giant_tile': {2: 64, 3: 16},  # 16^3 voxels ~= the 2D 64^2 tile by count
    'mop_margin': {2: 25, 3: 6},  # residual + 6/side -> a 13-17^3 mop window
    'max_window_area': {2: 3000, 3: 5000},  # Task 2: a 17^3 region solves whole (-43 % move)
    'reanchor_tile': {2: 48, 3: 16},
    'reanchor_overlap': {2: _REANCHOR_OVERLAP, 3: 8},  # Task 3 (module-internal, not a kwarg)
    'qp_max_iter': {2: 1000, 3: 1000},  # Task 3
    'ip_after_admm_iters': {2: 800, 3: 800},  # Task 3
}


def resolve_dim_defaults(dim, **knobs):
    """``knobs`` with every entry that sits at its 2D default replaced by the
    ``DEFAULTS_BY_DIM`` column for ``dim`` (the identity for ``dim == 2``)."""
    out = dict(knobs)
    for k, col in DEFAULTS_BY_DIM.items():
        if k in out and out[k] == col[2]:
            out[k] = col[dim]
    return out
```

(substitute the ruled Task 2-4 values). In the entry block, right after `is3d = dim == 3`:

```python
    r = resolve_dim_defaults(
        dim,
        giant_tile=giant_tile,
        mop_margin=mop_margin,
        max_window_area=max_window_area,
        reanchor_tile=reanchor_tile,
        reanchor_overlap=_REANCHOR_OVERLAP,
        qp_max_iter=qp_max_iter,
        ip_after_admm_iters=ip_after_admm_iters,
    )
    giant_tile, mop_margin, max_window_area = r['giant_tile'], r['mop_margin'], r['max_window_area']
    reanchor_tile, reanchor_overlap = r['reanchor_tile'], r['reanchor_overlap']
    qp_max_iter, ip_after_admm_iters = r['qp_max_iter'], r['ip_after_admm_iters']
```

`_ReanchorOpts` gains an `overlap: int` field set from `reanchor_overlap` where it is built, and `_reanchor_pass` reads `opts.overlap` in its `step = max(tile - overlap, tile // 2, 1)` line instead of the module constant (2D: the same 8). Then delete: the `giant_tile_3d if is3d else giant_tile` argument (pass `giant_tile`), the `giant_tile_3d=giant_tile_3d` keyword, the `if is3d: mop_margin = ...` line, the `mop_margin_3d=mop_margin_3d` entries in both recursive-call kwarg dicts, the `giant_tile_3d if is3d else reanchor_tile` expression (pass `reanchor_tile`), the `_InnerOpts.giant_tile_3d` field, and both `_3d` parameters from the signature. The recursive calls (coarse warm start, re-seed polish) pass the RESOLVED values, which on 3D are no longer at their 2D default and so pass through `resolve_dim_defaults` unchanged — state this in a comment at the first recursive call. The coarse gate `min(shape) >= 4 * max(opts.giant_tile, coarse_factor)` needs no change (it reads the resolved tile). Docstrings: one paragraph in `windowed_correct` ("**Per-dimension defaults.** ...", the table's rule in two sentences) replacing every `giant_tile_3d` / `mop_margin_3d` mention; same in `strategies/windowed.py`'s class docstring.

- [ ] **Step 4: Update the strategy, the crop script and the stage tests**

`dvfopt/strategies/windowed.py`: delete the two fields and their docstring entries; add to the `giant_tile` / `mop_margin` / `max_window_area` / `reanchor_tile` entries "(3D: the per-dimension default — see `DEFAULTS_BY_DIM`)". `benchmarks/make_hard_crops_3d.py`: rename the flags and kwargs (file list above); `tests/test_windowed_3d_stages.py`: rename every kwarg (`sed -i 's/giant_tile_3d=/giant_tile=/; s/mop_margin_3d=/mop_margin=/'` then fix the two rewritten tests by hand). One caveat the rename exposes: a stage test that passes `giant_tile_3d=10` and `mop_margin_3d=0` keeps its meaning under the table (10 is explicit, 0 disables); a test that passed `giant_tile_3d=16` now passes `giant_tile=16`, which is explicit and honoured — the same tile. Check `grep -rn "giant_tile_3d\|mop_margin_3d" dvfopt dvfopt_gui tests benchmarks asv_bench` returns nothing (the CHANGELOG's phase-2 history and this plan may keep the names).

- [ ] **Step 5: Run the tests**

Run (foreground, 900000 ms): `pytest tests/test_windowed_3d.py tests/test_windowed_3d_stages.py tests/test_windowed_strategy.py tests/test_gui_strategy_parity.py -k "not tiled_and_cleared and not ras" -q -p no:cacheprovider`
Expected: all pass (the two ~170 s tiler tests are excluded here; the controller runs the full suite).

- [ ] **Step 6: Lint, format, mypy, commit**

```bash
ruff check dvfopt dvfopt_gui tests benchmarks asv_bench && ruff format dvfopt/core/windowed/_common.py dvfopt/strategies/windowed.py benchmarks/make_hard_crops_3d.py tests/test_windowed_3d.py tests/test_windowed_3d_stages.py && mypy
git add -A dvfopt tests benchmarks
git commit -m "3D windowed, phase 3: DEFAULTS_BY_DIM — one per-dimension defaults table replaces the giant_tile_3d / mop_margin_3d twin knobs (3D cap, QP caps, re-anchor tile/overlap)"
```

**Controller after the commit:** the 2D identity A/B (must stay `IDENTITY PASS`) and the full suite.

---

### Task 6 (controller + one docs subagent): the phase-3 gate and docs

**Runs** (serial; log `benchmarks/output/windowed_3d/chain_p3_final.log`; engine defaults = the table): `make_hard_crops_3d.py --cfg l2_rows` for all four crops (pass no tile / margin flags — the defaults are the table) and `windowed_3d_sweep.py --case subvol16 --tag p3_final`.

**Gate:** every crop and the 16^3 at 0 folds (threshold and 0) / 0 floor / damage 0. Compare with the phase-2 rows (twist 468 it / 576 s / L2 27.0; sliver 699 / 633 / 23.3; moderate 672 / 878 / 58.7; cluster 2818 / 8250 / 90.4; 16^3 tiled 299 / 267 / 96.9 vs whole 107 / 638 / 55.8).

**Docs subagent** (one commit `3D windowed, phase 3: docs`): `CHANGELOG.md` gets a "### Added — 3D windowed engine, phase 3: the per-dimension defaults table, the 3D cap, the cubic line model" entry under `[Unreleased]` above the phase-2 entry, with (a) the table and its rule, (b) the Task-2 cap table (subvol16 / sub20 whole vs tiled), (c) the Task-3 QP table (both cases, every tag) and the re-anchor overlap pair, (d) the Task-4 line-model table (`ls_tr` vs `ls_cubic`, four cases) and the ruling, (e) the final crop-pack table next to the phase-2 numbers, (f) the twin-knob removal ("unreleased; `giant_tile=` / `mop_margin=` now work on 3D fields"). `CLAUDE.md`: the 3D bullet's phase-3 sentence (table rule, cap, line model, the numbers) and the windowed-engine knob list (`giant_tile_3d` / `mop_margin_3d` gone). `ARCHITECTURE.md` if it names the twin knobs. `benchmarks/make_hard_crops_3d.py` docstring. Verification: `ruff`, `mypy`, `pytest tests/test_windowed_3d.py -k defaults -q`.

Then: full suite, A/B on the final head, rebase onto `origin/main`, push, `gh pr create -R UCI-iGravi/dvfopt`, squash-merge on all-green, sync main, remove the worktree (copy the ledger to the main checkout's `.superpowers/sdd/` first), update memory.
