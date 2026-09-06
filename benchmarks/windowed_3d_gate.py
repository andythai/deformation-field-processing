"""3D windowed engine — the phase-1 gate and the two measured unknowns.

``--gate``  the three ``testcases_3d`` fields + the 16^3 B0039 sub-volume under
            {L2, none} x {edge rows on, off} on the ``'tr'`` step rule: fixed-6-tet
            folds (at threshold and at 0), the best-diagonal floor (both), damage,
            windows, SQP iterations, the exit reason of every inner call (i.e. every
            ladder rung), wall, L1 / L2 move and the largest |dz| moved. Asserts 0
            folds / damage 0 on the testcases under the default config (the plumbing
            gate) and damage 0 everywhere; the 16^3 result is reported with its floor.
``--cost``  per-SQP-iteration cost vs window volume: ONE frozen-ring window over the
            interior of 9^3 / 17^3 / 25^3 / 33^3 cubes, 8 SQP iterations, every QP
            solve timed (wall, ADMM iterations, status) through a proxy around
            ``isqp._make_qp``, the Jacobian build and constraint evaluation timed
            separately, and the QP size (free variables + one slack per row).
            ``--L`` restricts a ``--cost`` run to one cube edge (9/17/25/33) — the
            33^3 raw-B0039 cut is slow, so the sweep is meant to be run one edge at
            a time; each call reads any existing ``cost.json``, replaces the row for
            the edge just measured, and writes the merged table back, so four
            separate ``--cost --L <edge>`` calls build one table. Every QP solve
            also prints one line as it happens, so a run cut short by a timeout
            still leaves its per-QP measurements on stdout.

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
from dvfopt.jacobian.tetrahedron_sign import (  # noqa: E402
    n_neg_best_diagonal,
    six_tet_min_volume_3d,
)
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
    """Proxy over the QP object ``isqp._make_qp`` returns: every ``solve()`` is
    timed, logged and printed. Forwards every other attribute get AND set to the
    wrapped object, so it is a transparent drop-in wherever ``isqp_solve`` uses the
    object ``_make_qp`` hands back — checked against ``isqp.py``'s only call site
    (the ``_make_qp(...)`` at its ``prob = ...`` assignment) and ``_HybridQP``:
    both only ever call ``.setup()`` / ``.update()`` / ``.solve()`` on it, never
    set an attribute directly, but the ``__setattr__`` forward is kept so the proxy
    stays correct if that ever changes.
    """

    def __init__(self, qp):
        self._qp = qp

    def __getattr__(self, name):
        return getattr(self._qp, name)

    def __setattr__(self, name, value):
        if name == "_qp":
            object.__setattr__(self, name, value)
        else:
            setattr(self._qp, name, value)

    def solve(self):
        t = time.perf_counter()
        res = self._qp.solve()
        wall = time.perf_counter() - t
        iters = int(getattr(res.info, "iter", -1))
        status = str(getattr(res.info, "status", "?")).strip()
        QP_LOG.append((wall, iters, status))
        print(f"    qp {wall:.3f}s iters={iters} {status}", flush=True)
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
        phi.copy(),
        "isqp",
        constraint=c,
        objective=obj_cls(),
        threshold=THR,
        orientation_delta=od,
        orientation_rows="edges",
        verbose=0,
    )
    wall = time.perf_counter() - t
    mv1 = six_tet_min_volume_3d(out)
    move = out - phi
    qp = np.array([w for w, _i, _s in QP_LOG]) if QP_LOG else np.zeros(1)
    rec = dict(
        case=name,
        cfg=cfg,
        shape=list(map(int, phi.shape[1:])),
        folds_in=int((mv0 < THR).sum()),
        folds_in_zero=int((mv0 <= 0).sum()),
        floor_in=int(n_neg_best_diagonal(phi, THR)),
        floor_in_zero=int(n_neg_best_diagonal(phi, 0.0)),
        folds_out=int(rep.folds_after),
        folds_out_zero=int(rep.folds_after_zero),
        floor_out=int(rep.best_diag_floor_after),
        floor_out_zero=int(rep.best_diag_floor_after_zero),
        min_in=float(mv0.min()),
        min_out=float(mv1.min()),
        damage=int(rep.damage),
        n_windows=int(rep.n_windows),
        giant_regions=int(rep.giant_regions),
        rounds=int(rep.rounds),
        sqp_iters=int(sum(w.inner_iters for w in rep.windows)),
        feasible_windows=int(sum(w.feasible for w in rep.windows)),
        no_tr_fallbacks=int(sum(w.fallback for w in rep.windows)),
        backend_fallbacks=int(rep.backend_fallbacks),
        grown_windows=int(sum(w.grows > 0 for w in rep.windows)),
        exits=dict(collections.Counter(EXITS)),
        n_qp=len(QP_LOG),
        qp_s_median=float(np.median(qp)),
        qp_s_max=float(qp.max()),
        # mixes ADMM and IP (Clarabel cold-solve) iteration counts; the per-QP status string separates them.
        admm_iters_median=float(np.median([i for _w, i, _s in QP_LOG])) if QP_LOG else -1.0,
        wall_s=wall,
        l2_move=float(np.linalg.norm(move.ravel())),
        l1_move=float(np.abs(move).sum()),
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
        sub,
        "isqp",
        8,
        trace=tr,
        osqp_max_iter=1000,
        qp_backend="hybrid",
        step_rule="tr",
        feas_tol=5e-4,
        ftol=1e-2,
    )
    wall = time.perf_counter() - t
    return dict(
        L=L,
        n_free=int(sub.free_idx.size),
        n_rows=int(sub.n_enforced),
        qp_vars=int(sub.free_idx.size + sub.n_enforced),
        nnz_jac=int(J.nnz),
        jac_s=jac_s,
        cons_s=cons_s,
        sqp_iters=int(nit),
        feasible=bool(ok),
        wall_s=wall,
        s_per_sqp_iter=wall / max(1, nit),
        exit=str(tr.get("exit")),
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
    ap.add_argument(
        "--L", type=int, choices=sorted(COST_CUBES), help="restrict --cost to one cube edge"
    )
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
        cols = [
            "case",
            "cfg",
            "folds_in",
            "folds_out",
            "folds_out_zero",
            "floor_in",
            "floor_out",
            "floor_out_zero",
            "min_out",
            "damage",
            "n_windows",
            "sqp_iters",
            "exits",
            "wall_s",
            "l2_move",
            "max_abs_dz",
        ]
        with open(os.path.join(OUT, "gate.md"), "w") as fh:
            fh.write(_md(rows, cols))
    if args.cost:
        cubes = COST_CUBES if args.L is None else {args.L: COST_CUBES[args.L]}
        cost_json_path = os.path.join(OUT, "cost.json")
        existing = {}
        if os.path.exists(cost_json_path):
            with open(cost_json_path) as fh:
                existing = {r["L"]: r for r in json.load(fh)}
        for L, src in cubes.items():
            if src == "raw":
                phi, off, frac = cut_raw(L)
                print(f"raw cut at {off}, fold fraction {frac:.3f}", flush=True)
            else:
                phi = _load(src)
            rec = cost_case(L, phi)
            existing[L] = rec
            print(json.dumps(rec), flush=True)
        rows = [existing[L] for L in sorted(existing)]
        with open(cost_json_path, "w") as fh:
            json.dump(rows, fh, indent=1)
        cols = [
            "L",
            "n_free",
            "n_rows",
            "qp_vars",
            "nnz_jac",
            "jac_s",
            "cons_s",
            "sqp_iters",
            "s_per_sqp_iter",
            "exit",
        ]
        for r in rows:
            r["qp_s_median"] = float(np.median([w for w, _i, _s in r["qp"]])) if r["qp"] else -1
            r["admm_median"] = float(np.median([i for _w, i, _s in r["qp"]])) if r["qp"] else -1
        with open(os.path.join(OUT, "cost.md"), "w") as fh:
            fh.write(_md(rows, [*cols, "qp_s_median", "admm_median"]))


if __name__ == "__main__":
    main()
