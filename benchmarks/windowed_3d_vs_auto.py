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
import numpy as np
from benchmark_utils import load_cohort_field

from dvfopt.constraints import SimplexConstraint3D
from dvfopt.jacobian.tetrahedron_sign import (
    n_neg_best_diagonal,
    six_tet_min_volume_3d,
)
from dvfopt.solver import auto_strategy, correct_dvf

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
            phi.copy(),
            constraint="simplex_3d",
            strategy=method,
            objective=objective,
            threshold=THR,
            # The windowed strategy only lifts the SliceReport's final stats
            # (damage/rounds/...) to SolveInfo.extras -- and total_iter only
            # gets a real count -- when record_history=True is asked for
            # (dvfopt/strategies/windowed.py, dvfopt/strategies/base.py
            # _build_solve_info: an unrecorded run returns a bare, empty
            # SolveInfo). Other strategies are unaffected either way.
            record_history=(method == "isqp_windowed"),
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
    "case",
    "method",
    "objective",
    "auto_pick",
    "folds_in",
    "folds_out",
    "floor_out",
    "new_folds",
    "damage",
    "moved_frac",
    "sqp_iters",
    "wall_s",
    "l2_move",
    "l1_move",
    "feasible",
]


def table():
    rows = []
    for f in sorted(os.listdir(OUT)):
        if f.startswith("h2h_") and f.endswith(".json"):
            with open(os.path.join(OUT, f)) as fh:
                rows.append(json.load(fh))
    rows.sort(
        key=lambda r: (r["case"], METHODS.index(r["method"]) if r["method"] in METHODS else 9)
    )
    lines = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for r in rows:
        cells = [
            f"{r.get(k, ''):.4g}" if isinstance(r.get(k), float) else str(r.get(k, ""))
            for k in COLS
        ]
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
