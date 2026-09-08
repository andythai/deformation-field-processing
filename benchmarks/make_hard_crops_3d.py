"""Cut the 3D hard-crop pack from the raw B0039 field and run the phase-2 gate on it.

Four 24^3 crops (offsets from a stride-12 scan of the raw field, 2026-09-07):
twist (min tet volume -13.4, sparse), cluster (25 % of cubes below threshold),
sliver (880 cubes in [-0.001, 0.01), 1.8 % negative), moderate (10 % below).
``--build-only`` cuts them into data/dvfs/crops_3d/ (gitignored). Without it the
script also runs ``windowed_correct`` on every crop under the config given by
``--cfg`` (default: l2_rows, the engine default) and the tiler/mop knobs
``--giant-tile`` / ``--mop-margin`` (defaults 16 / 6, the engine defaults),
writing crops_<name>_<cfg>_t{tile}_m{margin}.json + crops.md to
benchmarks/output/windowed_3d/ — the reference table of the 3D port's
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
from dvfopt.jacobian.tetrahedron_sign import (  # noqa: E402
    n_neg_best_diagonal,
    six_tet_min_volume_3d,
)
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
# <objective>_<rows|norows>: 'none' is NoneObjective (pure feasibility), 'rows' = the 3D edge rows on
CFGS = {
    "l2_rows": (L2Objective, 0.01),
    "none_rows": (NoneObjective, 0.01),
    "l2_norows": (L2Objective, None),
}


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


def run(name, cfg, giant_tile, mop_margin):
    obj_cls, od = CFGS[cfg]
    phi = np.load(os.path.join(OUT_CROPS, f"{name}.npy"))
    c = SimplexConstraint3D(shape=phi.shape[1:])
    mv0 = six_tet_min_volume_3d(phi)
    t = time.perf_counter()
    out, rep = windowed_correct(
        phi.copy(),
        "isqp",
        constraint=c,
        objective=obj_cls(),
        threshold=THR,
        orientation_delta=od,
        orientation_rows="edges",
        giant_tile=giant_tile,
        mop_margin=mop_margin,
        verbose=0,
    )
    wall = time.perf_counter() - t
    move = out - phi
    rec = dict(
        case=name,
        cfg=cfg,
        giant_tile=giant_tile,
        mop_margin=mop_margin,
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
        mop_cleared=int(rep.mop_cleared),
        reseed_rounds_run=int(rep.reseed_rounds_run),
        coarse_folds_before=int(rep.coarse_folds_before),
        sqp_iters=int(sum(w.inner_iters for w in rep.windows)),
        wall_s=wall,
        l2_move=float(np.linalg.norm(move.ravel())),
        l1_move=float(np.abs(move).sum()),
    )
    with open(os.path.join(OUT, f"crops_{name}_{cfg}_t{giant_tile}_m{mop_margin}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps(rec), flush=True)
    assert rec["damage"] == 0, rec
    return rec


def table():
    rows = []
    for f in sorted(os.listdir(OUT)):
        if f.startswith("crops_") and f.endswith(".json"):
            with open(os.path.join(OUT, f)) as fh:
                rows.append(json.load(fh))
    cols = [
        "case",
        "cfg",
        "giant_tile",
        "mop_margin",
        "folds_in",
        "floor_in",
        "folds_out",
        "floor_out",
        "min_out",
        "damage",
        "rounds",
        "n_windows",
        "giant_regions",
        "mop_windows",
        "reseed_rounds_run",
        "mop_cleared",
        "coarse_folds_before",
        "folds_out_zero",
        "floor_out_zero",
        "l1_move",
        "sqp_iters",
        "wall_s",
        "l2_move",
    ]
    with open(os.path.join(OUT, "crops.md"), "w") as fh:
        fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(str(r.get(k, "")) for k in cols) + " |\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--case", choices=list(CROPS))
    ap.add_argument("--cfg", choices=list(CFGS), default="l2_rows")
    ap.add_argument("--giant-tile", type=int, default=16)
    ap.add_argument("--mop-margin", type=int, default=6)
    a = ap.parse_args()
    print(f"dvfopt from {dvfopt.__file__}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    crops_built = all(os.path.isfile(os.path.join(OUT_CROPS, f"{name}.npy")) for name in CROPS)
    if not crops_built or a.build_only:
        build()
    if a.build_only:
        return
    for name in CROPS:
        if a.case and name != a.case:
            continue
        run(name, a.cfg, a.giant_tile, a.mop_margin)
    table()


if __name__ == "__main__":
    main()
