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
    cases["coarse260"] = (
        coarse,
        ["bilinear"],
        ["none"],
    )  # min(H, W) >= 4 * 64 -> coarse warm start
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
        [
            w.fy0,
            w.fx0,
            w.ph,
            w.pw,
            w.n_free,
            w.n_enforced,
            w.inner_iters,
            w.grows,
            w.fallback,
            w.backend_fallback,
            w.patience_fallback,
            w.min_after,
            w.feasible,
        ]
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
                    phi.copy(),
                    "isqp",
                    constraint=FAMILIES[fam](shape=phi.shape[1:]),
                    objective=OBJECTIVES[obj](),
                    threshold=THR,
                    verbose=0,
                )
                np.save(os.path.join(out_dir, f"{key}.npy"), out)
                with open(os.path.join(out_dir, f"{key}.json"), "w") as fh:
                    json.dump(_scalars(rep), fh, sort_keys=True)
                print(
                    f"  {key}: folds {rep.folds_before}->{rep.folds_after} damage {rep.damage} "
                    f"windows {rep.n_windows} {rep.time_s:.1f}s",
                    flush=True,
                )


def compare(a, b):
    keys_a = sorted(f[:-4] for f in os.listdir(a) if f.endswith(".npy"))
    keys_b = sorted(f[:-4] for f in os.listdir(b) if f.endswith(".npy"))
    ok = keys_a == keys_b
    if not ok:
        print(f"key sets differ: {set(keys_a) ^ set(keys_b)}")
    for key in keys_a:
        same_arr = np.array_equal(
            np.load(os.path.join(a, f"{key}.npy")), np.load(os.path.join(b, f"{key}.npy"))
        )
        with open(os.path.join(a, f"{key}.json")) as fa, open(os.path.join(b, f"{key}.json")) as fb:
            same_rep = json.load(fa) == json.load(fb)
        print(
            f"  {key}: field {'same' if same_arr else 'DIFFERENT'}, report {'same' if same_rep else 'DIFFERENT'}"
        )
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
