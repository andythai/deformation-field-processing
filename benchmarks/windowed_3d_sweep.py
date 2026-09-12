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

``resolved`` in the record is the per-dimension resolution of the table knobs
(``DEFAULTS_BY_DIM``); a ``--set`` at a 2D default value resolves to the 3D column on a
3D case. To sweep a 2D-default value on a 3D case, add ``--set dim_defaults=False``,
which makes ``windowed_correct`` take every knob literally.
"""

import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from windowed_3d_gate import (
    CFGS,
    EXITS,
    QP_LOG,
    RAW,
    THR,
    _load,
)

import dvfopt.core.windowed._common as _cm
from dvfopt.constraints import SimplexConstraint3D
from dvfopt.core.windowed import windowed_correct
from dvfopt.jacobian.tetrahedron_sign import (
    n_neg_best_diagonal,
    six_tet_min_volume_3d,
)

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
    # these three are also passed explicitly below (the cfg / THR set their defaults), so a
    # `--set` touching any of them collides with a duplicate-keyword TypeError unless popped
    # here first -- this is what lets e.g. `--set orientation_delta=None` override the default.
    od = kw.pop("orientation_delta", od)
    rows = kw.pop("orientation_rows", "edges")
    thr = kw.pop("threshold", THR)
    EXITS.clear()
    QP_LOG.clear()
    TRACES.clear()
    t = time.perf_counter()
    out, rep = windowed_correct(
        phi.copy(),
        "isqp",
        constraint=c,
        objective=obj_cls(),
        threshold=thr,
        orientation_delta=od,
        orientation_rows=rows,
        verbose=0,
        **kw,
    )
    wall = time.perf_counter() - t
    move = out - phi
    admm = [(w, i) for w, i, s in QP_LOG if not s.startswith("clarabel")]
    qp_w = np.array([w for w, _i, _s in QP_LOG]) if QP_LOG else np.zeros(1)
    dim = phi.ndim - 1
    base = {
        k: col[2] for k, col in _cm.DEFAULTS_BY_DIM.items()
    }  # the 2D defaults = the engine signature defaults
    base.update({k: v for k, v in kw.items() if k in _cm.DEFAULTS_BY_DIM})
    # `--set dim_defaults=False` makes the engine take the knobs literally, so does this
    resolved = _cm.resolve_dim_defaults(dim, **base) if kw.get("dim_defaults", True) else base
    cap = resolved["qp_max_iter"]
    rec = dict(
        case=case,
        cfg=cfg,
        tag=tag,
        settings=settings,
        resolved=resolved,
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
        qp_n=len(QP_LOG),
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
    "case",
    "tag",
    "folds_out",
    "floor_out",
    "damage",
    "rounds",
    "n_windows",
    "mop_windows",
    "sqp_iters",
    "rejected_steps",
    "qp_n",
    "qp_s_median",
    "admm_median",
    "admm_at_cap",
    "ip_solves",
    "wall_s",
    "l2_move",
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
    ap.add_argument(
        "--case", choices=["subvol16", "sub20", "twist", "cluster", "sliver", "moderate"]
    )
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
