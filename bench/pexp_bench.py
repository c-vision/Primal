#!/usr/bin/env python3
"""Mixed SOC + exponential-cone benchmark (the README table), with the instance
defined in the repository instead of in a scratch directory.

The C side is bench/pexp_bench.c: one primal exponential cone per observation,
one rotated quadratic cone for the penalty,

    min   0.1*w + sum_i v_i
    s.t.  u_i = a_i'x + b_i,   (v_i, 1, u_i) in PEXP,   (w, 1, x) in RQUAD

solved through PRIMAL_optimize and timed internally (so process start-up is not
in the number).  This script generates the same data for the Clarabel
reference, checks every Primal answer against the known optimum, and reports
medians of an interleaved series.

The reference is the cone form Clarabel wants (Ax + s = b, s in K, with the
quadratic penalty written as 1+w >= ||(w-1, sqrt(2) x)||), which is the same
optimization problem as the C model above — not a different one.

Run with the interpreter that has clarabel/scipy/numpy:

    /Users/gaetano/ai/env-bench/bin/python3.14 bench/pexp_bench.py
    /Users/gaetano/ai/env-bench/bin/python3.14 bench/pexp_bench.py --runs 9
    /Users/gaetano/ai/env-bench/bin/python3.14 bench/pexp_bench.py \\
        --primal out/bench/pexp_bench --compare out/f202/out/bench/pexp_bench

`--compare` runs a second Primal binary in the same interleaved series: that is
the only way to compare two builds when the per-cell time is a millisecond and
the machine drifts between runs.
"""
from __future__ import annotations

import argparse
import math
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIZES = [(20, 4), (40, 8), (80, 12)]
# Validated optima of the model above, independent of either solver.
EXPECTED = {
    20: 18.718151648415301,
    40: 47.09337729630259,
    80: 81.84315908327622,
}
TOL = 1e-8


def instance(n: int, d: int):
    """The instance data, exactly as bench/pexp_bench.c draws it."""
    state = 11235813

    def draw():
        nonlocal state
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        return ((state >> 16) & 0x7FFF) / 32767.0

    feats = [(draw() * 2 - 1) * 0.5 for _ in range(n * d)]
    bias = [draw() * 2 - 1 for _ in range(n)]
    return feats, bias


def run_primal(binary: Path, n: int, d: int):
    line = subprocess.check_output([str(binary), str(n), str(d)], text=True, timeout=120)
    if not line.startswith("metrics "):
        raise SystemExit(f"unexpected output from {binary}: {line!r}")
    t, p, q, a, b = map(float, line.split()[1:6])
    gap = abs(p - q) / (1.0 + abs(p) + abs(q))
    rel = abs(p - EXPECTED[n]) / (1.0 + abs(EXPECTED[n]))
    bad = [(name, v) for name, v in
           (("rel_pri", a), ("rel_dual", b), ("rel_gap", gap), ("obj_vs_known", rel))
           if not (math.isfinite(v) and v <= TOL)]
    if bad:
        raise SystemExit(f"n={n} d={d} {binary}: " + ", ".join(f"{k}={v:.3e}" for k, v in bad))
    return t, p, q


def make_reference(n: int, d: int):
    import numpy as np
    from scipy import sparse
    import clarabel

    feats, bias = instance(n, d)
    features = np.array(feats).reshape(n, d)
    bias = np.array(bias)
    rows = 3 * n + d + 2
    mat = np.zeros((rows, d + n + 1))
    rhs = np.zeros(rows)
    for i in range(n):
        mat[3 * i, :d] = -features[i]      # s_t = bias_i + a_i'x
        rhs[3 * i] = bias[i]
        rhs[3 * i + 1] = 1.0               # s_u = 1
        mat[3 * i + 2, d + i] = -1.0       # s_v = v_i
    rhs[3 * n] = 1.0                       # s = (1 + w, w - 1, sqrt(2) x)
    rhs[3 * n + 1] = -1.0
    mat[3 * n:3 * n + 2, -1] = -1.0
    mat[3 * n + 2:, :d] = -math.sqrt(2) * np.eye(d)
    A = sparse.csc_matrix(mat)
    P = sparse.csc_matrix((d + n + 1, d + n + 1))
    c = np.r_[np.zeros(d), np.ones(n), 0.1]
    cones = [clarabel.ExponentialConeT() for _ in range(n)]
    cones.append(clarabel.SecondOrderConeT(d + 2))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = 1
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-9
    return (P, c, A, rhs, cones, settings)


def run_clarabel(ref, n: int, d: int):
    import clarabel

    P, c, A, rhs, cones, settings = ref
    t0 = time.perf_counter()
    solver = clarabel.DefaultSolver(P, c, A, rhs, cones, settings)
    sol = solver.solve()
    dt = time.perf_counter() - t0
    if sol.status != clarabel.SolverStatus.Solved:
        raise SystemExit(f"Clarabel n={n}: {sol.status}")
    rel = abs(sol.obj_val - EXPECTED[n]) / (1.0 + abs(EXPECTED[n]))
    if rel > 1e-7:
        raise SystemExit(f"Clarabel n={n}: obj={sol.obj_val:.12g} vs known {EXPECTED[n]:.12g}")
    return dt, sol.obj_val


def median(xs):
    return statistics.median(xs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=21, help="interleaved series (default 21)")
    ap.add_argument("--primal", default=str(ROOT / "out" / "bench" / "pexp_bench"),
                    help="Primal binary to time")
    ap.add_argument("--compare", default=None,
                    help="second Primal binary, timed in the same series")
    ap.add_argument("--no-clarabel", action="store_true")
    ap.add_argument("--sizes", default=",".join(f"{n}x{d}" for n, d in SIZES))
    args = ap.parse_args()

    primal = Path(args.primal).resolve()
    if not primal.exists():
        raise SystemExit(f"missing {primal}: run `make bench` first")
    other = Path(args.compare).resolve() if args.compare else None
    if other is not None and not other.exists():
        raise SystemExit(f"missing {other}")

    sizes = []
    for spec in args.sizes.split(","):
        n, d = spec.lower().split("x")
        sizes.append((int(n), int(d)))

    ref = None
    if not args.no_clarabel:
        try:
            import clarabel  # noqa: F401
            import numpy  # noqa: F401
            import scipy  # noqa: F401
        except ImportError as exc:
            raise SystemExit(f"Clarabel/numpy/scipy unavailable ({exc}): "
                             f"use the bench interpreter or --no-clarabel")

    print(f"series: {args.runs} per cell, interleaved"
          + (f"; comparing {primal.name} against {other}" if other else ""))
    head = "| Instance | Primal (ms) |"
    if other:
        head += " Compare (ms) | ratio |"
    if not args.no_clarabel:
        head += " Clarabel (ms) | ratio |"
    print(head)
    if other and not args.no_clarabel:
        print("|---|---:|---:|---:|---:|---:|")
    elif other or not args.no_clarabel:
        print("|---|---:|---:|---:|" if not other else "|---|---:|---:|---:|")
    else:
        print("|---|---:|")

    for n, d in sizes:
        ref = None if args.no_clarabel else make_reference(n, d)
        tp, to, tc = [], [], []
        for _ in range(args.runs):
            tp.append(run_primal(primal, n, d)[0])
            if other is not None:
                to.append(run_primal(other, n, d)[0])
            if ref is not None:
                tc.append(run_clarabel(ref, n, d)[0])
        mp = median(tp) * 1e3
        row = f"| {n}×{d} | {mp:.3f} |"
        if other is not None:
            mo = median(to) * 1e3
            row += f" {mo:.3f} | {mp / mo:.3f}× |"
        if ref is not None:
            mc = median(tc) * 1e3
            row += f" {mc:.3f} | {mp / mc:.2f}× |"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
