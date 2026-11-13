#!/usr/bin/env python3
"""The SOC / exp / mixed measurement matrix: one row per cell, medians.

bench/pexp_matrix.c builds the instance; this script runs it and reads the
GMB_DBG trace for what the public API does not expose — the KKT order, the
structural nonzeros, whether the route is dense or sparse, and the iteration
count. Those four numbers are what tells a *cost* from a *trajectory* problem,
and they are the reason this runner exists.

    python3 bench/pexp_matrix.py                 # the three experiments
    python3 bench/pexp_matrix.py --runs 9
    python3 bench/pexp_matrix.py --only family

Experiments (one knob at a time):

  family    soc | exp | mix   over the number of blocks, at d=8, rho=0.5
  density   soc | exp | mix   over the link density rho, at n=40, d=12
  degen     exp | mix         over rep (identical rows and cones), at n=40

Columns: blocks = cone blocks, rows = linking rows, KKT n = augmented order,
nnz = structural nonzeros of the sparse form, route = dense/sparse, it =
iterations, ms = median wall time of the solve, ms/it = the two divided.

With --ref (the default when clarabel is importable) each cell also runs a
Clarabel reference built on the SAME instance: the data generator below repeats
the C one draw for draw, and the two objectives are asserted to agree, so the
columns are comparable. Clarabel timings include solver construction.  SCS is
not used: the log/power cones are not in its default cone set.
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "out" / "bench" / "pexp_matrix"

KKT = re.compile(r"\[kkt\] pass=0 n=(\d+) reuse=0(.*)")
ITER = re.compile(r"^\s*(?:socp(?:_sp)? )?it=(\d+)")
ROUTE = re.compile(r"\[route\] status=(\d)")
RETRY = re.compile(r"\[retry\]")


# --------------------------------------------------------------------------
# Clarabel reference: the same instance, built from the same draws.
# --------------------------------------------------------------------------

class _Lcg:
    """The C generator of bench/pexp_matrix.c, step for step."""

    def __init__(self, seed=11235813):
        self.state = seed

    def draw(self) -> float:
        self.state = (self.state * 1103515245 + 12345) & 0xFFFFFFFF
        return ((self.state >> 16) & 0x7FFF) / 32767.0


def instance(fam: str, n: int, d: int, rho: float, rep: int):
    """(rows, bias, cx) with rows[i] = (j, a) for the linked features."""
    nb = n * rep
    m = max(1, min(d, int(rho * d + 0.5)))
    rng = _Lcg()
    A = [[(rng.draw() * 2 - 1) * 0.5 for _ in range(d)] for _ in range(n)]
    bias = [rng.draw() * 2 - 1 for _ in range(n)]
    cx = [rng.draw() * 2 - 1 for _ in range(nb * d)]
    links = []
    for i in range(n):
        step = 1 + int(rng.draw() * max(0, d - 2) + 0.5)
        if step >= d:
            step = 1
        while d > 1 and step > 1:
            a, b = step, d
            while b:
                a, b = b, a % b
            if a == 1:
                break
            step += 1
        j0 = int(rng.draw() * d)
        links.append([(j0 + k * step) % d for k in range(m)])
    return nb, m, A, bias, cx, links


def build_reference(fam: str, n: int, d: int, rho: float, rep: int):
    import numpy as np
    from scipy import sparse
    import clarabel

    nb, m, A, bias, cx, links = instance(fam, n, d, rho, rep)
    if fam == "soc":
        # 2 w >= ||x||^2  <=>  ||(w-1, sqrt2 x)|| <= w+1: an ordinary SOC per
        # block (Clarabel's Python API has no rotated cone type).
        rowsk, colsk, valsk, rhs, cones = [], [], [], [], []
        r = 0
        for k in range(nb):
            rowsk.append(r); colsk.append(k); valsk.append(-1.0)      # s = 1 + w_k
            rhs.append(1.0)
            r += 1
            rowsk.append(r); colsk.append(k); valsk.append(-1.0)      # s = w_k - 1
            rhs.append(-1.0)
            r += 1
            for j in range(d):                                        # s = sqrt2 x
                rowsk.append(r); colsk.append(nb + k * d + j); valsk.append(-2.0 ** 0.5)
                rhs.append(0.0)
                r += 1
            cones.append(clarabel.SecondOrderConeT(d + 2))
        nv = nb + nb * d
        q = np.zeros(nv)
        q[:nb] = 0.1
        q[nb:] = np.array(cx[:nb * d])
        Aeq = sparse.csc_matrix((valsk, (rowsk, colsk)), shape=(r, nv))
        return sparse.csc_matrix((nv, nv)), q, Aeq, np.array(rhs), cones

    nvar = d + nb + 1 + (1 if fam == "mix" else 0)
    V0, W = d, d + nb
    rowsk, colsk, valsk, rhs, cones = [], [], [], [], []
    r = 0
    for i in range(nb):
        base = i % n
        for pos in range(3):
            if pos == 0:            # s = bias + a'x  (the exp argument)
                for j in links[base]:
                    rowsk.append(r); colsk.append(j); valsk.append(-A[base][j])
                rhs.append(bias[base])
            elif pos == 1:          # s = 1
                rhs.append(1.0)
            else:                   # s = v_i
                rowsk.append(r); colsk.append(V0 + i); valsk.append(-1.0)
                rhs.append(0.0)
            r += 1
        cones.append(clarabel.ExponentialConeT())
    if fam == "mix":
        # (1+w, w-1, sqrt2 x) in SOC, i.e. w >= 0.5||x||^2
        rhs.extend([1.0, -1.0])
        rowsk.extend([r, r + 1]); colsk.extend([W, W]); valsk.extend([-1.0, -1.0])
        r += 2
        for j in range(d):
            rowsk.append(r); colsk.append(j); valsk.append(-2.0 ** 0.5)
            rhs.append(0.0)
            r += 1
        cones.append(clarabel.SecondOrderConeT(d + 2))
    q = np.zeros(nvar)
    q[V0:V0 + nb] = 1.0
    if fam == "mix":
        q[W] = 0.1
    Aeq = sparse.csc_matrix((valsk, (rowsk, colsk)), shape=(r, nvar))
    return sparse.csc_matrix((nvar, nvar)), q, Aeq, np.array(rhs), cones


def run_clarabel(fam, n, d, rho, rep):
    import clarabel
    P, q, A, b, cones = build_reference(fam, n, d, rho, rep)
    st = clarabel.DefaultSettings()
    st.verbose = False
    st.max_threads = 1
    st.tol_gap_abs = st.tol_gap_rel = st.tol_feas = 1e-9
    t0 = time.perf_counter()
    solver = clarabel.DefaultSolver(P, q, A, b, cones, st)
    sol = solver.solve()
    dt = time.perf_counter() - t0
    if sol.status != clarabel.SolverStatus.Solved:
        raise SystemExit(f"Clarabel {fam} n={n}: {sol.status}")
    return dt, float(sol.obj_val)


def run(fam: str, n: int, d: int, rho: float, rep: int):
    env = dict(os.environ, GMB_DBG="1")
    p = subprocess.run([str(BIN), fam, str(n), str(d), f"{rho:.2f}", str(rep)],
                       capture_output=True, text=True, env=env, timeout=600)
    out = [l for l in p.stdout.splitlines() if l.startswith("matrix,")]
    if not out:
        raise SystemExit(f"{fam} {n} {d} {rho} {rep}: no matrix line\n{p.stderr[-800:]}")
    f = out[-1].split(",")
    secs, pobj, dobj = float(f[8]), float(f[9]), float(f[10])
    pinf, dinf = float(f[11]), float(f[12])
    relgap = abs(pobj - dobj) / (1.0 + abs(pobj) + abs(dobj))
    err = p.stderr
    order = nnz = None
    sparse = None
    for line in err.splitlines():
        m = KKT.search(line)
        if m:
            order = int(m.group(1))
            rest = m.group(2)
            sparse = "sparse" in rest
            mm = re.search(r"nnz=(\d+)", rest)
            nnz = int(mm.group(1)) if mm else None
            break
    its = 0
    for line in err.splitlines():
        m = ITER.match(line)
        if m:
            its = max(its, int(m.group(1)) + 1)
    route = None
    m = ROUTE.search(err)
    if m:
        route = "native" if m.group(1) == "0" else "cuts"
    if "socp it=" in err or "socp_sp it=" in err:
        route = "socp"
    retries = len(RETRY.findall(err))
    return dict(secs=secs, pobj=pobj, pinf=pinf, dinf=dinf, relgap=relgap, order=order,
                nnz=nnz, sparse=sparse, its=its, route=route, retries=retries)


def cell(fam, n, d, rho, rep, runs):
    vals = [run(fam, n, d, rho, rep) for _ in range(runs)]
    med = {k: statistics.median([v[k] for v in vals])
           for k in ("secs", "pobj", "pinf", "dinf", "relgap")}
    first = vals[0]
    med.update(order=first["order"], nnz=first["nnz"], sparse=first["sparse"],
               its=first["its"], route=first["route"], retries=first["retries"])
    return med


def table(title, cases, runs, ref=False):
    print(f"\n### {title}\n")
    head = "| case | blocks | rows | KKT n | nnz | route | it | ms | ms/it |"
    if ref:
        head += " clarabel ms | ratio |"
    head += " rel_gap | retry |"
    print(head)
    print("|---|---:|---:|---:|---:|---|---:|---:|---:|" + ("---:|---:|" if ref else "---:|---:|"))
    for fam, n, d, rho, rep in cases:
        c = cell(fam, n, d, rho, rep, runs)
        rms = ratio = None
        if ref:
            tref, oref = zip(*[run_clarabel(fam, n, d, rho, rep) for _ in range(max(3, runs // 2))])
            rms = statistics.median(tref) * 1e3
            ratio = (c["secs"] * 1e3) / rms
            rel = abs(c["pobj"] - statistics.median(oref)) / (1.0 + abs(c["pobj"]))
            if rel > 1e-6:
                raise SystemExit(f"objective mismatch {fam} n={n} rho={rho} rep={rep}: "
                                 f"ours {c['pobj']:.12g} reference {statistics.median(oref):.12g}")
        blocks, rows = n * rep, 0 if fam == "soc" else n * rep
        gap = c["relgap"]
        route = "dense" if c["sparse"] is False else "sparse" if c["sparse"] else (c["route"] or "-")
        order = c["order"] if c["order"] is not None else "-"
        nnz = c["nnz"] if c["nnz"] is not None else "-"
        ms = c["secs"] * 1e3
        per = ms / c["its"] if c["its"] else 0.0
        label = f"{fam} n={n} rho={rho:.2f} rep={rep}"
        row = (f"| {label} | {blocks} | {rows} | {order} | {nnz} | {route} | {c['its']} | "
               f"{ms:.2f} | {per:.3f} |")
        if ref:
            row += f" {rms:.2f} | {ratio:.2f}x |"
        row += f" {gap:.1e} | {c['retries']} |"
        print(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--only", default="family,density,degen")
    ap.add_argument("--no-ref", action="store_true", help="skip the Clarabel reference")
    a = ap.parse_args()
    if not BIN.exists():
        raise SystemExit(f"missing {BIN}: run `make bench` first")

    ref = not a.no_ref
    if ref:
        try:
            import clarabel  # noqa: F401
        except ImportError:
            print("clarabel not importable: --no-ref implied", flush=True)
            ref = False
    only = set(a.only.split(","))
    if "family" in only:
        cases = [(f, n, 8, 0.5, 1) for f in ("soc", "exp", "mix")
                 for n in (10, 20, 40, 80)]
        table("family and blocks (d=8, rho=0.50, rep=1)", cases, a.runs, ref)
    if "density" in only:
        cases = [(f, 40, 12, r, 1) for f in ("soc", "exp", "mix")
                 for r in (0.25, 0.50, 1.00)]
        table("link density (n=40, d=12, rep=1)", cases, a.runs, ref)
    if "degen" in only:
        cases = [(f, 40, 8, 0.5, r) for f in ("exp", "mix") for r in (1, 2, 4)]
        table("degeneracy by replication (n=40, d=8, rho=0.50)", cases, a.runs, ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
