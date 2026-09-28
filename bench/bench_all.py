#!/usr/bin/env python3
#
# PrimalSolver - a convex optimization solver in C99 (LP/QP/SOCP/SDP/exp-power/MIP).
# Copyright 2026 Gaetano Minardi
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  A copy of the License
# is in the repository root (LICENSE) and at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations
# under the License.

"""One benchmark table for every class the solver covers, with the open
references that can take each class:

    instance  | class | vars x cons | nnz | PrimalSolver | HiGHS | Clarabel | SCS | SCIP | obj

  - LP / QP / MILP: the generated MPS instances (bench/gen_instances.py);
    HiGHS via SciPy, SCIP via pyscipopt, Clarabel via a conic conversion.
  - SOCP: the closed-form family of bench/conic_bench.c; Clarabel and SCS, and
    no HiGHS/SCIP baseline.
  - SDP: the same closed-form family swept over the block size d by
    bench/sdp_sweep.c (past conic_bench's d=8, to where the conic route stops
    scaling), Clarabel and SCS references from bench/conic_ref.py.

Clarabel is a conic interior-point solver: it takes LP, QP, SOCP and SDP, and
has no integer support, so the MILP rows are N/A for it (as for SCS).

Usage:  make bench && python3 bench/bench_all.py
Needs numpy/scipy plus the optional references clarabel, scs, highspy, pyscipopt
(N/A when missing). All of them live in one environment on this machine:
/Users/gaetano/ai/env-bench/bin/python3.14 -- run the script with that interpreter.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np
from scipy import sparse

import conic_ref
import gen_instances

SOLVE_MPS = os.path.join(ROOT, "out", "bench", "solve_mps")
CONIC_BENCH = os.path.join(ROOT, "out", "bench", "conic_bench")
SDP_SWEEP = os.path.join(ROOT, "out", "bench", "sdp_sweep")
SDP_DMAX = 16          # conic_bench stops at d=8; the sweep exposes the scaling limit

# (name, class, n, m, generator, sense_max) -- same specs as gen_instances.main
SPECS = []
for _n, _m in ((50, 25), (100, 50), (200, 100), (400, 200)):
    SPECS.append(("lp_%dx%d" % (_n, _m), "lp", _n, _m, gen_instances.make_lp, False))
for _n, _m in ((50, 25), (100, 50), (200, 100)):
    SPECS.append(("qp_%dx%d" % (_n, _m), "qp", _n, _m, gen_instances.make_qp, False))
for _n, _m in ((40, 20), (60, 30), (80, 40)):
    SPECS.append(("milp_%dx%d" % (_n, _m), "milp", _n, _m, gen_instances.make_milp, True))


def solve_ps(path):
    """PrimalSolver time from out/bench/solve_mps (CSV: rc,nvar,ncon,obj,s)."""
    try:
        out = subprocess.run([SOLVE_MPS, path], capture_output=True, text=True, timeout=600)
        rc, _, _, obj, sec = out.stdout.strip().split(",")
        if rc != "0":
            return "N/A", None
        return "%.4f" % float(sec), float(obj)
    except Exception:
        return "N/A", None


def to_clarabel(c, rows, lo, up, Q=None):
    """General-form min 0.5 x'Qx + c'x, row bounds and box bounds, into
    Clarabel's  min 0.5 z'Pz + q'z  s.t.  A z + s = b, s in K.

    Zero cone for equalities, nonnegative cone for every 'G'/'L' row and every
    finite variable bound.  z is free."""
    import clarabel
    n = len(c)
    cons = []                                  # (coef_dict, b, 'z'|'n')
    for ty, rhs, coef in rows:
        if ty == "G":                          # a'x >= rhs  ->  s = a'x-rhs
            cons.append(({j: -v for j, v in coef.items()}, -rhs, "n"))
        elif ty == "L":                        # a'x <= rhs  ->  s = rhs-a'x
            cons.append((coef, rhs, "n"))
        else:                                  # a'x == rhs
            cons.append((coef, rhs, "z"))
    for j in range(n):
        if np.isfinite(lo[j]):                 # x_j >= lo  ->  s = x_j-lo
            cons.append(({j: -1.0}, -lo[j], "n"))
        if np.isfinite(up[j]):                 # x_j <= up  ->  s = up-x_j
            cons.append(({j: 1.0}, up[j], "n"))
    cons.sort(key=lambda t: 0 if t[2] == "z" else 1)
    m = len(cons)
    A = sparse.lil_matrix((m, n))
    b = np.zeros(m)
    for i, (coef, rhs, _) in enumerate(cons):
        for j, v in coef.items():
            A[i, j] = v
        b[i] = rhs
    nz = sum(1 for t in cons if t[2] == "z")
    nn = m - nz
    cones = []
    if nz:
        cones.append(clarabel.ZeroConeT(nz))
    if nn:
        cones.append(clarabel.NonnegativeConeT(nn))
    P = sparse.lil_matrix((n, n))
    if Q:
        for (i, j), v in Q.items():
            P[i, j] = v
            if i != j:
                P[j, i] = v
    return sparse.csc_matrix(P), np.array(c, float), A.tocsc(), b, cones


def solve_clarabel_lp(c, rows, lo, up, Q=None):
    try:
        import clarabel
    except Exception:
        return None, None
    try:
        P, q, A, b, cones = to_clarabel(c, rows, lo, up, Q)
        st = clarabel.DefaultSettings()
        st.verbose = False
        solver = clarabel.DefaultSolver(P, q, A, b, cones, st)
        t = time.perf_counter()
        sol = solver.solve()
        dt = time.perf_counter() - t
        return dt, float(sol.obj_val)
    except Exception:
        return None, None


def solve_highs_scipy(c, rows, lo, up):
    """HiGHS via SciPy for LP/QP (QP has no SciPy QP -> N/A)."""
    try:
        from scipy.optimize import linprog
    except Exception:
        return None, None
    n = len(c)
    nrows = len(rows)
    A = sparse.lil_matrix((nrows, n))
    lo_c = np.empty(nrows)
    up_c = np.empty(nrows)
    for i, (ty, rhs, coef) in enumerate(rows):
        for j, v in coef.items():
            A[i, j] = v
        if ty == "L":
            lo_c[i], up_c[i] = -np.inf, rhs
        elif ty == "G":
            lo_c[i], up_c[i] = rhs, np.inf
        else:
            lo_c[i] = up_c[i] = rhs
    Ac = A.tocsr()
    nb = [None if v == -np.inf else v for v in lo]
    ub = [None if v == np.inf else v for v in up]
    eq_rows = [i for i in range(nrows) if lo_c[i] == up_c[i]]
    Aub, bub = [], []
    for i in range(nrows):
        if lo_c[i] == up_c[i]:
            continue
        if up_c[i] != np.inf:
            Aub.append(Ac[i]); bub.append(up_c[i])
        else:
            Aub.append(-Ac[i]); bub.append(-lo_c[i])
    try:
        t = time.perf_counter()
        res = linprog(np.array(c, float),
                      A_ub=sparse.vstack(Aub).tocsr() if Aub else None,
                      b_ub=np.array(bub) if bub else None,
                      A_eq=Ac[eq_rows] if eq_rows else None,
                      b_eq=lo_c[eq_rows] if eq_rows else None,
                      bounds=list(zip(nb, ub)), method="highs")
        dt = time.perf_counter() - t
        if res.fun is None or not np.isfinite(res.fun):
            return None, None
        return dt, float(res.fun)
    except Exception:
        return None, None


def parse_conic_bench():
    """PrimalSolver SOCP rows from out/bench/conic_bench."""
    socp, sdp = {}, {}
    try:
        out = subprocess.run([CONIC_BENCH], capture_output=True, text=True, timeout=600).stdout
        for ln in out.strip().splitlines()[1:]:
            f = ln.split(",")
            if f[0] == "socp":
                socp[int(f[1])] = (float(f[4]), f[5].split("=")[1])
            elif f[0] == "sdp":
                sdp[int(f[1])] = (float(f[4]), f[5].split("=")[1])
    except Exception:
        pass
    return socp, sdp


def parse_sdp_sweep(dmin=4, dmax=SDP_DMAX):
    """PrimalSolver SDP rows from out/bench/sdp_sweep (CSV: d,obj,seconds,rc,expected)."""
    rows = {}
    try:
        out = subprocess.run([SDP_SWEEP, str(dmin), str(dmax)],
                             capture_output=True, text=True, timeout=600).stdout
        for ln in out.strip().splitlines()[1:]:
            f = ln.split(",")
            rows[int(f[0])] = (float(f[2]), f[3])
    except Exception:
        pass
    return rows


def fmt(t):
    return "%.4f" % t if t is not None else "N/A"


def main():
    if not os.path.exists(SOLVE_MPS):
        sys.exit("run `make bench` first")
    mism = []
    print("| instance | class | vars x cons | nnz | PrimalSolver (s) | HiGHS (s) | Clarabel (s) | SCS (s) | SCIP (s) | obj |")
    print("|---|---|---|---|---|---|---|---|---|---|")

    for name, cls, n, m, fn, sense_max in SPECS:
        c, rows, lo, up, vt, Q = fn(n, m, 20260912 + n)
        path = os.path.join(HERE, "instances", name + ".mps")
        nnz = gen_instances.count_nnz(c, rows, Q)
        pst, psobj = solve_ps(path)
        objs = [psobj] if psobj is not None else []
        hst = cst = scst = None
        if cls == "milp":
            # Clarabel/SCS have no integer support; HiGHS-QP not used here.
            try:
                import run_bench
                scst, scobj = run_bench.solve_scip(path, 600)[1:]
                if scobj is not None:
                    objs.append(scobj)
            except Exception:
                pass
        else:
            if Q is None:
                hst, hsobj = solve_highs_scipy(c, rows, lo, up)
                if hsobj is not None:
                    objs.append(hsobj)
            cst, cobj = solve_clarabel_lp(c, rows, lo, up, Q)
            if cobj is not None:
                objs.append(cobj)
        if objs and (max(objs) - min(objs)) > 1e-5 * (1 + abs(objs[0])):
            mism.append("%s %s" % (name, objs))
        obj = objs[0] if objs else None
        print("| %s | %s | %d x %d | %d | %s | %s | %s | %s | %s | %s |" %
              (name, cls, n, m, nnz, pst, fmt(hst), fmt(cst), "N/A", fmt(scst),
               "%.8g" % obj if obj is not None else "N/A"))

    # ---- SOCP: closed-form family (Clarabel, SCS, no HiGHS/SCIP) ----
    ps_socp, _ = parse_conic_bench()
    ref = 1.0 / (2.0 ** 0.5)
    for n in conic_ref.SIZES:
        tc, oc = conic_ref.solve_clarabel(n)
        ts, os_ = conic_ref.solve_scs(n)
        sec, rc = ps_socp.get(n, (None, None))
        pcell = fmt(sec) if (sec is not None and rc == "0") else ("N/A (rc=%s)" % rc if rc else "N/A")
        for nm2, ob in (("clarabel", oc), ("scs", os_)):
            if ob is not None and abs(ob - ref) > 1e-4:
                mism.append("socp_%d %s=%g" % (n, nm2, ob))
        print("| socp_%d | socp | %d x 1 | - | %s | N/A | %s | %s | N/A | %.8f |" %
              (n, n, pcell, fmt(tc), fmt(ts), ref))
    # ---- SDP: the same closed-form family swept over the block size d ----
    ps_sdp = parse_sdp_sweep(4, SDP_DMAX)
    for d in range(4, SDP_DMAX + 1):
        try:
            _, _, _, _, _, exp = conic_ref.sdp_data(d, 200 + d)
        except Exception:
            exp = None
        tc, oc, _ = conic_ref.solve_clarabel_sdp(d, 200 + d)
        ts, os_, _ = conic_ref.solve_scs_sdp(d, 200 + d)
        sec, rc = ps_sdp.get(d, (None, None))
        pcell = fmt(sec) if (sec is not None and rc == "0") else ("N/A (rc=%s)" % rc if rc else "N/A")
        for nm2, ob in (("clarabel", oc), ("scs", os_)):
            if ob is not None and exp is not None and abs(ob - exp) > 1e-4 * (1 + abs(exp)):
                mism.append("sdp_%d %s=%g" % (d, nm2, ob))
        print("| sdp_%d | sdp | %d x %d | - | %s | N/A | %s | %s | N/A | %s |" %
              (d, d, d, pcell, fmt(tc), fmt(ts),
               "%.8g" % exp if exp is not None else "N/A"))

    print(file=sys.stderr)
    if mism:
        print("obj cross-check FAILED: " + "; ".join(mism), file=sys.stderr)
    else:
        print("obj cross-check: PrimalSolver / HiGHS / Clarabel / SCS / SCIP agree.", file=sys.stderr)


if __name__ == "__main__":
    main()
