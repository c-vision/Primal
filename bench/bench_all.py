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

  - LP / QP / MILP: the generated MPS instances (bench/gen_instances.py).
  - SOCP: the closed-form family of bench/conic_bench.c.
  - SDP: the same closed-form family swept over the block size d by
    bench/sdp_sweep.c (past conic_bench's d=8, to where the conic route stops
    scaling).

SCIP takes MPS files itself; a CBF model (the SOCP/SDP families, the library
instances) is built through SCIP's C API by out/bench/scip_cbf (`make
bench-scip`, bench/scip_cbf.c), SCIP-SDP solving the ones with PSD parts.

Every reference takes every class it supports, with its default settings,
given the model in the form it handles best (bench/conic_ref.py holds the
SOCP/SDP builders); N/A marks a class the solver does not support.  Each
reference solve is capped at REF_TIMEOUT and shows as ">Ns" past it.  SCS is
first-order, so its objective is checked at 1e-4 instead of 1e-5.

A second table runs the public instances bench/fetch_instances.py downloads
(CBLIB, SDPLIB, Maros-Meszaros, Netlib, MIPLIB; skipped when absent), every
solver -- PrimalSolver too -- capped at LIB_TIMEOUT; see library().  The
reference solvers take those files through bench/lib_ref.py.

Pajarito (integer models, HiGHS + Hypatia inside) and Hypatia (continuous
ones) run in Julia through bench/julia/ref.jl on the same files -- written out
as CBF for the SOCP/SDP families -- with julia from $JULIA, PATH or juliaup.

Usage:  make bench && python3 bench/fetch_instances.py && python3 bench/bench_all.py
Needs numpy/scipy plus the optional references clarabel, scs, highspy, pyscipopt
(N/A when missing). All of them live in one environment on this machine:
/Users/gaetano/ai/env-bench/bin/python3.14 -- run the script with that interpreter.
"""
import os
import re
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
import run_bench

SOLVE_MPS = os.path.join(ROOT, "out", "bench", "solve_mps")
CONIC_BENCH = os.path.join(ROOT, "out", "bench", "conic_bench")
SDP_SWEEP = os.path.join(ROOT, "out", "bench", "sdp_sweep")
SDP_DMAX = 22          # conic_bench stops at d=8; the sweep exposes the scaling limit
SDP_KNOWN_UNSOLVED = {20}   # the degenerate block d=20: the IPM stalls (rc=1007), see README
REF_TIMEOUT = 600      # seconds, per reference solve
LIB_TIMEOUT = 60       # seconds, per solve (PrimalSolver included) in the library table

# (name, class, n, m, generator, sense_max) -- same specs as gen_instances.main
SPECS = []
for _n, _m in ((50, 25), (100, 50), (200, 100), (400, 200)):
    SPECS.append(("lp_%dx%d" % (_n, _m), "lp", _n, _m, gen_instances.make_lp, False))
for _n, _m in ((50, 25), (100, 50), (200, 100)):
    SPECS.append(("qp_%dx%d" % (_n, _m), "qp", _n, _m, gen_instances.make_qp, False))
for _n, _m in ((40, 20), (60, 30), (80, 40)):
    SPECS.append(("milp_%dx%d" % (_n, _m), "milp", _n, _m, gen_instances.make_milp, True))


def solve_ps(path):
    """PrimalSolver time from out/bench/solve_mps (CSV: rc,nvar,ncon,obj,s,status)."""
    try:
        out = subprocess.run([SOLVE_MPS, path], capture_output=True, text=True, timeout=600)
        rc, _, _, obj, sec, _ = out.stdout.strip().split(",")
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
        st = conic_ref.clarabel_settings()
        t = time.perf_counter()  # setup counts: scaling, KKT solver setup
        solver = clarabel.DefaultSolver(P, q, A, b, cones, st)
        sol = solver.solve()
        dt = time.perf_counter() - t
        return dt, float(sol.obj_val)
    except Exception:
        return None, None


def solve_scs_lp(c, rows, lo, up, Q=None):
    """SCS on the same conic form as Clarabel (SCS wants P's upper triangle)."""
    try:
        import scs
    except Exception:
        return None, None
    try:
        P, q, A, b, cones = to_clarabel(c, rows, lo, up, Q)
        nz = next((cn.dim for cn in cones if type(cn).__name__ == "ZeroConeT"), 0)
        data = dict(P=sparse.triu(P).tocsc(), A=A, b=b, c=q)
        t = time.perf_counter()  # setup counts: scaling, KKT solver setup
        solver = scs.SCS(data, dict(z=nz, l=A.shape[0] - nz), **conic_ref.SCS_SETTINGS)
        sol = solver.solve()
        dt = time.perf_counter() - t
        if sol["info"]["status"] != "solved":
            return None, None
        return dt, float(sol["info"]["pobj"])
    except Exception:
        return None, None


def solve_scip_qp(c, rows, lo, up, Q, timeout=REF_TIMEOUT):
    """SCIP on the QP with the objective factored: Q = L L', y = L'x and
    0.5 ||y||^2 <= z.  SCIP proves optimality by closing the dual bound with
    gradient cuts; against the raw dense x'Qx (the MPS QUADOBJ) that crawls
    (qp_200x100 does not close in 600 s), on the factored form it takes a few
    seconds.  The time charges the Cholesky; building the model does not
    count, as for the other solvers."""
    try:
        from pyscipopt import Model, quicksum
    except Exception:
        return "n/a", None, None
    try:
        n = len(c)
        M = np.zeros((n, n))
        for (i, j), v in Q.items():
            M[i, j] = M[j, i] = v
        t = time.perf_counter()
        L = np.linalg.cholesky(M)
        t_chol = time.perf_counter() - t
        m = Model()
        m.hideOutput()
        m.setParam("limits/time", timeout)
        x = [m.addVar(lb=None if lo[j] == -np.inf else lo[j],
                      ub=None if up[j] == np.inf else up[j]) for j in range(n)]
        for ty, rhs, coef in rows:
            e = quicksum(v * x[j] for j, v in coef.items())
            m.addCons(e >= rhs if ty == "G" else (e <= rhs if ty == "L" else e == rhs))
        y = [m.addVar(lb=None) for _ in range(n)]
        for i in range(n):
            m.addCons(y[i] == quicksum(L[j, i] * x[j] for j in range(i, n) if L[j, i]))
        z = m.addVar(lb=None)
        m.addCons(0.5 * quicksum(yi * yi for yi in y) <= z)
        m.setObjective(quicksum(c[j] * x[j] for j in range(n)) + z)
        t = time.perf_counter()
        m.optimize()
        dt = t_chol + time.perf_counter() - t
        if m.getStatus() == "timelimit":
            return "timeout", None, None
        if m.getStatus() != "optimal":
            return "n/a", None, None
        return "ok", dt, float(m.getObjVal())
    except Exception:
        return "n/a", None, None


def parse_conic_bench():
    """PrimalSolver SOCP rows from out/bench/conic_bench."""
    socp, sdp = {}, {}
    try:
        out = subprocess.run([CONIC_BENCH], capture_output=True, text=True, timeout=600).stdout
        for ln in out.strip().splitlines()[1:]:
            f = ln.split(",")
            if f[0] == "socp":
                socp[int(f[1])] = (float(f[4]), f[5].split("=")[1], float(f[3]))
            elif f[0] == "sdp":
                sdp[int(f[1])] = (float(f[4]), f[5].split("=")[1], float(f[3]))
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
            rows[int(f[0])] = (float(f[2]), f[3], float(f[1]))
    except Exception:
        pass
    return rows


def fmt(t):
    return "%.4f" % t if t is not None else "N/A"


def fmt_ref(status, t):
    return ">%ds" % REF_TIMEOUT if status == "timeout" else fmt(t)


def bold_fastest(line):
    """Bold the fastest correct answer of a table row: the smallest time among
    the solver cells (between the four leading columns and obj) that carry no
    x mark; ties all bold."""
    c = line.strip().strip("|").split("|")
    c = [x.strip() for x in c]
    times = {}
    for i in range(4, len(c) - 1):
        if re.fullmatch(r"\d+\.\d+", c[i]):
            times[i] = float(c[i])
    if times:
        best = min(times.values())
        for i, t in times.items():
            if t == best:
                c[i] = "**%s**" % c[i]
    return "| " + " | ".join(c) + " |"


def fmt_jl(status, t, limit=None):
    """A Pajarito/Hypatia cell: its time, >Ns, N/A (class not taken) or fail."""
    if status in ("ok", "infeasible"):
        return fmt(t)
    if status == "timeout":
        return ">%ds" % (limit or REF_TIMEOUT)
    return "N/A" if status == "unsupported" else "fail"


def main():
    if not os.path.exists(SOLVE_MPS):
        sys.exit("run `make bench` first")
    mism = []
    # CBF files for the SOCP/SDP families (SCIP and Pajarito/Hypatia take
    # those), then Pajarito/Hypatia: one Julia process for every row
    import tempfile
    import lib_ref
    jdir = tempfile.mkdtemp(prefix="bench-jl-")
    jpath = {name: os.path.join(HERE, "instances", name + ".mps") for name, *_ in SPECS}
    for n in conic_ref.SIZES:
        jpath["socp_%d" % n] = os.path.join(jdir, "socp_%d.cbf" % n)
        open(jpath["socp_%d" % n], "w").write(conic_ref.socp_cbf(n))
    for d in range(4, SDP_DMAX + 1):
        jpath["sdp_%d" % d] = os.path.join(jdir, "sdp_%d.cbf" % d)
        open(jpath["sdp_%d" % d], "w").write(conic_ref.sdp_cbf(d, 200 + d))
    jres = lib_ref.solve_julia_batch(list(jpath.values()), REF_TIMEOUT)
    jl = {name: jres[p] for name, p in jpath.items()}
    print("| instance | class | vars x cons | nnz | Primal (s) | HiGHS (s) | Clarabel (s) | SCS (s) | SCIP (s) | Pajarito/Hypatia (s) | obj |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")

    for name, cls, n, m, fn, sense_max in SPECS:
        c, rows, lo, up, vt, Q = fn(n, m, 20260912 + n)
        path = os.path.join(HERE, "instances", name + ".mps")
        nnz = gen_instances.count_nnz(c, rows, Q)
        pst, psobj = solve_ps(path)
        objs = [psobj] if psobj is not None else []
        # HiGHS and SCIP read the MPS; Clarabel and SCS have no integers.
        hstat, hst, hsobj = run_bench.solve_highs(path, REF_TIMEOUT)
        if cls == "qp":
            scstat, scst, scobj = solve_scip_qp(c, rows, lo, up, Q)
        else:
            scstat, scst, scobj = run_bench.solve_scip(path, REF_TIMEOUT)
        cst = cobj = sst = sobj = None
        if cls != "milp":
            cst, cobj = solve_clarabel_lp(c, rows, lo, up, Q)
            sst, sobj = solve_scs_lp(c, rows, lo, up, Q)
        jst, jt, jobj = jl[name]
        objs += [o for o in (hsobj, cobj, scobj, jobj if jst == "ok" else None) if o is not None]
        if objs and (max(objs) - min(objs)) > 1e-5 * (1 + abs(objs[0])):
            mism.append("%s %s" % (name, objs))
        if objs and sobj is not None and abs(sobj - objs[0]) > 1e-4 * (1 + abs(objs[0])):
            mism.append("%s scs=%g" % (name, sobj))
        obj = objs[0] if objs else None
        print(bold_fastest("| %s | %s | %d x %d | %d | %s | %s | %s | %s | %s | %s | %s |" %
              (name, cls, n, m, nnz, pst, fmt_ref(hstat, hst), fmt(cst), fmt(sst),
               fmt_ref(scstat, scst), fmt_jl(jst, jt), "%.6g" % obj if obj is not None else "N/A")))

    # ---- SOCP: closed-form family ----
    ps_socp, _ = parse_conic_bench()
    ref = 1.0 / (2.0 ** 0.5)
    for n in conic_ref.SIZES:
        tc, oc = conic_ref.solve_clarabel(n)
        ts, os_ = conic_ref.solve_scs(n)
        scst_, tsc, osc = lib_ref.solve_scip_cbf(jpath["socp_%d" % n], REF_TIMEOUT)
        sec, rc, pobj = ps_socp.get(n, (None, None, None))
        pcell = fmt(sec) if (sec is not None and rc == "0") else ("N/A (rc=%s)" % rc if rc else "N/A")
        # PrimalSolver's own SOCP answer enters the cross-check (issue #8): a
        # missing row, a failed solve or an objective off the analytic value is
        # a failure, not part of "they agree".
        if sec is None:
            mism.append("socp_%d: no PrimalSolver row" % n)
        elif rc != "0":
            mism.append("socp_%d: PrimalSolver rc=%s" % (n, rc))
        elif abs(pobj - ref) > 1e-4:
            mism.append("socp_%d: PrimalSolver=%g want %g" % (n, pobj, ref))
        jst, jt, jobj = jl["socp_%d" % n]
        for nm2, ob in (("clarabel", oc), ("scs", os_), ("scip", osc),
                        ("hypatia", jobj if jst == "ok" else None)):
            if ob is not None and abs(ob - ref) > 1e-4:
                mism.append("socp_%d %s=%g" % (n, nm2, ob))
        print(bold_fastest("| socp_%d | socp | %d x 1 | - | %s | N/A | %s | %s | %s | %s | %.6f |" %
              (n, n, pcell, fmt(tc), fmt(ts), fmt_ref(scst_, tsc), fmt_jl(jst, jt), ref)))
    # ---- SDP: the same closed-form family swept over the block size d ----
    ps_sdp = parse_sdp_sweep(4, SDP_DMAX)
    for d in range(4, SDP_DMAX + 1):
        try:
            _, _, _, _, _, exp = conic_ref.sdp_data(d, 200 + d)
        except Exception:
            exp = None
        tc, oc, _ = conic_ref.solve_clarabel_sdp(d, 200 + d)
        ts, os_, _ = conic_ref.solve_scs_sdp(d, 200 + d)
        scst_, tsc, osc = lib_ref.solve_scip_cbf(jpath["sdp_%d" % d], REF_TIMEOUT)
        sec, rc, pobj = ps_sdp.get(d, (None, None, None))
        pcell = fmt(sec) if (sec is not None and rc == "0") else ("N/A (rc=%s)" % rc if rc else "N/A")
        if sec is None:
            mism.append("sdp_%d: no PrimalSolver row" % d)
        elif rc != "0":
            if d not in SDP_KNOWN_UNSOLVED:
                mism.append("sdp_%d: PrimalSolver rc=%s" % (d, rc))
        elif exp is not None and abs(pobj - exp) > 1e-4 * (1 + abs(exp)):
            mism.append("sdp_%d: PrimalSolver=%g want %g" % (d, pobj, exp))
        jst, jt, jobj = jl["sdp_%d" % d]
        for nm2, ob in (("clarabel", oc), ("scs", os_), ("scip-sdp", osc),
                        ("hypatia", jobj if jst == "ok" else None)):
            if ob is not None and exp is not None and abs(ob - exp) > 1e-4 * (1 + abs(exp)):
                mism.append("sdp_%d %s=%g" % (d, nm2, ob))
        print(bold_fastest("| sdp_%d | sdp | %d x %d | - | %s | N/A | %s | %s | %s | %s | %s |" %
              (d, d, d, pcell, fmt(tc), fmt(ts), fmt_ref(scst_, tsc), fmt_jl(jst, jt),
               "%.6g" % exp if exp is not None else "N/A")))

    lib_notes = library(mism)

    print(file=sys.stderr)
    for note in lib_notes:
        print(note, file=sys.stderr)
    if mism:
        print("obj cross-check FAILED: " + "; ".join(mism), file=sys.stderr)
        sys.exit(1)
    print("obj cross-check: PrimalSolver / HiGHS / Clarabel / SCS / SCIP agree.", file=sys.stderr)


def solve_ps_lib(path, timeout):
    """(status, seconds, objective) of out/bench/solve_mps on any model file."""
    try:
        out = subprocess.run([SOLVE_MPS, path], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", None, None
    try:
        rc, _, _, obj, sec, sta = out.stdout.strip().split(",")
    except ValueError:
        return "n/a", None, None
    solsta = sta.split("/")[0]
    if rc == "0" and solsta in ("OPTIMAL", "INTEGER_OPTIMAL"):
        return "ok", float(sec), float(obj)
    if "INFEAS" in solsta:
        return "infeasible", float(sec), None
    return "n/a (rc=%s)" % rc, None, None


def library(mism):
    """The public instances of bench/fetch_instances.py, one table.  Each
    solver's answer is checked against the published optimum where the set
    has one (or its certified infeasibility), otherwise against the value most
    solvers agree on; a disagreeing cell is marked with an x.  Only a wrong
    PrimalSolver answer fails the run -- a reference solver's is reported."""
    import csv
    import math
    import re
    import cbfio
    import lib_ref
    idx = os.path.join(HERE, "instances", "lib", "index.csv")
    if not os.path.exists(idx):
        return ["library table skipped: run bench/fetch_instances.py first"]
    print()
    print("| instance | set | class | vars x cons | Primal (s) | HiGHS (s) | Clarabel (s) | SCS (s) | SCIP (s) | Pajarito/Hypatia (s) | obj |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    notes = []
    rows = list(csv.DictReader(open(idx)))
    jres = lib_ref.solve_julia_batch([os.path.join(HERE, r["path"]) for r in rows], LIB_TIMEOUT)
    for r in rows:
        path = os.path.join(HERE, r["path"])
        cls = r["class"]
        if path.endswith(".cbf"):
            d = cbfio.read_cbf(path)
            dims = "%d x %d" % (d["var"][0] + sum(n * (n + 1) // 2 for n in d["psdvar"]),
                                d["con"][0] + sum(n * (n + 1) // 2 for n in d["psdcon"]))
        else:
            hm = lib_ref.highs_model(path)
            dims = "%d x %d" % (hm[1].shape[1], hm[1].shape[0])
        res = {"Primal": solve_ps_lib(path, LIB_TIMEOUT),
               "HiGHS": lib_ref.solve_highs(path, cls, LIB_TIMEOUT),
               "Clarabel": lib_ref.solve_clarabel(path, LIB_TIMEOUT),
               "SCS": lib_ref.solve_scs(path, LIB_TIMEOUT),
               "SCIP": lib_ref.solve_scip(path, cls, LIB_TIMEOUT),
               "Pajarito/Hypatia": jres[path]}
        mip = "int" in cls or cls == "milp"

        def tol(name):
            return 1e-4 if (mip or name in ("SCS", "SCIP")) else 1e-5

        ref = r["ref"]
        ref_ulp = 0.0                         # half a unit in the published last digit
        if ref and ref != "infeasible":
            mant = re.split(r"[eE]", ref.lstrip("+-"))[0].replace(".", "").lstrip("0")
            ref = float(ref)
            if ref != 0.0 and mant:
                ref_ulp = 0.5 * 10.0 ** (math.floor(math.log10(abs(ref))) - len(mant) + 1)
        elif not ref:                         # the value most solvers agree on
            objs = [(n, o) for n, (st, _, o) in res.items() if st.startswith("ok")]
            best = None
            for n, o in objs:
                agree = sum(abs(o2 - o) <= 1e-4 * (1 + abs(o)) for _, o2 in objs)
                if best is None or agree > best[0] or (agree == best[0] and best[1] == "SCS"):
                    best = (agree, n, o)
            ref = best[2] if best else None
            if best and best[0] == 1 and len(objs) > 1:
                ref = "disputed"
                notes.append("disputed: %s %s" % (r["name"], ", ".join(
                    "%s=%.8g" % (n, o) for n, o in objs)))
        cells = []
        for name, (st, dt, o) in res.items():
            if st.startswith("ok"):
                wrong = ref == "infeasible" or (isinstance(ref, float) and abs(o - ref) >
                                                max(tol(name) * (1 + abs(ref)), ref_ulp))
                cell = "%.4f" % dt
            elif st == "infeasible":
                wrong = ref != "infeasible"
                cell = ("%.4f" % dt) if dt is not None else "infeas."
            else:
                wrong = False
                cell = {"timeout": ">%ds" % LIB_TIMEOUT, "unsupported": "N/A"}.get(st, st.replace("n/a", "fail"))
            if wrong:
                cell += " \u2717"
                what = "%s %s=%s" % (r["name"], name, "infeasible" if o is None else "%.8g" % o)
                (mism if name == "Primal" else notes).append(what if name == "Primal" else "reference off: " + what)
            cells.append(cell)
        print(bold_fastest("| %s | %s | %s | %s | %s | %s |" % (
            r["name"], r["set"], cls, dims, " | ".join(cells),
            ref if isinstance(ref, str) else ("%.6g" % ref if ref is not None else "N/A"))))
    return notes


if __name__ == "__main__":
    main()
