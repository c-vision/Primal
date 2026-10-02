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
# Estratto della licenza (Apache License 2.0, §2 "Grant of Copyright License"):
#   "Subject to the terms and conditions of this License, each Contributor
#    hereby grants to You a perpetual, worldwide, non-exclusive, no-charge,
#    royalty-free, irrevocable copyright license to reproduce, prepare
#    Derivative Works of, publicly display, publicly perform, sublicense, and
#    distribute the Work and such Derivative Works in Source or Object form."
# 
# Esonero di responsabilita' e assenza di garanzia (Apache License 2.0, §7-§8):
#   [§7] Unless required by applicable law or agreed to in writing, Licensor
#   provides the Work (and each Contributor provides its Contributions) on an
#   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#   or implied, including, without limitation, any warranties or conditions of
#   TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR
#   PURPOSE.  You are solely responsible for determining the appropriateness of
#   using or redistributing the Work.
#   [§8] In no event and under no legal theory, whether in tort (including
#   negligence), contract, or otherwise, unless required by applicable law or
#   agreed to in writing, shall any Contributor be liable to You for damages,
#   including any direct, indirect, special, incidental, or consequential
#   damages arising as a result of this License or out of the use or inability
#   to use the Work.  This software is provided without any guarantee that it
#   will operate correctly or be free of defects.

"""Run the PrimalSolver benchmark and print a markdown table.

PrimalSolver is driven by out/bench/solve_mps (built with `make bench`);
HiGHS via highspy and SCIP via pyscipopt (optional: missing solvers show N/A).
Each solver's time covers the optimization call only (model read excluded),
single run.  Objectives are cross-checked and reported if they disagree.

Usage:  python3 bench/run_bench.py [--timeout SEC] [--instances DIR]
"""
import argparse
import csv
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def solve_primal(path, timeout):
    exe = os.path.join(ROOT, "out", "bench", "solve_mps")
    if not os.path.exists(exe):
        return "not-built", None, None
    try:
        out = subprocess.run([exe, path], capture_output=True, text=True,
                             timeout=timeout).stdout.strip()
    except subprocess.TimeoutExpired:
        return "timeout", None, None
    p = out.split(",")
    if len(p) != 6 or p[0] == "READ_ERROR":
        return "error", None, None
    return "ok", float(p[4]), float(p[3])


def solve_highs(path, timeout):
    try:
        import highspy
    except Exception:
        return solve_highs_scipy(path, timeout)
    try:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("time_limit", timeout)
        h.readModel(path)
        t = time.perf_counter()
        h.run()
        dt = time.perf_counter() - t
        st = h.getModelStatus()
        if st == highspy.HighsModelStatus.kTimeLimit:
            return "timeout", None, None
        if st == highspy.HighsModelStatus.kInfeasible:
            return "infeasible", dt, None
        if st != highspy.HighsModelStatus.kOptimal:
            return "n/a", None, None
        return "ok", dt, h.getObjectiveValue()
    except Exception:
        return "n/a", None, None


def solve_highs_scipy(path, timeout):
    """HiGHS through SciPy (linprog for LP, milp for MIP; both wrap HiGHS).

    Uses the bundled HiGHS when highspy is not installed.  The MPS is read by
    bench/mps_reader.py -- objectives are cross-checked in main(), which guards
    the reader."""
    try:
        import numpy as np
        from scipy.optimize import linprog, milp, LinearConstraint, Bounds
        from scipy.sparse import lil_matrix
        sys.path.insert(0, HERE)
        import mps_reader
    except Exception:
        return "n/a", None, None
    try:
        m = mps_reader.read_mps(path)
        n = m["nvar"]
        sgn = 1.0 if m["sense"] == "MIN" else -1.0
        c = sgn * np.array(m["c"], dtype=float)
        nrows = len(m["rows"])
        A = lil_matrix((nrows, n))
        lo_c = np.empty(nrows)
        up_c = np.empty(nrows)
        for i, (ty, rhs, coef) in enumerate(m["rows"]):
            for j, v in coef.items():
                A[i, j] = v
            if ty == "L":
                lo_c[i], up_c[i] = -np.inf, rhs
            elif ty == "G":
                lo_c[i], up_c[i] = rhs, np.inf
            else:
                lo_c[i] = up_c[i] = rhs
        t = time.perf_counter()
        if any(m["integ"]):
            res = milp(c=c, constraints=[LinearConstraint(A.tocsr(), lo_c, up_c)],
                       integrality=np.array(m["integ"]),
                       bounds=Bounds(np.array(m["lo"]),
                                     [u if u != np.inf else 1e30 for u in m["up"]]),
                       options={"time_limit": timeout})
            fun = res.fun
        else:
            from scipy.sparse import vstack
            Ac = A.tocsr()
            nb = [None if v == -np.inf else v for v in m["lo"]]
            ub = [None if v == np.inf else v for v in m["up"]]
            eq_rows = [i for i in range(nrows) if lo_c[i] == up_c[i]]
            Aub, bub = [], []
            for i in range(nrows):
                if lo_c[i] == up_c[i]:
                    continue
                if up_c[i] != np.inf:               # 'L' row: A x <= rhs
                    Aub.append(Ac[i]); bub.append(up_c[i])
                else:                                # 'G' row: -A x <= -rhs
                    Aub.append(-Ac[i]); bub.append(-lo_c[i])
            res = linprog(c, A_ub=vstack(Aub).tocsr() if Aub else None,
                          b_ub=np.array(bub) if bub else None,
                          A_eq=Ac[eq_rows] if eq_rows else None,
                          b_eq=lo_c[eq_rows] if eq_rows else None,
                          bounds=list(zip(nb, ub)), method="highs",
                          options={"time_limit": timeout})
            fun = res.fun
        dt = time.perf_counter() - t
        if fun is None or not np.isfinite(fun):
            return "n/a", None, None
        return "ok", dt, sgn * float(fun)
    except Exception:
        return "n/a", None, None


def solve_scip(path, timeout):
    try:
        from pyscipopt import Model
    except Exception:
        return "n/a", None, None
    try:
        m = Model()
        m.hideOutput()
        m.setParam("limits/time", timeout)
        m.readProblem(path)
        t = time.perf_counter()
        m.optimize()
        dt = time.perf_counter() - t
        # a time limit leaves an incumbent behind: not a solve to report
        if m.getStatus() == "timelimit":
            return "timeout", None, None
        if m.getStatus() == "infeasible":
            return "infeasible", dt, None
        if m.getStatus() != "optimal":
            return "n/a", None, None
        return "ok", dt, m.getObjVal()
    except Exception:
        return "n/a", None, None


def cell(status, t, timeout):
    if status != "ok":
        return {"n/a": "N/A", "timeout": ">%ds" % int(timeout), "not-built": "—",
                "error": "err"}.get(status, "—")
    return "%.3f" % t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--instances", default=os.path.join(HERE, "instances"))
    args = ap.parse_args()
    idx = os.path.join(args.instances, "index.csv")
    if not os.path.exists(idx):
        sys.exit("run bench/gen_instances.py first")
    rows = list(csv.DictReader(open(idx)))
    print("| instance | vars | cons | nnz | PrimalSolver (s) | HiGHS (s) | SCIP (s) |")
    print("|---|---|---|---|---|---|---|")
    warn = 0
    for r in rows:
        path = os.path.join(args.instances, r["name"] + ".mps")
        sp, tp, op = solve_primal(path, args.timeout)
        sh, th, oh = solve_highs(path, args.timeout)
        ss, ts, os_ = solve_scip(path, args.timeout)
        # objective cross-check (where both available)
        objs = [(x, o) for x, o in (("P", op), ("H", oh), ("S", os_))
                if o is not None]
        if len(objs) >= 2:
            lo = min(o for _, o in objs)
            hi = max(o for _, o in objs)
            if hi - lo > 1e-4 * (1 + abs(lo)):
                warn += 1
                print("<!-- objective mismatch %s: %s -->" % (r["name"], objs))
        print("| %s | %s | %s | %s | %s | %s | %s |" %
              (r["name"], r["vars"], r["cons"], r["nnz"],
               cell(sp, tp, args.timeout), cell(sh, th, args.timeout),
               cell(ss, ts, args.timeout)))
    if warn:
        print("\nWARNING: %d objective mismatches" % warn, file=sys.stderr)


if __name__ == "__main__":
    main()
