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

"""Generate deterministic benchmark instances (MPS) for the PrimalSolver
benchmark.  All instances are reproducible from fixed seeds.

Usage:  python3 bench/gen_instances.py [--out DIR]

Classes: LP, QP, MILP (MPS files, readable by PrimalSolver/HiGHS/SCIP).
SOCP and SDP are generated directly in C (bench/conic_bench.c) because MPS
cannot express cones.
"""
import argparse
import os
import random

INF = 1e30


def write_mps(path, name, sense_max, c, rows, lo, up, vartype, Q=None):
    """rows: list of (type, rhs, {j: aij}); type in 'LEG'; vartype 'C'/'I'/'B'."""
    L = ["NAME " + name]
    if sense_max:
        L += ["OBJSENSE", "    MAX"]
    L.append("ROWS")
    L.append(" N obj")
    for i, (ty, rhs, _) in enumerate(rows):
        L.append(" %s c%d" % (ty, i))
    L.append("COLUMNS")
    nvar = len(c)
    for j in range(nvar):
        if vartype[j] != 'C':
            L.append("    MARKER                 'MARKER'                 'INTORG'")
        if c[j] != 0.0:
            L.append("    x%d        obj        %.17g" % (j, c[j]))
        for i, (_, _, coef) in enumerate(rows):
            if coef.get(j, 0.0) != 0.0:
                L.append("    x%d        c%d         %.17g" % (j, i, coef[j]))
        if vartype[j] != 'C':
            L.append("    MARKER                 'MARKER'                 'INTEND'")
    L.append("RHS")
    for i, (_, rhs, _) in enumerate(rows):
        L.append("    rhs       c%d         %.17g" % (i, rhs))
    L.append("BOUNDS")
    for j in range(nvar):
        if vartype[j] == 'B':
            L.append(" BV bnd       x%d" % j)
        elif lo[j] == up[j]:
            L.append(" FX bnd       x%d         %.17g" % (j, lo[j]))
        else:
            L.append(" LO bnd       x%d         %.17g" % (j, lo[j]))
            L.append(" UP bnd       x%d         %.17g" % (j, up[j]))
    if Q:
        # QUADOBJ (formato CPLEX) DOPO BOUNDS e SENZA nome di riga: HiGHS scarta
        # l'intero obiettivo (anche lineare) se la riga e' "QUADOBJ obj"; con il
        # solo "QUADOBJ" lo legge correttamente. La semantica e' 0.5 x'Qx. Il
        # nostro reader accetta QUADOBJ (senza nome) e QSECTION/QMATRIX (con nome).
        L.append("QUADOBJ")
        for (i, j), v in sorted(Q.items()):
            if v != 0.0:
                L.append("    x%d        x%d         %.17g" % (i, j, v))
    L.append("ENDATA")
    open(path, "w").write("\n".join(L) + "\n")


def count_nnz(c, rows, Q=None):
    nz = sum(1 for v in c if v != 0.0)
    for _, _, coef in rows:
        nz += sum(1 for v in coef.values() if v != 0.0)
    if Q:
        nz += sum(1 for v in Q.values() if v != 0.0)
    return nz


def make_lp(n, m, seed):
    rng = random.Random(seed)
    c = [rng.uniform(1.0, 10.0) for _ in range(n)]
    rows = []
    for i in range(m):
        coef = {}
        for _ in range(max(2, n // 10)):
            coef[rng.randrange(n)] = rng.uniform(0.1, 2.0)
        rhs = 0.4 * sum(coef.values())      # feasible at x = 1
        rows.append(('G', rhs, coef))
    lo = [0.0] * n
    up = [1.0] * n
    vt = ['C'] * n
    return c, rows, lo, up, vt, None


def make_qp(n, m, seed):
    rng = random.Random(seed)
    c = [rng.uniform(0.0, 1.0) for _ in range(n)]
    # Q = R'R + I  (PSD, dense upper triangle)
    R = [[rng.uniform(-1, 1) for _ in range(n)] for _ in range(max(2, n // 5))]
    Q = {}
    for i in range(n):
        for j in range(i, n):
            s = sum(R[k][i] * R[k][j] for k in range(len(R)))
            if i == j:
                s += 1.0
            if abs(s) > 1e-12:
                Q[(i, j)] = s
    rows = []
    for i in range(m):
        coef = {}
        for _ in range(max(2, n // 10)):
            coef[rng.randrange(n)] = rng.uniform(0.1, 2.0)
        rhs = 0.4 * sum(coef.values())
        rows.append(('G', rhs, coef))
    lo = [0.0] * n
    up = [1.0] * n
    vt = ['C'] * n
    return c, rows, lo, up, vt, Q


def make_milp(n, m, seed):
    """max c'x s.t. A x <= b, first half binary, rest continuous in [0,1]."""
    rng = random.Random(seed)
    c = [rng.uniform(1.0, 10.0) for _ in range(n)]
    rows = []
    for i in range(m):
        coef = {}
        for _ in range(max(2, n // 8)):
            coef[rng.randrange(n)] = rng.uniform(0.5, 3.0)
        rhs = 0.5 * sum(coef.values())      # feasible at x = 1, nontrivial
        rows.append(('L', rhs, coef))
    lo = [0.0] * n
    up = [1.0] * n
    vt = ['B' if j < n // 2 else 'C' for j in range(n)]
    return c, rows, lo, up, vt, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "instances"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    specs = []
    for n, m in ((50, 25), (100, 50), (200, 100), (400, 200)):
        specs.append(("lp", n, m, make_lp, False))
    for n, m in ((50, 25), (100, 50), (200, 100)):
        specs.append(("qp", n, m, make_qp, False))
    for n, m in ((40, 20), (60, 30), (80, 40)):
        specs.append(("milp", n, m, make_milp, True))
    index = []
    for cls, n, m, fn, sense_max in specs:
        c, rows, lo, up, vt, Q = fn(n, m, 20260912 + n)
        name = "%s_%dx%d" % (cls, n, m)
        path = os.path.join(args.out, name + ".mps")
        write_mps(path, name, sense_max, c, rows, lo, up, vt, Q)
        nnz = count_nnz(c, rows, Q)
        index.append((name, cls, n, m, nnz))
        print("%-14s vars=%4d cons=%4d nnz=%5d -> %s" % (name, n, m, nnz, path))
    with open(os.path.join(args.out, "index.csv"), "w") as f:
        f.write("name,class,vars,cons,nnz\n")
        for name, cls, n, m, nnz in index:
            f.write("%s,%s,%d,%d,%d\n" % (name, cls, n, m, nnz))


if __name__ == "__main__":
    main()
