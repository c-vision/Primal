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

"""Conic cross-check: solve the SAME SOCP family as bench/conic_bench.c with
Clarabel and SCS (optional; show N/A if not installed).

Family (conic_bench.c):  min sum_k t_k
    s.t. (t_k, x_2k, x_2k+1) in QUAD,  sum_i x_i = 1,
closed-form optimum 1/sqrt(2) (concentrate the unit mass in one cone).

Clarabel/SCS take the conic standard form  min q'z  s.t.  A z + s = b, s in K:
    * zero cone (dim 1): sum_i x_i = 1;
    * K second-order cones (dim 3): s_k = (t_k, x_2k, x_2k+1).

Usage:  python3 bench/conic_ref.py            (any interpreter with numpy/scipy)
"""
import math
import time

import numpy as np
from scipy import sparse

SIZES = [40, 120, 200]


def build_socp(n):
    """Return (N, q, A, b, zero_dim, soc_dims) matching conic_bench.c."""
    K = n // 2
    N = n + K
    rows = 1 + 3 * K
    A = sparse.lil_matrix((rows, N))
    b = np.zeros(rows)
    # zero cone: sum_i x_i = 1
    A[0, 0:n] = 1.0
    b[0] = 1.0
    # second-order cones: s_k = (t_k, x_2k, x_2k+1)
    for k in range(K):
        A[1 + 3 * k, n + k] = -1.0            # s0 = t_k
        A[2 + 3 * k, 2 * k] = -1.0            # s1 = x_2k
        A[3 + 3 * k, 2 * k + 1] = -1.0        # s2 = x_2k+1
    q = np.concatenate([np.zeros(n), np.ones(K)])
    return N, q, A.tocsc(), b, 1, [3] * K


def solve_clarabel(n):
    try:
        import clarabel
    except Exception:
        return None, None
    N, q, A, b, z, soc = build_socp(n)
    P = sparse.csc_matrix((N, N))
    cones = [clarabel.ZeroConeT(z)] + [clarabel.SecondOrderConeT(d) for d in soc]
    try:
        st = clarabel.DefaultSettings()
        st.verbose = False
        solver = clarabel.DefaultSolver(P, q, A, b, cones, st)
        t = time.perf_counter()
        sol = solver.solve()
        dt = time.perf_counter() - t
        return dt, float(sol.obj_val)
    except Exception:
        return None, None


def solve_scs(n):
    try:
        import scs
    except Exception:
        return None, None
    N, q, A, b, z, soc = build_socp(n)
    data = dict(A=A, b=b, c=q)
    cone = dict(z=z, q=soc)
    try:
        t = time.perf_counter()
        sol = scs.solve(data, cone, verbose=False)
        dt = time.perf_counter() - t
        return dt, float(sol["info"]["pobj"])
    except Exception:
        return None, None


def _rng(seed):
    """Same LCG as conic_bench.c urand()."""
    st = seed
    while True:
        st = (st * 1103515245 + 12345) & 0xFFFFFFFF
        yield ((st >> 16) & 0x7FFF) / 32767.0


def sdp_data(d, seed, order="colmajor"):
    """Same SDP as conic_bench.c run_sdp: min <C,X>, X_ii=1, X>=0, C=-vv'.
    Returns (L, c, A, b, dim, expected) in the svec convention. order selects
    the triangle: "colmajor" (SCS) or "rowmajor" (Clarabel)."""
    g = _rng(seed)
    v = [next(g) * 2.0 - 1.0 for _ in range(d)]
    expected = -sum(abs(x) for x in v) ** 2
    idx, k = {}, 0
    if order == "rowmajor":
        for i in range(d):
            for j in range(i + 1):
                idx[(i, j)] = k
                k += 1
    else:
        for j in range(d):
            for i in range(j, d):
                idx[(i, j)] = k
                k += 1
    L = k

    def svec(M):
        out = np.zeros(L)
        for (i, j), kk in idx.items():
            out[kk] = M[i][j] * (math.sqrt(2.0) if i != j else 1.0)
        return out

    C = [[-v[i] * v[j] for j in range(d)] for i in range(d)]
    c = svec(C)
    A = sparse.lil_matrix((d + L, L))
    b = np.zeros(d + L)
    for i in range(d):
        A[i, idx[(i, i)]] = 1.0
        b[i] = 1.0
    for kk in range(L):
        A[d + kk, kk] = -1.0
    return L, c, A.tocsc(), b, d, expected


def solve_scs_sdp(d, seed):
    try:
        import scs
    except Exception:
        return None, None, None
    L, c, A, b, dim, expected = sdp_data(d, seed)
    data = dict(A=A, b=b, c=c)
    try:
        t = time.perf_counter()
        sol = scs.solve(data, dict(z=dim, s=[dim]), verbose=False)
        dt = time.perf_counter() - t
        return dt, float(sol["info"]["pobj"]), expected
    except Exception:
        return None, None, None


def solve_clarabel_sdp(d, seed):
    try:
        import clarabel
    except Exception:
        return None, None, None
    L, c, A, b, dim, expected = sdp_data(d, seed, order="rowmajor")
    try:
        st = clarabel.DefaultSettings()
        st.verbose = False
        cones = [clarabel.ZeroConeT(dim), clarabel.PSDTriangleConeT(dim)]
        solver = clarabel.DefaultSolver(sparse.csc_matrix((L, L)), c, A, b, cones, st)
        t = time.perf_counter()
        sol = solver.solve()
        dt = time.perf_counter() - t
        return dt, float(sol.obj_val), expected
    except Exception:
        return None, None, None


def main():
    print("| n | K | PrimalSolver (s) | Clarabel (s) | SCS (s) | obj (expected 1/sqrt2) |")
    print("|---|---|---|---|---|---|")
    print("# PrimalSolver: python3 bench/conic_bench.py  -> out/bench/conic_bench")
    for n in SIZES:
        K = n // 2
        tc, oc = solve_clarabel(n)
        ts, os_ = solve_scs(n)
        objs = [o for o in (oc, os_) if o is not None]
        ref = 1.0 / math.sqrt(2.0)
        ok = all(abs(o - ref) < 1e-4 for o in objs) if objs else None
        print("| %d | %d | — | %s | %s | %s |" %
              (n, K,
               "%.4f" % tc if tc is not None else "N/A",
               "%.4f" % ts if ts is not None else "N/A",
               ("%.8f %s" % (oc, "OK" if abs(oc - ref) < 1e-4 else "DIFF"))
               if oc is not None else ("N/A" if not ok else "%.8f" % os_)))
    print()
    print("| d | PrimalSolver (s) | Clarabel (s) | SCS (s) | obj | chiuso |")
    print("|---|---|---|---|---|---|")
    for d in range(4, 9):
        tc, oc, exp = solve_clarabel_sdp(d, 200 + d)
        ts, os_, _ = solve_scs_sdp(d, 200 + d)
        ob = oc if oc is not None else os_
        print("| %d | — | %s | %s | %s | %.8g |" %
              (d, "%.4f" % tc if tc is not None else "N/A",
               "%.4f" % ts if ts is not None else "N/A",
               "%.8g" % ob if ob is not None else "N/A", exp))


if __name__ == "__main__":
    main()
