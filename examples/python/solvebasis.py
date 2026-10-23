#!/usr/bin/env python3
"""MOSEK port "solvebasis" - write a basis to file, read it back and solve from it.

Flow (as in the official example):
  1. build a known LP and solve it
  2. derive the basis of the optimal solution and write it (writebasis)
  3. build a NEW task and read the basis (readbasis)
  4. solve with PRIMAL_solvebasis and check the solution matches

    max 3x0 + x1 + 5x2 + x3
    s.t. 3x0 + x1 + 2x2 + x3  = 30
         2x0 + x1 + 3x2 + x3 >= 15
                2x1 +     x3 <= 9
    x >= 0, x3 in [0,5]          -> x* = (0, 0, 15, 0), pobj = 75

Port of examples/c/solvebasis.c.
"""

import ctypes
import math

import primalsolver as ps
from primalsolver import Model, BK, SENSE, SOL

BASIS_FILE = "/tmp/mc_solvebasis.bas"
# PRIMALstakeye as numbered in primal.h. (The SK class of primalsolver 0.1.3 numbers
# them differently - LOW=0, UPR=1, FIX=2, BAS=3 - and is not used here.)
SK_BAS, SK_LOW, SK_UPR = 1, 3, 4


def build(m):
    """Build the reference basis LP."""
    m.add_vars(4)
    m.add_cons(3)
    m.sense(SENSE.MAX)
    for j, c in enumerate([3, 1, 5, 1]):
        m.cj(j, c)
    for j in range(4):
        m.var_bounds(j, BK.LO, 0.0, math.inf)
    m.var_bounds(3, BK.RA, 0.0, 5.0)
    m.a_row(0, [0, 1, 2, 3], [3, 1, 2, 1]); m.con_bounds(0, BK.FX, 30.0, 30.0)
    m.a_row(1, [0, 1, 2, 3], [2, 1, 3, 1]); m.con_bounds(1, BK.LO, 15.0, math.inf)
    m.a_row(2, [1, 3], [2, 1]);             m.con_bounds(2, BK.UP, -math.inf, 9.0)


def getxx(m, which, n):
    buf = (ctypes.c_double * n)()
    ps.PRIMAL_getxx(m.task, which, buf)
    return list(buf)


def getprimalobj(m, which):
    v = ctypes.c_double()
    ps.PRIMAL_getprimalobj(m.task, which, ctypes.byref(v))
    return v.value


def basis_from_solution(m):
    """Variables BAS strictly between their bounds, LOW/UPR at a bound; rows BAS
    (basic slack) if inactive, LOW/UPR if active on the lower/upper side."""
    nvar, ncon = m.numvar(), m.numcon()
    x = getxx(m, SOL.BAS, nvar)
    skx = []
    for j in range(nvar):
        bk, bl, bu = m.getvarbound(j)
        if bk == BK.FX:
            skx.append(SK_BAS)
        elif abs(x[j] - bl) < 1e-6:
            skx.append(SK_LOW)
        elif abs(x[j] - bu) < 1e-6:
            skx.append(SK_UPR)
        else:
            skx.append(SK_BAS)
    skc = []
    for i in range(ncon):
        s = 0.0
        for j in range(nvar):
            s += m.getaij(i, j) * x[j]
        bk, bl, bu = m.getconbound(i)
        if abs(s - bl) < 1e-6:
            skc.append(SK_LOW)
        elif abs(s - bu) < 1e-6:
            skc.append(SK_UPR)
        else:
            skc.append(SK_BAS)
    return skc, skx


def main() -> int:
    with Model() as t1, Model() as t2:
        build(t1)                                   # 1. solve and derive the basis
        r = t1.solve()
        if r.rc != 0:
            print(f"optimize rc={r.rc}")
            return 1
        po1 = getprimalobj(t1, SOL.BAS)
        print(f"phase 1: pobj = {po1:.4f} (expected 75)")
        skc, skx = basis_from_solution(t1)

        rc = ps.PRIMAL_putskc(t1.task, SOL.BAS, (ctypes.c_int * len(skc))(*skc))   # 2. write it
        rc |= ps.PRIMAL_putskx(t1.task, SOL.BAS, (ctypes.c_int * len(skx))(*skx))
        if rc:
            print(f"putsk rc={rc}")
            return 1
        rc = ps.PRIMAL_writebasis(t1.task, BASIS_FILE.encode())
        if rc != 0:
            print(f"writebasis rc={rc}")
            return 1
        print(f"phase 2: basis written to {BASIS_FILE}")

        build(t2)                                   # 3. new task: read the basis, solvebasis
        rc = ps.PRIMAL_readbasis(t2.task, BASIS_FILE.encode())
        if rc != 0:
            print(f"readbasis rc={rc}")
            return 1
        rc = ps.PRIMAL_solvebasis(t2.task)
        if rc != 0:
            print(f"solvebasis rc={rc}")
            return 1
        po2 = getprimalobj(t2, SOL.BAS)
        x2 = getxx(t2, SOL.BAS, 4)
    print(f"phase 3: solvebasis pobj = {po2:.4f}, x = ({x2[0]:.4f}, {x2[1]:.4f}, {x2[2]:.4f}, {x2[3]:.4f})")
    ok = (abs(po1 - 75.0) < 1e-6 and abs(po2 - po1) < 1e-6 and abs(x2[0]) < 1e-6
          and abs(x2[1]) < 1e-6 and abs(x2[2] - 15.0) < 1e-5 and abs(x2[3]) < 1e-6)
    print(f"{'OK' if ok else 'FAIL'} (expected pobj=75, x=(0,0,15,0), basis restored)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
