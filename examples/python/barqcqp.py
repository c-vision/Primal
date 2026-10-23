#!/usr/bin/env python3
"""Symmetric bars and quadratic part (constraint and objective) in the same model,
solved by the QCQP-to-conic route through the public API alone; plus the nonconvex
domain, which must be refused. The optimal values are derived by hand.

A) min -x0 - x1 - 4 <E00,B>
   s.t. x0^2 + x1^2 + <E00,B> <= 3,  <I,B> = 1,  B in S^2_+,  x >= 0
   -> x* = (1,1), B = diag(1,0), obj = -6.
B) min x0^2 + x1^2 - 2 x0 - 2 x1 - 3 <E00,B>   s.t.  <I,B> = 1,  B in S^2_+,  x >= 0
   -> x* = (1,1), B = diag(1,0), obj = -5 (dropping the quadratic objective would
   leave an unbounded model, so the number certifies the whole encoding).
C) as A but with x0^2 - 2 x1^2 + <E00,B> <= 3 and 0 <= x <= 2: diag(2,-4) has an
   eigenvalue on the wrong side, the domain is not convex and must be refused:
   rc = ERR_ARG, and the solution getters say "no" with the same code.
Port of examples/c/barqcqp.c.
"""

import ctypes
import math

import primalsolver as ps
from primalsolver import Model, BK, SENSE, SOL

RES_OK = 0
RES_ERR_ARG = 1001


def put_syms(m):
    """E00 and the 2x2 identity, shared by the three cases."""
    return m.sparsesymmat(2, [(0, 0, 1.0)]), m.sparsesymmat(2, [(0, 0, 1.0), (1, 1, 1.0)])


def results(m):
    """Optimize and read x, obj and B with the getters' own return codes."""
    x = (ctypes.c_double * 2)()
    bar = (ctypes.c_double * 4)()
    po = ctypes.c_double()
    ok = (ps.PRIMAL_optimize(m.task) == RES_OK
          and ps.PRIMAL_getxx(m.task, SOL.ITR, x) == RES_OK
          and ps.PRIMAL_getprimalobj(m.task, SOL.ITR, ctypes.byref(po)) == RES_OK
          and ps.PRIMAL_getbarxj(m.task, SOL.ITR, 0, bar) == RES_OK)
    return ok, list(x), po.value, list(bar)


def report(name, ok, x, po, bar, want):
    ok = (ok and abs(x[0] - 1.0) < 1e-4 and abs(x[1] - 1.0) < 1e-4 and abs(po - want) < 1e-5
          and abs(bar[0] - 1.0) < 1e-5 and abs(bar[1]) < 1e-5 and abs(bar[3]) < 1e-5)
    print(f"barqcqp {name}  obj={po:.6f} (expected {want:.0f})  x=({x[0]:.6f}, {x[1]:.6f})  "
          f"B=[[{bar[0]:.4f},{bar[1]:.4f}],[{bar[2]:.4f},{bar[3]:.4f}]]  {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    passed = True
    with Model() as m:                              # A) quadratic row and bar in the same row
        m.add_cons(2)
        m00, mi = put_syms(m)
        m.barvars([2])
        m.add_vars(2)
        m.sense(SENSE.MIN)
        for j in range(2):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.cj(0, -1.0); m.cj(1, -1.0)
        m.barcj(0, [(m00, -4.0)])
        m.con_bounds(0, BK.UP, -math.inf, 3.0)
        m.qconk(0, [(0, 0, 2.0), (1, 1, 2.0)])
        m.baraij(0, 0, [(m00, 1.0)])
        m.baraij(1, 0, [(mi, 1.0)])
        m.con_bounds(1, BK.FX, 1.0, 1.0)
        passed &= report("A", *results(m), -6.0)

    with Model() as m:                              # B) convex quadratic objective + bar
        m.add_cons(1)
        m00, mi = put_syms(m)
        m.barvars([2])
        m.add_vars(2)
        m.sense(SENSE.MIN)
        for j in range(2):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.cj(0, -2.0); m.cj(1, -2.0)
        m.qobj([(0, 0, 2.0), (1, 1, 2.0)])
        m.barcj(0, [(m00, -3.0)])
        m.baraij(0, 0, [(mi, 1.0)])
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        passed &= report("B", *results(m), -5.0)

    with Model() as m:                              # C) nonconvex domain: refusal, not an answer
        m.add_cons(2)
        m00, mi = put_syms(m)
        m.barvars([2])
        m.add_vars(2)
        m.sense(SENSE.MIN)
        for j in range(2):
            m.var_bounds(j, BK.RA, 0.0, 2.0)
        m.cj(0, -1.0); m.cj(1, -1.0)
        m.con_bounds(0, BK.UP, -math.inf, 3.0)
        m.qconk(0, [(0, 0, 2.0), (1, 1, -4.0)])
        m.baraij(0, 0, [(m00, 1.0)])
        m.baraij(1, 0, [(mi, 1.0)])
        m.con_bounds(1, BK.FX, 1.0, 1.0)
        rc = ps.PRIMAL_optimize(m.task)
        x = (ctypes.c_double * 2)()
        po = ctypes.c_double()
        ok = (rc == RES_ERR_ARG and ps.PRIMAL_getxx(m.task, SOL.ITR, x) == RES_ERR_ARG
              and ps.PRIMAL_getprimalobj(m.task, SOL.ITR, ctypes.byref(po)) == RES_ERR_ARG)
        print(f"barqcqp C  rc={rc} (expected {RES_ERR_ARG} = ERR_ARG)  obj not published  {'OK' if ok else 'FAIL'}")
        passed &= ok
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
