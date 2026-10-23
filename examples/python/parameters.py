#!/usr/bin/env python3
"""MOSEK port "parameters" - reading/writing parameters.

Every parameter set (putintparam/putdouparam) is read back with the same value; an
invalid parameter is rejected (ERR_ARG); changing INTPNT_MAX_ITERATIONS really limits
the interior-point iterations (TRM). Port of examples/c/parameters.c.
"""

import ctypes
import math

import primalsolver as ps
from primalsolver import Model, BK

IPAR_LOG = 4
IPAR_SIMPLEX_MAX_ITERATIONS = 10
IPAR_INTPNT_MAX_ITERATIONS = 11
DPAR_INTPNT_TOL_PFEAS = 0
DPAR_INTPNT_TOL_DFEAS = 1
DPAR_INTPNT_TOL_REL_GAP = 2
RES_OK = 0


def main() -> int:
    errors = 0
    with Model() as m:
        m.int_param(IPAR_LOG, 0)                    # int params
        m.int_param(IPAR_SIMPLEX_MAX_ITERATIONS, 5000)
        m.int_param(IPAR_INTPNT_MAX_ITERATIONS, 123)
        errors += m.getintparam(IPAR_LOG) != 0
        errors += m.getintparam(IPAR_SIMPLEX_MAX_ITERATIONS) != 5000
        errors += m.getintparam(IPAR_INTPNT_MAX_ITERATIONS) != 123

        m.dou_param(DPAR_INTPNT_TOL_PFEAS, 1e-7)    # double params
        m.dou_param(DPAR_INTPNT_TOL_DFEAS, 2e-7)
        m.dou_param(DPAR_INTPNT_TOL_REL_GAP, 3e-7)
        errors += abs(m.getdouparam(DPAR_INTPNT_TOL_PFEAS) - 1e-7) > 1e-15
        errors += abs(m.getdouparam(DPAR_INTPNT_TOL_DFEAS) - 2e-7) > 1e-15
        errors += abs(m.getdouparam(DPAR_INTPNT_TOL_REL_GAP) - 3e-7) > 1e-15

        # invalid parameters rejected: validation comes from the ranges of the
        # declarative table, so 9999 exists in neither namespace and a tolerance
        # outside [0,1] is an API error
        vi = ctypes.c_int()
        errors += ps.PRIMAL_putintparam(m.task, 9999, 1) == RES_OK
        errors += ps.PRIMAL_putdouparam(m.task, 9999, 1.0) == RES_OK
        errors += ps.PRIMAL_putdouparam(m.task, DPAR_INTPNT_TOL_PFEAS, -1.0) == RES_OK
        errors += ps.PRIMAL_getintparam(m.task, 9999, ctypes.byref(vi)) == RES_OK

    # INTPNT_MAX_ITERATIONS as a real limit: QP with 1 iteration ->
    # TRM_MAX_ITER (1007), or anyway non-OK if the QP does not converge
    with Model() as q:
        q.add_vars(2)
        q.add_cons(1)
        q.var_bounds(0, BK.LO, 0.0, math.inf)
        q.var_bounds(1, BK.LO, 0.0, math.inf)
        q.qobj([(0, 0, 1.0), (1, 1, 1.0)])
        q.a_row(0, [0, 1], [1.0, 1.0])
        q.con_bounds(0, BK.LO, 1.0, math.inf)
        q.int_param(IPAR_INTPNT_MAX_ITERATIONS, 1)
        rq = q.solve().rc
    print(f"QP with max_iter=1: rc={rq} (expected non-OK: 1007 TRM or 1002)")
    rc_trm = rq != RES_OK                           # the limit really interrupts

    print(f"parameter error count: {errors}")
    ok = errors == 0 and rc_trm
    print(f"{'OK' if ok else 'FAIL'} (parameters round-trip + real iteration limit)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
