#!/usr/bin/env python3
"""MOSEK port "callback" - progress callback during optimization.

Registers PRIMAL_setprogresscb + PRIMAL_setinfoconnname and solves a small
minimum-variance QP; checks that the callback receives >= 1 progress notification
and that the solution is the expected one:
    min 0.5*(x0^2 + x1^2)  s.t.  x0 + x1 = 1   ->  x* = (0.5, 0.5), var = 0.25
Port of examples/c/callback.c.
"""

import ctypes
import math

import primalsolver as ps
from primalsolver import Model, BK

PROGRESSCB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_char_p)

n_calls = 0
last_info = ""


@PROGRESSCB
def progcb(handle, info):
    """Progress callback counting calls and storing the latest info."""
    global n_calls, last_info
    n_calls += 1
    last_info = info.decode()[:255]
    print(f"  [progress] {info.decode()}")


def main() -> int:
    global n_calls
    with Model() as m:
        m.add_vars(2)
        m.add_cons(1)
        for j in range(2):
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.qobj([(0, 0, 1.0), (1, 1, 1.0)])
        m.a_row(0, [0, 1], [1.0, 1.0])
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        ps.PRIMAL_setprogresscb(m.task, ctypes.cast(progcb, ctypes.c_void_p), None)
        ps.PRIMAL_setinfoconnname(m.task, b"callback-demo")
        n_calls = 0
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    x, obj = r.x, r.objective
    print(f"x = ({x[0]:.6f}, {x[1]:.6f}), obj = {obj:.6f}, callback calls = {n_calls}")
    print(f"last info: {last_info}")
    ok = abs(x[0] - 0.5) < 1e-6 and abs(x[1] - 0.5) < 1e-6 and abs(obj - 0.25) < 1e-6 and n_calls >= 1
    print(f"{'OK' if ok else 'FAIL'} (expected x=(0.5,0.5), obj=0.25, >=1 calls)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
