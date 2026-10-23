#!/usr/bin/env python3
"""Finite-horizon linear MPC as a QP (the linearized branch of jonaylton/mpc-missile;
the full missile NLP is out of scope for a convex solver).

Discretized double integrator p'' = u, h = 1:
    p_{k+1} = p_k + v_k + u_k/2,   v_{k+1} = v_k + u_k
i.e. x_{k+1} = A x_k + B u_k, A = [[1,1],[0,1]], B = (1/2,1). Minimum-energy
regulation to a terminal state over N steps:
    minimize sum_k u_k^2  s.t.  x_0 = x_ini,  x_{k+1} = A x_k + B u_k,  x_N = x_term.

Hand instance: N = 2, x_ini = (0,0), x_term = (1,0) -> u = (1,-1), sum u^2 = 2.
Port of examples/c/mpc_linear.c.
"""

import math

from primalsolver import Model, BK, SENSE

NSTEP = 2
NST = NSTEP + 1
A = [[1, 1], [0, 1]]
B = [0.5, 1.0]
XINI = [0, 0]


def P(k): return 2 * k                              # noqa: E704 - state layout (p_k, v_k)
def V(k): return 2 * k + 1                          # noqa: E704
def U(k): return 2 * NST + k                        # noqa: E704 - controls after the states


def main() -> int:
    nv = 2 * NST + NSTEP
    nrow = 2 + 2 * NSTEP + 2
    with Model() as m:
        m.add_vars(nv)
        m.add_cons(nrow)
        for v in range(nv):
            m.var_bounds(v, BK.FR, -math.inf, math.inf)
        m.sense(SENSE.MIN)
        for k in range(NSTEP):
            m.qobj_ij(U(k), U(k), 2.0)              # u_k^2
        row = 0
        for c in range(2):                          # x_0 = x_ini
            m.a_row(row, [c], [1.0])
            m.con_bounds(row, BK.FX, XINI[c], XINI[c])
            row += 1
        for k in range(NSTEP):                      # x_{k+1} - A x_k - B u_k = 0
            for c in range(2):
                sub = [P(k + 1) if c == 0 else V(k + 1)]
                val = [1.0]
                for d in range(2):
                    if A[c][d] != 0.0:
                        sub.append(P(k) if d == 0 else V(k))
                        val.append(-A[c][d])
                if B[c] != 0.0:
                    sub.append(U(k))
                    val.append(-B[c])
                m.a_row(row, sub, val)
                m.con_bounds(row, BK.FX, 0.0, 0.0)
                row += 1
        m.a_row(row, [P(NSTEP)], [1.0]); m.con_bounds(row, BK.FX, 1.0, 1.0); row += 1   # p_N = 1
        m.a_row(row, [V(NSTEP)], [1.0]); m.con_bounds(row, BK.FX, 0.0, 0.0); row += 1   # v_N = 0
        r = m.solve()
    ok = r.rc == 0
    obj, x = (r.objective, r.x) if ok else (0.0, [0.0] * nv)
    u = [x[U(k)] for k in range(NSTEP)]
    good = (ok and abs(obj - 2.0) < 1e-7 and abs(u[0] - 1.0) < 1e-6 and abs(u[1] + 1.0) < 1e-6
            and abs(x[P(NSTEP)] - 1.0) < 1e-6 and abs(x[V(NSTEP)]) < 1e-6)
    print(f"mpc_linear: obj={obj:.8f}  u=({u[0]:.6f}, {u[1]:.6f})  x_N=({x[P(NSTEP)]:.6f}, {x[V(NSTEP)]:.6f})  "
          f"(expected 2, u=(1,-1), x_N=(1,0))  {'OK' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
