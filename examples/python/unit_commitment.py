#!/usr/bin/env python3
"""A tiny thermal unit-commitment MILP (MOSEK Modeling Cookbook / MOSEK Tutorials).

Two generators over two periods; commitment and startup are binary, the dispatch
is continuous:

    min  sum_{i,t} ( no_load_i u_{i,t} + marginal_i p_{i,t} + startup_i v_{i,t} )
    s.t. sum_i p_{i,t} = d_t                         (demand, both periods)
         min_i u_{i,t} <= p_{i,t} <= max_i u_{i,t}   (dispatch bounds)
         v_{i,t} >= u_{i,t} - u_{i,t-1},  v >= 0     (startup, u_{i,-1} = 0)
         u_{i,t}, v_{i,t} in {0,1}

Instance: gen 0 (min 1, max 3, no-load 5, marginal 2, startup 10), gen 1 (min 0,
max 4, no-load 0, marginal 6, startup 0), demand (2, 4): hand-derived optimum 36.
Port of examples/c/unit_commitment.c.
"""

import math

from primalsolver import Model, BK, VAR_TYPE

NG = 2
NT = 2
LO = [1, 0]
UP = [3, 4]
NOLOAD = [5, 0]
MARG = [2, 6]
STARTC = [10, 0]
DEMAND = [2, 4]
NVAR = 3 * NG * NT


def U(i, t): return i * NT + t                      # noqa: E704 - variable layout
def V(i, t): return NG * NT + i * NT + t            # noqa: E704
def P(i, t): return 2 * NG * NT + i * NT + t        # noqa: E704


def main() -> int:
    with Model() as m:
        m.add_vars(NVAR)
        m.add_cons(NT + 3 * NG * NT)                # demand + (max, min, startup) per (i,t)
        for i in range(NG):
            for k in range(NT):
                m.var_bounds(U(i, k), BK.RA, 0.0, 1.0)
                m.var_type(U(i, k), VAR_TYPE.INT_BIN)
                m.var_bounds(V(i, k), BK.RA, 0.0, 1.0)
                m.var_type(V(i, k), VAR_TYPE.INT_BIN)
                m.var_bounds(P(i, k), BK.LO, 0.0, math.inf)
                m.cj(U(i, k), NOLOAD[i])
                m.cj(V(i, k), STARTC[i])
                m.cj(P(i, k), MARG[i])
        row = 0
        for k in range(NT):                         # demand: sum_i p_i,k = d_k
            m.a_row(row, [P(i, k) for i in range(NG)], [1.0] * NG)
            m.con_bounds(row, BK.FX, DEMAND[k], DEMAND[k])
            row += 1
        for i in range(NG):
            for k in range(NT):
                m.a_row(row, [P(i, k), U(i, k)], [1.0, -UP[i]])        # p - max*u <= 0
                m.con_bounds(row, BK.UP, -math.inf, 0.0)
                row += 1
                m.a_row(row, [P(i, k), U(i, k)], [1.0, -LO[i]])        # p - min*u >= 0
                m.con_bounds(row, BK.LO, 0.0, math.inf)
                row += 1
                if k == 0:                          # v - u_k + u_{k-1} >= 0  (u_{i,-1} = 0)
                    m.a_row(row, [V(i, k), U(i, k)], [1.0, -1.0])
                else:
                    m.a_row(row, [V(i, k), U(i, k), U(i, k - 1)], [1.0, -1.0, 1.0])
                m.con_bounds(row, BK.LO, 0.0, math.inf)
                row += 1
        r = m.solve()
    if r.rc != 0:
        print(f"unit_commitment  rc={r.rc} FAIL")
        return 1
    x, obj = r.x, r.objective
    ok = abs(obj - 36.0) < 1e-6
    print(f"unit_commitment  d=[2,4]  p0=({x[P(0, 0)]:.1f},{x[P(0, 1)]:.1f}) p1=({x[P(1, 0)]:.1f},{x[P(1, 1)]:.1f}) "
          f"obj={obj:.4f} (expected 36) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
