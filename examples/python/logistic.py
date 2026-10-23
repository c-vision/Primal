#!/usr/bin/env python3
"""MOSEK port "logistic" - conic regression via the softmax (sum-exp) function.

    min  t + 0.5*gamma*||x||^2
    s.t. y_i = a_i' x                    (i = 1..m)
         t >= sum_i exp(y_i)

Each exponential is a cone: (v_i, 1, y_i) in PEXP => v_i >= exp(y_i), t = sum_i v_i.
The quadratic term is the rotated cone (w, 1, x0, x1) => w >= 0.5*||x||^2.
Independent check: the objective in x is smooth, so the optimum is confirmed by
finite-difference gradient descent. Port of examples/c/logistic.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE

M = 4
NX = 2
A = [[2.0, 1.0], [-1.0, 1.0], [1.0, -2.0], [-2.0, -1.0]]
GAMMA = 0.1
IPAR_INTPNT_MAX_ITERATIONS = 11


def fval(x):
    """Smooth objective f(x) = sum_i exp(a_i'x) + 0.5*gamma*||x||^2."""
    s = 0.0
    for i in range(M):
        s += math.exp(A[i][0] * x[0] + A[i][1] * x[1])
    return s + 0.5 * GAMMA * (x[0] * x[0] + x[1] * x[1])


def main() -> int:
    # variables: [x0, x1, v_0..v_{M-1}, u_0..u_{M-1}, one, w]
    nv = 2 + 2 * M + 2
    v0, u0, one = 2, 2 + M, 2 + 2 * M
    w = one + 1
    with Model() as m:
        m.add_vars(nv)
        m.add_cons(M)
        m.sense(SENSE.MIN)
        for i in range(M):
            m.cj(v0 + i, 1.0)
        m.cj(w, GAMMA)
        for i in range(M):                          # a_i'x - u_i = 0
            m.a_row(i, [0, 1, u0 + i], [A[i][0], A[i][1], -1.0])
            m.con_bounds(i, BK.FX, 0.0, 0.0)
        for j in range(nv):
            m.var_bounds(j, BK.FR, -math.inf, math.inf)
        m.var_bounds(one, BK.FX, 1.0, 1.0)
        # big-M on the u_i: makes the round-0 LP bounded (the true optimum is near 0)
        for i in range(M):
            m.var_bounds(u0 + i, BK.RA, -30.0, 30.0)
        for i in range(M):
            m.cone(CT.PEXP, [v0 + i, one, u0 + i])
        m.cone(CT.RQUAD, [w, one, 0, 1])
        m.int_param(IPAR_INTPNT_MAX_ITERATIONS, 5000)
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    xx, po = r.x, r.objective
    print(f"x = ({xx[0]:.6f}, {xx[1]:.6f}), obj = {po:.6f}")

    xg = [0.0, 0.0]                                 # independent check: gradient descent on f
    fbest, step = fval(xg), 0.5
    for _ in range(20000):
        g = [0.0, 0.0]
        for d in range(2):
            h = 1e-6
            xp, xm = list(xg), list(xg)
            xp[d] += h
            xm[d] -= h
            g[d] = (fval(xp) - fval(xm)) / (2 * h)
        xn = [xg[0] - step * g[0], xg[1] - step * g[1]]
        fn = fval(xn)
        if fn < fbest:
            fbest, xg = fn, xn
            step *= 1.05
        elif fn < fbest + 1e-12:
            step *= 0.7                             # close: shrink the step
        else:
            step *= 0.5
        if step < 1e-12:
            break
    print(f"numerical gradient: f* = {fbest:.6f} at ({xg[0]:.6f}, {xg[1]:.6f})")
    ok = abs(po - fbest) < 1e-3 and abs(xx[0] - xg[0]) < 1e-2 and abs(xx[1] - xg[1]) < 1e-2
    print(f"{'OK' if ok else 'FAIL'} (conic = numerical gradient)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
