#!/usr/bin/env python3
"""Robust optimization with a hard uncertain inequality, solved EXACTLY by evaluating
the log-sum-exp at the box vertices (MOSEK Tutorials, approx-uncertain-ineq).

For x in R^n, matrices B_i (L x n), i = 1..p, and the uncertainty box [-1,1]^L with
vertices z_k, the "exact" model is

    minimize    sum(x)
    subject to  sum_i exp( (B_i x)^T z_k - sum(x) ) <= 1   for every vertex z_k,

each term an exponential cone: h1_i >= exp(h3_i), h2 = 1, sum_i h1_i <= 1.
Instance L=1 (two vertices +/-1), n=1, p=2, B_0 = (1/2), B_1 = (1/5): the feasible
set is x >= x*, the largest root of exp(-x/2) + exp(-4x/5) = 1, checked against a
dense scan of x. Port of examples/c/hard_uncertain.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE

L = 1
NP = 2
NVX = 1
NVERT = 1 << L
NH = NP * NVERT
B = [0.5, 0.2]                                      # B_i (1x1)
VERT = [-1.0, 1.0]
X, H1, H3, ONE = 0, 1, 1 + NH, 1 + 2 * NH
NV = ONE + 1


def main() -> int:
    with Model() as m:
        m.add_vars(NV)
        m.add_cons(NH + NVERT)
        m.var_bounds(X, BK.FR, -math.inf, math.inf)
        for q in range(NH):
            m.var_bounds(H1 + q, BK.LO, 0.0, math.inf)
        for q in range(NH):
            m.var_bounds(H3 + q, BK.FR, -math.inf, math.inf)
        m.var_bounds(ONE, BK.FX, 1.0, 1.0)
        row = 0
        for i in range(NP):
            for k in range(NVERT):                  # h3_ik - (B_i z_k - 1) x = 0
                coef = B[i] * VERT[k] - 1.0
                m.a_row(row, [H3 + i * NVERT + k, X], [1.0, -coef])
                m.con_bounds(row, BK.FX, 0.0, 0.0)
                row += 1
        for k in range(NVERT):                      # sum_i h1_ik <= 1
            m.a_row(row, [H1 + i * NVERT + k for i in range(NP)], [1.0] * NP)
            m.con_bounds(row, BK.UP, -math.inf, 1.0)
            row += 1
        for q in range(NH):
            m.cone(CT.PEXP, [H1 + q, ONE, H3 + q])
        m.cj(X, 1.0)
        m.sense(SENSE.MIN)
        r = m.solve()
    ok = r.rc == 0
    x = r.x[X] if ok else 0.0

    xstar, found = 0.0, False                       # reference: smallest x where every vertex holds
    for q in range(-400000, 400001):
        xx = q * 1e-5
        if all(sum(math.exp(B[i] * VERT[k] * xx - xx) for i in range(NP)) <= 1.0 for k in range(NVERT)):
            xstar, found = xx, True
            break
    okall = ok and found and abs(x - xstar) < 1e-4 * (1.0 + abs(xstar))
    print(f"hard_uncertain  L={L} n={NVX} p={NP}")
    print(f"  x* = {x:.8f}   [scan {xstar:.8f}]")
    print("OK" if okall else "FAIL")
    return 0 if okall else 1


if __name__ == "__main__":
    raise SystemExit(main())
