#!/usr/bin/env python3
"""Minimum-variance portfolio under a factor risk model.

Sigma = F F' + diag(D), so w'Sigma w = ||F'w||^2 + sum_j D_j w_j^2; the QP is built with
the explicit Sigma and cross-checked against a hand value. Hand case: N = 3,
F = (1,1,0), D = (0,0,1): with e'w = 1, w'Sigma w = (1-w2)^2 + w2^2, minimised at
w2 = 1/2, value 1/2. Port of examples/c/factor_model.c.
"""

import math

from primalsolver import Model, BK


def main() -> int:
    F = [1.0, 1.0, 0.0]
    D = [0.0, 0.0, 1.0]
    Sigma = [[F[i] * F[j] + (D[i] if i == j else 0.0) for j in range(3)] for i in range(3)]
    with Model() as m:
        m.add_vars(3)
        m.add_cons(1)
        for j in range(3):
            m.var_bounds(j, BK.FR, -math.inf, math.inf)
        m.a_row(0, [0, 1, 2], [1.0, 1.0, 1.0])  # e'w = 1
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        for i in range(3):                      # w'Sigma w: 1/2 w'Qw with Q = 2 Sigma
            for j in range(i, 3):
                m.qobj([(i, j, 2.0 * Sigma[i][j])])
        r = m.solve()
    if r.rc != 0:
        print("optimize rc=%d" % r.rc)
        return 1
    obj, w = r.objective, r.x
    quad = 0.0                                  # the operator, rebuilt from the published w
    for i in range(3):
        for j in range(3):
            quad += w[i] * Sigma[i][j] * w[j]
    ok = (abs(obj - 0.5) < 1e-7 and abs(w[2] - 0.5) < 1e-6 and abs(w[0] + w[1] - 0.5) < 1e-6
          and abs(w[0] + w[1] + w[2] - 1.0) < 1e-6 and abs(quad - obj) < 1e-7)
    print("w = (%.4f, %.4f, %.4f), obj = %.6f (expected 0.5)" % (w[0], w[1], w[2], obj))
    print("w'Sigma w reconstructed = %.6f, e'w = %.2e" % (quad, w[0] + w[1] + w[2]))
    print("%s (Sigma = F F' + diag(D), minimum at w2 = 1/2)" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
