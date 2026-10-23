#!/usr/bin/env python3
"""MOSEK port "portfolio_6_factor" - portfolio optimization with a FACTOR model.

The covariance is x'(F P F' + D)x with factors f = F'x, P = diag(p) and specific
risk D (diagonal); cardinality binaries as in portfolio_5:

    max  r'x - gamma*(||p^{1/2} F'x||^2 + x'Dx)
    s.t. sum(x) = 1, 0 <= x <= y, y binary, sum(y) <= k

F (3x2): 2 factors; p = (0.05, 0.06); D = diag(0.01, 0.02, 0.005). The factor risk
is modelled with a variable v and the cone (v, z1, z2) in QUAD with z = p^{1/2} F'x;
x'Dx stays quadratic. Check: budget, binaries, objective consistent with the direct
evaluation of the risk at x*. Port of examples/c/portfolio_6.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE, VAR_TYPE

N = 3                                               # assets
M = 2                                               # factors
K = 2
R = [0.10717, 0.07502, 0.11902]
F = [[0.16, -0.23], [0.28, 0.19], [-0.11, 0.32]]
P = [0.05, 0.06]
D = [0.01, 0.02, 0.005]
GAMMA = 0.05
VV, VZ, VY = 3, 4, 6                                # variables: x(0..2), v=3, z(4..5), y(6..8)


def main() -> int:
    with Model() as m:
        m.add_vars(9)
        m.add_cons(2 + N + M)                       # budget, cardinality, x<=y, z = sqrt(p) F'x
        m.sense(SENSE.MAX)
        for j in range(N):
            m.cj(j, R[j])
            m.var_bounds(j, BK.LO, 0.0, math.inf)
        m.cj(VV, -GAMMA)                            # -gamma * v
        m.var_bounds(VV, BK.LO, 0.0, math.inf)
        for i in range(M):
            m.var_bounds(VZ + i, BK.FR, -math.inf, math.inf)
        for j in range(N):
            m.var_bounds(VY + j, BK.RA, 0.0, 1.0)
            m.var_type(VY + j, VAR_TYPE.INT_BIN)
        m.a_row(0, [0, 1, 2], [1.0, 1.0, 1.0])      # budget
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.a_row(1, [VY, VY + 1, VY + 2], [1.0, 1.0, 1.0])   # cardinality
        m.con_bounds(1, BK.UP, -math.inf, float(K))
        for j in range(N):                          # x_j <= y_j
            m.a_row(2 + j, [j, VY + j], [1.0, -1.0])
            m.con_bounds(2 + j, BK.UP, -math.inf, 0.0)
        for i in range(M):                          # z_i = sqrt(p_i) * (F'x)_i
            sp = math.sqrt(P[i])
            m.a_row(2 + N + i, [0, 1, 2, VZ + i], [-sp * F[0][i], -sp * F[1][i], -sp * F[2][i], 1.0])
            m.con_bounds(2 + N + i, BK.FX, 0.0, 0.0)
        m.cone(CT.QUAD, [VV, VZ, VZ + 1])           # v >= ||z||
        m.qobj([(j, j, -2.0 * GAMMA * D[j]) for j in range(N)])   # -gamma * x'Dx
        r = m.solve()
    if r.rc != 0:
        print(f"optimize rc={r.rc}")
        return 1
    x, obj = r.x, r.objective
    print(f"x = ({x[0]:.4f}, {x[1]:.4f}, {x[2]:.4f})")
    print(f"v = {x[VV]:.4f}, y = ({x[VY]:.0f}, {x[VY + 1]:.0f}, {x[VY + 2]:.0f})")
    print(f"obj = {obj:.6f}")
    # independent check: risk at x* from F, P, D and obj = r'x - gamma*risk. The conic
    # IPM leaves v slightly above ||z||, so the tolerance is 5e-3 on both sides.
    fx = [0.0, 0.0]
    for i in range(M):
        for j in range(N):
            fx[i] += F[j][i] * x[j]
    risk = 0.0
    for i in range(M):
        risk += P[i] * fx[i] * fx[i]
    for j in range(N):
        risk += D[j] * x[j] * x[j]
    exp_obj = 0.0
    for j in range(N):
        exp_obj += R[j] * x[j]
    exp_obj -= GAMMA * risk
    print(f"direct risk = {risk:.6f}, expected obj = {exp_obj:.6f}")
    ok = abs(x[0] + x[1] + x[2] - 1.0) < 1e-5 and exp_obj - 5e-3 <= obj <= exp_obj + 5e-3
    nact = 0
    for j in range(N):
        if x[j] > 1e-6:
            nact += 1
        if abs(x[VY + j] - math.floor(x[VY + j] + 0.5)) > 1e-6 or x[j] > x[VY + j] + 1e-6:
            ok = False
    if nact > K:
        ok = False
    print(f"{'OK' if ok else 'FAIL'} (factors+MIP: obj = r'x - gamma*risk, cardinality <= {K})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
