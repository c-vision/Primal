#!/usr/bin/env python3
"""The MOSEK "acc1.c" tutorial (affine conic constraints), the only example in the
OptimalCNC/MOSEK_EXAMPLE CMake template. NOT the same model as acc1.py, which ports
the Julia "acc1" (min x1):

    maximize   c'x,             c = (2, 3, -1)
    subject to sum(x) = 1
               gamma >= || G x + h ||_2   (gamma = 0.03, one quadratic cone)

with G = [[1.5, 0.1], [0.3, 2.1]], h = (0, 0.1). In the MOSEK example the cone is
written with the ACC/AFE API; here the same rows are spelled as direct variables
(z1, z2, the two AFE rows) plus the cone.

Hand value: obj = 3.2805112648 at x = (-0.07838011, 1.12891290, -0.05053279).
Port of examples/c/acc1_max.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE

X0, X1, X2, Z1, Z2, GAMMA = 0, 1, 2, 3, 4, 5


def main() -> int:
    with Model() as m:
        m.add_vars(6)
        m.add_cons(3)
        m.sense(SENSE.MAX)
        for j in range(X0, Z2 + 1):
            m.var_bounds(j, BK.FR, -math.inf, math.inf)
        m.var_bounds(GAMMA, BK.FX, 0.03, 0.03)
        m.a_row(0, [X0, X1, X2], [1.0, 1.0, 1.0])   # sum(x) = 1
        m.con_bounds(0, BK.FX, 1.0, 1.0)
        m.a_row(1, [Z1, X0, X1], [1.0, -1.5, -0.1])  # z1 = 1.5 x0 + 0.1 x1
        m.con_bounds(1, BK.FX, 0.0, 0.0)
        m.a_row(2, [Z2, X0, X2], [1.0, -0.3, -2.1])  # z2 = 0.3 x0 + 2.1 x2 + 0.1
        m.con_bounds(2, BK.FX, 0.1, 0.1)
        m.cone(CT.QUAD, [GAMMA, Z1, Z2])             # gamma >= ||(z1, z2)||
        m.obj([X0, X1, X2], [2.0, 3.0, -1.0])
        r = m.solve()
    ok = r.rc == 0
    x, obj = (r.x, r.objective) if ok else ([0.0] * 6, 0.0)
    z1 = 1.5 * x[X0] + 0.1 * x[X1]
    z2 = 0.3 * x[X0] + 2.1 * x[X2] + 0.1
    nrm = math.sqrt(z1 * z1 + z2 * z2)
    total = x[X0] + x[X1] + x[X2]
    good = (ok and abs(obj - 3.2805112648) < 1e-6 and abs(x[X0] + 0.0783801115) < 1e-6
            and abs(x[X1] - 1.1289128998) < 1e-6 and abs(x[X2] + 0.0505327883) < 1e-6
            and nrm <= 0.03 + 1e-9 and abs(total - 1.0) < 1e-9)
    print(f"acc1_max  obj={obj:.10f} (expected 3.2805112648)  x=({x[X0]:.8f},{x[X1]:.8f},{x[X2]:.8f})")
    print(f"  ||Gx+h||={nrm:.10f} (<=0.03)  sum={total:.10f}")
    print("OK" if good else "FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
