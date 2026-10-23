#!/usr/bin/env python3
"""Wasserstein barycenter of discrete distributions (LP).

Source: MOSEK/Tutorials, wasserstein/wasserstein-bary.py. For distributions
nu_1..nu_K on a common finite support {x_1..x_n} and ground cost d_ij = |x_i - x_j|,
the Wasserstein-1 barycenter minimizes the total transport cost

    minimize    sum_k sum_ij d_ij * pi^k_ij
    subject to  sum_j pi^k_ij = mu_i        (row marginals, common mu)
                sum_i pi^k_ij = nu^k_j      (column marginals)
                pi^k_ij >= 0,  mu >= 0,  sum_i mu_i = 1

Hand-derived instance: support x = (0,1,2), nu_1 = (1/2,1/2,0), nu_2 = (0,1/2,1/2):
the optimal value is 1 (the barycenter is not unique here).
Port of examples/c/wasserstein.c.
"""

import math

from primalsolver import Model, BK, SENSE

NB = 3                                              # support points
NK = 2                                              # distributions
XS = [0.0, 1.0, 2.0]
NU = [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]]


def main() -> int:
    npi = NB * NB
    vmu, vpi, nv = 0, NB, NB + NK * npi
    with Model() as m:
        m.add_vars(nv)
        m.add_cons(NK * (NB + NB))                  # row + column marginals per k
        for i in range(NB):
            m.var_bounds(vmu + i, BK.LO, 0.0, math.inf)
        for q in range(NK * npi):
            m.var_bounds(vpi + q, BK.LO, 0.0, math.inf)
        row = 0
        for k in range(NK):
            base = vpi + k * npi
            for i in range(NB):                     # sum_j pi^k_ij - mu_i = 0
                m.a_row(row, [base + i * NB + j for j in range(NB)] + [vmu + i], [1.0] * NB + [-1.0])
                m.con_bounds(row, BK.FX, 0.0, 0.0)
                row += 1
            for j in range(NB):                     # sum_i pi^k_ij = nu^k_j
                m.a_row(row, [base + i * NB + j for i in range(NB)], [1.0] * NB)
                m.con_bounds(row, BK.FX, NU[k][j], NU[k][j])
                row += 1
        for k in range(NK):                         # objective: sum d_ij pi^k_ij
            for i in range(NB):
                for j in range(NB):
                    m.cj(vpi + k * npi + i * NB + j, abs(XS[i] - XS[j]))
        m.sense(SENSE.MIN)
        r = m.solve()
    ok = r.rc == 0
    obj, mu = 0.0, [0.0] * NB
    if ok:
        x = r.x
        mu = [x[vmu + i] for i in range(NB)]
        for k in range(NK):
            for i in range(NB):
                for j in range(NB):
                    obj += abs(XS[i] - XS[j]) * x[vpi + k * npi + i * NB + j]
    msum = mu[0] + mu[1] + mu[2]
    okall = ok and abs(obj - 1.0) < 1e-8 and abs(msum - 1.0) < 1e-7 and all(v >= -1e-9 for v in mu)
    print(f"wasserstein  n={NB} K={NK}")
    print(f"  total cost = {obj:.10f} (expected 1.0)")
    print(f"  barycenter mu = ({mu[0]:.6f}, {mu[1]:.6f}, {mu[2]:.6f})  sum={msum:.8f}")
    print("OK" if okall else "FAIL")
    return 0 if okall else 1


if __name__ == "__main__":
    raise SystemExit(main())
