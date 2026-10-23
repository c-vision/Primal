#!/usr/bin/env python3
"""Best-subset selection by rank-one convexification (MOSEK Tutorials,
rank-one-regression), with a disjunction (DJC) selecting the support:

    minimize    y'y - 2 y'X b + t + mu * sum_i u_i
    subject to  t >= b' (X'X + lambda I) b     (rotated cone)
                sum_i z_i <= k,   z_i in {0,1}
                |b_i| <= u_i
                (z_i == 1) OR (b_i == 0)        (one DJC per i)

Hand-derived instance: X = [[1,0],[1,1],[0,1]], y = (1,2,1), k=1, lambda = mu = 0.
With card(b) <= 1 the best is b = (3/2, 0) (or (0, 3/2)), objective 3/2.
Port of examples/c/rank_one.c.
"""

import math

from primalsolver import Model, APPROX, BK, CT, SENSE, SolverError, VAR_TYPE

NPTS = 3
NF = 2
LAM = 0.0
MU = 0.0
KCAP = 1
XP = [[1, 0], [1, 1], [0, 1], [0, 0], [0, 0]]       # X stacked over sqrt(lambda) I (lambda = 0)
Y = [1, 2, 1]
B, U, TT, Z, W = 0, NF, 2 * NF, 2 * NF + 1, 3 * NF + 1
HT = 3 * NF + 1 + (NPTS + NF)
ONE = HT + 1
NV = HT + 2


def main() -> int:
    with Model() as m:
        m.add_vars(NV)
        m.add_cons((NPTS + NF) + 1 + 1 + 2 * NF)    # W links + HT + card + |b|<=u
        for i in range(NF):
            m.var_bounds(B + i, BK.FR, -math.inf, math.inf)
            m.var_bounds(U + i, BK.LO, 0.0, math.inf)
            m.var_bounds(Z + i, BK.RA, 0.0, 1.0)
            m.var_type(Z + i, VAR_TYPE.INT_BIN)
        m.var_bounds(TT, BK.LO, 0.0, math.inf)
        for k in range(NPTS + NF):
            m.var_bounds(W + k, BK.FR, -math.inf, math.inf)
        m.var_bounds(HT, BK.LO, 0.0, math.inf)
        m.var_bounds(ONE, BK.FX, 1.0, 1.0)
        row = 0
        for k in range(NPTS + NF):                  # W_k - sum_j Xp[k][j] B_j = 0
            cols = [j for j in range(NF) if XP[k][j] != 0.0]
            m.a_row(row, [W + k] + [B + j for j in cols], [1.0] + [-XP[k][j] for j in cols])
            m.con_bounds(row, BK.FX, 0.0, 0.0)
            row += 1
        m.a_row(row, [HT, TT], [1.0, -0.5])
        m.con_bounds(row, BK.FX, 0.0, 0.0)
        row += 1
        m.a_row(row, [Z + i for i in range(NF)], [1.0] * NF)
        m.con_bounds(row, BK.UP, -math.inf, KCAP)
        row += 1
        for i in range(NF):                         # |B_i| <= U_i
            m.a_row(row, [B + i, U + i], [1.0, -1.0]); m.con_bounds(row, BK.UP, -math.inf, 0.0); row += 1
            m.a_row(row, [B + i, U + i], [-1.0, -1.0]); m.con_bounds(row, BK.UP, -math.inf, 0.0); row += 1
        m.cone(CT.RQUAD, [ONE, HT] + [W + k for k in range(NPTS + NF)])
        for j in range(NF):
            coeff = 0.0
            for k in range(NPTS):
                coeff += XP[k][j] * Y[k]
            m.cj(B + j, -2.0 * coeff)
        m.cj(TT, 1.0)
        for i in range(NF):
            m.cj(U + i, MU)
        m.cfix(float(sum(v * v for v in Y)))
        m.sense(SENSE.MIN)

        m.afes(2 * NF)                              # disjunction (z_i == 1) OR (b_i == 0)
        for i in range(NF):
            m.afe_entry(i, Z + i, 1.0)
            m.afe_entry(NF + i, B + i, 1.0)
        rzero = m.domain(APPROX.RZERO, 1)
        m.djcs(NF)
        for i in range(NF):
            try:
                m.djc(i, [rzero, rzero], [i, NF + i], [1.0, 0.0], [1, 1])
            except SolverError as err:
                print(f"rank_one  putdjc {err} FAIL")
                return 1
        r = m.solve()
    ok = r.rc == 0
    obj, b0, b1 = (r.objective, r.x[B], r.x[B + 1]) if ok else (0.0, 0.0, 0.0)
    onezero = abs(b0) < 1e-6 or abs(b1) < 1e-6
    okall = (ok and abs(obj - 1.5) < 1e-6 and onezero
             and (abs(abs(b0) - 1.5) < 1e-5 or abs(abs(b1) - 1.5) < 1e-5))
    print(f"rank_one  k={KCAP}")
    print(f"  b = ({b0:.8f}, {b1:.8f})   objective = {obj:.10f} (expected 1.5)")
    print("OK" if okall else "FAIL")
    return 0 if okall else 1


if __name__ == "__main__":
    raise SystemExit(main())
