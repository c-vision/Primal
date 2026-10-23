#!/usr/bin/env python3
"""Smallest k balls covering a point set (MISOCP).

Source: MOSEK/Tutorials, facility-location/small_disks.py. Given points p_i in R^2
and a number k of balls, choose centers x_j and radii r_j, plus binary assignment
s_ij, so that every point is covered:

    minimize    t
    subject to  || p_i - x_j ||_2 <= r_j + M (1 - s_ij)   for all i,j  (SOC)
                sum_j s_ij = 1,   s_ij in {0,1}
                t >= r_j,   r_j >= 0

Hand-derived instance: four collinear points 0,1,2,3 and k=2 balls -> t = 0.5
(centers at 0.5 and 2.5). Port of examples/c/facility_location.c.
"""

import math

from primalsolver import Model, BK, CT, SENSE, VAR_TYPE

NP = 4                                              # points
NK = 2                                              # balls
DIM = 2
BIGM = 10.0
PTS = [(0, 0), (1, 0), (2, 0), (3, 0)]


def main() -> int:
    # vars: T | R[NK] | X[NK*DIM] | S[NP*NK] | W[NP*NK] | D[NP*NK*DIM]
    vt = 0
    vr = vt + 1
    vx = vr + NK
    vs = vx + NK * DIM
    vw = vs + NP * NK
    vd = vw + NP * NK
    nv = vd + NP * NK * DIM
    with Model() as m:
        m.add_vars(nv)
        m.add_cons(NP * NK + NP * NK * DIM + NP + NK)   # W + D + assign + t>=R
        m.var_bounds(vt, BK.LO, 0.0, math.inf)
        for j in range(NK):
            m.var_bounds(vr + j, BK.LO, 0.0, math.inf)
        for q in range(NK * DIM):
            m.var_bounds(vx + q, BK.FR, -math.inf, math.inf)
        for q in range(NP * NK):
            m.var_bounds(vs + q, BK.RA, 0.0, 1.0)
            m.var_type(vs + q, VAR_TYPE.INT_BIN)
        for q in range(NP * NK):
            m.var_bounds(vw + q, BK.FR, -math.inf, math.inf)
        for q in range(NP * NK * DIM):
            m.var_bounds(vd + q, BK.FR, -math.inf, math.inf)
        row = 0
        for i in range(NP):
            for j in range(NK):                     # W_ij - R_j + M*S_ij = M
                m.a_row(row, [vw + i * NK + j, vr + j, vs + i * NK + j], [1.0, -1.0, BIGM])
                m.con_bounds(row, BK.FX, BIGM, BIGM)
                row += 1
        for i in range(NP):
            for j in range(NK):                     # D_ijq + X_jq = P_iq
                for q in range(DIM):
                    m.a_row(row, [vd + (i * NK + j) * DIM + q, vx + j * DIM + q], [1.0, 1.0])
                    m.con_bounds(row, BK.FX, PTS[i][q], PTS[i][q])
                    row += 1
        for i in range(NP):                         # sum_j s_ij = 1
            m.a_row(row, [vs + i * NK + j for j in range(NK)], [1.0] * NK)
            m.con_bounds(row, BK.FX, 1.0, 1.0)
            row += 1
        for j in range(NK):                         # t >= R_j
            m.a_row(row, [vt, vr + j], [1.0, -1.0])
            m.con_bounds(row, BK.LO, 0.0, math.inf)
            row += 1
        for i in range(NP):
            for j in range(NK):                     # W_ij >= || D_ij ||
                d0 = vd + (i * NK + j) * DIM
                m.cone(CT.QUAD, [vw + i * NK + j, d0, d0 + 1])
        m.cj(vt, 1.0)
        m.sense(SENSE.MIN)
        r = m.solve()
    ok = r.rc == 0
    obj, rad, ctr = 0.0, [0.0] * NK, [0.0] * (NK * DIM)
    if ok:
        obj = r.x[vt]
        rad = [r.x[vr + j] for j in range(NK)]
        ctr = [r.x[vx + q] for q in range(NK * DIM)]
    covered = 1                                     # every point within R[j] of its ball center
    if ok:
        for i in range(NP):
            best = min(math.sqrt((PTS[i][0] - ctr[2 * j]) ** 2 + (PTS[i][1] - ctr[2 * j + 1]) ** 2) - rad[j]
                       for j in range(NK))
            if best > 1e-6:
                covered = 0
    okall = ok and covered and abs(obj - 0.5) < 1e-6
    print(f"facility_location  n={NP} k={NK}")
    print(f"  t* = {obj:.8f} (expected 0.5)   radii=({rad[0]:.4f},{rad[1]:.4f}) coverage ok={covered}")
    print("OK" if okall else "FAIL")
    return 0 if okall else 1


if __name__ == "__main__":
    raise SystemExit(main())
