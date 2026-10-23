#!/usr/bin/env python3
"""k-means clustering as a mixed-integer conic program with a disjunction
(MOSEK Tutorials, kmeans-clustering).

    minimize    sum_i d_i
    subject to  dAux_ij >= || p_i - c_j ||^2        (rotated cone per i,j)
                d_i >= dAux_i,j  AND  y_i == j       for SOME j   (DJC per i)
                y_i in {0,..,K-1},  c_0 <= c_1

The "some j" is written with the DJC API (appendafes + putdjc), as the notebook's
M.disjunction over the K (distance, label) terms. Hand-derived instance: four
collinear points 0,1,2,3 and K=2 clusters -> centroids 0.5 and 2.5, inertia 1.
Port of examples/c/kmeans.c.
"""

import math

from primalsolver import Model, APPROX, BK, CT, SENSE, SolverError, VAR_TYPE

NPTS = 4
NK = 2
PT = [0.0, 1.0, 2.0, 3.0]
C = 0
DIST = NK
AUX = NK + NPTS
D3 = NK + NPTS + NPTS * NK
Y = NK + NPTS + 2 * NPTS * NK
HALF = Y + NPTS
NV = HALF + 1


def main() -> int:
    naux = NPTS * NK
    nafe = naux + NPTS                              # one per (i,j) distance + one label per i
    with Model() as m:
        m.add_vars(NV)
        m.add_cons(naux + 1)                        # D3 links + centroid ordering
        # Centroids are bound to the data range [0,3] (a k-means centroid is the mean
        # of the points it owns). DECLARED DEVIATION: the notebook leaves them unbounded.
        for j in range(NK):
            m.var_bounds(C + j, BK.RA, 0.0, 3.0)
        for i in range(NPTS):
            m.var_bounds(DIST + i, BK.LO, 0.0, math.inf)
        for q in range(naux):
            m.var_bounds(AUX + q, BK.LO, 0.0, math.inf)
        for q in range(naux):
            m.var_bounds(D3 + q, BK.FR, -math.inf, math.inf)
        for i in range(NPTS):
            m.var_bounds(Y + i, BK.RA, 0.0, float(NK - 1))
            m.var_type(Y + i, VAR_TYPE.INT)
        m.var_bounds(HALF, BK.FX, 0.5, 0.5)
        row = 0
        for i in range(NPTS):
            for j in range(NK):                     # D3_ij - C_j = -point_i
                m.a_row(row, [D3 + i * NK + j, C + j], [1.0, -1.0])
                m.con_bounds(row, BK.FX, -PT[i], -PT[i])
                row += 1
        m.a_row(row, [C, C + 1], [1.0, -1.0])       # c0 <= c1
        m.con_bounds(row, BK.UP, -math.inf, 0.0)
        for q in range(naux):                       # dAux_ij >= D3_ij^2
            m.cone(CT.RQUAD, [AUX + q, HALF, D3 + q])
        for i in range(NPTS):
            m.cj(DIST + i, 1.0)
        m.sense(SENSE.MIN)

        # disjunction: for each point, d_i >= dAux_ij AND y_i == j, for SOME j
        m.afes(nafe)
        for i in range(NPTS):
            for j in range(NK):                     # afe iK+j = d_i - dAux_ij
                a = i * NK + j
                m.afe_entry(a, DIST + i, 1.0)
                m.afe_entry(a, AUX + a, -1.0)
        for i in range(NPTS):                       # afe nAux+i = y_i
            m.afe_entry(naux + i, Y + i, 1.0)
        rplus = m.domain(APPROX.RPLUS, 1)
        rzero = m.domain(APPROX.RZERO, 1)
        m.djcs(NPTS)
        for i in range(NPTS):
            dom, afe, bb = [], [], []
            for j in range(NK):
                dom += [rplus, rzero]
                afe += [i * NK + j, naux + i]
                bb += [0.0, float(j)]
            try:
                m.djc(i, dom, afe, bb, [2] * NK)
            except SolverError as err:
                print(f"kmeans  putdjc {err} FAIL")
                return 1
        r = m.solve()
    ok = r.rc == 0
    obj, cen = 0.0, [0.0] * NK
    if ok:
        cen = [r.x[C], r.x[C + 1]]
        obj = sum(r.x[DIST + i] for i in range(NPTS))
    okall = ok and abs(obj - 1.0) < 1e-6
    print(f"kmeans  n={NPTS} K={NK}")
    print(f"  objective (inertia) = {obj:.8f} (expected 1.0)   centers = ({cen[0]:.4f}, {cen[1]:.4f})")
    print("OK" if okall else "FAIL")
    return 0 if okall else 1


if __name__ == "__main__":
    raise SystemExit(main())
