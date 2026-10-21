#!/usr/bin/env python3
"""PrimalSolver - worst-case CVaR over a box of return uncertainty (LP).

Port of c_examples/finance/robust_cvar.c. CVaR at level beta (Rockafellar-
Uryasev) is an LP in (w, alpha, z). If each return may deviate in a box
r_s +/- d_s and w >= 0, the worst case is the low corner r_s - d_s, so the robust
CVaR is the same LP on the shifted returns.

Hand case: 2 equiprobable scenarios, beta=1/2, nominal r0=(0.10,0.20),
r1=(0.30,0.00), box d=(0.05,0.05): robust CVaR = -0.10 at w=(1/2,1/2), nominal
CVaR = -0.15 (robustness costs 0.05).

    python python_examples/robust_cvar.py
"""

from primalsolver import Model, BK, SOLSTA

R0 = [0.10, 0.20]
R1 = [0.30, 0.00]
DBOX = [0.05, 0.05]


def solve(robust: bool):
    r0 = [R0[j] - (DBOX[j] if robust else 0.0) for j in range(2)]
    r1 = [R1[j] - (DBOX[j] if robust else 0.0) for j in range(2)]
    with Model(maxcon=3, maxvar=5) as m:     # w0, w1, alpha, z0, z1
        for j in (0, 1, 3, 4):
            m.var_bounds(j, BK.LO, 0.0)
        m.var_bounds(2, BK.FR)
        m.cj(2, 1.0); m.cj(3, 1.0); m.cj(4, 1.0)   # minimize alpha + z0 + z1
        m.a_row(0, [3, 0, 1, 2], [1.0, r0[0], r0[1], 1.0])
        m.con_bounds(0, BK.LO, 0.0)
        m.a_row(1, [4, 0, 1, 2], [1.0, r1[0], r1[1], 1.0])
        m.con_bounds(1, BK.LO, 0.0)
        m.a_row(2, [0, 1], [1.0, 1.0])
        m.con_bounds(2, BK.FX, 1.0, 1.0)
        r = m.solve()
    return r.objective, r.x[0], r.x[1], r.solsta


def main() -> int:
    nom, wn0, wn1, _ = solve(False)
    rob, wr0, wr1, sta = solve(True)
    ok = (sta == SOLSTA.OPTIMAL and abs(rob + 0.10) < 1e-7
          and abs(wr0 - 0.5) < 1e-6 and abs(wr1 - 0.5) < 1e-6
          and abs(nom + 0.15) < 1e-7 and rob >= nom - 1e-9)
    print(f"nominal CVaR = {nom:.6f} (-0.15), w=({wn0:.4f},{wn1:.4f})")
    print(f"robust  CVaR = {rob:.6f} (-0.10), w=({wr0:.4f},{wr1:.4f}) "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
