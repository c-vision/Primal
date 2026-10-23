#!/usr/bin/env python3
"""QP speed planning of a fuel-cell vehicle through signalized intersections
(liuboer/MOSEK-ADMM, Eco_MOSEK/MOSEK_main.m, the upper level of the bi-level paper
Energy 2022, doi 10.1016/j.energy.2022.123956).

With a the vector of accelerations (dt = 1 s), speed and position are prefix sums
v = Psi a and s = Psi (Psi a), and the upper level solves

    min_a   m a'(Psi a) + m g rrc Phi'(Psi a) + 1/2 rho CD FA v_mean (Psi a)'(Psi a)
    s.t.    acc_min <= a_k <= acc_max,   0 <= v_k <= spd_max (k < N),  v_N <= 1
            s_lower_k <= s_k <= s_upper_k           (traffic-light corridor)

This sample uses a hand-built corridor on a single short route. Verification: the
returned a is checked feasible against every original constraint, its objective is
recomputed directly, and it must not be worse than a feasible constant-speed profile.
Port of examples/c/eco_driving_qp.c.
"""

import math

from primalsolver import Model, BK, SENSE

N = 10
SPD_MAX, ACC_MIN, ACC_MAX = 60.0 / 3.6, -2.0, 1.4
MASS, RHO, CD, FA, G, RRC = 1380.0, 1.2, 0.335, 2.0, 9.81, 0.009
S_LO = [0.5, 1.5, 3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 16.5, 17.0]
S_UP = [1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 20.0, 20.5, 21.0]


def main() -> int:
    v_mean = 17.0 / N
    # QP: minimize 1/2 a'Q a + c'a, Q = m(Psi+Psi') + rho CD FA v_mean Psi'Psi, c = m g rrc Psi'Phi
    psi = [[1.0 if i <= k else 0.0 for i in range(N)] for k in range(N)]
    c = []
    for i in range(N):
        acc = 0.0
        for k in range(N):
            acc += MASS * G * RRC * psi[k][i]
        c.append(acc)
    q = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            ptp = 0.0
            for k in range(N):
                ptp += psi[k][i] * psi[k][j]        # (Psi'Psi)_ij
            q[i][j] = MASS * (psi[i][j] + psi[j][i]) + RHO * CD * FA * v_mean * ptp
    av, vv_, sv_ = 0, N, 2 * N                      # variables: a, v, s (linked by rows)
    with Model() as m:
        m.add_vars(3 * N)
        m.sense(SENSE.MIN)
        for k in range(N):
            m.var_bounds(av + k, BK.RA, ACC_MIN, ACC_MAX)
            m.var_bounds(vv_ + k, BK.LO, 0.0, SPD_MAX if k < N - 1 else 1.0)
            m.var_bounds(sv_ + k, BK.RA, S_LO[k], S_UP[k])
        m.add_cons(2 * N)
        for k in range(N):                          # v_k - sum_{i<=k} a_i = 0
            m.a_row(k, [vv_ + k] + [av + i for i in range(k + 1)], [1.0] + [-1.0] * (k + 1))
            m.con_bounds(k, BK.FX, 0.0, 0.0)
        for k in range(N):                          # s_k - sum_{j<=k} v_j = 0
            m.a_row(N + k, [sv_ + k] + [vv_ + j for j in range(k + 1)], [1.0] + [-1.0] * (k + 1))
            m.con_bounds(N + k, BK.FX, 0.0, 0.0)
        for i in range(N):                          # lower triangle, full x'Qx coefficient
            for j in range(i + 1):
                m.qobj_ij(i, j, q[i][j])
        for i in range(N):
            m.cj(i, c[i])
        r = m.solve()
    ok = r.rc == 0
    a, v, s, obj = [0.0] * N, [0.0] * N, [0.0] * N, 0.0
    if ok:
        a = r.x[av:av + N]
        v = r.x[vv_:vv_ + N]
        s = r.x[sv_:sv_ + N]
        obj = r.objective

    worst = -1e300                                  # independent checks: constraints, objective
    vv, ss = [0.0] * N, [0.0] * N
    acc_v = acc_s = 0.0
    for k in range(N):
        acc_v += a[k]
        vv[k] = acc_v
        acc_s += acc_v
        ss[k] = acc_s
        worst = max(worst, a[k] - ACC_MAX, ACC_MIN - a[k],
                    vv[k] - SPD_MAX if k < N - 1 else vv[k] - 1.0, -vv[k],
                    ss[k] - S_UP[k], S_LO[k] - ss[k])
    direct = 0.0
    for k in range(N):
        direct += MASS * a[k] * vv[k]               # m a'(Psi a)
    for k in range(N):
        direct += MASS * G * RRC * vv[k]            # m g rrc sum v
    for k in range(N):
        direct += 0.5 * RHO * CD * FA * v_mean * vv[k] * vv[k]
    # feasible comparison profile: the midpoint of the corridor
    vc = [0.5 * (S_LO[k] + S_UP[k]) - (0.5 * (S_LO[k - 1] + S_UP[k - 1]) if k else 0.0) for k in range(N)]
    ac = [vc[k] - (vc[k - 1] if k else 0.0) for k in range(N)]
    sc = []
    for k in range(N):
        svc = 0.0
        for j in range(k + 1):
            svc += vc[j]
        sc.append(svc)
    cons_feas, cons = 1, 0.0
    for k in range(N):
        if ac[k] > ACC_MAX + 1e-9 or ac[k] < ACC_MIN - 1e-9:
            cons_feas = 0
        if vc[k] < -1e-9 or (k < N - 1 and vc[k] > SPD_MAX + 1e-9) or (k == N - 1 and vc[k] > 1.0 + 1e-9):
            cons_feas = 0
        if sc[k] < S_LO[k] - 1e-9 or sc[k] > S_UP[k] + 1e-9:
            cons_feas = 0
        cons += MASS * ac[k] * vc[k] + MASS * G * RRC * vc[k] + 0.5 * RHO * CD * FA * v_mean * vc[k] * vc[k]
    good = (ok and worst < 1e-6 and abs(direct - obj) < 1e-6 * (1.0 + abs(obj))
            and (not cons_feas or obj <= cons + 1e-6))
    print(f"eco_driving_qp  N={N}  obj={obj:.6f}  worst_viol={worst:.2e}  constant={cons:.6f} (feasible={cons_feas})")
    print(f"  a=({a[0]:.4f}..{a[N - 1]:.4f})  v=({v[0]:.4f}..{v[N - 1]:.4f})  "
          f"s=({s[0]:.4f},{s[1]:.4f},..,{s[N - 1]:.4f})  {'OK' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
