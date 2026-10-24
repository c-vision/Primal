#!/usr/bin/env julia
# eco_driving_qp.jl — QP speed planning of a fuel-cell vehicle through
# signalized intersections (liuboer/MOSEK-ADMM, the upper level of the
# bi-level paper, Energy 2022). With accelerations a (dt = 1 s), v = Psi a and
# s = Psi v are prefix sums, and
#
#   min_a  m a'(Psi a) + m g rrc Phi'(Psi a) + 1/2 rho CD FA v_mean (Psi a)'(Psi a)
#   s.t.   acc_min <= a <= acc_max,  0 <= v <= spd_max (v_N <= 1),  s_lower <= s <= s_upper
#
# on a hand-built traffic-light corridor. The returned a is checked feasible,
# the objective recomputed directly, and compared with a feasible
# constant-speed profile (the QP must not be worse). Port of examples/c/eco_driving_qp.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const N = 10

function main()
    spd_max, acc_min, acc_max = 60.0 / 3.6, -2.0, 1.4   # vehicle parameters
    mass, rho, CD, FA, g, rrc = 1380.0, 1.2, 0.335, 2.0, 9.81, 0.009
    s_lo = [0.5, 1.5, 3.0, 5.0, 7.5, 10.0, 12.5, 15.0, 16.5, 17.0]   # traffic-light corridor
    s_up = [1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 20.0, 20.5, 21.0]
    v_mean = 17.0 / N

    # QP: minimize 1/2 a'Qa + c'a, Q = m(Psi + Psi') + rho CD FA v_mean Psi'Psi, c = m g rrc Psi'Phi
    Psi = [i <= k ? 1.0 : 0.0 for k in 1:N, i in 1:N]
    c = zeros(N)
    for i in 1:N, k in 1:N
        c[i] += mass * g * rrc * Psi[k, i]
    end
    Q = zeros(N, N)
    for i in 1:N, j in 1:N
        q = mass * (Psi[i, j] + Psi[j, i])
        PsitPsi = 0.0
        for k in 1:N
            PsitPsi += Psi[k, i] * Psi[k, j]        # (Psi'Psi)_ij
        end
        q += rho * CD * FA * v_mean * PsitPsi
        Q[i, j] = q
    end

    A, V, S, nv = 0, N, 2N, 3N                      # variables: a, v, s (linked by rows)
    m = Model(0, 0)
    r = try
        add_vars(m, nv)
        sense(m, SENSE_MIN)
        for k in 0:N-1
            var_bounds(m, A + k, BK_RA; lo=acc_min, up=acc_max)
            var_bounds(m, V + k, BK_LO; lo=0.0, up=k < N - 1 ? spd_max : 1.0)
            var_bounds(m, S + k, BK_RA; lo=s_lo[k+1], up=s_up[k+1])
        end
        add_cons(m, 2N)                             # v_k - sum_{i<=k} a_i = 0 ; s_k - sum_{j<=k} v_j = 0
        for k in 0:N-1
            a_row(m, k, [V + k; [A + i for i in 0:k]], [1.0; fill(-1.0, k + 1)])
            con_bounds(m, k, BK_FX; lo=0.0, up=0.0)
        end
        for k in 0:N-1
            a_row(m, N + k, [S + k; [V + j for j in 0:k]], [1.0; fill(-1.0, k + 1)])
            con_bounds(m, N + k, BK_FX; lo=0.0, up=0.0)
        end
        for i in 0:N-1, j in 0:i                    # lower triangle, full x'Qx coefficient
            qobj_ij(m, i, j, Q[i+1, j+1])
        end
        for i in 0:N-1
            cj(m, i, c[i+1])
        end
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    a, v, s, obj = zeros(N), zeros(N), zeros(N), 0.0
    if ok
        a, v, s, obj = r.x[A+1:A+N], r.x[V+1:V+N], r.x[S+1:S+N], r.objective
    end

    # independent checks: the constraints and the direct objective
    worst = -1e300
    vv, ss = zeros(N), zeros(N)
    av, sv = 0.0, 0.0
    for k in 1:N
        av += a[k]; vv[k] = av; sv += av; ss[k] = sv
        worst = max(worst, a[k] - acc_max); worst = max(worst, acc_min - a[k])
        worst = max(worst, k < N ? vv[k] - spd_max : vv[k] - 1.0); worst = max(worst, -vv[k])
        worst = max(worst, ss[k] - s_up[k]); worst = max(worst, s_lo[k] - ss[k])
    end
    direct = 0.0
    for k in 1:N
        direct += mass * a[k] * vv[k]               # m a'(Psi a)
    end
    for k in 1:N
        direct += mass * g * rrc * vv[k]            # m g rrc sum v
    end
    for k in 1:N
        direct += 0.5 * rho * CD * FA * v_mean * vv[k] * vv[k]
    end
    # feasible comparison profile: the midpoint of the corridor
    vc = [0.5 * (s_lo[k] + s_up[k]) - (k > 1 ? 0.5 * (s_lo[k-1] + s_up[k-1]) : 0.0) for k in 1:N]
    ac = [vc[k] - (k > 1 ? vc[k-1] : 0.0) for k in 1:N]
    sc = zeros(N)
    for k in 1:N
        svc = 0.0
        for j in 1:k
            svc += vc[j]
        end
        sc[k] = svc
    end
    consFeas, cons = true, 0.0
    for k in 1:N
        (ac[k] > acc_max + 1e-9 || ac[k] < acc_min - 1e-9) && (consFeas = false)
        (vc[k] < -1e-9 || (k < N && vc[k] > spd_max + 1e-9) || (k == N && vc[k] > 1.0 + 1e-9)) && (consFeas = false)
        (sc[k] < s_lo[k] - 1e-9 || sc[k] > s_up[k] + 1e-9) && (consFeas = false)
        cons += mass * ac[k] * vc[k] + mass * g * rrc * vc[k] + 0.5 * rho * CD * FA * v_mean * vc[k] * vc[k]
    end
    good = ok && worst < 1e-6 && abs(direct - obj) < 1e-6 * (1.0 + abs(obj)) && (!consFeas || obj <= cons + 1e-6)
    @printf("eco_driving_qp  N=%d  obj=%.6f  worst_viol=%.2e  constant=%.6f (feasible=%d)\n", N, obj, worst, cons, consFeas)
    @printf("  a=(%.4f..%.4f)  v=(%.4f..%.4f)  s=(%.4f,%.4f,..,%.4f)  %s\n",
            a[1], a[N], v[1], v[N], s[1], s[2], s[N], good ? "OK" : "FAIL")
    return good ? 0 : 1
end

exit(main())
