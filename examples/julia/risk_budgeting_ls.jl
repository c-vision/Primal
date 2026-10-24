#!/usr/bin/env julia
# risk_budgeting_ls.jl — LONG-SHORT risk budgeting, mixed-integer conic (MOSEK
# Portfolio Optimization Cookbook, ch9):
#
#   minimize s - a sum_i b_i t_i
#   s.t.  s >= 1/2 x'Sx  ((s, 1, G'x) in Qr),  t_i <= log|x_i|  ((|x_i|, 1, t_i) in K_exp)
#         x = xp - xm,  xp <= M yp,  xm <= M ym,  yp + ym <= 1,  y binary
#
# The binaries keep |x_i| = xp_i + xm_i exact (without them the log term is
# unbounded). Hand case S = I, a = 1, b = (1/2, 1/2): |x_i| = 1/sqrt(2),
# obj = 0.5 + 0.34657359. Port of examples/c/risk_budgeting_ls.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# Solve for N assets with S = G G' (G row-major), budgets b, aversion a;
# returns (ok, objective, |x|).
function rb_ls(N, G, b, a)
    # var layout: x(N) xp(N) xm(N) yp(N) ym(N) t(N) s(1) z(N) u(N) ONE(1)
    X0, XP, XM, YP, YM, T0, S0 = 0, N, 2N, 3N, 4N, 5N, 6N
    Z0, U0, ONE = 6N + 1, 7N + 1, 8N + 1
    nv = ONE + 1
    m = Model(0, 0)
    r = try
        add_vars(m, nv)
        sense(m, SENSE_MIN)
        for i in 0:N-1
            var_bounds(m, X0 + i, BK_FR; lo=-Inf, up=Inf)
            var_bounds(m, XP + i, BK_LO; lo=0.0, up=Inf)
            var_bounds(m, XM + i, BK_LO; lo=0.0, up=Inf)
            var_type(m, YP + i, VT_INT)             # binary via [0,1] + INT
            var_bounds(m, YP + i, BK_RA; lo=0.0, up=1.0)
            var_type(m, YM + i, VT_INT)
            var_bounds(m, YM + i, BK_RA; lo=0.0, up=1.0)
            var_bounds(m, T0 + i, BK_FR; lo=-Inf, up=Inf)
            var_bounds(m, Z0 + i, BK_FR; lo=-Inf, up=Inf)
            var_bounds(m, U0 + i, BK_LO; lo=0.0, up=Inf)
        end
        var_bounds(m, S0, BK_FR; lo=-Inf, up=Inf)
        var_bounds(m, ONE, BK_FX; lo=1.0, up=1.0)
        add_cons(m, 6N)
        bigM = 20.0
        for i in 0:N-1
            a_row(m, i, [X0 + i, XP + i, XM + i], [1.0, -1.0, 1.0])
            con_bounds(m, i, BK_FX; lo=0.0, up=0.0)               # x - xp + xm = 0
            a_row(m, N + i, [XP + i, YP + i], [1.0, -bigM])
            con_bounds(m, N + i, BK_UP; lo=-Inf, up=0.0)          # xp <= M yp
            a_row(m, 2N + i, [XM + i, YM + i], [1.0, -bigM])
            con_bounds(m, 2N + i, BK_UP; lo=-Inf, up=0.0)         # xm <= M ym
            a_row(m, 3N + i, [YP + i, YM + i], [1.0, 1.0])
            con_bounds(m, 3N + i, BK_UP; lo=-Inf, up=1.0)         # yp + ym <= 1
            sub, val = [Z0 + i], [1.0]                            # z_i - (G'x)_i = 0
            for k in 0:N-1
                G[k*N+i+1] != 0.0 && (push!(sub, X0 + k); push!(val, -G[k*N+i+1]))
            end
            a_row(m, 4N + i, sub, val)
            con_bounds(m, 4N + i, BK_FX; lo=0.0, up=0.0)
            a_row(m, 5N + i, [U0 + i, XP + i, XM + i], [1.0, -1.0, -1.0])
            con_bounds(m, 5N + i, BK_FX; lo=0.0, up=0.0)          # u - xp - xm = 0
        end
        cone(m, CT_RQUAD, [S0; ONE; collect(Z0:Z0+N-1)])         # s >= ||z||^2/2 = 1/2 x'Sx
        for i in 0:N-1                              # t_i <= log(u_i) = log|x_i|
            cone(m, CT_PEXP, [U0 + i, ONE, T0 + i])
        end
        cj(m, S0, 1.0)
        for i in 0:N-1
            cj(m, T0 + i, -a * b[i+1])
        end
        solve(m)
    finally
        free(m)
    end
    r.rc == 0 || return false, -1.0, zeros(N)
    return true, r.objective, [abs(r.x[X0+i+1]) for i in 0:N-1]
end

function main()
    println("risk_budgeting_ls (long-short MIO)")
    allok = true
    N = 2                                           # hand case N=2, G=I, a=1, b=(1/2,1/2)
    ok, obj, xabs = rb_ls(N, [1.0, 0.0, 0.0, 1.0], [0.5, 0.5], 1.0)
    want = 0.5 + 0.34657359027997264                # 0.5 - log(1/sqrt2)
    xw = 1.0 / sqrt(2.0)
    good = ok && abs(obj - want) < 1e-4 && abs(xabs[1] - xw) < 1e-3 && abs(xabs[2] - xw) < 1e-3
    @printf("  N=2 S=I: obj=%.6f (expected %.6f)  |x|=(%.6f,%.6f) (expected %.6f)  %s\n",
            obj, want, xabs[1], xabs[2], xw, good ? "OK" : "FAIL")
    allok &= good
    @printf("%s\n", allok ? "OK" : "FAIL")
    return allok ? 0 : 1
end

exit(main())
