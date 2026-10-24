#!/usr/bin/env julia
# dist_robust.jl — distributionally robust portfolio (Wasserstein DRO), an LP
# (MOSEK/Tutorials, dist-robust-portfolio):
#
#   minimize eps lambda + (1/N) sum_i s_i
#   s.t.  b_k t + a_k (data_i . x) <= s_i  (all i, k),  |a_k x_j| <= lambda,  sum x = 1,  x >= 0
#
# with the two loss pieces a = (-1, -51), b = (10, -40) of the notebook.
# Checked against a brute force over the weight simplex with t on a grid.
# Port of examples/c/dist_robust.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const M = 2
const NS = 3
const AK = (-1.0, -51.0)
const BK = (10.0, -40.0)
const DATA = [0.02 0.01; -0.01 0.03; 0.04 -0.02]
const EPS = 0.05

# Worst-case DRO objective at weights x with auxiliary t.
function worst_objective(x, t)
    lambda, ssum = 0.0, 0.0
    for j in 1:M
        v0, v1 = abs(AK[1] * x[j]), abs(AK[2] * x[j])
        v0 > lambda && (lambda = v0)
        v1 > lambda && (lambda = v1)
    end
    for i in 1:NS
        d = 0.0
        for j in 1:M
            d += DATA[i, j] * x[j]
        end
        s0, s1 = BK[1] * t + AK[1] * d, BK[2] * t + AK[2] * d
        ssum += s0 > s1 ? s0 : s1
    end
    return EPS * lambda + ssum / NS
end

function main()
    X, S, LAM, TT, NV = 0, M, M + NS, M + NS + 1, M + NS + 2
    m = Model(0, 0)
    r = try
        add_vars(m, NV)
        add_cons(m, 2NS + 4M + 1)
        for j in 0:M-1
            var_bounds(m, X + j, BK_LO; lo=0.0, up=Inf)
        end
        for i in 0:NS-1
            var_bounds(m, S + i, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(m, LAM, BK_FR; lo=-Inf, up=Inf)
        var_bounds(m, TT, BK_FR; lo=-Inf, up=Inf)
        row = 0
        for i in 0:NS-1, k in 1:2                   # C1 for both k
            a_row(m, row, [TT; [X + j for j in 0:M-1]; S + i], [BK[k]; [AK[k] * DATA[i+1, j+1] for j in 0:M-1]; -1.0])
            con_bounds(m, row, BK_UP; lo=-Inf, up=0.0); row += 1
        end
        for j in 0:M-1, k in 1:2                    # C2 pos/neg
            a_row(m, row, [X + j, LAM], [AK[k], -1.0])
            con_bounds(m, row, BK_UP; lo=-Inf, up=0.0); row += 1
            a_row(m, row, [X + j, LAM], [-AK[k], -1.0])
            con_bounds(m, row, BK_UP; lo=-Inf, up=0.0); row += 1
        end
        a_row(m, row, [X + j for j in 0:M-1], ones(M))
        con_bounds(m, row, BK_FX; lo=1.0, up=1.0); row += 1
        cj(m, LAM, EPS)
        for i in 0:NS-1
            cj(m, S + i, 1.0 / NS)
        end
        sense(m, SENSE_MIN)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    obj, x = 0.0, zeros(M)
    if ok
        xx = r.x
        x = xx[X+1:X+M]
        obj = EPS * xx[LAM+1]
        for i in 0:NS-1
            obj += xx[S+i+1] / NS
        end
    end

    best = 1e300                                    # brute force: x = (w, 1-w), t on a grid
    for a in 0:4000
        w = a / 4000.0
        xr = (w, 1.0 - w)
        for b in -2000:2000
            f = worst_objective(xr, b * 0.01)
            f < best && (best = f)
        end
    end
    okall = ok && abs(obj - best) < 2e-3 * (1.0 + abs(best)) && abs(x[1] + x[2] - 1.0) < 1e-7
    @printf("dist_robust  m=%d N=%d eps=%.2f\n", M, NS, EPS)
    @printf("  x = (%.8f, %.8f)   LP objective = %.8f\n", x[1], x[2], obj)
    @printf("  brute force = %.8f\n", best)
    @printf("%s\n", okall ? "OK" : "FAIL")
    return okall ? 0 : 1
end

exit(main())
