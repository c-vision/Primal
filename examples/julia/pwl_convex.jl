#!/usr/bin/env julia
# pwl_convex.jl — least-squares fit by a convex piecewise-linear function
# (MOSEK/Tutorials, pwl-convex-approximation):
#
#   minimize m  s.t.  m >= sum_i (t_i - Y_i)^2  (rotated cone (1/2, m, t - Y)),
#                     t_i >= t_j + <s_j, X_i - X_j>  for all i != j  (convexity)
#
# Instance d = 1, X = (0, 1, 2), Y = (0, 1, 0): t = (1/3, 1/3, 1/3), m = 2/3.
# Port of examples/c/pwl_convex.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NP = 3
const XD = [0.0, 1.0, 2.0]
const YD = [0.0, 1.0, 0.0]

function main()
    M, T, S, AD, HALF = 0, 1, 1 + NP, 1 + 2NP, 1 + 3NP
    NV = 1 + 3NP + 1
    m = Model(0, 0)
    try
        add_vars(m, NV)
        add_cons(m, NP + NP * (NP - 1))             # auxDiff + off-diagonal convexity
        var_bounds(m, M, BK_LO; lo=0.0, up=Inf)
        for i in 0:NP-1
            var_bounds(m, T + i, BK_FR; lo=-Inf, up=Inf)
            var_bounds(m, S + i, BK_FR; lo=-Inf, up=Inf)
            var_bounds(m, AD + i, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(m, HALF, BK_FX; lo=0.5, up=0.5)
        row = 0
        for i in 0:NP-1                             # auxDiff_i = t_i - Y_i
            a_row(m, row, [AD + i, T + i], [1.0, -1.0])
            con_bounds(m, row, BK_FX; lo=-YD[i+1], up=-YD[i+1])
            row += 1
        end
        for i in 0:NP-1, j in 0:NP-1                # t_j - t_i + s_j(X_i - X_j) <= 0
            i == j && continue                      # the i=j row is the trivial 0<=0
            a_row(m, row, [T + j, T + i, S + j], [1.0, -1.0, XD[i+1] - XD[j+1]])
            con_bounds(m, row, BK_UP; lo=-Inf, up=0.0)
            row += 1
        end
        cone(m, CT_RQUAD, [HALF, M, AD, AD + 1, AD + 2])   # 2 HALF M >= sum AD_i^2
        cj(m, M, 1.0)
        sense(m, SENSE_MIN)
        r = solve(m)
        ok = r.rc == 0
        mv, tv = 0.0, zeros(NP)
        if ok
            mv = r.x[M+1]
            tv = [r.x[T+i+1] for i in 0:NP-1]
        end
        sse = 0.0
        for i in 1:NP
            sse += (tv[i] - YD[i]) * (tv[i] - YD[i])
        end
        okall = ok && abs(mv - 2.0 / 3.0) < 1e-7 && abs(tv[1] - 1.0 / 3.0) < 1e-6 &&
                abs(tv[2] - 1.0 / 3.0) < 1e-6 && abs(tv[3] - 1.0 / 3.0) < 1e-6 && abs(mv - sse) < 1e-7
        @printf("pwl_convex  n=%d\n", NP)
        @printf("  t = (%.8f, %.8f, %.8f)   (expected 1/3,1/3,1/3)\n", tv[1], tv[2], tv[3])
        @printf("  m = SSE = %.8f   (expected 2/3=%.8f)   direct sse=%.8f\n", mv, 2.0 / 3.0, sse)
        @printf("%s\n", okall ? "OK" : "FAIL")
        return okall ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
