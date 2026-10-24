#!/usr/bin/env julia
# mle_density.jl — MLE of a log-concave density via exponential cones
# (MOSEK/Tutorials, mle-convex-density-function). On a grid y_i with
# spacings dy_i, estimate x_i = g(y_i) of a concave g = log f:
#
#   minimize sum_i weight_i u_i
#   s.t.  u_i >= -log(x_i)  (PEXP cone (x_i, 1, -u_i)),  x concave,
#         sum_i ((x_i + x_{i+1})/2) dy_i = 1
#
# Instance n = 3, dy = 1/2, weights (1, 2, 3): KKT gives x = (2/3, 2/3, 2),
# objective -3 log(4/3) = -0.8630462174. Port of examples/c/mle_density.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const ND = 3

function main()
    dy = [0.5, 0.5, 0.5]
    w = [1.0, 2.0, 3.0]
    X, U, NNEG, ONE, NV = 0, ND, 2ND, 3ND, 3ND + 1
    m = Model(0, 0)
    r = try
        add_vars(m, NV)
        add_cons(m, (ND - 2) + 1 + ND)              # concavity + normalization + links
        for i in 0:ND-1
            var_bounds(m, X + i, BK_FR; lo=-Inf, up=Inf)
        end
        for i in 0:ND-1
            var_bounds(m, U + i, BK_FR; lo=-Inf, up=Inf)
        end
        for i in 0:ND-1
            var_bounds(m, NNEG + i, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(m, ONE, BK_FX; lo=1.0, up=1.0)
        row = 0
        for i in 0:ND-3                             # concavity of x
            a_row(m, row, [X + i, X + i + 1, X + i + 2], [-dy[i+2], dy[i+1] + dy[i+2], -dy[i+1]])
            con_bounds(m, row, BK_UP; lo=-Inf, up=0.0); row += 1
        end
        val = zeros(ND)                             # normalization: sum ((x_i+x_{i+1})/2) dy_i = 1
        for i in 1:ND-1
            val[i] += 0.5 * dy[i]
            val[i+1] += 0.5 * dy[i]
        end
        a_row(m, row, [X + i for i in 0:ND-1], val)
        con_bounds(m, row, BK_FX; lo=1.0, up=1.0); row += 1
        for i in 0:ND-1                             # -u_i - N_i = 0
            a_row(m, row, [NNEG + i, U + i], [1.0, 1.0])
            con_bounds(m, row, BK_FX; lo=0.0, up=0.0); row += 1
        end
        for i in 0:ND-1
            cone(m, CT_PEXP, [X + i, ONE, NNEG + i])
        end
        for i in 0:ND-1
            cj(m, U + i, w[i+1])
        end
        sense(m, SENSE_MIN)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    obj, xv = 0.0, zeros(ND)
    if ok
        x = r.x
        xv = [x[X+i+1] for i in 0:ND-1]
        for i in 1:ND
            obj += w[i] * x[U+i]
        end
    end
    want = -3.0 * ccall(:log, Cdouble, (Cdouble,), 4.0 / 3.0)
    okall = ok && abs(xv[1] - 2.0 / 3.0) < 2e-3 && abs(xv[2] - 2.0 / 3.0) < 2e-3 &&
            abs(xv[3] - 2.0) < 2e-3 && abs(obj - want) < 1e-6
    @printf("mle_density  n=%d\n", ND)
    @printf("  x = (%.8f, %.8f, %.8f)   (expected 2/3, 2/3, 2)\n", xv[1], xv[2], xv[3])
    @printf("  objective = %.10f   (expected %.10f)\n", obj, want)
    @printf("%s\n", okall ? "OK" : "FAIL")
    return okall ? 0 : 1
end

exit(main())
