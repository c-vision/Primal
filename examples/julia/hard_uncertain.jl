#!/usr/bin/env julia
# hard_uncertain.jl — robust optimization with a hard uncertain inequality,
# solved EXACTLY at the box vertices (MOSEK Tutorials, approx-uncertain-ineq):
#
#   minimize sum(x)  s.t.  sum_i exp((B_i x)'z_k - sum(x)) <= 1  for every vertex z_k
#
# each term an exponential cone: h1_i >= exp(h3_i), h2 = 1, sum_i h1_i <= 1.
# Instance L = 1 (vertices +-1), n = 1, p = 2, B = (1/2), (1/5): x* is the
# largest root of exp(-x/2) + exp(-4x/5) = 1, checked against a dense scan.
# Port of examples/c/hard_uncertain.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const L = 1
const NP = 2
const NVX = 1
const NVERT = 1 << L
const NH = NP * NVERT
const B = [0.5, 0.2]
const VERT = [-1.0, 1.0]

cexp(x) = ccall(:exp, Cdouble, (Cdouble,), x)      # the C library's exp

function main()
    X, H1, H3, ONE = 0, 1, 1 + NH, 1 + 2NH
    NV = ONE + 1
    m = Model(0, 0)
    r = try
        add_vars(m, NV)
        add_cons(m, NH + NVERT)
        var_bounds(m, X, BK_FR; lo=-Inf, up=Inf)
        for q in 0:NH-1
            var_bounds(m, H1 + q, BK_LO; lo=0.0, up=Inf)
        end
        for q in 0:NH-1
            var_bounds(m, H3 + q, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(m, ONE, BK_FX; lo=1.0, up=1.0)
        row = 0
        for i in 0:NP-1, k in 0:NVERT-1             # h3_ik - (B_i z_k - 1) x = 0
            coef = B[i+1] * VERT[k+1] - 1.0
            a_row(m, row, [H3 + i * NVERT + k, X], [1.0, -coef])
            con_bounds(m, row, BK_FX; lo=0.0, up=0.0); row += 1
        end
        for k in 0:NVERT-1                          # sum_i h1_ik <= 1
            a_row(m, row, [H1 + i * NVERT + k for i in 0:NP-1], ones(NP))
            con_bounds(m, row, BK_UP; lo=-Inf, up=1.0); row += 1
        end
        for q in 0:NH-1
            cone(m, CT_PEXP, [H1 + q, ONE, H3 + q])
        end
        cj(m, X, 1.0)
        sense(m, SENSE_MIN)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    x = ok ? r.x[X+1] : 0.0

    xstar, found = 0.0, false                       # reference: the smallest x where every vertex holds
    for q in -400000:400000
        xx = q * 1e-5
        feas = true
        for k in 1:NVERT
            s = 0.0
            for i in 1:NP
                s += cexp(B[i] * VERT[k] * xx - xx)
            end
            s > 1.0 && (feas = false; break)
        end
        if feas
            xstar, found = xx, true
            break
        end
    end
    okall = ok && found && abs(x - xstar) < 1e-4 * (1.0 + abs(xstar))
    @printf("hard_uncertain  L=%d n=%d p=%d\n", L, NVX, NP)
    @printf("  x* = %.8f   [scan %.8f]\n", x, xstar)
    @printf("%s\n", okall ? "OK" : "FAIL")
    return okall ? 0 : 1
end

exit(main())
