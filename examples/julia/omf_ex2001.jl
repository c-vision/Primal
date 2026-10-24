#!/usr/bin/env julia
# omf_ex2001.jl — Example 20.1 of "Optimization Methods in Finance"
# (Cornuejols & Tutuncu, 2007), Section 20.4: mean-variance portfolio with a
# tracking-error constraint against the 50/50 benchmark:
#
#   maximize mu'x  s.t.  TE(x) <= 0.10,  x1 + x2 + x3 = 1,  x >= 0
#
# TE = ||L'(x1 - 0.5, x2 - 0.5)||_2 with L the Cholesky factor of the 2x2
# covariance block: the quadratic cone (0.10, r1, r2). The book: mu = (6,4,0)
# -> (0.831, 0.169, 0), objective 5.662; mu = (5,5,0) -> objective 5.0.
# Port of examples/c/omf_ex2001.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# 2x2 covariance block and its Cholesky factor (Sigma2 = L L')
const S11, S12, S22 = 0.1764, 0.09702, 0.1089
const l11 = sqrt(S11)
const l21 = S12 / l11
const l22 = sqrt(S22 - l21 * l21)

# Solve one tracking-error SOCP case and check the book optimum.
function solve_case(name, mu1, mu2, want_obj, wx1, wx2)
    m = Model(3, 6)                                 # x1 x2 x3 r1 r2 t(=0.10); rows budget, r1, r2
    r = try
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 1, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 2, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, 3, BK_FR; lo=-Inf, up=Inf)    # r1
        var_bounds(m, 4, BK_FR; lo=-Inf, up=Inf)    # r2
        var_bounds(m, 5, BK_FX; lo=0.10, up=0.10)   # TE budget
        cj(m, 0, -mu1); cj(m, 1, -mu2)              # maximize mu'x
        a_row(m, 0, [0, 1, 2], [1.0, 1.0, 1.0])    # budget
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        # r1 = l11 (x1-0.5) + l21 (x2-0.5)  ->  r1 - l11 x1 - l21 x2 = -0.5(l11+l21)
        a_row(m, 1, [0, 1, 3], [-l11, -l21, 1.0])
        con_bounds(m, 1, BK_FX; lo=-0.5 * (l11 + l21), up=-0.5 * (l11 + l21))
        # r2 = l22 (x2-0.5)  ->  r2 - l22 x2 = -0.5 l22
        a_row(m, 2, [1, 4], [-l22, 1.0])
        con_bounds(m, 2, BK_FX; lo=-0.5 * l22, up=-0.5 * l22)
        cone(m, CT_QUAD, [5, 3, 4])                 # t >= ||(r1, r2)||
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        z, x = r.objective, r.x
        d1, d2 = x[1] - 0.5, x[2] - 0.5
        te = sqrt(S11 * d1 * d1 + 2.0 * S12 * d1 * d2 + S22 * d2 * d2)
        # wx1 < 0: a continuum of optima (the book says so), check only the
        # objective and the constraints, not a specific point
        xcheck = wx1 < 0.0 || (abs(x[1] - wx1) < 2e-3 && abs(x[2] - wx2) < 2e-3)
        ok = abs(z + want_obj) < 1e-3 && xcheck && te <= 0.10 + 1e-6 && abs(x[1] + x[2] + x[3] - 1.0) < 1e-6
        @printf("omf_ex2001 %-10s x=(%.4f,%.4f,%.4f) TE=%.4f obj=%.4f (book %.4f) %s\n",
                name, x[1], x[2], x[3], te, -z, want_obj, ok ? "OK" : "FAIL")
    else
        @printf("omf_ex2001 %-10s rc=%d FAIL\n", name, r.rc)
    end
    return ok
end

function main()
    a = solve_case("mu=(6,4,0)", 6.0, 4.0, 5.662, 0.831, 0.169)
    b = solve_case("mu=(5,5,0)", 5.0, 5.0, 5.000, -1.0, -1.0)   # continuum of optima
    return (a && b) ? 0 : 1
end

exit(main())
