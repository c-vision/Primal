#!/usr/bin/env julia
# gp_toolbox.jl — a small geometric program (MOSEK Tutorials, gp-toolbox):
#
#   minimize x + y^2 z  s.t.  0.1 sqrt(x) + 2 y^-1 <= 1,  z^-1 + y x^-2 <= 1
#
# Under y = log x every posynomial becomes a log-sum-exp:
#   sum_i c_i exp(a_i.y) <= 1  <=>  p_i >= exp(a_i.y + ln c_i),  sum p_i <= 1,
# one exponential cone per monomial plus a linear row; the objective adds -t
# and minimizes t. Port of examples/c/gp_toolbox.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NV = 3
clog(x) = ccall(:log, Cdouble, (Cdouble,), x)      # the C library's log/exp
cexp(x) = ccall(:exp, Cdouble, (Cdouble,), x)

function main()
    # each monomial is (coef, [(var, exponent), ...]); each posynomial a list of them
    OBJ = [(1.0, [(0, 1.0)]), (1.0, [(1, 2.0), (2, 1.0)])]     # x + y^2 z
    C0 = [(0.1, [(0, 0.5)]), (2.0, [(1, -1.0)])]               # 0.1 sqrt(x) + 2/y
    C1 = [(1.0, [(2, -1.0)]), (1.0, [(0, -2.0), (1, 1.0)])]    # 1/z + y/x^2
    posys = [OBJ, C0, C1]
    isobj = [true, false, false]
    nposy, nmulti = 3, 6

    # vars: y[NV] | t | e_i (per monomial) | p_i (per monomial) | ONE
    YBASE, TV, EBASE = 0, NV, NV + 1
    PBASE = EBASE + nmulti
    ONEV = PBASE + nmulti
    NVAR = ONEV + 1
    m = Model(0, 0)
    r = try
        add_vars(m, NVAR)
        for i in 0:NV-1
            var_bounds(m, YBASE + i, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(m, TV, BK_FR; lo=-Inf, up=Inf)
        for i in 0:nmulti-1
            var_bounds(m, EBASE + i, BK_FR; lo=-Inf, up=Inf)
        end
        for i in 0:nmulti-1
            var_bounds(m, PBASE + i, BK_LO; lo=0.0, up=Inf)
        end
        var_bounds(m, ONEV, BK_FX; lo=1.0, up=1.0)
        add_cons(m, nposy + 2nmulti)                # one sum row per posynomial + one link row per monomial
        row, ecur, pcur = 0, 0, 0
        for i in 1:nposy
            p0 = pcur
            for (coef, terms) in posys[i]
                sub, val = [EBASE + ecur], [1.0]
                for (v, a) in terms
                    push!(sub, YBASE + v); push!(val, -a)
                end
                isobj[i] && (push!(sub, TV); push!(val, 1.0))
                a_row(m, row, sub, val)
                con_bounds(m, row, BK_FX; lo=clog(coef), up=clog(coef)); row += 1
                cone(m, CT_PEXP, [PBASE + pcur, ONEV, EBASE + ecur])
                ecur += 1; pcur += 1
            end
            nm = length(posys[i])
            a_row(m, row, [PBASE + p0 + q for q in 0:nm-1], ones(nm))
            con_bounds(m, row, BK_UP; lo=-Inf, up=1.0); row += 1
        end
        cj(m, TV, 1.0)
        sense(m, SENSE_MIN)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    x, topt = zeros(NV), 0.0
    if ok
        x = [cexp(r.x[YBASE+i+1]) for i in 0:NV-1]
        topt = r.x[TV+1]
    end
    # objective and the two constraint posynomials, by direct substitution
    obj = x[1] + x[2] * x[2] * x[3]
    c0 = 0.1 * sqrt(x[1]) + 2.0 / x[2]
    c1 = 1.0 / x[3] + x[2] / (x[1] * x[1])
    okall = ok && c0 < 1.0 + 1e-8 && c1 < 1.0 + 1e-8 && abs(obj - cexp(topt)) < 1e-7 * (1.0 + obj)
    @printf("gp_toolbox  x=%.6f y=%.6f z=%.6f  obj = %.6f\n", x[1], x[2], x[3], obj)
    @printf("  c0 = %.9f  c1 = %.9f  (both <= 1)   exp(t) = %.6f\n", c0, c1, cexp(topt))
    @printf("%s\n", okall ? "OK" : "FAIL")
    return okall ? 0 : 1
end

exit(main())
