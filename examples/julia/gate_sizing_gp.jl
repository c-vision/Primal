#!/usr/bin/env julia
# gate_sizing_gp.jl — gate sizing of a digital circuit as a geometric program
# (lrakai/gate-sizing, from Boyd's "A Tutorial on Geometric Programming").
# Gate i's delay is the posynomial gamma_i s_i^-1 sum_{j driven by i}
# (beta_j s_j + alpha_j); minimise the critical path D subject to the timing
# chain and the total area sum_i a_i s_i <= Amax. Under y = log x each
# posynomial becomes exponential cones plus a linear row (as gp_toolbox).
#
# Hand case: two inverters in a chain driving a load of 10, Amax = 10:
# D(s) = (s2 + 1)/s1 + 10/s2, s2 = 10 sqrt(10)/(sqrt(11) + sqrt(10)),
# D* = 3.19761770. Port of examples/c/gate_sizing_gp.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NV = 6                                        # s1 s2 T1 T2 V2 D
const S1, S2, T1, T2, V2, D = 0:5
clog(x) = ccall(:log, Cdouble, (Cdouble,), x)      # the C library's log/exp
cexp(x) = ccall(:exp, Cdouble, (Cdouble,), x)

function main()
    Amax = 10.0
    # posynomials: each monomial is (coef, [(var, exponent), ...])
    P = [[(1.0, [(S2, 1.0), (S1, -1.0), (T1, -1.0)]), (1.0, [(S1, -1.0), (T1, -1.0)])],
         [(10.0, [(S2, -1.0), (T2, -1.0)]), (1.0, [(V2, 1.0), (T2, -1.0)])],
         [(1.0, [(T1, 1.0), (V2, -1.0)])],
         [(1.0, [(T2, 1.0), (D, -1.0)])],
         [(1.0 / Amax, [(S1, 1.0)]), (1.0 / Amax, [(S2, 1.0)])]]
    NPF = length(P)
    nmulti = sum(length(p) for p in P if length(p) > 1)
    NRQ = count(p -> length(p) == 1, P)
    YBASE, EBASE = 0, NV
    PBASE = EBASE + nmulti
    ONEV = PBASE + nmulti
    nv = ONEV + 1
    m = Model(0, 0)
    r = try
        add_vars(m, nv)
        sense(m, SENSE_MIN)
        for i in 0:nv-1
            var_bounds(m, i, BK_FR; lo=-Inf, up=Inf)
        end
        for i in 0:nmulti-1
            var_bounds(m, PBASE + i, BK_LO; lo=0.0, up=Inf)
        end
        var_bounds(m, ONEV, BK_FX; lo=1.0, up=1.0)
        add_cons(m, NPF + 2nmulti + NRQ)
        row, ecur, pcur = 0, 0, 0
        for posy in P
            if length(posy) == 1                    # linear: a.y + ln c <= 0
                coef, terms = posy[1]
                a_row(m, row, [YBASE + v for (v, _) in terms], [a for (_, a) in terms])
                con_bounds(m, row, BK_UP; lo=-Inf, up=-clog(coef)); row += 1
            else
                p0 = pcur
                for (coef, terms) in posy
                    a_row(m, row, [EBASE + ecur; [YBASE + v for (v, _) in terms]], [1.0; [-a for (_, a) in terms]])
                    con_bounds(m, row, BK_FX; lo=clog(coef), up=clog(coef)); row += 1
                    cone(m, CT_PEXP, [PBASE + pcur, ONEV, EBASE + ecur])
                    ecur += 1; pcur += 1
                end
                a_row(m, row, [PBASE + p0 + q for q in 0:length(posy)-1], ones(length(posy)))
                con_bounds(m, row, BK_UP; lo=-Inf, up=1.0); row += 1
            end
        end
        cj(m, YBASE + D, 1.0)                       # minimize log D
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    y = ok ? [cexp(r.x[YBASE+i+1]) for i in 0:NV-1] : zeros(NV)

    # independent checks: the delay and the area at the recovered sizes
    s1, s2 = y[S1+1], y[S2+1]
    delay = (s2 + 1.0) / s1 + 10.0 / s2             # D(s) = D1 + D2
    area = s1 + s2
    want = 10.0 * sqrt(10.0) / (sqrt(11.0) + sqrt(10.0))   # s2*
    good = ok && abs(delay - 3.1976177) < 1e-4 && abs(area - Amax) < 1e-6 &&
           abs(s2 - want) < 1e-2 && abs(y[D+1] - delay) < 1e-4
    @printf("gate_sizing_gp  delay=%.8f (expected 3.19761770)  s1=%.6f s2=%.6f (expected %.6f)\n", delay, s1, s2, want)
    @printf("  area=%.6f (<= %.0f)  D(var)=%.8f  %s\n", area, Amax, y[D+1], good ? "OK" : "FAIL")
    return good ? 0 : 1
end

exit(main())
