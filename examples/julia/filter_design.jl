#!/usr/bin/env julia
# filter_design.jl — FIR filter design as a trigonometric-polynomial SDP
# (MOSEK/Tutorials, filterdesign). H(w) = x_0 + 2 x_1 cos(w) + ... + 2 x_n cos(n w)
# is nonnegative on an interval iff its coefficients lie in a cone described by
# PSD Toeplitz matrices (<T_i^m, Y> = the i-th diagonal sum of Y). The design
# minimizes the stopband level t s.t. H >= 0, 1-d <= H <= 1+d on [0, wp] and
# H <= t on [ws, pi]; the returned response is checked on a dense grid
# (declared deviation: numerical sampling, not a second solver).
# Port of examples/c/filter_design.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NF = 3                                        # polynomial degree n
const M_PI = 3.14159265358979323846
ccos(x) = ccall(:cos, Cdouble, (Cdouble,), x)      # the C library's cos

# add coef * <T_i^dim, X_bar> to row (one symmat per diagonal entry)
function addT(m, row, bar, dim, i, coef)
    ii = abs(i)
    ii >= dim && return
    for k in 0:dim-ii-1
        id = sparsesymmat(m, dim, [(k, k + ii, ii == 0 ? 1.0 : 0.5)])   # <m,X> = X_{k,k+ii}
        baraij(m, row, bar, [(id, coef)])
    end
end

function main()
    DELTA, WP, WS = 0.05, M_PI / 4.0, M_PI / 4.0 + M_PI / 8.0
    X, TT, NV = 0, NF + 1, NF + 2                   # scalar vars: x[0..NF], t
    m = Model(0, 0)
    r = try
        add_vars(m, NV)
        for i in 0:NF
            var_bounds(m, X + i, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(m, TT, BK_LO; lo=0.0, up=Inf)
        # bars: 0 global X0(NF+1); 1,2 passband upper; 3,4 passband lower; 5,6 stopband
        barvars(m, [NF + 1, NF + 1, NF, NF + 1, NF, NF + 1, NF])
        add_cons(m, 4 * (NF + 1))
        row = 0
        for i in 0:NF                               # global nonnegativity: <T_i^{n+1},X0> - x_i = 0
            addT(m, row, 0, NF + 1, i, 1.0)
            a_row(m, row, [X + i], [-1.0])
            con_bounds(m, row, BK_FX; lo=0.0, up=0.0); row += 1
        end
        for i in 0:NF                               # passband upper (level 1+delta)
            addT(m, row, 1, NF + 1, i, 1.0)
            addT(m, row, 2, NF, i + 1, 1.0)
            addT(m, row, 2, NF, i - 1, 1.0)
            addT(m, row, 2, NF, i, -2.0 * ccos(WP))
            rhs = i == 0 ? (1.0 + DELTA) : 0.0
            a_row(m, row, [X + i], [1.0])
            con_bounds(m, row, BK_FX; lo=rhs, up=rhs); row += 1
        end
        for i in 0:NF                               # passband lower (level 1-delta)
            addT(m, row, 3, NF + 1, i, 1.0)
            addT(m, row, 4, NF, i + 1, 1.0)
            addT(m, row, 4, NF, i - 1, 1.0)
            addT(m, row, 4, NF, i, -2.0 * ccos(WP))
            rhs = i == 0 ? -(1.0 - DELTA) : 0.0
            a_row(m, row, [X + i], [-1.0])
            con_bounds(m, row, BK_FX; lo=rhs, up=rhs); row += 1
        end
        for i in 0:NF                               # stopband (level t)
            addT(m, row, 5, NF + 1, i, 1.0)
            addT(m, row, 6, NF, i + 1, -1.0)
            addT(m, row, 6, NF, i - 1, -1.0)
            addT(m, row, 6, NF, i, 2.0 * ccos(WS))
            i == 0 ? a_row(m, row, [X, TT], [1.0, -1.0]) : a_row(m, row, [X + i], [1.0])
            con_bounds(m, row, BK_FX; lo=0.0, up=0.0); row += 1
        end
        cj(m, TT, 1.0)
        sense(m, SENSE_MIN)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    x = ok ? r.x : zeros(NV)
    topt = x[TT+1]

    neg, pa_hi, pa_lo, sb = false, false, false, false   # verify the specs on a dense grid
    for q in 0:4000
        w = M_PI * q / 4000.0
        H = x[X+1]
        for i in 1:NF
            H += 2.0 * x[X+i+1] * ccos(i * w)
        end
        H < -1e-5 && (neg = true)
        if w <= WP + 1e-9
            H > 1.0 + DELTA + 1e-4 && (pa_hi = true)
            H < 1.0 - DELTA - 1e-4 && (pa_lo = true)
        end
        w >= WS - 1e-9 && H > topt + 1e-4 && (sb = true)
    end
    okall = ok && !neg && !pa_hi && !pa_lo && !sb && topt < 1.0
    @printf("filter_design  n=%d  wp=pi/4  ws=3pi/8  delta=%.2f\n", NF, DELTA)
    @printf("  t* (stopband) = %.6f   H>=0:%d  passband ok:%d  stopband ok:%d\n", topt, !neg, !(pa_hi || pa_lo), !sb)
    print("  x = (")
    for i in 0:NF
        @printf("%s%.5f", i > 0 ? ", " : "", x[X+i+1])
    end
    println(")")
    @printf("%s\n", okall ? "OK" : "FAIL")
    return okall ? 0 : 1
end

exit(main())
