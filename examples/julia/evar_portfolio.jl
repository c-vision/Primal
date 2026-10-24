#!/usr/bin/env julia
# evar_portfolio.jl — entropic value-at-risk portfolio (MOSEK Portfolio
# Optimization Cookbook, chapter 8). Maximize m'x - delta EVaR_alpha(loss)
# over the simplex, with
#
#   EVaR_alpha(L) = inf_{s>0} s (log(sum_i p_i exp(L_i/s)) - log(1-alpha))
#
# modelled with one exponential cone per scenario plus a linear row:
# (u_i, s, L_i - z) in K_exp, sum p_i u_i <= s, EVaR = z - s log(1-alpha).
# Checked against a brute force over the simplex with EVaR minimized by
# golden section. Port of examples/c/evar_portfolio.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NA = 2                                        # assets
const NS = 4                                        # scenarios
const RET = [0.10 0.02; -0.05 0.04; 0.20 0.01; -0.10 0.03]
const PROB = [0.25, 0.25, 0.25, 0.25]
const ALPHA = 0.95
const DELTA = 1.0
clog(x) = ccall(:log, Cdouble, (Cdouble,), x)      # the C library's log/exp
cexp(x) = ccall(:exp, Cdouble, (Cdouble,), x)

# loss in scenario i for weights x: L_i = -sum_a R[i][a] x[a]
function loss(i, x)
    r = 0.0
    for a in 1:NA
        r += RET[i, a] * x[a]
    end
    return -r
end

# g(s) = s (logsumexp(L/s) - log(1-alpha)), written stably around max(L)
function evar_g(s, x)
    mx = loss(1, x)
    for i in 2:NS
        l = loss(i, x); l > mx && (mx = l)
    end
    acc = 0.0
    for i in 1:NS
        acc += PROB[i] * cexp((loss(i, x) - mx) / s)
    end
    return mx + s * (clog(acc) - clog(1.0 - ALPHA))
end

# EVaR by golden section on log(s)
function evar(x)
    a, b = -40.0, 12.0                              # log(s) bracket (the infimum may be at s -> 0)
    gr = 0.6180339887498949
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = evar_g(cexp(c), x), evar_g(cexp(d), x)
    for _ in 1:200
        if fc < fd
            b = d; d = c; fd = fc; c = b - gr * (b - a); fc = evar_g(cexp(c), x)
        else
            a = c; c = d; fc = fd; d = a + gr * (b - a); fd = evar_g(cexp(d), x)
        end
    end
    return evar_g(cexp(0.5 * (a + b)), x)
end

# The conic model: max m'x - delta EVaR; returns (ok, objective, x, evar).
function evar_conic()
    X0, Z, S, U, V = 0, NA, NA + 1, NA + 2, NA + 2 + NS
    NV = NA + 2 + 2NS
    R_BUD, R_ULIN, R_V = 0, 1, 2
    mvec = zeros(NA)
    for a in 1:NA, i in 1:NS
        mvec[a] += PROB[i] * RET[i, a]
    end
    m = Model(0, 0)
    r = try
        add_vars(m, NV)
        add_cons(m, 2 + NS)
        for a in 0:NA-1
            var_bounds(m, X0 + a, BK_LO; lo=0.0, up=Inf)
        end
        var_bounds(m, Z, BK_FR; lo=-Inf, up=Inf)
        var_bounds(m, S, BK_LO; lo=0.0, up=Inf)
        for i in 0:NS-1
            var_bounds(m, U + i, BK_LO; lo=0.0, up=Inf)
        end
        for i in 0:NS-1
            var_bounds(m, V + i, BK_FR; lo=-Inf, up=Inf)
        end
        a_row(m, R_BUD, collect(X0:X0+NA-1), ones(NA))   # budget
        con_bounds(m, R_BUD, BK_FX; lo=1.0, up=1.0)
        a_row(m, R_ULIN, [collect(U:U+NS-1); S], [PROB; -1.0])   # sum p_i u_i - s <= 0
        con_bounds(m, R_ULIN, BK_UP; lo=-Inf, up=0.0)
        for i in 0:NS-1                             # v_i + R x + z = 0 (v_i = loss_i - z)
            a_row(m, R_V + i, [V + i; collect(X0:X0+NA-1); Z], [1.0; RET[i+1, :]; 1.0])
            con_bounds(m, R_V + i, BK_FX; lo=0.0, up=0.0)
        end
        for i in 0:NS-1
            cone(m, CT_PEXP, [U + i, S, V + i])
        end
        for a in 0:NA-1                             # max m'x - delta z + delta log(1-alpha) s
            cj(m, X0 + a, mvec[a+1])
        end
        cj(m, Z, -DELTA)
        cj(m, S, DELTA * clog(1.0 - ALPHA))
        sense(m, SENSE_MAX)
        solve(m)
    finally
        free(m)
    end
    r.rc == 0 || return false, 0.0, zeros(NA), 0.0
    x = r.x
    obj = 0.0
    for a in 1:NA
        obj += mvec[a] * x[X0+a]
    end
    obj -= DELTA * (x[Z+1] - x[S+1] * clog(1.0 - ALPHA))
    return true, obj, x[X0+1:X0+NA], x[Z+1] - x[S+1] * clog(1.0 - ALPHA)
end

# expected return of the weights xr
function mean_return(xr)
    ret = 0.0
    for a in 1:NA
        mr = 0.0
        for i in 1:NS
            mr += PROB[i] * RET[i, a]
        end
        ret += mr * xr[a]
    end
    return ret
end

function main()
    ok, obj, x, ev = evar_conic()
    best, bw = -1e300, 0.0                          # reference: brute force over the simplex
    for k in 0:20000
        w = k / 20000.0
        xr = [w, 1.0 - w]
        f = mean_return(xr) - DELTA * evar(xr)
        f > best && ((best, bw) = (f, w))
    end
    ret_sol = mean_return(x)
    ev_direct = evar(x)
    okall = ok && abs(obj - best) < 1e-3 * (1.0 + abs(best)) && abs(ev - ev_direct) < 1e-3 * (1.0 + abs(ev_direct)) &&
            abs(x[1] + x[2] - 1.0) < 1e-7 && x[1] >= -1e-9 && x[2] >= -1e-9
    @printf("evar_portfolio  N=%d T=%d alpha=%.2f delta=%.2f\n", NA, NS, ALPHA, DELTA)
    @printf("  x = (%.8f, %.8f)   return = %.8f\n", x[1], x[2], ret_sol)
    @printf("  EVaR = %.8f   [brute force obj %.8f at w=%.5f]\n", ev, best, bw)
    @printf("  conic objective = %.8f   [brute force %.8f]\n", obj, best)
    @printf("%s\n", okall ? "OK" : "FAIL")
    return okall ? 0 : 1
end

exit(main())
