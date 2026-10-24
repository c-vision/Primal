#!/usr/bin/env julia
# logcontrast_lasso.jl — log-contrast LASSO logistic regression
# (audreyolmsted/admm-portfolio, benchmarked against MOSEK/CVXR):
#
#   minimize (1/n) sum_i log(1 + exp(-y_i eta_i)) + lambda sum_j s_j |b_j|
#   s.t.  eta = [1 X] beta,  sum_j b_j = 0  (log-contrast)
#
# Two independent solvers on the same deterministic instance: the exact conic
# program (two exponential cones per sample for the softplus, a linear L1
# epigraph) and the repository's ADMM (Newton beta-step, closed-form joint
# proximal z-step), both checked against a brute force over the reduced space.
# Instance n = 16, p = 3, lambda = 0.3: feature 0 is thresholded to zero.
# Port of examples/c/logcontrast_lasso.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NF = 3                                        # features
const NS = 16                                       # samples
const NL = 4                                        # p + 1 unknowns
const LAM = 0.3                                     # LASSO penalty
const WTRUE = [-0.5, -0.5, 1.0]                     # true log-contrast signal (sum zero)
cexp(x) = ccall(:exp, Cdouble, (Cdouble,), x)      # the C library's exp/log1p
clog1p(x) = ccall(:log1p, Cdouble, (Cdouble,), x)

const st = Ref{UInt32}(20260924)
# Deterministic pseudo-random draw in [0,1] from the fixed seed.
function rnd()
    st[] = st[] * 0x41c64e6d + 0x00003039           # 1103515245, 12345 (mod 2^32)
    return Float64((st[] >> 16) & 0x7fff) / 32767.0
end

# Build the deterministic CLR-transformed instance, its labels and the L1 scales.
function make_instance()
    X = zeros(NS, NF)
    for i in 1:NS
        raw, m = zeros(NF), 0.0
        for j in 1:NF
            raw[j] = rnd() * 2.0 - 1.0; m += raw[j]
        end
        m /= NF
        for j in 1:NF
            X[i, j] = raw[j] - m                    # CLR: row mean 0
        end
    end
    Y = zeros(NS)
    for i in 1:NS
        eta = 0.0
        for j in 1:NF
            eta += X[i, j] * WTRUE[j]
        end
        eta += rnd() - 0.5                          # label noise
        Y[i] = eta > 0.0 ? 1.0 : -1.0
    end
    SIG = zeros(NF)
    for j in 1:NF
        mu, v = 0.0, 0.0
        for i in 1:NS
            mu += X[i, j]
        end
        mu /= NS
        for i in 1:NS
            v += (X[i, j] - mu) * (X[i, j] - mu)
        end
        SIG[j] = sqrt((Float64(NS - 1) / NS) * (v / (NS - 1)))
        SIG[j] == 0.0 && (SIG[j] = 1.0)
    end
    return X, Y, SIG
end

softplus(z) = z >= 0.0 ? z + clog1p(cexp(-z)) : clog1p(cexp(z))
sigmoid(z) = z >= 0.0 ? 1.0 / (1.0 + cexp(-z)) : cexp(z) / (1.0 + cexp(z))

# objective at beta (intercept first)
function objective(b, X, Y, SIG)
    s = 0.0
    for i in 1:NS
        eta = b[1]
        for j in 1:NF
            eta += X[i, j] * b[1+j]
        end
        s += softplus(-Y[i] * eta)
    end
    s /= NS
    for j in 1:NF
        s += LAM * SIG[j] * abs(b[1+j])
    end
    return s
end

# conic model: vars b0 | b_j | g_j | per sample (t,m,s,a,b) | one
BJ(j) = 1 + j
GJ(j) = 1 + NF + j
SB(i) = 1 + 2NF + 5i
const ONE = 1 + 2NF + 5NS
const NVAR = 1 + 2NF + 5NS + 1
const NR = 1 + 2NF + 3NS

# Solve the log-contrast LASSO through the PEXP conic model; returns (ok, beta).
function conic_solve(X, Y, SIG)
    m = Model(NR, NVAR)
    r = try
        for j in 0:NVAR-1
            var_bounds(m, j, BK_FR; lo=-Inf, up=Inf)
        end
        for j in 0:NF-1
            var_bounds(m, GJ(j), BK_LO; lo=0.0, up=Inf)
        end
        for i in 0:NS-1
            var_bounds(m, SB(i) + 3, BK_LO; lo=0.0, up=Inf)
            var_bounds(m, SB(i) + 4, BK_LO; lo=0.0, up=Inf)
        end
        var_bounds(m, ONE, BK_FX; lo=1.0, up=1.0)
        row = 0
        a_row(m, row, [BJ(j) for j in 0:NF-1], ones(NF))   # log-contrast: sum_j b_j = 0
        con_bounds(m, row, BK_FX; lo=0.0, up=0.0); row += 1
        for j in 0:NF-1                             # g_j >= |b_j|
            a_row(m, row, [GJ(j), BJ(j)], [1.0, -1.0]); con_bounds(m, row, BK_LO; lo=0.0, up=Inf); row += 1
            a_row(m, row, [GJ(j), BJ(j)], [1.0, 1.0]);  con_bounds(m, row, BK_LO; lo=0.0, up=Inf); row += 1
        end
        for i in 0:NS-1
            T, Mv, S, A, B = SB(i), SB(i) + 1, SB(i) + 2, SB(i) + 3, SB(i) + 4
            a_row(m, row, [Mv, T], [1.0, 1.0])     # m + t = 0
            con_bounds(m, row, BK_FX; lo=0.0, up=0.0); row += 1
            # s + t + y_i eta_i = 0,  eta_i = b0 + sum_j X_ij b_j
            a_row(m, row, [S; T; 0; [BJ(j) for j in 0:NF-1]], [1.0; 1.0; Y[i+1]; [Y[i+1] * X[i+1, j+1] for j in 0:NF-1]])
            con_bounds(m, row, BK_FX; lo=0.0, up=0.0); row += 1
            a_row(m, row, [A, B], [1.0, 1.0])      # a + b <= 1
            con_bounds(m, row, BK_UP; lo=-Inf, up=1.0); row += 1
        end
        for i in 0:NS-1
            cone(m, CT_PEXP, [SB(i) + 3, ONE, SB(i) + 1])
            cone(m, CT_PEXP, [SB(i) + 4, ONE, SB(i) + 2])
        end
        for i in 0:NS-1
            cj(m, SB(i), 1.0 / NS)
        end
        for j in 0:NF-1
            cj(m, GJ(j), LAM * SIG[j+1])
        end
        solve(m)
    finally
        free(m)
    end
    r.rc == 0 || return false, zeros(NL)
    return true, [r.x[1]; [r.x[BJ(j)+1] for j in 0:NF-1]]
end

# smooth part of the beta-step objective: mean softplus loss + (rho/2)||b[1:] - target||^2
function f_aug(b, target, rho, X, Y)
    s = 0.0
    for i in 1:NS
        eta = b[1]
        for j in 1:NF
            eta += X[i, j] * b[1+j]
        end
        s += softplus(-Y[i] * eta)
    end
    f = s / NS
    for j in 1:NF
        rr = b[1+j] - target[j]
        f += 0.5 * rho * rr * rr
    end
    return f
end

# ADMM (the repo's method); returns (beta, iterations, converged).
function admm_solve(X, Y, SIG)
    rho, abs_tol, rel_tol, max_iter = 1.0, 1e-6, 1e-4, 5000
    beta, z, u = zeros(NL), zeros(NF), zeros(NF)
    conv, it = false, 0
    while it < max_iter
        zold = copy(z)
        target = z .- u                             # beta-step: Newton with backtracking
        for _ in 1:20
            g, H = zeros(NL), zeros(NL, NL)
            for i in 1:NS
                eta = beta[1]
                xi = zeros(NL); xi[1] = 1.0
                for j in 1:NF
                    eta += X[i, j] * beta[1+j]; xi[1+j] = X[i, j]
                end
                p = sigmoid(Y[i] * eta)             # P(correct)
                gi = -Y[i] * (1.0 - p) / NS         # d loss / d eta
                hi = p * (1.0 - p) / NS
                for a in 1:NL
                    g[a] += gi * xi[a]
                    for b in 1:NL
                        H[a, b] += hi * xi[a] * xi[b]
                    end
                end
            end
            for j in 1:NF                           # penalty gradient/Hessian
                g[1+j] += rho * (beta[1+j] - target[j])
                H[1+j, 1+j] += rho
            end
            A = [H -g]                              # solve H d = -g (Gauss elimination, partial pivoting)
            d = zeros(NL)
            for c in 1:NL
                piv = c
                for r in c+1:NL
                    abs(A[r, c]) > abs(A[piv, c]) && (piv = r)
                end
                for b in 1:NL+1
                    A[c, b], A[piv, b] = A[piv, b], A[c, b]
                end
                if abs(A[c, c]) < 1e-14
                    d[c] = 0.0; continue
                end
                for r in c+1:NL
                    f = A[r, c] / A[c, c]
                    for b in c:NL+1
                        A[r, b] -= f * A[c, b]
                    end
                end
            end
            for c in NL:-1:1
                s = A[c, NL+1]
                for b in c+1:NL
                    s -= A[c, b] * d[b]
                end
                d[c] = abs(A[c, c]) < 1e-14 ? 0.0 : s / A[c, c]
            end
            dn = 0.0
            for a in 1:NL
                dn += d[a] * d[a]
            end
            dn < 1e-18 && break
            step = 1.0                              # backtracking on the objective
            f0 = f_aug(beta, target, rho, X, Y)
            for _ in 1:30
                bn = [beta[a] + step * d[a] for a in 1:NL]
                if f_aug(bn, target, rho, X, Y) <= f0 - 1e-12 * step * step
                    beta = bn
                    break
                end
                step *= 0.5
            end
            step < 1e-9 && break
        end
        w = beta[2:end]
        # z-step: soft-threshold, shifted so that sum(z) = 0 (monotone root by bisection)
        thr = [LAM * SIG[j] / rho for j in 1:NF]
        v = [w[j] + u[j] for j in 1:NF]
        lo, hi = v[1], v[1]
        for j in 1:NF
            v[j] - thr[j] < lo && (lo = v[j] - thr[j])
            v[j] + thr[j] < lo && (lo = v[j] + thr[j])
            v[j] + thr[j] > hi && (hi = v[j] + thr[j])
            v[j] - thr[j] > hi && (hi = v[j] - thr[j])
        end
        c = 0.0
        for _ in 1:80
            c = 0.5 * (lo + hi)
            s = 0.0
            for j in 1:NF
                zz = v[j] - c
                if zz > thr[j]
                    s += zz - thr[j]
                elseif zz < -thr[j]
                    s += zz + thr[j]
                end
            end
            s > 0.0 ? (lo = c) : (hi = c)
        end
        for j in 1:NF
            zz = v[j] - c
            z[j] = zz > thr[j] ? zz - thr[j] : (zz < -thr[j] ? zz + thr[j] : 0.0)
        end
        for j in 1:NF                               # dual update
            u[j] += w[j] - z[j]
        end
        rn, sn, wn, zn, un = 0.0, 0.0, 0.0, 0.0, 0.0   # convergence
        for j in 1:NF
            rn += (w[j] - z[j]) * (w[j] - z[j])
            sn += (z[j] - zold[j]) * (z[j] - zold[j])
            wn += w[j] * w[j]; zn += z[j] * z[j]; un += u[j] * u[j]
        end
        rn = sqrt(rn); sn = rho * sqrt(sn)
        eps_pri = sqrt(Float64(NF)) * abs_tol + rel_tol * max(sqrt(wn), sqrt(zn))
        eps_dual = sqrt(Float64(NF)) * abs_tol + rel_tol * rho * sqrt(un)
        if rn < eps_pri && sn < eps_dual
            conv = true; it += 1
            break
        end
        it += 1
    end
    return [beta[1]; z], it, conv
end

# brute force over the reduced space (b3 = -b1 - b2); returns (objective, beta).
function brute_force(X, Y, SIG)
    span, NG = 6.0, 120
    best, bb = 1e300, zeros(NL)
    for a in 0:NG
        b0 = -span + 2 * span * a / NG
        for c in 0:NG
            t1 = -span + 2 * span * c / NG
            for e in 0:NG
                t2 = -span + 2 * span * e / NG
                b = [b0, t1, t2, -t1 - t2]
                f = objective(b, X, Y, SIG)
                f < best && ((best, bb) = (f, b))
            end
        end
    end
    return best, bb
end

function main()
    X, Y, SIG = make_instance()
    ok, bc = conic_solve(X, Y, SIG)
    if !ok
        println("logcontrast_lasso  conic rc!=OK FAIL")
        return 1
    end
    oc = objective(bc, X, Y, SIG)
    ba, iters, conv = admm_solve(X, Y, SIG)
    oa = objective(ba, X, Y, SIG)
    ob, bb = brute_force(X, Y, SIG)
    sumc, suma, dbmax = 0.0, 0.0, 0.0
    for j in 1:NF
        sumc += bc[1+j]; suma += ba[1+j]
        abs(ba[1+j] - bc[1+j]) > dbmax && (dbmax = abs(ba[1+j] - bc[1+j]))
    end
    ok_conic = abs(oc - ob) < 1e-3 * (1.0 + abs(ob))
    ok_admm = conv && abs(oa - oc) < 1e-4 * (1.0 + abs(oc)) && dbmax < 1e-3
    ok_sum = abs(sumc) < 1e-6 && abs(suma) < 1e-6
    ok_sparse = abs(bc[2]) < 1e-6 && abs(ba[2]) < 1e-6   # feature 0 penalised to 0
    ok = ok_conic && ok_admm && ok_sum && ok_sparse
    @printf("logcontrast_lasso  n=%d p=%d lambda=%.3f\n", NS, NF, LAM)
    @printf("  conic : obj=%.8f  beta=(%.5f, %.5f, %.5f, %.5f)  sum=%.2e\n", oc, bc[1], bc[2], bc[3], bc[4], sumc)
    @printf("  admm  : obj=%.8f  beta=(%.5f, %.5f, %.5f, %.5f)  sum=%.2e  iters=%d %s\n",
            oa, ba[1], ba[2], ba[3], ba[4], suma, iters, conv ? "converged" : "NOT-converged")
    @printf("  brute : obj=%.8f  beta=(%.5f, %.5f, %.5f, %.5f)\n", ob, bb[1], bb[2], bb[3], bb[4])
    @printf("  [conic vs brute %s] [admm vs conic: obj %s, max|db| %.2e] [zero-sum %s] [sparse %s]\n",
            ok_conic ? "OK" : "FAIL", ok_admm ? "OK" : "FAIL", dbmax, ok_sum ? "OK" : "FAIL", ok_sparse ? "OK" : "FAIL")
    @printf("%s\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
