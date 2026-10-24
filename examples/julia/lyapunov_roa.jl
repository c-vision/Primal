#!/usr/bin/env julia
# lyapunov_roa.jl — SOS Lyapunov region-of-attraction certificate (from
# Ali-Eimaan/lyapunov-toolbox, BSD-3-Clause). Time-reversed Van der Pol
# x1' = -x2, x2' = x1 + (x1^2 - 1) x2 with V(x) = 1.5 x1^2 - x1 x2 + x2^2; the
# sublevel set {V <= rho} is certified by the SOS S-procedure
#
#   -(Vdot + eps ||x||^4) + lambda (V - rho)  in  Sigma[x],  lambda SOS,
#
# an SDP with QS PSD (10x10), QL PSD (6x6) and 28 coefficient rows (the tables
# below are the exact coefficient data; Mc = rho AL - ALV). The certified level
# is found by bisection: the repo reports 2.304375, the true tangency 2.30448.
# Port of examples/c/lyapunov_roa.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NROW = 28
const TS = [(i, j) for i in 0:9 for j in i:9]       # upper-triangle pairs of QS (55)
const TL = [(i, j) for i in 0:5 for j in i:5]       # upper-triangle pairs of QL (21)
const AS = Float64[
    1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  2  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  2  0  0  0  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  2  0  0  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  2  0  0  0  0  0  2  0  0  0  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  2  0  0  0  0  0  2  0  0  0  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  0  0  0  2  0  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  0  0  0  2  0  0  0  0  2  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  0  0  0  2  0  0  0  0  0  0  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  0  2  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  0  2  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  1  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  2  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  1  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1
]
const AL = Float64[
    1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  2  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  2  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  2  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  2  0  0  2  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  2  0  0  2  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  1  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
]
const ALV = Float64[
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    -1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    1.5  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  -2  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  3  -2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  3  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  2  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  -2  2  0  -1  2  0  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  3  -2  2  1.5  -2  0  0  0  1  0  0  0  0  0  0  0  0  0
    0  0  0  0  3  -2  0  3  0  0  0  -1  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  3  0  0  0  0  0  1.5  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  -2  2  0  0  2  0  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  3  -2  2  0  -2  2  0  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  3  -2  0  3  -2  2  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  3  0  0  3  -2  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  3  0  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  -1  2  0  0  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1.5  -2  2  1  0  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  3  -2  -1  2  0
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  3  1.5  -2  1
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  3  -1
    0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1.5
]
const RHS = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, -1e-06, 0, -2.000002, 1, -1e-06, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0]

# symmetric eigenvalue routine (Jacobi), for the PSD check of the Gram blocks
function jacobi(n, A)
    a = copy(A)                                     # row-major n x n
    for _ in 1:100
        off = 0.0
        for p in 0:n-1, q in p+1:n-1
            off += a[p*n+q+1] * a[p*n+q+1]
        end
        off < 1e-28 && break
        for p in 0:n-1, q in p+1:n-1
            apq = a[p*n+q+1]
            abs(apq) < 1e-300 && continue
            th = (a[q*n+q+1] - a[p*n+p+1]) / (2 * apq)
            t = (th >= 0 ? 1 : -1) / (abs(th) + sqrt(th * th + 1))
            c = 1 / sqrt(t * t + 1)
            s = t * c
            for k in 0:n-1
                ap, aq = a[p*n+k+1], a[q*n+k+1]
                a[p*n+k+1] = c * ap - s * aq; a[q*n+k+1] = s * ap + c * aq
            end
            for k in 0:n-1
                ap, aq = a[k*n+p+1], a[k*n+q+1]
                a[k*n+p+1] = c * ap - s * aq; a[k*n+q+1] = s * ap + c * aq
            end
        end
    end
    return [a[i*n+i+1] for i in 0:n-1]
end

# Test SOS feasibility of the Lyapunov certificate at level rho; returns (ok, QS, QL).
function solve_feas(rho)
    m = Model(0, 0)
    try
        barvars(m, [10, 6])
        add_cons(m, NROW)
        for r in 1:NROW
            terms = [(i, j, AS[r, k] * (i == j ? 1.0 : 0.5)) for (k, (i, j)) in enumerate(TS)]
            terms = filter(t -> t[3] != 0.0, terms)
            if !isempty(terms)
                id = sparsesymmat(m, 10, terms)
                baraij(m, r - 1, 0, [(id, 1.0)])
            end
            terms = [(i, j, (rho * AL[r, k] - ALV[r, k]) * (i == j ? 1.0 : 0.5)) for (k, (i, j)) in enumerate(TL)]
            terms = filter(t -> t[3] != 0.0, terms)
            if !isempty(terms)
                id = sparsesymmat(m, 6, terms)
                baraij(m, r - 1, 1, [(id, 1.0)])
            end
            con_bounds(m, r - 1, BK_FX; lo=RHS[r], up=RHS[r])
        end
        res = solve(m)
        res.rc == 0 || return false, zeros(100), zeros(36)
        return true, getbarxj(m, 0), getbarxj(m, 1)
    finally
        free(m)
    end
end

function main()
    lo, hi = 0.0, 4.0                               # bisection on the certified level
    for _ in 1:26
        mid = 0.5 * (lo + hi)
        solve_feas(mid)[1] ? (lo = mid) : (hi = mid)
    end
    ok = lo > 2.30 && lo < 2.31
    good, QS, QL = solve_feas(lo)                   # certificate at the certified level
    if good
        mnS, mnL = minimum(jacobi(10, QS)), minimum(jacobi(6, QL))
        ids = 0.0                                   # identity residual: AS vech(QS) + Mc vech(QL) - RHS
        for r in 1:NROW
            acc = 0.0
            for (k, (i, j)) in enumerate(TS)
                acc += AS[r, k] * QS[i*10+j+1]
            end
            for (k, (i, j)) in enumerate(TL)
                acc += (lo * AL[r, k] - ALV[r, k]) * QL[i*6+j+1]
            end
            d = abs(acc - RHS[r]); d > ids && (ids = d)
        end
        ok = ok && mnS > -1e-6 && mnL > -1e-6 && ids < 1e-6
        @printf("lyapunov_roa  rho=%.6f  (repo 2.304375, true tangency 2.30448)\n", lo)
        @printf("              Gram PSD: min eig QS=%.3e QL=%.3e  identity residual=%.3e  %s\n",
                mnS, mnL, ids, ok ? "OK" : "FAIL")
    else
        println("lyapunov_roa  certificate solve failed")
        ok = false
    end
    return ok ? 0 : 1
end

exit(main())
