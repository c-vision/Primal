#!/usr/bin/env julia
# nearestcorrelation.jl — port of the "nearestcorrelation" example: nearest
# correlation matrix (Higham).
#
#   min ||A - X||_F  s.t.  X PSD,  diag(X) = 1
#
# modelled as min t s.t. (t, d) in QUAD with d_ij = sqrt(w_ij)(A_ij - X_ij)
# (w = 2 off the diagonal), a 3x3 bar X and rows X_ii = 1. Higham's
# alternating projections provide the independent check.
# Port of examples/c/nearestcorrelation.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const N = 3
const A = [2.0 -1.0 0.0; -1.0 2.0 -1.0; 0.0 -1.0 2.0]

# Frobenius distance to A
function fro(X)
    s = 0.0
    for i in 1:N, j in 1:N
        d = X[i, j] - A[i, j]
        s += d * d
    end
    return sqrt(s)
end

# self-contained 3x3 Jacobi eigendecomposition: returns (eigenvalues, eigenvectors as columns)
function eig3(M)
    W = copy(M)
    V = [i == j ? 1.0 : 0.0 for i in 1:N, j in 1:N]
    for _ in 1:100
        off = 0.0
        for p in 1:N, q in p+1:N
            off += W[p, q] * W[p, q]
        end
        off < 1e-30 && break
        for p in 1:N, q in p+1:N
            apq = W[p, q]
            abs(apq) < 1e-32 && continue
            theta = (W[q, q] - W[p, p]) / (2.0 * apq)
            t = (theta >= 0.0 ? 1.0 : -1.0) / (abs(theta) + sqrt(theta * theta + 1.0))
            c = 1.0 / sqrt(t * t + 1.0)
            sn = t * c
            for k in 1:N
                wp, wq = W[p, k], W[q, k]
                W[p, k] = c * wp - sn * wq; W[q, k] = sn * wp + c * wq
            end
            for k in 1:N
                wp, wq = W[k, p], W[k, q]
                W[k, p] = c * wp - sn * wq; W[k, q] = sn * wp + c * wq
            end
            for k in 1:N
                vp, vq = V[k, p], V[k, q]
                V[k, p] = c * vp - sn * vq; V[k, q] = sn * vp + c * vq
            end
        end
    end
    return [W[i, i] for i in 1:N], V
end

# Higham alternating projection: nearest correlation matrix (reference)
function higham()
    X = copy(A)
    for _ in 1:500
        ev, V = eig3(X)                             # PSD projection: clip negative eigenvalues
        S = zeros(N, N)
        for k in 1:N
            lk = ev[k] > 0.0 ? ev[k] : 0.0
            for i in 1:N, j in 1:N
                S[i, j] += lk * V[i, k] * V[j, k]
            end
        end
        for i in 1:N                                # diag = 1
            S[i, i] = 1.0
        end
        diff = maximum(abs.(S .- X))                # convergence check
        X = S
        diff < 1e-12 && break
    end
    return X
end

function main()
    nd = N * (N + 1) ÷ 2
    m = Model(0, 0)
    X = zeros(N, N)
    po = 0.0
    try
        add_vars(m, 1)                              # t
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        cj(m, 0, 1.0)                               # min t
        msym = [sparsesymmat(m, N, [(i, j, 1.0)]) for i in 0:N-1 for j in i:N-1]
        barvars(m, [N])
        add_vars(m, nd)                             # d_ij (i <= j)
        for k in 0:nd-1
            var_bounds(m, 1 + k, BK_FR; lo=-Inf, up=Inf)
        end
        add_cons(m, nd + N)
        # (i,i): X_ii + d_ii = A_ii ; (i<j): 2 X_ij + sqrt(2) d_ij = 2 A_ij
        row = 0
        for i in 0:N-1, j in i:N-1
            w = i == j ? 1.0 : 2.0
            rhs = i == j ? A[i+1, j+1] : 2.0 * A[i+1, j+1]
            baraij(m, row, 0, [(msym[row+1], 1.0)])
            a_row(m, row, [1 + row], [sqrt(w)])
            con_bounds(m, row, BK_FX; lo=rhs, up=rhs)
            row += 1
        end
        me = [msym[k] for (k, (i, j)) in enumerate([(i, j) for i in 0:N-1 for j in i:N-1]) if i == j]
        for k in 1:N                                # diag rows: <E_kk, X> = 1
            baraij(m, row, 0, [(me[k], 1.0)])
            con_bounds(m, row, BK_FX; lo=1.0, up=1.0)
            row += 1
        end
        cone(m, CT_QUAD, collect(0:nd))             # t >= ||d|| = ||A - X||_F
        r = solve(m)
        if r.rc != 0
            @printf("optimize rc=%d\n", r.rc)
            return 1
        end
        Xr = getbarxj(m, 0)
        po = r.objective
        X = [Xr[(i-1)*N+j] for i in 1:N, j in 1:N]
    finally
        free(m)
    end
    println("X =")
    for i in 1:N
        print("   ")
        for j in 1:N
            @printf(" %8.4f", X[i, j])
        end
        println()
    end
    @printf("obj (Frobenius norm) = %.6f\n", po)
    Xh = higham()                                   # independent check: Higham
    println("Higham:")
    for i in 1:N
        print("   ")
        for j in 1:N
            @printf(" %8.4f", Xh[i, j])
        end
        println()
    end
    fh, fx = fro(Xh), fro(X)
    @printf("||A-X_higham||_F = %.6f, ||A-X_clone||_F = %.6f\n", fh, fx)
    ev, _ = eig3(X)                                 # feasible: diag = 1, PSD
    psd_ok = all(e -> e >= -1e-6, ev)
    diag_ok = all(i -> abs(X[i, i] - 1.0) <= 1e-5, 1:N)
    ok = psd_ok && diag_ok && fx <= fh + 1e-3
    @printf("%s (PSD=%s, diag=1=%s, norm ~ Higham)\n", ok ? "OK" : "FAIL", psd_ok ? "yes" : "no", diag_ok ? "yes" : "no")
    return ok ? 0 : 1
end

exit(main())
