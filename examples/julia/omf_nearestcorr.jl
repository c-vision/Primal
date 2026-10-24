#!/usr/bin/env julia
# omf_nearestcorr.jl — nearest correlation matrix, "Optimization Methods in
# Finance" (Cornuejols & Tutuncu, 2007), Example 10.1. Given an invalid
# correlation estimate Sigma_hat (smallest eigenvalue -0.1337), solve
#
#   min ||Sigma - Sigma_hat||_F  s.t.  Sigma PSD,  diag(Sigma) = 1.
#
# Model: bar variable X (4x4 PSD), an auxiliary d_ij per i <= j with
# sum d_ij^2 = ||Sigma_hat - X||_F^2, and the cone (t, d). Checked against
# Higham's alternating projections and the book's 2-digit values.
# Port of examples/c/omf_nearestcorr.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const N = 4
const A = [1.0 0.8 0.5 0.2; 0.8 1.0 0.9 0.1; 0.5 0.9 1.0 0.7; 0.2 0.1 0.7 1.0]
# reference from the book (2 significant digits)
const REF = [1.00 0.76 0.53 0.18; 0.76 1.00 0.82 0.15; 0.53 0.82 1.00 0.65; 0.18 0.15 0.65 1.00]
# independent reference: Higham alternating projections (computed offline)
const HIGHAM = [1.0000 0.7698 0.5301 0.1823; 0.7698 1.0000 0.8169 0.1488;
                0.5301 0.8169 1.0000 0.6513; 0.1823 0.1488 0.6513 1.0000]

function main()
    nd = N * (N + 1) ÷ 2
    m = Model(0, 0)
    try
        add_vars(m, 1 + nd)                         # 0 = t (Frobenius norm), 1..nd = d_ij
        var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
        cj(m, 0, 1.0)
        for k in 0:nd-1
            var_bounds(m, 1 + k, BK_FR; lo=-Inf, up=Inf)
        end
        msym = [sparsesymmat(m, N, [(i, j, 1.0)]) for i in 0:N-1 for j in i:N-1]
        barvars(m, [N])
        add_cons(m, nd + N)
        row = 0
        for i in 0:N-1, j in i:N-1
            w = i == j ? 1.0 : 2.0
            rhs = i == j ? A[i+1, j+1] : 2.0 * A[i+1, j+1]
            baraij(m, row, 0, [(msym[row+1], 1.0)])
            a_row(m, row, [1 + row], [sqrt(w)])
            con_bounds(m, row, BK_FX; lo=rhs, up=rhs)
            row += 1
        end
        for k in 0:N-1                              # X_kk = 1
            baraij(m, row, 0, [(msym[k*(2N-k+1)÷2+1], 1.0)])
            con_bounds(m, row, BK_FX; lo=1.0, up=1.0)
            row += 1
        end
        cone(m, CT_QUAD, collect(0:nd))             # t >= ||d|| = ||Sigma_hat - X||_F
        r = solve(m)
        ok = r.rc == 0
        if ok
            Xr, po = getbarxj(m, 0), r.objective
            fro, dev_book, dev_high = 0.0, 0.0, 0.0
            for i in 1:N, j in 1:N
                xij = Xr[(i-1)*N+j]
                d = xij - A[i, j]; fro += d * d
                abs(xij - REF[i, j]) > dev_book && (dev_book = abs(xij - REF[i, j]))
                abs(xij - HIGHAM[i, j]) > dev_high && (dev_high = abs(xij - HIGHAM[i, j]))
            end
            fro = sqrt(fro)
            ok = dev_high < 5e-3 && abs(Xr[1] - 1.0) < 1e-5 && abs(po - fro) < 1e-4
            @printf("omf_nearestcorr  ||Sig-X||_F=%.6g  max|X-Higham|=%.4g  max|X-book|=%.4g (book 2 sig figs) %s\n",
                    fro, dev_high, dev_book, ok ? "OK" : "FAIL")
        else
            @printf("omf_nearestcorr  rc=%d FAIL\n", r.rc)
        end
        return ok ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
