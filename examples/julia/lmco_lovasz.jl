#!/usr/bin/env julia
# lmco_lovasz.jl — Lovasz capacity number of C5, "Lectures on Modern Convex
# Optimization 2020-2023" (Ben-Tal & Nemirovski). With the 5-cycle C5 and the
# affine matrix L(x) (1 between non-adjacent nodes, an arc variable between
# adjacent ones),
#
#   theta(C5) = min { lambda : lambda I - L(x) PSD } = sqrt(5)  (the book).
#
# Model: bar variable X (5x5 PSD), one row per entry (i <= j):
# <E_ij, X> = (lambda I - L(x))_ij. Port of examples/c/lmco_lovasz.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# index of the arc variable for pair (i,j), i<j; -1 if non-adjacent
const ARC = [-1 1 -1 -1 5; -1 -1 2 -1 -1; -1 -1 -1 3 -1; -1 -1 -1 -1 4; -1 -1 -1 -1 -1]

function main()
    m = Model(0, 0)
    try
        add_vars(m, 6)                              # 0 = lambda, 1..5 = xAB, xBC, xCD, xDE, xEA
        var_bounds(m, 0, BK_LO; lo=1.0, up=Inf)
        for k in 1:5
            var_bounds(m, k, BK_RA; lo=-10.0, up=10.0)
        end
        cj(m, 0, 1.0)                               # min lambda
        msym = Int[]
        for i in 0:4, j in i:4
            push!(msym, sparsesymmat(m, 5, [(i, j, 1.0)]))
        end
        barvars(m, [5])
        add_cons(m, 15)
        row = 0
        for i in 0:4, j in i:4
            baraij(m, row, 0, [(msym[row+1], 1.0)])
            if i == j                               # <E_ii,X> = lambda - 1
                a_row(m, row, [0], [-1.0])
                con_bounds(m, row, BK_FX; lo=-1.0, up=-1.0)
            elseif ARC[i+1, j+1] >= 0               # <E_ij,X> = 2X_ij = -2 x_arc
                a_row(m, row, [ARC[i+1, j+1]], [2.0])
                con_bounds(m, row, BK_FX; lo=0.0, up=0.0)
            else                                    # non-arc: <E_ij,X> = -2
                con_bounds(m, row, BK_FX; lo=-2.0, up=-2.0)
            end
            row += 1
        end
        r = solve(m)
        ok = r.rc == 0
        if ok
            x = r.x
            ok = abs(x[1] - sqrt(5.0)) < 1e-6
            @printf("lmco_lovasz  theta(C5)=%.9g (book sqrt(5)=%.9g) %s\n", x[1], sqrt(5.0), ok ? "OK" : "FAIL")
        else
            @printf("lmco_lovasz  rc=%d FAIL\n", r.rc)
        end
        return ok ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
