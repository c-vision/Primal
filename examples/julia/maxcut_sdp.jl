#!/usr/bin/env julia
# maxcut_sdp.jl — max-cut SDP relaxation on a complete bipartite graph:
#
#   min (1/4) sum_{(i,j) in E} X_ij  s.t.  X_ii = 1,  X PSD
#
# For a bipartite graph the relaxation is exact: the optimum is -|E|/4
# (all edges cut), i.e. -p q/4 for K_{p,q}.
#
#   julia maxcut_sdp.jl [p] [q]   (default 4 4)
#
# Port of examples/c/maxcut_sdp.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    p = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 4
    q = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 4
    (p < 1 || q < 1) && return 2
    n = p + q
    E = p * q
    expected = -0.25 * E

    m = Model(0, 0)
    try
        add_vars(m, 1)
        var_bounds(m, 0, BK_FX; lo=0.0, up=0.0)     # dummy scalar
        barvars(m, [n])
        add_cons(m, n)                              # X_ii = 1
        # objective: 1/4 on each edge X_ij (i<p<=j); <E_ij,X> = 2 X_ij, so val = 1/8
        mC = sparsesymmat(m, n, [(i, j, 0.125) for i in 0:p-1 for j in p:n-1])
        barcj(m, 0, [(mC, 1.0)])
        for i in 0:n-1
            mE = sparsesymmat(m, n, [(i, i, 1.0)])
            baraij(m, i, 0, [(mE, 1.0)])
            con_bounds(m, i, BK_FX; lo=1.0, up=1.0)
        end
        # valid inequalities |X_ij| <= 1 on the edges (implied by PSD + diag=1):
        # they tighten the outer LP relaxation substantially. <E_ij,X> = 2 X_ij.
        row = n
        add_cons(m, E)
        for i in 0:p-1, j in p:n-1
            mO = sparsesymmat(m, n, [(i, j, 1.0)])
            baraij(m, row, 0, [(mO, 1.0)])
            con_bounds(m, row, BK_RA; lo=-2.0, up=2.0)
            row += 1
        end

        t0 = time()
        r = solve(m)
        sec = time() - t0
        obj = 0.0
        ok = r.rc == 0
        if ok
            obj = r.objective
            ok = abs(obj - expected) < 1e-4 * (1.0 + abs(expected))
        end
        @printf("maxcut_sdp p=%d q=%d n=%d edges=%d obj=%.6f ref=%.6f rc=%d t=%.4fs %s\n",
                p, q, n, E, obj, expected, r.rc, sec, ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
