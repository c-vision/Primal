#!/usr/bin/env julia
# lp_mincut.jl — minimum s-t cut as a linear program, and the CORRELATED
# min-cut (maxdan94/lp_mincut):
#
#   minimize sum_e w_e d_e  s.t.  d_e - p_s(e) + p_t(e) >= 0,  p_s - p_t = 1,  d, p >= 0
#
# The optimum is the min-cut capacity (= max flow). The correlated variant
# adds d_ij = d_kl for a pair of arcs. Instance (the repo's graph.txt, s=0,
# t=3): arcs 0->1 (0.5), 0->2 (0.3), 1->3 (0.4), 2->3 (0.5); min cut 0.7.
# Port of examples/c/lp_mincut.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NE = 4                                        # arcs
const NN = 4                                        # nodes
const SRC = [0, 0, 1, 2]
const DST = [1, 2, 3, 3]
const W = [0.5, 0.3, 0.4, 0.5]

# Solve min-cut with optional correlated arc pairs forced equal; returns (cut value or -1, d).
function mincut(cor)
    D0, P0, nv = 0, NE, NE + NN
    m = Model(0, 0)
    r = try
        add_vars(m, nv)
        sense(m, SENSE_MIN)
        for i in 0:nv-1
            var_bounds(m, i, BK_LO; lo=0.0, up=Inf)
        end
        for e in 0:NE-1
            cj(m, D0 + e, W[e+1])
        end
        add_cons(m, NE + 1 + length(cor))
        row = 0
        for e in 0:NE-1                             # d_e - p_s + p_t >= 0
            a_row(m, row, [D0 + e, P0 + SRC[e+1], P0 + DST[e+1]], [1.0, -1.0, 1.0])
            con_bounds(m, row, BK_LO; lo=0.0, up=Inf); row += 1
        end
        a_row(m, row, [P0 + 0, P0 + 3], [1.0, -1.0])   # p_s - p_t = 1
        con_bounds(m, row, BK_FX; lo=1.0, up=1.0); row += 1
        for (a, b) in cor                           # d_ij = d_kl
            a_row(m, row, [D0 + a, D0 + b], [1.0, -1.0])
            con_bounds(m, row, BK_FX; lo=0.0, up=0.0); row += 1
        end
        solve(m)
    finally
        free(m)
    end
    r.rc == 0 || return -1.0, zeros(NE)
    return r.objective, r.x[D0+1:D0+NE]
end

function main()
    println("lp_mincut (min-cut LP; graph.txt of maxdan94/lp_mincut)")
    all = true
    cut, d = mincut(Tuple{Int,Int}[])
    # independent check: the value must equal the hand max flow 0.4 + 0.3 = 0.7
    good = cut > 0 && abs(cut - 0.7) < 1e-7
    @printf("  min cut = %.8f (expected 0.7)  d=(%.4f,%.4f,%.4f,%.4f)  %s\n",
            cut, d[1], d[2], d[3], d[4], good ? "OK" : "FAIL")
    all &= good

    # correlated variant: force d(0->1) = d(2->3) (arcs 0 and 3); the cut may
    # only rise (a correlation restricts the feasible set)
    cc, d2 = mincut([(0, 3)])
    good2 = cc > 0 && cc >= cut - 1e-9 && abs(d2[1] - d2[4]) < 1e-7
    @printf("  correlated d01=d23: cut = %.8f (d01=%.4f=d23=%.4f)  >= 0.7  %s\n",
            cc, d2[1], d2[4], good2 ? "OK" : "FAIL")
    all &= good2

    @printf("%s\n", all ? "OK" : "FAIL")
    return all ? 0 : 1
end

exit(main())
