#!/usr/bin/env julia
# nesting_dotted_board.jl — nesting of convex polygons on a dotted board, a
# MILP (eellookkaa/mddm, the "Dotted-Board Model"). Each polygon picks one dot,
# the board size is minimised:
#
#   minimise z_length + z_width
#   s.t.  sum_d delta[i,d] = 1,  z >= (edge of i at d) delta[i,d],
#         delta[i,d] + delta[j,e] <= 1 for e in NFP(i,j,d),  delta binary
#
# Case: two unit squares on a 3x3 grid of dots at spacing 0.5; two squares
# overlap iff both index differences are <= 1. The optimum is exactly 3.
# Port of examples/c/nesting_dotted_board.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NG = 3                                        # dots per axis (indices 0..2)
const NDOT = NG * NG
const NP = 2                                        # two squares

overlap(d1, d2) = abs(div(d1, NG) - div(d2, NG)) <= 1 && abs(d1 % NG - d2 % NG) <= 1

function main()
    X0, ZL, ZW, NV = 0, NP * NDOT, NP * NDOT + 1, NP * NDOT + 2
    m = Model(0, 0)
    r = try
        add_vars(m, NV)
        sense(m, SENSE_MIN)
        for i in 0:NP-1, d in 0:NDOT-1
            var_type(m, X0 + i * NDOT + d, VT_INT)
            var_bounds(m, X0 + i * NDOT + d, BK_RA; lo=0.0, up=1.0)
        end
        var_bounds(m, ZL, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, ZW, BK_LO; lo=0.0, up=Inf)
        cj(m, ZL, 1.0)
        cj(m, ZW, 1.0)
        pairs = [(i, j, d1, d2) for i in 0:NP-1 for j in 0:NP-1 if i != j
                 for d1 in 0:NDOT-1 for d2 in 0:NDOT-1 if overlap(d1, d2)]
        add_cons(m, 2 * NP * NDOT + NP + length(pairs))
        row = 0
        for i in 0:NP-1, d in 0:NDOT-1              # (edge) delta - z <= 0
            id, jd = div(d, NG), d % NG
            a_row(m, row, [X0 + i * NDOT + d, ZL], [0.5 * id + 1.0, -1.0])
            con_bounds(m, row, BK_UP; lo=-Inf, up=0.0); row += 1
            a_row(m, row, [X0 + i * NDOT + d, ZW], [0.5 * jd + 1.0, -1.0])
            con_bounds(m, row, BK_UP; lo=-Inf, up=0.0); row += 1
        end
        for i in 0:NP-1                             # sum_d delta[i,d] = 1
            a_row(m, row, [X0 + i * NDOT + d for d in 0:NDOT-1], ones(NDOT))
            con_bounds(m, row, BK_FX; lo=1.0, up=1.0); row += 1
        end
        for (i, j, d1, d2) in pairs                 # no-overlap
            a_row(m, row, [X0 + i * NDOT + d1, X0 + j * NDOT + d2], [1.0, 1.0])
            con_bounds(m, row, BK_UP; lo=-Inf, up=1.0); row += 1
        end
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    v, obj = ok ? (r.x, r.objective) : (zeros(NV), 0.0)
    chosen, legal = fill(-1, NP), true              # recover the chosen dots and check them
    for i in 0:NP-1
        for d in 0:NDOT-1
            v[X0+i*NDOT+d+1] > 0.5 && (chosen[i+1] = d)
        end
        chosen[i+1] < 0 && (legal = false)
    end
    if legal
        for i in 1:NP, j in 1:NP
            i != j && overlap(chosen[i], chosen[j]) && (legal = false)
        end
    end
    good = ok && abs(obj - 3.0) < 1e-6 && legal
    coord(c, f) = c >= 0 ? f(c) : -1
    println("nesting_dotted_board (2 squares, board 2x2, fine grid)")
    @printf("  z = %.6f (expected 3)  chosen dots = (%d,%d) / (%d,%d)  legal=%d  %s\n",
            obj, coord(chosen[1], c -> div(c, NG)), coord(chosen[1], c -> c % NG),
            coord(chosen[2], c -> div(c, NG)), coord(chosen[2], c -> c % NG), legal, good ? "OK" : "FAIL")
    return good ? 0 : 1
end

exit(main())
