#!/usr/bin/env julia
# exact_cover.jl — exact planar cover with rectangular bricks, by binary MIP
# (MOSEK/Tutorials, exact-planar-cover). On an R x C board, choose placements
# x_{s,r,c} in {0,1} covering every cell exactly once, with the fewest bricks:
#
#   sum_{placements covering (i,j)} x = 1  for every cell,  minimize sum x
#
# Instances: A) 2x3 board, dominoes 1x2 / 2x1 -> 3 bricks;
#            B) 3x3 board, 2x2 squares        -> infeasible (9 not divisible by 4).
# Port of examples/c/exact_cover.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# Returns (1, bricks) if solved to optimality, (0, 0) if infeasible, (-1, 0) on another failure.
function cover(R, C, h, w)
    ns = length(h)
    var = fill(-1, ns, R, C)                        # placement -> variable index (or -1)
    nv = 0
    for s in 1:ns, r in 0:R-1, c in 0:C-1
        if r + h[s] <= R && c + w[s] <= C
            var[s, r+1, c+1] = nv
            nv += 1
        end
    end
    m = Model(0, 0)
    try
        add_vars(m, nv)
        add_cons(m, R * C)
        for k in 0:nv-1
            var_bounds(m, k, BK_RA; lo=0.0, up=1.0)
            var_type(m, k, VT_INT_BIN)
            cj(m, k, 1.0)
        end
        for i in 0:R-1, j in 0:C-1
            sub = Int[]
            for s in 1:ns, r in 0:i, c in 0:j
                var[s, r+1, c+1] < 0 && continue
                (i < r + h[s] && j < c + w[s]) && push!(sub, var[s, r+1, c+1])
            end
            a_row(m, i * C + j, sub, ones(length(sub)))
            con_bounds(m, i * C + j, BK_FX; lo=1.0, up=1.0)
        end
        r = solve(m)
        r.rc == 0 && return 1, trunc(Int, r.objective + 0.5)
        (r.rc == RC[:ERR_INFEASIBLE] || r.rc == RC[:ERR_UNBOUNDED]) && return 0, 0   # no exact cover
        return -1, 0
    finally
        free(m)
    end
end

function main()
    a, na = cover(2, 3, [1, 2], [2, 1])             # dominoes
    b, _ = cover(3, 3, [2], [2])                    # 2x2 squares
    status(v) = v == 1 ? "optimal" : (v == 0 ? "infeasible" : "error")
    ok = (a == 1 && na == 3) && b == 0
    @printf("exact_cover\n")
    @printf("  A) 2x3, domino: %s  bricks=%d (expected 3)\n", status(a), na)
    @printf("  B) 3x3, 2x2:    %s (expected infeasible)\n", status(b))
    @printf("%s\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
