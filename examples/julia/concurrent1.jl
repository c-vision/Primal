#!/usr/bin/env julia
# concurrent1.jl — MOSEK port "concurrent1": several algorithms competing on
# the same problem, the best one is taken. The optimizers (FREE, INTPNT,
# PRIMAL_SIMPLEX, DUAL_SIMPLEX) run in sequence and their pobj are compared.
#
# Problem: max 3x0 + x1 + 5x2 + x3 with EQ/LO/UP rows, pobj 250/3 ~ 83.333.
# Port of examples/c/concurrent1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const IPAR_OPTIMIZER = 0
const OPTIMIZER_FREE, OPTIMIZER_INTPNT, OPTIMIZER_DUAL_SIMPLEX, OPTIMIZER_PRIMAL_SIMPLEX = 2, 4, 1, 8

# Build the reference LP used by every optimizer.
function build(m)
    add_vars(m, 4)
    add_cons(m, 3)
    sense(m, SENSE_MAX)
    c = [3.0, 1, 5, 1]
    for j in 0:3
        cj(m, j, c[j+1])
    end
    con_bounds(m, 0, BK_FX; lo=30, up=30)
    con_bounds(m, 1, BK_LO; lo=15, up=0)
    con_bounds(m, 2, BK_UP; lo=0, up=25)
    var_bounds(m, 0, BK_LO; lo=0, up=0)
    var_bounds(m, 1, BK_RA; lo=0, up=10)
    var_bounds(m, 2, BK_LO; lo=0, up=0)
    var_bounds(m, 3, BK_LO; lo=0, up=0)
    a_col(m, 0, [0, 1], [3, 2])
    a_col(m, 1, [0, 1, 2], [1, 1, 2])
    a_col(m, 2, [0, 1], [2, 3])
    a_col(m, 3, [1, 2], [1, 3])
end

function main()
    names = ["FREE", "INTPNT", "PRIMAL_SIMPLEX", "DUAL_SIMPLEX"]
    opt = [OPTIMIZER_FREE, OPTIMIZER_INTPNT, OPTIMIZER_PRIMAL_SIMPLEX, OPTIMIZER_DUAL_SIMPLEX]
    best, bopt = -Inf, -1
    vals = zeros(4)
    all_ok = true
    for k in 1:4
        m = Model(0, 0)
        po = -Inf
        try
            build(m)
            int_param(m, IPAR_OPTIMIZER, opt[k])
            r = solve(m)
            if r.rc == 0 && r.solsta == SOLSTA_OPTIMAL
                po = r.objective
                @printf("%-16s pobj = %.6f\n", names[k], po)
            else
                all_ok = false
            end
        finally
            free(m)
        end
        vals[k] = po
        po > best && ((best, bopt) = (po, k))
    end
    @printf("best: %s with pobj = %.6f (expected 83.333333 = 250/3)\n", bopt >= 1 ? names[bopt] : "?", best)
    ok = all_ok && abs(best - 250.0 / 3.0) < 1e-6
    any(k -> abs(vals[k] - best) > 1e-6, 1:4) && (ok = false)   # all agree
    @printf("%s (4 optimizers, same optimum — simulated concurrency)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
