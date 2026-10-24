#!/usr/bin/env julia
# diet.jl — Stigler's nutrition model (the DIET LP, GAMS library / Dantzig
# 1963), with its LP dual as an independent cross-check:
#
#   primal: min c'x  s.t.  N x >= b,  x >= 0
#   dual:   max b'y  s.t.  N'y <= c,  y >= 0
#
# Hand case: N = [2 1 0; 1 2 0; 0 0 1], b = (4, 5, 1), c = (1, 1, 2):
# x = (1, 2, 1), cost 5; y = (1/3, 1/3, 2), value 5. Port of examples/c/diet.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NF = 3
const NN = 3
const N = [2.0 1.0 0.0; 1.0 2.0 0.0; 0.0 0.0 1.0]
const BB = [4.0, 5.0, 1.0]
const CC = [1.0, 1.0, 2.0]

# Solve one LP; rows[i] are the coefficient rows, returns (ok, x, objective).
function solve_lp(nv, rows, bk, rhs, c, maximize)
    m = Model(0, 0)
    r = try
        add_vars(m, nv)
        add_cons(m, length(rows))
        sense(m, maximize ? SENSE_MAX : SENSE_MIN)
        for j in 0:nv-1
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
        end
        for (i, row) in enumerate(rows)
            a_row(m, i - 1, collect(0:nv-1), row)
            bk == BK_LO ? con_bounds(m, i - 1, BK_LO; lo=rhs[i], up=Inf) :
                          con_bounds(m, i - 1, BK_UP; lo=-Inf, up=rhs[i])
        end
        for j in 0:nv-1
            cj(m, j, c[j+1])
        end
        solve(m)
    finally
        free(m)
    end
    r.rc == 0 || return false, zeros(nv), 0.0
    return true, r.x[1:nv], r.objective
end

function main()
    println("diet (Stigler, 3 foods x 3 nutrients)")
    allok = true

    # primal: min c'x s.t. N x >= b, x >= 0
    ok, xp, cost = solve_lp(NF, [N[i, :] for i in 1:NN], BK_LO, BB, CC, false)
    mindef = 0.0                                    # nutrient totals at xp and the deficit
    for i in 1:NN
        s = 0.0
        for j in 1:NF
            s += N[i, j] * xp[j]
        end
        BB[i] - s > mindef && (mindef = BB[i] - s)
    end
    good = ok && abs(cost - 5.0) < 1e-7 && abs(xp[1] - 1.0) < 1e-7 && abs(xp[2] - 2.0) < 1e-7 &&
           abs(xp[3] - 1.0) < 1e-7 && mindef < 1e-9
    @printf("  primal: cost=%.6f (expected 5)  x=(%.4f,%.4f,%.4f)  deficit=%.2e  %s\n",
            cost, xp[1], xp[2], xp[3], mindef, good ? "OK" : "FAIL")
    allok &= good

    # dual: max b'y s.t. N'y <= c, y >= 0
    ok, yd, value = solve_lp(NN, [N[:, j] for j in 1:NF], BK_UP, CC, BB, true)
    vmax = 0.0
    for j in 1:NF
        s = 0.0
        for i in 1:NN
            s += N[i, j] * yd[i]
        end
        s - CC[j] > vmax && (vmax = s - CC[j])
    end
    good = ok && abs(value - 5.0) < 1e-7 && vmax < 1e-9 && abs(value - cost) < 1e-7
    @printf("  dual:    value=%.6f (expected 5)  y=(%.4f,%.4f,%.4f)  viol=%.2e  strong duality %s\n",
            value, yd[1], yd[2], yd[3], vmax, good ? "OK" : "FAIL")
    allok &= good

    @printf("%s\n", allok ? "OK" : "FAIL")
    return allok ? 0 : 1
end

exit(main())
