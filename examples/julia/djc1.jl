#!/usr/bin/env julia
# djc1.jl — MOSEK port "djc1": disjunctive constraints (DJC) with the
# reference API (appenddjcs + putdjc on AFEs and domains).
#
#   min / max x0  s.t.  0 <= x0 <= 10  and  [x0 <= 2] OR [6 <= x0 <= 7]
#
# min -> x0 = 0 (first disjunct), max -> x0 = 7 (second). One AFE f = x0;
# term 0 is RMINUS (f - 2 <= 0), term 1 is RMINUS (f - 7) AND RPLUS (f - 6).
# Port of examples/c/djc1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const DOM_RPLUS = 2
const DOM_RMINUS = 3

# Solve the disjunction for min or max of x0; returns (objective, x0).
function solve_djc(maximize)
    m = Model(0, 0)
    po, x0 = -1e30, NaN
    try
        add_vars(m, 1)
        var_bounds(m, 0, BK_RA; lo=0.0, up=10.0)
        cj(m, 0, 1.0)
        maximize && sense(m, SENSE_MAX)
        afes(m, 1)                                  # one AFE f = x0, two 1-D linear domains
        afe_entry(m, 0, 0, 1.0)
        dminus = domain(m, DOM_RMINUS; n=1)
        dplus = domain(m, DOM_RPLUS; n=1)
        djcs(m, 1)
        djc(m, 0, [dminus, dminus, dplus], [0, 0, 0], [2.0, 7.0, 6.0], [1, 2])
        r = solve(m)                                # x includes the DJC binaries
        if r.rc == 0
            po, x0 = r.objective, r.x[1]
        end
    finally
        free(m)
    end
    return po, x0
end

function main()
    pmin, xmin = solve_djc(false)
    pmax, xmax = solve_djc(true)
    @printf("min: x0 = %.4f (expected 0), obj = %.4f\n", xmin, pmin)
    @printf("max: x0 = %.4f (expected 7), obj = %.4f\n", xmax, pmax)
    ok = abs(xmin) < 1e-6 && abs(pmin) < 1e-6 && abs(xmax - 7.0) < 1e-6 && abs(pmax - 7.0) < 1e-6
    @printf("%s (disjunction: x<=2 OR 6<=x<=7)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
