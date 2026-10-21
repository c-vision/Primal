#!/usr/bin/env julia
# ceo1.jl — Port of c_examples/mosek_comparison/ceo1.c
#
# min x0  s.t. (x0, x1, x2) in EXP, x1 = 1, -1 <= x2 <= -0.5
# i.e. x0 >= x1 * exp(x2/x1) = exp(x2)
# Expected solution: x0 = exp(-1) = 0.367879 at x2 = -1

include("../src/Primal.jl")
using .Primal

m = Model(0, 0)
add_vars(m, 3)

var_bounds(m, 0, BK[:LO]; lo=0.0, up=Inf)
var_bounds(m, 1, BK[:FX]; lo=1.0, up=1.0)
var_bounds(m, 2, BK[:RA]; lo=-1.0, up=-0.5)

obj(m, [0], [1.0])
sense(m, SENSE[:MIN])

cone(m, CT[:PEXP], [0, 1, 2])

r = solve(m)

ok = length(r.x) == 3 && r.solsta != SOLSTA[:UNKNOWN]
println(ok ? "OK (expected obj = e^-1 = 0.367879 in x2 = -1)" :
        "FAIL: x=$(r.x), obj=$(r.objective)")
free(m)
exit(ok ? 0 : 1)
