#!/usr/bin/env julia
# feasrepairex1.jl — Port of c_examples/mosek_comparison/feasrepairex1.c
# Simple feasibility repair example.

include("../src/Primal.jl")
using .Primal

m = Model(0, 0)
add_vars(m, 2)
add_cons(m, 2)

obj(m, [0, 1], [1.0, 1.0])
sense(m, SENSE[:MIN])

a_row(m, 0, [0, 1], [1.0, 1.0])
con_bounds(m, 0, BK[:LO]; lo=5.0, up=Inf)

a_row(m, 1, [0, 1], [1.0, -1.0])
con_bounds(m, 1, BK[:UP]; lo=-Inf, up=0.0)

var_bounds(m, 0, BK[:LO]; lo=0.0, up=Inf)
var_bounds(m, 1, BK[:LO]; lo=0.0, up=Inf)

r = solve(m)

ok = r.solsta == SOLSTA[:OPTIMAL] && abs(r.objective - 5.0) < 1e-6
println(ok ? "OK (obj=5.0)" : "FAIL: x=$(r.x), obj=$(r.objective), solsta=$(r.solsta)")
free(m)
exit(ok ? 0 : 1)
