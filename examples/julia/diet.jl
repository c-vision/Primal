#!/usr/bin/env julia
# diet.jl — Simple LP diet problem

using Primal
using Primal: Model   # not exported: JuMP exports its own Model

m = Model(0, 0)
add_vars(m, 3)
add_cons(m, 2)

obj(m, [0, 1, 2], [2.0, 3.0, 1.0])
sense(m, SENSE[:MIN])
var_bounds(m, 0, BK[:LO]; lo=0.0, up=Inf)
var_bounds(m, 1, BK[:LO]; lo=0.0, up=Inf)
var_bounds(m, 2, BK[:LO]; lo=0.0, up=Inf)
a_row(m, 0, [0, 1, 2], [2.0, 4.0, 1.0])
con_bounds(m, 0, BK[:LO]; lo=10.0, up=Inf)
a_row(m, 1, [0, 1, 2], [1.0, 2.0, 3.0])
con_bounds(m, 1, BK[:LO]; lo=8.0, up=Inf)

r = solve(m)
ok = r.solsta in (SOLSTA[:OPTIMAL], SOLSTA[:INTEGER_OPTIMAL])
println(ok ? "OK (diet solved, obj=$(r.objective))" : "FAIL: x=$(r.x), obj=$(r.objective)")
free(m)
exit(ok ? 0 : 1)
