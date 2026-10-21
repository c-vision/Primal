#!/usr/bin/env julia
# mil1.jl — Simple MIP: max x0+x1 s.t. x0+x1<=2.5, x integer

include("../src/Primal.jl")
using .Primal

m = Model(0, 0)
add_vars(m, 2)
add_cons(m, 1)

obj(m, [0, 1], [1.0, 1.0])
sense(m, SENSE[:MAX])
var_bounds(m, 0, BK[:LO]; lo=0.0, up=Inf)
var_bounds(m, 1, BK[:LO]; lo=0.0, up=Inf)
a_row(m, 0, [0, 1], [1.0, 1.0])
con_bounds(m, 0, BK[:UP]; lo=-Inf, up=2.5)
var_type(m, 0, VAR_TYPE[:INT])
var_type(m, 1, VAR_TYPE[:INT])

r = solve(m)
ok = r.solsta == SOLSTA[:INTEGER_OPTIMAL] && abs(r.objective - 2.0) < 1e-6
println(ok ? "OK (mil1 obj=2.0)" : "FAIL: x=$(r.x), obj=$(r.objective)")
free(m)
exit(ok ? 0 : 1)
