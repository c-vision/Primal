#!/usr/bin/env julia
# binary_quadratic.jl — Simple MIQP

using Primal
using Primal: Model   # not exported: JuMP exports its own Model

m = Model(0, 0)
add_vars(m, 2)
add_cons(m, 1)

obj(m, [0, 1], [0.0, 0.0])
sense(m, SENSE[:MIN])
qobj(m, [(0, 0, 2.0), (1, 1, 2.0), (0, 1, 1.0)])
var_type(m, 0, VAR_TYPE[:INT_BIN])
var_type(m, 1, VAR_TYPE[:INT_BIN])
var_bounds(m, 0, BK[:LO]; lo=0.0, up=1.0)
var_bounds(m, 1, BK[:LO]; lo=0.0, up=1.0)
a_row(m, 0, [0, 1], [1.0, 1.0])
con_bounds(m, 0, BK[:LO]; lo=1.0, up=Inf)

r = solve(m)
ok = r.solsta == SOLSTA[:INTEGER_OPTIMAL]
println(ok ? "OK (binary_quadratic solved, obj=$(r.objective))" : "FAIL: x=$(r.x), obj=$(r.objective)")
free(m)
exit(ok ? 0 : 1)
