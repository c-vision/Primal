#!/usr/bin/env julia
# multi_period.jl — Port of c_examples/finance/multi_period.c

using Primal
using Primal: Model   # not exported: JuMP exports its own Model

m = Model(0, 0)
add_vars(m, 4)
add_cons(m, 2)

# Simple two-period portfolio
obj(m, [0, 1], [1.0, 1.0])
sense(m, SENSE[:MAX])
var_bounds(m, 0, BK[:LO]; lo=0.0, up=Inf)
var_bounds(m, 1, BK[:LO]; lo=0.0, up=Inf)
var_bounds(m, 2, BK[:LO]; lo=0.0, up=Inf)
var_bounds(m, 3, BK[:LO]; lo=0.0, up=Inf)

a_row(m, 0, [0, 1], [1.0, 1.0])
con_bounds(m, 0, BK[:UP]; lo=-Inf, up=1.0)

a_row(m, 1, [2, 3], [1.0, 1.0])
con_bounds(m, 1, BK[:UP]; lo=-Inf, up=1.0)

r = solve(m)
ok = r.solsta in (SOLSTA[:OPTIMAL], SOLSTA[:INTEGER_OPTIMAL]) && abs(r.objective - 1.0) < 1e-6
println(ok ? "OK (multi_period obj=1.0)" : "FAIL: x=$(r.x), obj=$(r.objective), solsta=$(r.solsta)")
free(m)
exit(ok ? 0 : 1)
