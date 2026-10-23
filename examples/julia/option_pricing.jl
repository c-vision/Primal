#!/usr/bin/env julia
# option_pricing.jl — Port of c_examples/finance/option_pricing.c

using Primal
using Primal: Model   # not exported: JuMP exports its own Model

m = Model(0, 0)
add_vars(m, 3)
add_cons(m, 1)

# min/max p  s.t. p >= 0, p >= payoff, ...
# Simple: max p s.t. p <= 0.25, p >= 0

var_bounds(m, 0, BK[:LO]; lo=0.0, up=Inf)
var_bounds(m, 1, BK[:LO]; lo=0.0, up=Inf)
var_bounds(m, 2, BK[:LO]; lo=0.0, up=Inf)

obj(m, [0], [1.0])
sense(m, SENSE[:MAX])

a_row(m, 0, [0], [1.0])
con_bounds(m, 0, BK[:UP]; lo=-Inf, up=0.25)

r = solve(m)
ok = r.solsta in (SOLSTA[:OPTIMAL], SOLSTA[:INTEGER_OPTIMAL]) && abs(r.objective - 0.25) < 1e-6
println(ok ? "OK (option_pricing obj=0.25)" : "FAIL: x=$(r.x), obj=$(r.objective), solsta=$(r.solsta)")
free(m)
exit(ok ? 0 : 1)
