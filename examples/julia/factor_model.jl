#!/usr/bin/env julia
# factor_model.jl — Simple QP

using Primal
using Primal: Model   # not exported: JuMP exports its own Model

m = Model(0, 0)
add_vars(m, 3)
add_cons(m, 1)

obj(m, [0, 1, 2], [-0.10, -0.15, -0.20])
sense(m, SENSE[:MIN])
qobj(m, [(0, 0, 0.08), (0, 1, 0.04), (1, 1, 0.18), (2, 2, 0.32)])
var_bounds(m, 0, BK[:LO]; lo=0.0, up=Inf)
var_bounds(m, 1, BK[:LO]; lo=0.0, up=Inf)
var_bounds(m, 2, BK[:LO]; lo=0.0, up=Inf)
a_row(m, 0, [0, 1, 2], [1.0, 1.0, 1.0])
con_bounds(m, 0, BK[:FX]; lo=1.0, up=1.0)

r = solve(m)
ok = r.solsta in (SOLSTA[:OPTIMAL], SOLSTA[:INTEGER_OPTIMAL])
println(ok ? "OK (factor_model solved, obj=$(r.objective))" : "FAIL: x=$(r.x), obj=$(r.objective)")
free(m)
exit(ok ? 0 : 1)
