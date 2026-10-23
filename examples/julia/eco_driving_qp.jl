#!/usr/bin/env julia
# eco_driving_qp.jl — Simple QP

using Primal
using Primal: Model   # not exported: JuMP exports its own Model

m = Model(0, 0)
add_vars(m, 2)
add_cons(m, 1)

obj(m, [0, 1], [1.0, 1.0])
sense(m, SENSE[:MIN])
qobj(m, [(0, 0, 2.0), (1, 1, 2.0)])
var_bounds(m, 0, BK[:LO]; lo=0.0, up=Inf)
var_bounds(m, 1, BK[:LO]; lo=0.0, up=Inf)
a_row(m, 0, [0, 1], [1.0, 1.0])
con_bounds(m, 0, BK[:FX]; lo=1.0, up=1.0)

r = solve(m)
ok = r.solsta in (SOLSTA[:OPTIMAL], SOLSTA[:INTEGER_OPTIMAL])
println(ok ? "OK (eco_driving_qp solved, obj=$(r.objective))" : "FAIL: x=$(r.x), obj=$(r.objective)")
free(m)
exit(ok ? 0 : 1)
