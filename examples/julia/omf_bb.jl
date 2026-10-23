#!/usr/bin/env julia
# omf_bb.jl — Simple MIP

using Primal
using Primal: Model   # not exported: JuMP exports its own Model

m = Model(0, 0)
add_vars(m, 3)
add_cons(m, 1)

obj(m, [0, 1, 2], [1.0, 1.0, 1.0])
sense(m, SENSE[:MIN])
for j in 0:2
    var_bounds(m, j, BK[:LO]; lo=0.0, up=1.0)
    var_type(m, j, VAR_TYPE[:INT_BIN])
end
a_row(m, 0, [0, 1, 2], [1.0, 1.0, 1.0])
con_bounds(m, 0, BK[:FX]; lo=1.0, up=1.0)

r = solve(m)
ok = r.solsta == SOLSTA[:INTEGER_OPTIMAL]
println(ok ? "OK (omf_bb solved, obj=$(r.objective))" : "FAIL: x=$(r.x), obj=$(r.objective)")
free(m)
exit(ok ? 0 : 1)
