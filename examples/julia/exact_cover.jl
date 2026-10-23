#!/usr/bin/env julia
# exact_cover.jl — Simple MIP

using Primal
using Primal: Model   # not exported: JuMP exports its own Model

m = Model(0, 0)
add_vars(m, 4)
add_cons(m, 2)

obj(m, [0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0])
sense(m, SENSE[:MIN])
for j in 0:3
    var_bounds(m, j, BK[:LO]; lo=0.0, up=1.0)
    var_type(m, j, VAR_TYPE[:INT_BIN])
end
a_row(m, 0, [0, 1], [1.0, 1.0])
con_bounds(m, 0, BK[:FX]; lo=1.0, up=1.0)
a_row(m, 1, [2, 3], [1.0, 1.0])
con_bounds(m, 1, BK[:FX]; lo=1.0, up=1.0)

r = solve(m)
ok = r.solsta == SOLSTA[:INTEGER_OPTIMAL]
println(ok ? "OK (exact_cover solved, obj=$(r.objective))" : "FAIL: x=$(r.x), obj=$(r.objective)")
free(m)
exit(ok ? 0 : 1)
