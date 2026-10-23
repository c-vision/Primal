#!/usr/bin/env julia
# lp_ineq_large.jl — Simple LP

using Primal
using Primal: Model   # not exported: JuMP exports its own Model

m = Model(0, 0)
add_vars(m, 5)
add_cons(m, 3)

obj(m, [0, 1, 2, 3, 4], [1.0, 1.0, 1.0, 1.0, 1.0])
sense(m, SENSE[:MIN])
for j in 0:4
    var_bounds(m, j, BK[:LO]; lo=0.0, up=Inf)
end
for i in 0:2
    a_row(m, i, collect(0:4), ones(5))
    con_bounds(m, i, BK[:LO]; lo=1.0, up=Inf)
end

r = solve(m)
ok = r.solsta in (SOLSTA[:OPTIMAL], SOLSTA[:INTEGER_OPTIMAL])
println(ok ? "OK (lp_ineq_large solved, obj=$(r.objective))" : "FAIL: x=$(r.x), obj=$(r.objective)")
free(m)
exit(ok ? 0 : 1)
