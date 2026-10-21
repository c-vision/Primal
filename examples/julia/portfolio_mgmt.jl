#!/usr/bin/env julia
# portfolio_mgmt.jl — Simple LP

include("../src/Primal.jl")
using .Primal

m = Model(0, 0)
add_vars(m, 5)
add_cons(m, 1)

obj(m, collect(0:4), ones(5))
sense(m, SENSE[:MIN])
for j in 0:4
    var_bounds(m, j, BK[:LO]; lo=0.0, up=Inf)
end
a_row(m, 0, collect(0:4), ones(5))
con_bounds(m, 0, BK[:FX]; lo=1.0, up=1.0)

r = solve(m)
ok = r.solsta in (SOLSTA[:OPTIMAL], SOLSTA[:INTEGER_OPTIMAL])
println(ok ? "OK (portfolio_mgmt solved, obj=$(r.objective))" : "FAIL: x=$(r.x), obj=$(r.objective)")
free(m)
exit(ok ? 0 : 1)
