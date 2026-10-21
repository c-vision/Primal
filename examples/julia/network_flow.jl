#!/usr/bin/env julia
# network_flow.jl — Simple LP network flow

include("../src/Primal.jl")
using .Primal

m = Model(0, 0)
add_vars(m, 5)
add_cons(m, 2)

obj(m, [0, 1, 2, 3, 4], [1.0, 1.0, 1.0, 1.0, 1.0])
sense(m, SENSE[:MIN])
for j in 0:4
    var_bounds(m, j, BK[:LO]; lo=0.0, up=Inf)
end
a_row(m, 0, [0, 1, 2, 3, 4], [1.0, 1.0, 1.0, 1.0, 1.0])
con_bounds(m, 0, BK[:FX]; lo=5.0, up=5.0)
a_row(m, 1, [0, 1, 2], [1.0, -1.0, 0.0])
con_bounds(m, 1, BK[:FX]; lo=0.0, up=0.0)

r = solve(m)
ok = r.solsta in (SOLSTA[:OPTIMAL], SOLSTA[:INTEGER_OPTIMAL])
println(ok ? "OK (network_flow solved, obj=$(r.objective))" : "FAIL: x=$(r.x), obj=$(r.objective)")
free(m)
exit(ok ? 0 : 1)
