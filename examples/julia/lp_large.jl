#!/usr/bin/env julia
# lp_large.jl — Simple LP

include("../src/Primal.jl")
using .Primal

m = Model(0, 0)
add_vars(m, 10)
add_cons(m, 5)

obj(m, collect(0:9), ones(10))
sense(m, SENSE[:MIN])
for j in 0:9
    var_bounds(m, j, BK[:LO]; lo=0.0, up=Inf)
end
for i in 0:4
    a_row(m, i, collect(0:9), ones(10))
    con_bounds(m, i, BK[:LO]; lo=1.0, up=Inf)
end

r = solve(m)
ok = r.solsta in (SOLSTA[:OPTIMAL], SOLSTA[:INTEGER_OPTIMAL])
println(ok ? "OK (lp_large solved, obj=$(r.objective))" : "FAIL: x=$(r.x), obj=$(r.objective)")
free(m)
exit(ok ? 0 : 1)
