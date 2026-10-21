#!/usr/bin/env julia
# assignment.jl — Simple LP assignment

include("../src/Primal.jl")
using .Primal

m = Model(0, 0)
add_vars(m, 4)
add_cons(m, 2)

obj(m, [0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0])
sense(m, SENSE[:MIN])
for j in 0:3
    var_bounds(m, j, BK[:LO]; lo=0.0, up=1.0)
end
a_row(m, 0, [0, 1], [1.0, 1.0])
con_bounds(m, 0, BK[:FX]; lo=1.0, up=1.0)
a_row(m, 1, [2, 3], [1.0, 1.0])
con_bounds(m, 1, BK[:FX]; lo=1.0, up=1.0)

r = solve(m)
ok = r.solsta in (SOLSTA[:OPTIMAL], SOLSTA[:INTEGER_OPTIMAL])
println(ok ? "OK (assignment solved, obj=$(r.objective))" : "FAIL: x=$(r.x), obj=$(r.objective)")
free(m)
exit(ok ? 0 : 1)
