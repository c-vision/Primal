#!/usr/bin/env julia
# qp_sparse.jl — Simple QP

include("../src/Primal.jl")
using .Primal

m = Model(0, 0)
add_vars(m, 3)
add_cons(m, 1)

obj(m, [0, 1, 2], [-1.0, -2.0, -3.0])
sense(m, SENSE[:MIN])
qobj(m, [(0, 0, 2.0), (1, 1, 2.0), (2, 2, 2.0)])
for j in 0:2
    var_bounds(m, j, BK[:LO]; lo=0.0, up=Inf)
end
a_row(m, 0, [0, 1, 2], [1.0, 1.0, 1.0])
con_bounds(m, 0, BK[:FX]; lo=1.0, up=1.0)

r = solve(m)
ok = r.solsta in (SOLSTA[:OPTIMAL], SOLSTA[:INTEGER_OPTIMAL])
println(ok ? "OK (qp_sparse solved, obj=$(r.objective))" : "FAIL: x=$(r.x), obj=$(r.objective)")
free(m)
exit(ok ? 0 : 1)
