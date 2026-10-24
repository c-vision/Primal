#!/usr/bin/env julia
# omf_bb.jl — branch-and-bound examples from "Optimization Methods in Finance"
# (Cornuejols & Tutuncu, 2007), Section 11.3.2. Two pure integer LPs on the
# same feasible region  -x1 + x2 <= 2,  8 x1 + 2 x2 <= 19,  x >= 0 integer:
#
#   (A) max x1 + x2    -> the book's optimum x = (1, 3), objective 4
#   (B) max 3 x1 + x2  -> the book's optimum x = (2, 1), objective 7
#
# The solver's own B&B must land on the same integral points.
# Port of examples/c/omf_bb.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# Solve one branch-and-bound MILP case and check the book optimum.
function solve_case(name, c1, c2, want_obj, wx1, wx2)
    m = Model(2, 2)
    r = try
        for j in 0:1
            var_bounds(m, j, BK_LO; lo=0.0, up=Inf)
            var_type(m, j, VT_INT)
        end
        cj(m, 0, -c1); cj(m, 1, -c2)
        a_row(m, 0, [0, 1], [-1.0, 1.0]); con_bounds(m, 0, BK_UP; lo=-Inf, up=2.0)
        a_row(m, 1, [0, 1], [8.0, 2.0]);  con_bounds(m, 1, BK_UP; lo=-Inf, up=19.0)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        x, z = r.x, r.objective
        ok = abs(z + want_obj) < 1e-6 && abs(x[1] - wx1) < 1e-6 && abs(x[2] - wx2) < 1e-6
        @printf("omf_bb %-3s x=(%.6g,%.6g) obj=%.6g (book x=(%.4g,%.4g), %.4g) %s\n",
                name, x[1], x[2], -z, wx1, wx2, want_obj, ok ? "OK" : "FAIL")
    else
        @printf("omf_bb %-3s rc=%d FAIL\n", name, r.rc)
    end
    return ok
end

function main()
    a = solve_case("(A)", 1.0, 1.0, 4.0, 1.0, 3.0)
    b = solve_case("(B)", 3.0, 1.0, 7.0, 2.0, 1.0)
    return (a && b) ? 0 : 1
end

exit(main())
