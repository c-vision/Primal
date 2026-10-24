#!/usr/bin/env julia
# transaction_cost.jl — transaction-cost LP with free variables and a
# degenerate (cost-free) column (StackOverflow 37586543, MOSEK Fusion).
# n = 3, initial holdings x0 = (-20, -50, -10), costs t = (.01, .01, .01):
#
#   min t'z  s.t.  l - x0 >= 0,  z - (x - x0) >= 0,  z - (x0 - x) >= 0,  e'x = 0
#   x free, z free, l >= 0
#
# z >= |x - x0| and e'(x - x0) = 80 give the value 0.8 on a whole FACE (every
# x_i >= x0_i); the asker's vector (60, -50, -10) is one of its points, so the
# check is the value plus face membership. Port of examples/c/transaction_cost.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    x0 = [-20.0, -50.0, -10.0]
    m = Model(0, 0)
    r = try
        add_vars(m, 9)                              # 0..2 x, 3..5 z, 6..8 l
        add_cons(m, 10)                             # 0..2 long1, 3..5 buy, 6..8 sell, 9 longeqshort
        for i in 0:2
            var_bounds(m, i, BK_FR; lo=-Inf, up=Inf)
            var_bounds(m, 3 + i, BK_FR; lo=-Inf, up=Inf)
            var_bounds(m, 6 + i, BK_LO; lo=0.0, up=Inf)
            cj(m, 3 + i, 0.01)
        end
        for i in 0:2
            a_row(m, i, [6 + i], [1.0])
            con_bounds(m, i, BK_LO; lo=x0[i+1], up=Inf)
            a_row(m, 3 + i, [3 + i, i], [1.0, -1.0])
            con_bounds(m, 3 + i, BK_LO; lo=-x0[i+1], up=Inf)
            a_row(m, 6 + i, [3 + i, i], [1.0, 1.0])
            con_bounds(m, 6 + i, BK_LO; lo=x0[i+1], up=Inf)
        end
        a_row(m, 9, [0, 1, 2], [1.0, 1.0, 1.0])
        con_bounds(m, 9, BK_FX; lo=0.0, up=0.0)
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    obj, x, pinf = r.objective, r.x, r.pinf
    sx, viol = 0.0, 0.0
    for i in 1:3
        sx += x[i]
        x0[i] - x[i] > viol && (viol = x0[i] - x[i])
        need = abs(x[i] - x0[i])
        need - x[3+i] > viol && (viol = need - x[3+i])
    end
    on_face = viol <= 1e-6 && abs(sx) <= 1e-6
    ok = abs(obj - 0.8) < 1e-6 && on_face && pinf <= 1e-6
    @printf("x = (%.4f, %.4f, %.4f), obj = %.6f (expected 0.8)\n", x[1], x[2], x[3], obj)
    @printf("e'x = %.2e, x_i >= x0_i: %s, getprimalinfeas = %.2e\n", sx, on_face ? "yes" : "NO", pinf)
    @printf("%s (optimum on a face: value 0.8, not the post's vector)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
