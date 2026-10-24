#!/usr/bin/env julia
# sensitivity.jl — MOSEK port "sensitivity": post-optimal analysis (cost
# sensitivity) on a small LP.
#
#   min x0 + x1  s.t.  x0 + 2 x1 >= 4,  x >= 0   ->  x* = (0, 2), obj = 2
#
# x0 (nonbasic at its lower bound) stays at 0 while c0 >= 1/2; x1 (basic) has
# the degenerate range [1, 1]. Checked by re-solving with c0 = 0.6 (in range,
# x0 stays 0) and c0 = 0.4 (out of range, x0 > 0). Port of examples/c/sensitivity.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model
const B = Primal.Bindings

# Build the LP with a given c0 (the c1 coefficient is 1).
function build(m, c0)
    add_vars(m, 2)
    add_cons(m, 1)
    var_bounds(m, 0, BK_LO; lo=0.0, up=Inf)
    var_bounds(m, 1, BK_LO; lo=0.0, up=Inf)
    cj(m, 0, c0)
    cj(m, 1, 1.0)
    a_row(m, 0, [0, 1], [1.0, 2.0])
    con_bounds(m, 0, BK_LO; lo=4.0, up=Inf)
end

# Re-solve the LP with a given c0; returns (objective, x0, x1).
function solve_with_c0(c0)
    m = Model(0, 0)
    r = try
        build(m, c0)
        solve(m)
    finally
        free(m)
    end
    r.rc == 0 || return -1e30, NaN, NaN
    return r.objective, r.x[1], r.x[2]
end

function main()
    m = Model(0, 0)
    try
        build(m, 1.0)
        r = solve(m)
        if r.rc != 0
            @printf("optimize rc=%d\n", r.rc)
            return 1
        end
        x, obj = r.x, r.objective
        @printf("base solution: x = (%.4f, %.4f), obj = %.4f\n", x[1], x[2], obj)

        lc0, uc0, lc1, uc1 = Ref{Cdouble}(0.0), Ref{Cdouble}(0.0), Ref{Cdouble}(0.0), Ref{Cdouble}(0.0)
        B.costsensitivity(m.task[], 0, lc0, uc0)
        B.costsensitivity(m.task[], 1, lc1, uc1)
        @printf("c0 in [%.4f, %s]\n", lc0[], uc0[] == Inf ? "+inf" : "...")
        @printf("c1 in [%.4f, %.4f] (basic: degenerate range)\n", lc1[], uc1[])

        p_in, xa, ya = solve_with_c0(0.6)           # external check
        p_out, xb, yb = solve_with_c0(0.4)
        @printf("c0=0.6 (in range):  x = (%.4f, %.4f), obj = %.4f\n", xa, ya, p_in)
        @printf("c0=0.4 (out range): x = (%.4f, %.4f), obj = %.4f\n", xb, yb, p_out)

        ok = abs(x[1]) < 1e-9 && abs(x[2] - 2.0) < 1e-9 && abs(lc0[] - 0.5) < 1e-9 && uc0[] == Inf &&
             abs(xa) < 1e-9 &&                      # in range: x0 stays 0
             abs(xb) > 1e-9 &&                      # out range: x0 > 0
             abs(p_out - (0.4 * 4.0)) < 1e-6        # all on x0: 1.6
        @printf("%s (c0 range confirmed by re-solve)\n", ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
