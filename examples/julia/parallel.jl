#!/usr/bin/env julia
# parallel.jl — MOSEK port "parallel": several independent tasks, solved in
# sequence (the example checks task isolation).
#
#   A: max x  s.t.  x <= 4            -> 4
#   B: min x  s.t.  x >= 2            -> 2
#   C: max x0 + x1  s.t.  x0 + x1 <= 7 -> 7
#
# Port of examples/c/parallel.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    A, Bt, C = Model(0, 0), Model(0, 0), Model(0, 0)
    try
        add_vars(A, 1); add_cons(A, 1)              # task A: max x s.t. x <= 4
        cj(A, 0, 1.0)
        sense(A, SENSE_MAX)
        a_row(A, 0, [0], [1.0]); con_bounds(A, 0, BK_UP; lo=-Inf, up=4.0)

        add_vars(Bt, 1); add_cons(Bt, 1)            # task B: min x s.t. x >= 2
        cj(Bt, 0, 1.0)
        a_row(Bt, 0, [0], [1.0]); con_bounds(Bt, 0, BK_LO; lo=2.0, up=Inf)

        add_vars(C, 2); add_cons(C, 1)              # task C: max x0+x1 s.t. x0+x1 <= 7
        sense(C, SENSE_MAX)
        cj(C, 0, 1.0); cj(C, 1, 1.0)
        a_row(C, 0, [0, 1], [1.0, 1.0]); con_bounds(C, 0, BK_UP; lo=-Inf, up=7.0)

        rA, rB, rC = solve(A), solve(Bt), solve(C)
        @printf("A: rc=%d pobj=%.4f (expected 4)\n", rA.rc, rA.objective)
        @printf("B: rc=%d pobj=%.4f (expected 2)\n", rB.rc, rB.objective)
        @printf("C: rc=%d pobj=%.4f (expected 7)\n", rC.rc, rC.objective)
        ok = (rA.rc == 0 && abs(rA.objective - 4.0) < 1e-9) &&
             (rB.rc == 0 && abs(rB.objective - 2.0) < 1e-9) &&
             (rC.rc == 0 && abs(rC.objective - 7.0) < 1e-9)
        @printf("%s (3 independent tasks, isolation verified)\n", ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(A); free(Bt); free(C)
    end
end

exit(main())
