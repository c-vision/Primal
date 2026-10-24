#!/usr/bin/env julia
# feasrepairex1.jl — MOSEK port "feasrepairex1": feasibility repair. An
# infeasible LP is "repaired" by finding the point of minimum total violation
# (elastic relaxation, PRIMAL_feasrepair).
#
#   min x0  s.t.  x0 >= 2,  x0 <= 1   (no feasible x)
#
# The minimum total violation is 1 (moving one bound by 1).
# Port of examples/c/feasrepairex1.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model
const B = Primal.Bindings

function main()
    m = Model(0, 0)
    try
        add_vars(m, 1)
        add_cons(m, 2)
        var_bounds(m, 0, BK_FR; lo=-Inf, up=Inf)
        cj(m, 0, 1.0)
        a_row(m, 0, [0], [1.0]); con_bounds(m, 0, BK_LO; lo=2.0, up=Inf)    # x0 >= 2
        a_row(m, 1, [0], [1.0]); con_bounds(m, 1, BK_UP; lo=-Inf, up=1.0)   # x0 <= 1

        # 1. optimize: infeasible
        rc = Int(B.optimize(m.task[]))
        @printf("optimize: rc=%d (expected 1002 infeasible)\n", rc)
        infeas = rc == RC[:ERR_INFEASIBLE]

        # 2. feasrepair: point of minimum violation
        rc = Int(B.feasrepair(m.task[]))
        if rc != 0
            @printf("feasrepair rc=%d\n", rc)
            return 1
        end
        x = zeros(1)
        B.getxx(m.task[], SOL_ITR, x)
        # total violation of the repaired point: (2 - x0)+ + (x0 - 1)+ = 1
        viol = 0.0
        x[1] < 2.0 && (viol += 2.0 - x[1])
        x[1] > 1.0 && (viol += x[1] - 1.0)
        @printf("feasrepair: x0 = %.4f, total violation = %.4f (expected = 1)\n", x[1], viol)

        ok = infeas && abs(viol - 1.0) < 1e-6
        @printf("%s (repair to minimum violation)\n", ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
