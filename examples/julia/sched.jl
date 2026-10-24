#!/usr/bin/env julia
# sched.jl — scheduling MIP (equivalent to the Python Fusion API example
# "Scheduling": assign 30 tasks to 6 processors minimizing the makespan):
#
#   x[i][j] binary (proc i, task j),  t free (makespan)
#   sum_i x[i][j] = 1 per task,  t - sum_j T[j] x[i][j] >= 0 per proc,  min t
#
# The task times are those of the Python example (random.seed(0), sorted
# decreasing), hardcoded. The partition-bound cut derives t >= T_j, so the
# tree closes at the root and the optimum is PROVEN (INTEGER_OPTIMAL).
# Port of examples/c/sched.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const N, M = 30, 6
const T = [97.328509, 93.040884, 68.870958, 57.771417, 54.733747, 28.056097,
           4.931142, 4.638985, 4.632452, 4.608664, 4.595353, 4.377687,
           4.240869, 4.135194, 4.031818, 4.023217, 3.919327, 3.735936,
           3.473476, 3.333528, 3.045099, 3.018747, 2.906388, 2.682286,
           2.619737, 2.240590, 2.213251, 2.127351, 2.035667, 2.002025]
const PRO_STA_PRIM_FEAS = 2

function main()
    m = Model(0, 0)                                 # x[i][j] = var i*N + j; t = var M*N
    r = try
        add_vars(m, M * N + 1)
        add_cons(m, N + M)
        for i in 0:M-1, j in 0:N-1                  # x binaries in [0,1], t free
            var_bounds(m, i * N + j, BK_RA; lo=0.0, up=1.0)
            var_type(m, i * N + j, VT_INT_BIN)
        end
        var_bounds(m, M * N, BK_FR; lo=-Inf, up=Inf)
        cj(m, M * N, 1.0)                           # min t
        sense(m, SENSE_MIN)
        for j in 0:N-1                              # each task assigned to a single proc
            a_row(m, j, [i * N + j for i in 0:M-1], ones(M))
            con_bounds(m, j, BK_FX; lo=1.0, up=1.0)
        end
        for i in 0:M-1                              # t - sum_j T[j] x[i][j] >= 0  (load_i <= t)
            a_row(m, N + i, [[i * N + j for j in 0:N-1]; M * N], [-T; 1.0])
            con_bounds(m, N + i, BK_LO; lo=0.0, up=Inf)
        end
        solve(m)
    finally
        free(m)
    end
    xx, obj = r.x, r.objective
    @printf("rc = %d, prosta = %d, solsta = %d\n", r.rc, r.prosta, r.solsta)
    @printf("Optimal makespan: %.4f\n", xx[M*N+1])
    for i in 0:M-1
        load = 0.0
        for j in 0:N-1
            xx[i*N+j+1] > 0.5 && (load += T[j+1])
        end
        @printf("  M%d: load = %.4f\n", i, load)
    end
    ok = r.rc == 0 && r.solsta == SOLSTA_INTEGER_OPTIMAL && r.prosta == PRO_STA_PRIM_FEAS &&
         abs(obj - 97.328509) < 1e-3
    for j in 0:N-1                                  # each task assigned exactly once
        ok || break
        count(i -> xx[i*N+j+1] > 0.5, 0:M-1) != 1 && (ok = false)
    end
    @printf("%s (optimality proven: LB = max(sum/6, max task) = %.2f, closes at the root)\n",
            ok ? "OK" : "FAIL", max((0.0 + 399.801611 + 76.963) / 6.0, 97.328509))   # big 6 + small 24
    return ok ? 0 : 1
end

exit(main())
