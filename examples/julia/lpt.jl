#!/usr/bin/env julia
# lpt.jl — multi-processor scheduling by an LPT-warm-started MIP (MOSEK
# Cookbook 11.4):
#
#   minimize t  s.t.  sum_i x_ij = 1 (each task on one machine),
#                     t >= sum_j T_j x_ij (every machine load),  x binary, t >= 0
#
# The Longest Processing Time heuristic seeds the MIP. Instance: 2 machines,
# tasks {3,3,2,2,2}: the optimum is 6 ({3,3} / {2,2,2}), LPT gives 7, so the
# MIP must improve on its warm start. Port of examples/c/lpt.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const M = 2
const NT = 5
const TASK = [3, 3, 2, 2, 2]                        # already sorted descending (LPT)
const DPAR_MIP_TOL_REL_GAP = 5

function main()
    TT = M * NT                                     # makespan variable
    m = Model(0, 0)
    try
        add_vars(m, M * NT + 1)
        add_cons(m, NT + M)
        sense(m, SENSE_MIN)
        for i in 0:M-1, j in 0:NT-1
            var_bounds(m, i * NT + j, BK_RA; lo=0.0, up=1.0)
            var_type(m, i * NT + j, VT_INT_BIN)
        end
        var_bounds(m, TT, BK_LO; lo=0.0, up=Inf)
        cj(m, TT, 1.0)
        row = 0
        for j in 0:NT-1                             # each task on exactly one machine
            a_row(m, row, [i * NT + j for i in 0:M-1], ones(M))
            con_bounds(m, row, BK_FX; lo=1.0, up=1.0); row += 1
        end
        for i in 0:M-1                              # t >= load of machine i
            a_row(m, row, [TT; [i * NT + j for j in 0:NT-1]], [1.0; -Float64.(TASK)])
            con_bounds(m, row, BK_LO; lo=0.0, up=Inf); row += 1
        end

        # LPT initial solution: least-loaded machine for each (descending) task.
        load = zeros(Int, M)
        warm = zeros(M * NT + 1)
        for j in 0:NT-1
            i = 0
            for k in 1:M-1
                load[k+1] < load[i+1] && (i = k)
            end
            warm[i*NT+j+1] = 1.0
            load[i+1] += TASK[j+1]
        end
        lpt_makespan = maximum(load)
        warm[TT+1] = Float64(lpt_makespan)
        warm_start(m; x=warm)
        dou_param(m, DPAR_MIP_TOL_REL_GAP, 1e-4)

        r = solve(m)
        ok = r.rc == 0
        obj, x = 0.0, zeros(M * NT + 1)
        if ok
            obj, x = r.objective, r.x
        end
        good = ok && abs(obj - 6.0) < 1e-6 && abs(x[TT+1] - 6.0) < 1e-6 &&
               lpt_makespan == 7                    # the MIP improved on the LPT warm start
        @printf("lpt: makespan=%.6f (LPT initial=%d)  expected 6 (LPT=7 suboptimal)  %s\n",
                x[TT+1], lpt_makespan, good ? "OK" : "FAIL")
        return good ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
