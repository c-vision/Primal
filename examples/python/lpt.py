#!/usr/bin/env python3
"""Multi-processor scheduling by an LPT-warm-started MIP (MOSEK Cookbook 11.4, lpt.cc).

Assign n tasks to m identical machines so that the makespan is minimal:
    minimize   t
    s.t.       sum_i x_ij = 1                       (each task on one machine)
               t >= sum_j T_j x_ij   for each i     (t >= every machine load)
               x_ij in {0,1},  t >= 0.
The answer of the Longest Processing Time heuristic (sort the tasks descending, put
each on the currently least-loaded machine) is fed as the MIP initial solution.

Hand instance: m = 2, tasks {3,3,2,2,2}. The total is 12, so any makespan is at
least 6, and {3,3} / {2,2,2} attains it: optimum 6. LPT gives 7, so the warm start
is NOT optimal and the MIP has to improve on it. Port of examples/c/lpt.c.
"""

import math

from primalsolver import Model, BK, SENSE, VAR_TYPE

M = 2
NT = 5
TASK = [3, 3, 2, 2, 2]                              # already sorted descending (LPT)
DPAR_MIP_TOL_REL_GAP = 5


def main() -> int:
    tt = M * NT                                     # makespan variable
    with Model() as m:
        m.add_vars(M * NT + 1)
        m.add_cons(NT + M)
        m.sense(SENSE.MIN)
        for i in range(M):
            for j in range(NT):
                m.var_bounds(i * NT + j, BK.RA, 0.0, 1.0)
                m.var_type(i * NT + j, VAR_TYPE.INT_BIN)
        m.var_bounds(tt, BK.LO, 0.0, math.inf)
        m.cj(tt, 1.0)
        row = 0
        for j in range(NT):                         # each task on exactly one machine
            m.a_row(row, [i * NT + j for i in range(M)], [1.0] * M)
            m.con_bounds(row, BK.FX, 1.0, 1.0)
            row += 1
        for i in range(M):                          # t >= load of machine i
            m.a_row(row, [tt] + [i * NT + j for j in range(NT)], [1.0] + [-TASK[j] for j in range(NT)])
            m.con_bounds(row, BK.LO, 0.0, math.inf)
            row += 1

        # LPT initial solution: least-loaded machine for each (descending) task.
        load = [0] * M
        warm = [0.0] * (M * NT + 1)
        for j in range(NT):
            i = 0
            for k in range(1, M):
                if load[k] < load[i]:
                    i = k
            warm[i * NT + j] = 1.0
            load[i] += TASK[j]
        lpt_makespan = max(load)
        warm[tt] = float(lpt_makespan)
        m.warm_start(x=warm)
        m.dou_param(DPAR_MIP_TOL_REL_GAP, 1e-4)
        r = m.solve()
    ok = r.rc == 0
    obj, x = (r.objective, r.x) if ok else (0.0, [0.0] * (M * NT + 1))
    good = ok and abs(obj - 6.0) < 1e-6 and abs(x[tt] - 6.0) < 1e-6 and lpt_makespan == 7
    print(f"lpt: makespan={x[tt]:.6f} (LPT initial={lpt_makespan})  expected 6 (LPT=7 suboptimal)  "
          f"{'OK' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
