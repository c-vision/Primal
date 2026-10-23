#!/usr/bin/env python3
"""Scheduling MIP (equivalent to the Python Fusion API example "Scheduling": assign
30 tasks to 6 processors minimizing the makespan).

    x[i][j] in {0,1}   assignment binary (processor i, task j)
    t                  free continuous variable (makespan)
    for each task j:   sum_i x[i][j] = 1
    for each proc i:   t - sum_j T[j]*x[i][j] >= 0
    objective:         min t

The task times are those of the Python example (random.seed(0), 24 tasks
uniform(1,5) + 6 tasks uniform(20,100), sorted decreasing), hardcoded. The
partition-bound cut derives t >= T_j, so the tree closes at the root and the integer
optimum is PROVEN (rc = OK, solsta = INTEGER_OPTIMAL). Port of examples/c/sched.c.
"""

import math

from primalsolver import Model, BK, SENSE, SOLSTA, VAR_TYPE

N, M = 30, 6
T = [97.328509, 93.040884, 68.870958, 57.771417, 54.733747, 28.056097,
     4.931142, 4.638985, 4.632452, 4.608664, 4.595353, 4.377687,
     4.240869, 4.135194, 4.031818, 4.023217, 3.919327, 3.735936,
     3.473476, 3.333528, 3.045099, 3.018747, 2.906388, 2.682286,
     2.619737, 2.240590, 2.213251, 2.127351, 2.035667, 2.002025]
PRO_STA_PRIM_FEAS = 2


def main() -> int:
    tv = M * N                                      # x[i][j] = var i*N + j; t = var M*N
    with Model() as m:
        m.add_vars(M * N + 1)
        m.add_cons(N + M)
        for i in range(M):
            for j in range(N):
                m.var_bounds(i * N + j, BK.RA, 0.0, 1.0)
                m.var_type(i * N + j, VAR_TYPE.INT_BIN)
        m.var_bounds(tv, BK.FR, -math.inf, math.inf)
        m.cj(tv, 1.0)                               # min t
        m.sense(SENSE.MIN)
        for j in range(N):                          # each task assigned to a single processor
            m.a_row(j, [i * N + j for i in range(M)], [1.0] * M)
            m.con_bounds(j, BK.FX, 1.0, 1.0)
        for i in range(M):                          # t - sum_j T[j]*x[i][j] >= 0  (load_i <= t)
            m.a_row(N + i, [i * N + j for j in range(N)] + [tv], [-T[j] for j in range(N)] + [1.0])
            m.con_bounds(N + i, BK.LO, 0.0, math.inf)
        r = m.solve()
    x = r.x
    print(f"rc = {r.rc}, prosta = {r.prosta}, solsta = {r.solsta}")
    print(f"Optimal makespan: {x[tv]:.4f}")
    for i in range(M):
        load = sum(T[j] for j in range(N) if x[i * N + j] > 0.5)
        print(f"  M{i}: load = {load:.4f}")
    ok = (r.rc == 0 and r.solsta == SOLSTA.INTEGER_OPTIMAL and r.prosta == PRO_STA_PRIM_FEAS
          and abs(r.objective - 97.328509) < 1e-3)
    if ok:                                          # each task assigned exactly once
        ok = all(sum(1 for i in range(M) if x[i * N + j] > 0.5) == 1 for j in range(N))
    lb = max((0.0 + 399.801611 + 76.963) / 6.0, 97.328509)   # big 6 + small 24
    print(f"{'OK' if ok else 'FAIL'} (optimality proven: LB = max(sum/6, max task) = {lb:.2f}, "
          f"closes at the root)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
