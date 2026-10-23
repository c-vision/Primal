#!/usr/bin/env python3
"""A directed TSP on four cities by iterative subtour elimination
(MOSEK Cookbook 11.8, tsp.cc).

Binary x[i][j] = 1 iff the tour goes i -> j. The base model is the assignment
relaxation
    minimize   sum_ij C_ij x_ij
    s.t.       sum_j x_ij = 1,  sum_i x_ij = 1     (one out, one in)
               x_ij <= A_ij                        (only the listed arcs)
A cover of disjoint cycles is repaired by adding, for every cycle found,
    sum_{i->j in the cycle} x_ij <= (cycle size) - 1
and re-solving until one Hamiltonian cycle remains.

Instance: the unit square (0,1),(1,2),(2,3),(3,0) of cost 1 plus the cheap arcs
(1,0),(0,2),(2,1),(0,3) of cost 0.1. The only Hamiltonian cycle is 0->1->2->3->0,
cost 4 (the cheap arcs form the 3-cycle 1->0->2->1, which subtour elimination
removes). Port of examples/c/tsp.c.
"""

import math

from primalsolver import Model, BK, SENSE, VAR_TYPE

N = 4
AI = [0, 1, 2, 3, 1, 0, 2, 0]
AJ = [1, 2, 3, 0, 0, 2, 1, 3]
CV = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1]


def idx(i, j):
    return i * N + j


def solve(m):
    """One solve of the model built so far; returns (objective, successor of each city)."""
    r = m.solve()
    succ = [-1] * N
    for i in range(N):
        for j in range(N):
            if r.x[idx(i, j)] > 0.5:
                succ[i] = j
    return r.objective, succ


def main() -> int:
    two_hop = False                                 # without 2-hop removal the first cover is disjoint cycles
    c = [0.0] * (N * N)
    for e in range(8):
        c[idx(AI[e], AJ[e])] = CV[e]
    with Model() as m:
        m.add_vars(N * N)
        m.add_cons(2 * N + (N if two_hop else 0))
        m.sense(SENSE.MIN)
        for i in range(N):
            for j in range(N):
                a = 1.0 if c[idx(i, j)] > 0.0 else 0.0
                m.var_bounds(idx(i, j), BK.RA, 0.0, a)   # x_ij <= A_ij
                m.var_type(idx(i, j), VAR_TYPE.INT_BIN)
                m.cj(idx(i, j), c[idx(i, j)])
        row = 0
        for i in range(N):                          # one arc out of each city
            m.a_row(row, [idx(i, j) for j in range(N)], [1.0] * N)
            m.con_bounds(row, BK.FX, 1.0, 1.0)
            row += 1
        for j in range(N):                          # one arc into each city
            m.a_row(row, [idx(i, j) for i in range(N)], [1.0] * N)
            m.con_bounds(row, BK.FX, 1.0, 1.0)
            row += 1
        # no 2-hop loops: the rows exist only with two_hop (the C example writes
        # them unconditionally and the library rejects the out-of-range rows)
        if two_hop:
            for i in range(N):
                m.a_row(row, [idx(i, (i + 1) % N), idx((i + 1) % N, i)], [1.0, 1.0])
                m.con_bounds(row, BK.UP, -math.inf, 1.0)
                row += 1

        it = 0                                      # iterative subtour elimination
        while True:
            po, succ = solve(m)
            it += 1
            cycle, cur = [], 0                      # walk the cycle containing city 0
            while True:
                cycle.append(cur)
                cur = succ[cur]
                if cur == 0 or len(cycle) >= N:
                    break
            if len(cycle) == N and cur == 0:
                break                               # one Hamiltonian cycle
            m.add_cons(1)                           # sum_{i in cycle} x_{i,succ[i]} <= len - 1
            last = m.numcon()
            m.a_row(last - 1, [idx(k, succ[k]) for k in cycle], [1.0] * len(cycle))
            m.con_bounds(last - 1, BK.UP, -math.inf, float(len(cycle) - 1))
    good = (abs(po - 4.0) < 1e-6 and succ == [1, 2, 3, 0]
            and it >= 2)                            # subtours were eliminated
    print(f"tsp: obj={po:.6f}  tour 0->{succ[0]}->{succ[1]}->{succ[2]}->{succ[3]}  "
          f"(expected 4, 0->1->2->3->0)  iterations={it}  {'OK' if good else 'FAIL'}")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
