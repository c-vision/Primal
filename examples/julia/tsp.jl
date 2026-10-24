#!/usr/bin/env julia
# tsp.jl — a directed TSP on four cities by iterative subtour elimination
# (MOSEK Cookbook 11.8). Binary x_ij = 1 iff the tour goes i -> j; the base
# model is the assignment relaxation (one arc out, one in, only listed arcs).
# Each cycle of the cover that is not Hamiltonian gets the cut
# sum_{i->j in cycle} x_ij <= size - 1, and the model is re-solved.
#
# Instance: the unit square (cost 1) plus cheap arcs (0.1) forming the
# 3-cycle 1->0->2->1; the only tour is 0->1->2->3->0, cost 4.
# Port of examples/c/tsp.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const N = 4
IDX(i, j) = i * N + j
const AI = [0, 1, 2, 3, 1, 0, 2, 0]                 # arcs present and their cost
const AJ = [1, 2, 3, 0, 0, 2, 1, 3]
const CV = [1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1]

# One solve of the model built so far; returns (objective, successor of each city).
function solve_succ(m)
    r = solve(m)
    succ = fill(-1, N)
    for i in 0:N-1, j in 0:N-1
        r.x[IDX(i, j)+1] > 0.5 && (succ[i+1] = j)
    end
    return r.objective, succ
end

function main()
    two_hop = false                                 # without 2-hop removal the first cover is disjoint cycles
    m = Model(0, 0)
    try
        add_vars(m, N * N)
        add_cons(m, 2N + (two_hop ? N : 0))
        sense(m, SENSE_MIN)
        C = zeros(N * N)
        for e in 1:8
            C[IDX(AI[e], AJ[e])+1] = CV[e]
        end
        for i in 0:N-1, j in 0:N-1
            a = C[IDX(i, j)+1] > 0.0 ? 1.0 : 0.0
            var_bounds(m, IDX(i, j), BK_RA; lo=0.0, up=a)   # x_ij <= A_ij
            var_type(m, IDX(i, j), VT_INT_BIN)
            cj(m, IDX(i, j), C[IDX(i, j)+1])
        end
        row = 0
        for i in 0:N-1                              # one arc out of each city
            a_row(m, row, [IDX(i, j) for j in 0:N-1], ones(N))
            con_bounds(m, row, BK_FX; lo=1.0, up=1.0); row += 1
        end
        for j in 0:N-1                              # one arc into each city
            a_row(m, row, [IDX(i, j) for i in 0:N-1], ones(N))
            con_bounds(m, row, BK_FX; lo=1.0, up=1.0); row += 1
        end
        if two_hop                                  # no 2-hop loops (rows exist only then)
            for i in 0:N-1
                a_row(m, row, [IDX(i, (i + 1) % N), IDX((i + 1) % N, i)], [1.0, 1.0])
                con_bounds(m, row, BK_UP; lo=-Inf, up=1.0); row += 1
            end
        end

        it, po, succ = 0, 0.0, fill(-1, N)          # iterative subtour elimination
        while true
            po, succ = solve_succ(m)
            it += 1
            cycle, cur = Int[], 0                   # walk the cycle containing city 0
            while true
                push!(cycle, cur); cur = succ[cur+1]
                (cur != 0 && length(cycle) < N) || break
            end
            len = length(cycle)
            (len == N && cur == 0) && break         # one Hamiltonian cycle
            add_cons(m, 1)                          # sum_{i in cycle} x_{i,succ[i]} <= len - 1
            last = numcon(m)
            a_row(m, last - 1, [IDX(c, succ[c+1]) for c in cycle], ones(len))
            con_bounds(m, last - 1, BK_UP; lo=-Inf, up=Float64(len - 1))
        end
        # the tour is 0 -> 1 -> 2 -> 3 -> 0
        good = abs(po - 4.0) < 1e-6 && succ == [1, 2, 3, 0] && it >= 2   # subtours were eliminated
        @printf("tsp: obj=%.6f  tour 0->%d->%d->%d->%d  (expected 4, 0->1->2->3->0)  iterations=%d  %s\n",
                po, succ[1], succ[2], succ[3], succ[4], it, good ? "OK" : "FAIL")
        return good ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
