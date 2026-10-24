#!/usr/bin/env julia
# unit_commitment.jl — a tiny thermal unit-commitment MILP (MOSEK Modeling
# Cookbook / Tutorials). Two generators over two periods; commitment u and
# startup v are binary, the dispatch p is continuous:
#
#   min sum (no_load u + marginal p + startup v)
#   s.t.  sum_i p_it = d_t,  min_i u <= p <= max_i u,  v_it >= u_it - u_i,t-1
#
# Instance: gen 0 (min 1, max 3, no-load 5, marginal 2, startup 10), gen 1
# (min 0, max 4, marginal 6), demand (2, 4): hand-derived optimum 36.
# Port of examples/c/unit_commitment.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NG = 2
const NT = 2
U(i, t) = i * NT + t                                # variable layout
V(i, t) = NG * NT + i * NT + t
P(i, t) = 2 * NG * NT + i * NT + t
const NVAR = 3 * NG * NT
const lo = [1.0, 0]
const up = [3.0, 4]
const noload = [5.0, 0]
const marg = [2.0, 6]
const startc = [10.0, 0]
const demand = [2.0, 4]

function main()
    m = Model(NT + 3 * NG * NT, NVAR)               # rows: demand + 3 per (generator, period)
    r = try
        for i in 0:NG-1, k in 0:NT-1
            var_bounds(m, U(i, k), BK_RA; lo=0.0, up=1.0)
            var_type(m, U(i, k), VT_INT_BIN)
            var_bounds(m, V(i, k), BK_RA; lo=0.0, up=1.0)
            var_type(m, V(i, k), VT_INT_BIN)
            var_bounds(m, P(i, k), BK_LO; lo=0.0, up=Inf)
            cj(m, U(i, k), noload[i+1])
            cj(m, V(i, k), startc[i+1])
            cj(m, P(i, k), marg[i+1])
        end
        row = 0
        for k in 0:NT-1                             # demand: sum_i p_i,k = d_k
            a_row(m, row, [P(i, k) for i in 0:NG-1], ones(NG))
            con_bounds(m, row, BK_FX; lo=demand[k+1], up=demand[k+1]); row += 1
        end
        for i in 0:NG-1, k in 0:NT-1
            a_row(m, row, [P(i, k), U(i, k)], [1.0, -up[i+1]])   # p - max u <= 0
            con_bounds(m, row, BK_UP; lo=-Inf, up=0.0); row += 1
            a_row(m, row, [P(i, k), U(i, k)], [1.0, -lo[i+1]])   # p - min u >= 0
            con_bounds(m, row, BK_LO; lo=0.0, up=Inf); row += 1
            if k == 0                               # v - u_k + u_{k-1} >= 0  (u_{i,-1} = 0)
                a_row(m, row, [V(i, k), U(i, k)], [1.0, -1.0])
            else
                a_row(m, row, [V(i, k), U(i, k), U(i, k - 1)], [1.0, -1.0, 1.0])
            end
            con_bounds(m, row, BK_LO; lo=0.0, up=Inf); row += 1
        end
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        obj, x = r.objective, r.x
        ok = abs(obj - 36.0) < 1e-6
        @printf("unit_commitment  d=[2,4]  p0=(%.1f,%.1f) p1=(%.1f,%.1f) obj=%.4f (expected 36) %s\n",
                x[P(0, 0)+1], x[P(0, 1)+1], x[P(1, 0)+1], x[P(1, 1)+1], obj, ok ? "OK" : "FAIL")
    else
        @printf("unit_commitment  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
