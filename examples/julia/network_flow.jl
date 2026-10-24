#!/usr/bin/env julia
# network_flow.jl — min-cost flow on a random graph (LP, scaling test).
#
#   min sum_e c_e f_e  s.t.  sum_{out of v} f_e - sum_{into v} f_e = b_v,  0 <= f <= u
#
# The balances come from a random reference flow, so the LP is feasible by
# construction. Verified: node balance, capacity, strong duality. The data come
# from the same 32-bit linear congruential generator as the C example.
#
#   julia network_flow.jl [nodes] [edges]   (default 200 800)
#
# Port of examples/c/network_flow.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const st = Ref{UInt32}(27182818)
# Deterministic pseudo-random draw in [0,1] from the fixed seed.
function rnd()
    st[] = st[] * 0x41c64e6d + 0x00003039           # 1103515245, 12345 (mod 2^32)
    return Float64((st[] >> 16) & 0x7fff) / 32767.0
end

function main()
    N = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 200
    E = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 800
    (N < 2 || E < 1) && return 2
    eu, ev = zeros(Int, E), zeros(Int, E)
    cost, cap, b = zeros(E), zeros(E), zeros(N)
    for e in 1:E
        u = trunc(Int, rnd() * N)
        v = trunc(Int, rnd() * N)
        v == u && (v = (v + 1) % N)
        eu[e], ev[e] = u, v
        cost[e] = 0.5 + rnd() * 10.0
        cap[e] = 5.0 + rnd() * 20.0
        flow0 = rnd() * cap[e] * 0.5                # reference flow
        b[u+1] += flow0                             # node u sends flow0 out
        b[v+1] -= flow0                             # node v receives flow0 in
    end

    m = Model(0, 0)
    try
        add_vars(m, E)
        add_cons(m, N)
        for e in 0:E-1
            cj(m, e, cost[e+1])
            var_bounds(m, e, BK_RA; lo=0.0, up=cap[e+1])
        end
        for v in 0:N-1
            sub, val = Int[], Float64[]
            for e in 0:E-1
                eu[e+1] == v && (push!(sub, e); push!(val, 1.0))
                ev[e+1] == v && (push!(sub, e); push!(val, -1.0))
            end
            isempty(sub) && (push!(sub, 0); push!(val, 0.0))
            a_row(m, v, sub, val)
            con_bounds(m, v, BK_FX; lo=b[v+1], up=b[v+1])
        end

        t0 = time()
        r = solve(m)
        sec = time() - t0
        obj, dobj = 0.0, 0.0
        ok = r.rc == 0
        if ok
            obj, dobj, f = r.objective, r.dual, r.x
            maxbal, maxcap, minf = 0.0, 0.0, 0.0
            bal = zeros(N)
            for e in 1:E
                bal[eu[e]+1] += f[e]
                bal[ev[e]+1] -= f[e]
                f[e] - cap[e] > maxcap && (maxcap = f[e] - cap[e])
                f[e] < minf && (minf = f[e])
            end
            for v in 1:N
                abs(bal[v] - b[v]) > maxbal && (maxbal = abs(bal[v] - b[v]))
            end
            ok = maxbal < 1e-5 && maxcap < 1e-5 && minf > -1e-6 && abs(obj - dobj) < 1e-5 * (1.0 + abs(obj))
        end
        @printf("netflow    N=%d E=%d vars=%d cons=%d obj=%.6f dobj=%.6f t=%.4fs %s\n",
                N, E, E, N, obj, dobj, sec, ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(m)
    end
end

exit(main())
