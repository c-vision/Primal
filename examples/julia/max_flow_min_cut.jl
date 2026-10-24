#!/usr/bin/env julia
# max_flow_min_cut.jl — maximum flow / minimum cut as a linear program
# (LightConvex example list):
#
#   maximize sum_{e out of s} f_e  s.t.  0 <= f_e <= u_e,  conservation at v != s, t
#
# Instance: source 0, sink 3, arcs 0->1 (3), 0->2 (2), 1->3 (2), 2->3 (3),
# 1->2 (1): max flow = min cut = 5. The flow is re-checked and all cuts are
# enumerated independently. Port of examples/c/max_flow_min_cut.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NARC = 5
const ARCF = [0, 0, 1, 2, 1]                        # arc table: from, to, capacity
const ARCT = [1, 2, 3, 3, 2]
const CAP = [3.0, 2.0, 2.0, 3.0, 1.0]
const NNODE = 4
const SRC = 0
const SNK = 3

# minimum cut by brute force over the partitions with source in S, sink in T
function min_cut_bruteforce()
    best = 1e30
    for mask in 0:(1<<(NNODE-2))-1
        inS = zeros(Int, NNODE)
        inS[SRC+1], inS[SNK+1] = 1, 0
        for v in 0:NNODE-1
            if v != SRC && v != SNK
                bit = v < SRC ? v : v - 1
                inS[v+1] = (mask >> bit) & 1
            end
        end
        cut = 0.0
        for e in 1:NARC
            (inS[ARCF[e]+1] == 1 && inS[ARCT[e]+1] == 0) && (cut += CAP[e])
        end
        cut < best && (best = cut)
    end
    return best
end

function main()
    m = Model(NARC + (NNODE - 2), NARC)             # 5 flows; 5 capacity + 2 conservation rows
    r = try
        for e in 0:NARC-1
            var_bounds(m, e, BK_LO; lo=0.0, up=Inf)
        end
        cj(m, 0, -1.0)                              # maximize f0 + f1: min -(f0 + f1)
        cj(m, 1, -1.0)
        row = 0
        for e in 0:NARC-1                           # capacity: f_e <= u_e
            a_row(m, row, [e], [1.0])
            con_bounds(m, row, BK_UP; lo=-Inf, up=CAP[e+1]); row += 1
        end
        for v in 1:NNODE-2                          # conservation: in - out = 0
            sub, val = Int[], Float64[]
            for e in 1:NARC
                c = 0.0
                ARCT[e] == v && (c += 1.0)          # inflow
                ARCF[e] == v && (c -= 1.0)          # outflow
                c != 0.0 && (push!(sub, e - 1); push!(val, c))
            end
            a_row(m, row, sub, val)
            con_bounds(m, row, BK_FX; lo=0.0, up=0.0); row += 1
        end
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    if ok
        x = r.x
        flow = x[1] + x[2]
        maxcapviol, maxconsviol = 0.0, 0.0          # recompute capacities and conservation
        for e in 1:NARC
            x[e] > CAP[e] + 1e-7 && (maxcapviol = max(maxcapviol, x[e] - CAP[e]))
        end
        for v in 1:NNODE-2
            bal = 0.0
            for e in 1:NARC
                ARCT[e] == v && (bal += x[e])
                ARCF[e] == v && (bal -= x[e])
            end
            maxconsviol = max(maxconsviol, abs(bal))
        end
        cut = min_cut_bruteforce()
        ok = abs(flow - 5.0) < 1e-6 && maxcapviol < 1e-7 && maxconsviol < 1e-7 && abs(cut - flow) < 1e-9
        @printf("max_flow_min_cut  flow=%.6f (expected 5)  min_cut=%.6f  max_cap_viol=%.2e max_cons_viol=%.2e  f=(%.2f,%.2f,%.2f,%.2f,%.2f) %s\n",
                flow, cut, maxcapviol, maxconsviol, x[1], x[2], x[3], x[4], x[5], ok ? "OK" : "FAIL")
    else
        @printf("max_flow_min_cut  rc=%d FAIL\n", r.rc)
    end
    return ok ? 0 : 1
end

exit(main())
