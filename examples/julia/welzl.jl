#!/usr/bin/env julia
# welzl.jl — Welzl's exact minimum enclosing circle (tschm/min_circle),
# cross-checked against this library's SOCP (min r s.t. ||x - p_i|| <= r) on
# twenty deterministic pseudo-random point sets: two independent methods must
# agree exactly.
#
#   welzl(P, R): if P empty or |R| = 3: return circle(R)
#                D = welzl(P - p, R); p inside D ? D : welzl(P - p, R + p)
#
# The points come from the same 64-bit linear congruential generator as the C
# example. Port of examples/c/welzl.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# Circle through 0, 1, 2 or 3 boundary points: (cx, cy, r).
function circ_of(m, R)
    m == 0 && return (0.0, 0.0, -1.0)               # empty: contains nothing
    m == 1 && return (R[1, 1], R[1, 2], 0.0)
    if m == 2
        c = (0.5 * (R[1, 1] + R[2, 1]), 0.5 * (R[1, 2] + R[2, 2]))
        s = 0.0
        for d in 1:2
            s += (R[1, d] - R[2, d]) * (R[1, d] - R[2, d])
        end
        return (c[1], c[2], 0.5 * sqrt(s))
    end
    ax, ay, bx, by, cx, cy = R[1, 1], R[1, 2], R[2, 1], R[2, 2], R[3, 1], R[3, 2]
    dd = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    abs(dd) < 1e-14 && return (0.0, 0.0, Inf)       # collinear
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    zx = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / dd
    zy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / dd
    return (zx, zy, sqrt((ax - zx) * (ax - zx) + (ay - zy) * (ay - zy)))
end

# True when point p lies inside circle z (within tolerance).
function inside(z, px, py)
    z[3] < 0 && return false
    dx, dy = px - z[1], py - z[2]
    return sqrt(dx * dx + dy * dy) <= z[3] + 1e-9
end

# Welzl's recursive minimum-enclosing-circle routine on P[first:end, :].
function welzl(P, first, R, m)
    (first > size(P, 1) || m == 3) && return circ_of(m, R)
    D = welzl(P, first + 1, R, m)
    inside(D, P[first, 1], P[first, 2]) && return D
    R[m+1, 1], R[m+1, 2] = P[first, 1], P[first, 2]
    return welzl(P, first + 1, R, m + 1)
end

# SOCP oracle (same formulation as min_circle); returns (ok, r).
function socp_circle(P)
    k = size(P, 1)
    PX, PY, R, D = 0, 1, 2, 3
    m = Model(0, 0)
    r = try
        add_vars(m, 3 + 2k)
        add_cons(m, 2k)
        var_bounds(m, PX, BK_FR; lo=-Inf, up=Inf)
        var_bounds(m, PY, BK_FR; lo=-Inf, up=Inf)
        var_bounds(m, R, BK_LO; lo=0.0, up=Inf)
        for i in 0:k-1
            var_bounds(m, D + 2i, BK_FR; lo=-Inf, up=Inf)
            var_bounds(m, D + 2i + 1, BK_FR; lo=-Inf, up=Inf)
            a_row(m, 2i, [D + 2i, PX], [1.0, -1.0])
            con_bounds(m, 2i, BK_FX; lo=-P[i+1, 1], up=-P[i+1, 1])
            a_row(m, 2i + 1, [D + 2i + 1, PY], [1.0, -1.0])
            con_bounds(m, 2i + 1, BK_FX; lo=-P[i+1, 2], up=-P[i+1, 2])
        end
        for i in 0:k-1
            cone(m, CT_QUAD, [R, D + 2i, D + 2i + 1])
        end
        cj(m, R, 1.0)
        sense(m, SENSE_MIN)
        solve(m)
    finally
        free(m)
    end
    return r.rc == 0, (r.rc == 0 ? r.x[R+1] : 0.0)
end

const KP = 10
const g_seed = Ref{UInt64}(0)
# Deterministic pseudo-random draw in [0,1] from the fixed seed.
function rnd()
    g_seed[] = g_seed[] * 1103515245 + 12345
    return Float64((g_seed[] >> 16) % 1000) / 1000.0
end

function main()
    allok, nsets = true, 20
    println("welzl")
    for s in 0:nsets-1
        g_seed[] = 1000 + 7s
        P = zeros(KP, 2)
        for i in 1:KP
            P[i, 1] = rnd()
            P[i, 2] = rnd()
        end
        for i in KP-1:-1:1                          # Fisher-Yates shuffle with the same PRNG
            j = trunc(Int, rnd() * (i + 1))
            j > i && (j = i)
            P[i+1, 1], P[i+1, 2], P[j+1, 1], P[j+1, 2] = P[j+1, 1], P[j+1, 2], P[i+1, 1], P[i+1, 2]
        end
        w = welzl(P, 1, zeros(3, 2), 0)
        ok, rs = socp_circle(P)
        good = ok && abs(w[3] - rs) < 1e-6
        if !good
            @printf("  set %2d  welzl=%.8f  socp=%.8f  FAIL\n", s, w[3], rs)
            allok = false
        elseif s == 0
            @printf("  set  0  welzl=%.8f  socp=%.8f  OK\n", w[3], rs)
        end
    end
    @printf("  %d point sets, Welzl == SOCP: %s\n", nsets, allok ? "OK" : "FAIL")
    @printf("%s\n", allok ? "OK" : "FAIL")
    return allok ? 0 : 1
end

exit(main())
