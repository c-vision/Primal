#!/usr/bin/env julia
# min_circle.jl — minimum enclosing circle, SOCP vs exact (tschm/min_circle):
#
#   minimize r  s.t.  ||x - p_i||_2 <= r  (one quadratic cone per point)
#
# cross-checked against an EXACT oracle: the circle through the best pair or
# triple of points, by brute force. Cases: the repo's tests (r = 2, r = 2.5)
# plus a deterministic pseudo-random point set (the same 64-bit LCG as the C
# example). Port of examples/c/min_circle.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# Solve the minimum enclosing circle SOCP; returns (ok, r).
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

# Circumcircle of 2 (pair) or 3 (triple) indexed points: (r, cx, cy).
function circum(P, idx)
    if length(idx) == 2
        a, b = idx
        c = (0.5 * (P[a, 1] + P[b, 1]), 0.5 * (P[a, 2] + P[b, 2]))
        s = 0.0
        for d in 1:2
            s += (P[a, d] - P[b, d]) * (P[a, d] - P[b, d])
        end
        return 0.5 * sqrt(s), c[1], c[2]
    end
    ax, ay = P[idx[1], 1], P[idx[1], 2]
    bx, by = P[idx[2], 1], P[idx[2], 2]
    cx, cy = P[idx[3], 1], P[idx[3], 2]
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    abs(d) < 1e-14 && return Inf, 0.0, 0.0
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    zx = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    zy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    dx, dy = ax - zx, ay - zy
    return sqrt(dx * dx + dy * dy), zx, zy
end

# True when the circle (c, r) contains all points.
contains_all(P, cx, cy, r) =
    all(i -> sqrt((P[i, 1] - cx) * (P[i, 1] - cx) + (P[i, 2] - cy) * (P[i, 2] - cy)) <= r + 1e-9, 1:size(P, 1))

# Exact minimum enclosing circle by brute force over pairs/triples.
function exact_circle(P)
    k = size(P, 1)
    best = Inf
    for i in 1:k, j in i+1:k
        r, cx, cy = circum(P, (i, j))
        r < best && contains_all(P, cx, cy, r) && (best = r)
    end
    for i in 1:k, j in i+1:k, l in j+1:k
        r, cx, cy = circum(P, (i, j, l))
        r < best && contains_all(P, cx, cy, r) && (best = r)
    end
    return best
end

# Run one point set through SOCP and exact oracle and compare.
function run(name, P)
    ok, rs = socp_circle(P)
    re = exact_circle(P)
    good = ok && abs(rs - re) < 1e-6
    @printf("  %-14s k=%d  SOCP r=%.8f  exact r=%.8f  %s\n", name, size(P, 1), rs, re, good ? "OK" : "FAIL")
    return good
end

function main()
    allok = true
    println("min_circle")
    allok &= run("two points", [0.0 0.0; 4.0 0.0])
    allok &= run("three points", [0.0 0.0; 4.0 0.0; 2.0 4.0])
    PR = zeros(8, 2)                                # deterministic pseudo-random 8 points (LCG)
    s0 = UInt64(12345)
    for i in 1:8, d in 1:2
        s0 = s0 * 1103515245 + 12345
        PR[i, d] = Float64((s0 >> 16) % 1000) / 1000.0
    end
    allok &= run("random (seed)", PR)
    @printf("%s\n", allok ? "OK" : "FAIL")
    return allok ? 0 : 1
end

exit(main())
