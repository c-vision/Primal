#!/usr/bin/env julia
# facility.jl — capacitated facility location MILP (scaling test).
#
#   min sum_j f_j y_j + sum_ij c_ij x_ij
#   s.t.  sum_j x_ij = d_i,  sum_i x_ij <= q_j y_j,  y binary,  x >= 0
#
# Verified: demands met, capacities respected, y integral, and the MIP value
# is >= the LP relaxation bound. The data come from the same 32-bit linear
# congruential generator as the C example.
#
#   julia facility.jl [m] [n]   (default 15 40)
#
# Port of examples/c/facility.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const st = Ref{UInt32}(55555)
# Deterministic pseudo-random draw in [0,1] from the fixed seed.
function rnd()
    st[] = st[] * 0x41c64e6d + 0x00003039           # 1103515245, 12345 (mod 2^32)
    return Float64((st[] >> 16) & 0x7fff) / 32767.0
end

# Build the facility location model (integral = MIP, otherwise its LP relaxation).
function build(t, m, n, integral, f, q, d, c)
    add_vars(t, m + m * n)                          # y_0..y_{m-1}, x_ij at m + i*m + j
    add_cons(t, n + m)
    for j in 0:m-1
        cj(t, j, f[j+1])
        var_bounds(t, j, BK_RA; lo=0.0, up=1.0)
        integral && var_type(t, j, VT_INT_BIN)
    end
    for i in 0:n-1, j in 0:m-1
        k = m + i * m + j
        cj(t, k, c[i*m+j+1])
        var_bounds(t, k, BK_LO; lo=0.0, up=Inf)
    end
    for i in 0:n-1                                  # sum_j x_ij = d_i
        a_row(t, i, [m + i * m + j for j in 0:m-1], ones(m))
        con_bounds(t, i, BK_FX; lo=d[i+1], up=d[i+1])
    end
    for j in 0:m-1                                  # sum_i x_ij - q_j y_j <= 0
        a_row(t, n + j, [[m + i * m + j for i in 0:n-1]; j], [ones(n); -q[j+1]])
        con_bounds(t, n + j, BK_UP; lo=-Inf, up=0.0)
    end
end

function main()
    m = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 15
    n = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 40
    (m < 1 || n < 1) && return 2
    f, q, d = zeros(m), zeros(m), zeros(n)
    totald = 0.0
    for j in 1:m
        f[j] = 100.0 + rnd() * 200.0
        q[j] = 40.0 + rnd() * 40.0
    end
    for i in 1:n
        d[i] = 1.0 + rnd() * 8.0
        totald += d[i]
    end
    c = [1.0 + rnd() * 20.0 for _ in 1:m*n]
    totalq = 0.0                                    # scale capacities so total capacity is ample
    for j in 1:m
        totalq += q[j]
    end
    for j in 1:m
        q[j] *= 1.5 * totald / totalq
    end

    t = Model(0, 0)
    obj, ok, sec = 0.0, false, 0.0
    try
        build(t, m, n, true, f, q, d, c)
        t0 = time()
        r = solve(t)
        sec = time() - t0
        ok = r.rc == 0
        if ok
            obj, x = r.objective, r.x
            maxdem, maxcap, maxfrac = 0.0, 0.0, 0.0
            for i in 0:n-1
                s = 0.0
                for j in 0:m-1
                    s += x[m+i*m+j+1]
                end
                abs(s - d[i+1]) > maxdem && (maxdem = abs(s - d[i+1]))
            end
            for j in 0:m-1
                s = 0.0
                for i in 0:n-1
                    s += x[m+i*m+j+1]
                end
                cap = q[j+1] * x[j+1]
                s - cap > maxcap && (maxcap = s - cap)
                fr = abs(x[j+1] - floor(x[j+1] + 0.5))
                fr > maxfrac && (maxfrac = fr)
            end
            ok = maxdem < 1e-5 && maxcap < 1e-5 && maxfrac < 1e-6
        end
    finally
        free(t)
    end
    lp = 0.0                                        # LP relaxation lower bound
    rl = Model(0, 0)
    try
        build(rl, m, n, false, f, q, d, c)
        r = solve(rl)
        r.rc == 0 && (lp = r.objective)
    finally
        free(rl)
    end
    obj + 1e-6 < lp && (ok = false)
    @printf("facility   m=%d n=%d vars=%d cons=%d obj=%.4f lp=%.4f t=%.4fs %s\n",
            m, n, m + m * n, n + m, obj, lp, sec, ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
