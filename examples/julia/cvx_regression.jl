#!/usr/bin/env julia
# cvx_regression.jl — L1 (least absolute deviations) regression as an LP
# (scaling test: many rows).
#
#   min ||A x - b||_1  <=>  min sum_i t_i  s.t.  -t_i <= (A x - b)_i <= t_i,  t >= 0
#
# Verified: feasibility and strong duality pobj ~ dobj. The data come from the
# same 32-bit linear congruential generator as the C example.
#
#   julia cvx_regression.jl [m] [d]   (default 400 30)
#
# Port of examples/c/cvx_regression.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const st = Ref{UInt32}(14142135)
# Deterministic pseudo-random draw in [0,1] from the fixed seed.
function rnd()
    st[] = st[] * 0x41c64e6d + 0x00003039           # 1103515245, 12345 (mod 2^32)
    return Float64((st[] >> 16) & 0x7fff) / 32767.0
end

function main()
    m = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 400
    d = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 30
    (m < 1 || d < 1) && return 2
    A = [rnd() * 2.0 - 1.0 for _ in 1:m*d]          # row-major, as the C example
    b = [rnd() * 2.0 - 1.0 for _ in 1:m]

    t = Model(0, 0)
    try
        add_vars(t, d + m)                          # x (d) free, t (m) >= 0
        add_cons(t, 2m)
        for j in 0:d-1
            var_bounds(t, j, BK_FR; lo=-Inf, up=Inf)
        end
        for i in 0:m-1
            var_bounds(t, d + i, BK_LO; lo=0.0, up=Inf)
            cj(t, d + i, 1.0)
        end
        sub = [collect(0:d-1); 0]
        for i in 0:m-1
            sub[d+1] = d + i
            row = A[i*d+1:i*d+d]
            a_row(t, i, sub, [row; -1.0])           # row i: A_i x - t_i <= b_i
            con_bounds(t, i, BK_UP; lo=-Inf, up=b[i+1])
            a_row(t, m + i, sub, [-row; -1.0])      # row m+i: -A_i x - t_i <= -b_i
            con_bounds(t, m + i, BK_UP; lo=-Inf, up=-b[i+1])
        end

        t0 = time()
        r = solve(t)
        sec = time() - t0
        obj, dobj = 0.0, 0.0
        ok = r.rc == 0
        if ok
            obj, dobj, x = r.objective, r.dual, r.x
            maxviol, mint = 0.0, 0.0
            for i in 0:m-1
                res = -b[i+1]
                for j in 0:d-1
                    res += A[i*d+j+1] * x[j+1]
                end
                abs(res) - x[d+i+1] > maxviol && (maxviol = abs(res) - x[d+i+1])
                x[d+i+1] < mint && (mint = x[d+i+1])
            end
            ok = maxviol < 1e-5 && mint > -1e-6 && abs(obj - dobj) < 1e-5 * (1.0 + abs(obj))
        end
        @printf("cvx_reg    m=%d d=%d vars=%d cons=%d obj=%.6f dobj=%.6f t=%.4fs %s\n",
                m, d, d + m, 2m, obj, dobj, sec, ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(t)
    end
end

exit(main())
