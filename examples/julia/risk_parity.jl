#!/usr/bin/env julia
# risk_parity.jl — equal risk contribution portfolio via exponential cones:
#
#   min sqrt(x'Sx) - c sum_i log(x_i)
#   conic form: (t, G'x) in Q,  (x_i, 1, s_i) in EXP (s_i <= log x_i),  min t - c sum s_i
#
# with S = F F' + diag(D), G = [F, sqrt(D)]. After rescaling to sum(x) = 1 the
# risk contributions x_i (Sx)_i are equal (verified). The data come from the
# same 32-bit linear congruential generator as the C example.
#
#   julia risk_parity.jl [n] [k] [c]   (default 20 4 0.01)
#
# Port of examples/c/risk_parity.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const st = Ref{UInt32}(909090)
# Deterministic pseudo-random draw in [0,1] from the fixed seed.
function rnd()
    st[] = st[] * 0x41c64e6d + 0x00003039           # 1103515245, 12345 (mod 2^32)
    return Float64((st[] >> 16) & 0x7fff) / 32767.0
end

function main()
    n = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 20
    k = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 4
    (n < 2 || k < 1) && return 2
    c = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 0.01
    F, D = zeros(n * k), zeros(n)
    for i in 0:n-1
        D[i+1] = 0.001 + rnd() * 0.004
        for j in 0:k-1
            F[i*k+j+1] = (rnd() - 0.5) * 0.2
        end
    end
    m2 = k + n
    X0, Z0, TV, S0, ONE = 0, n, n + m2, n + m2 + 1, n + m2 + 1 + n   # x (n), z (m2), t, s (n), one
    t = Model(0, 0)
    try
        add_vars(t, n + m2 + 1 + n + 1)
        add_cons(t, m2)
        for i in 0:n-1
            var_bounds(t, X0 + i, BK_RA; lo=1e-4, up=1e4)
        end
        for j in 0:m2-1
            var_bounds(t, Z0 + j, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(t, TV, BK_LO; lo=0.0, up=Inf)
        cj(t, TV, 1.0)
        for i in 0:n-1
            # big-M on the log variable: keeps the round-0 tangent LP bounded
            var_bounds(t, S0 + i, BK_RA; lo=-30.0, up=10.0)
            cj(t, S0 + i, -c)
        end
        var_bounds(t, ONE, BK_FX; lo=1.0, up=1.0)
        for j in 0:m2-1                             # z_j - (G'x)_j = 0
            sub, val = Int[], Float64[]
            for i in 0:n-1
                g = j < k ? F[i*k+j+1] : (j - k == i ? sqrt(D[i+1]) : 0.0)
                g != 0.0 && (push!(sub, X0 + i); push!(val, -g))
            end
            push!(sub, Z0 + j); push!(val, 1.0)
            a_row(t, j, sub, val)
            con_bounds(t, j, BK_FX; lo=0.0, up=0.0)
        end
        cone(t, CT_QUAD, [TV; collect(Z0:Z0+m2-1)]) # (t, z) in QUAD
        for i in 0:n-1                              # x_i >= exp(s_i)
            cone(t, CT_PEXP, [X0 + i, ONE, S0 + i])
        end

        t0 = time()
        r = solve(t)
        sec = time() - t0
        ok = r.rc == 0
        spread = 0.0
        if ok
            x = r.x[1:n]
            s = 0.0                                 # rescale to sum 1
            for i in 1:n
                s += x[i]
            end
            for i in 1:n
                x[i] /= s
            end
            Sx = zeros(n)
            for i in 0:n-1
                for a2 in 0:k-1
                    ss = 0.0
                    for l in 0:n-1
                        ss += F[l*k+a2+1] * x[l+1]
                    end
                    Sx[i+1] += F[i*k+a2+1] * ss
                end
                Sx[i+1] += D[i+1] * x[i+1]
            end
            rcmin, rcmax = 1e300, -1e300
            for i in 1:n
                rci = x[i] * Sx[i]
                rci < rcmin && (rcmin = rci)
                rci > rcmax && (rcmax = rci)
            end
            spread = rcmin > 0 ? (rcmax / rcmin - 1.0) : 1.0
            ok = spread < 1e-3                      # risk contributions equal within 0.1%
        end
        @printf("riskparity n=%d k=%d RC spread=%.3e rc=%d t=%.4fs %s\n", n, k, spread, r.rc, sec, ok ? "OK" : "FAIL")
        return ok ? 0 : 1
    finally
        free(t)
    end
end

exit(main())
