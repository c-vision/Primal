#!/usr/bin/env julia
# socp_robust.jl — least-squares regression as a second-order cone program
# (scaling test for the conic path).
#
#   min ||A x - b||_2  <=>  min t  s.t.  (t, r) in QUAD,  r = A x - b
#
# The optimum t* = ||A x* - b|| is cross-checked against the normal equations
# A'A x = A'b, solved with the library's dense LU (dmat_solve_lu, as the C
# sample). The data come from the same 32-bit linear congruential generator.
#
#   julia socp_robust.jl [m] [d]   (default 300 40)
#
# Port of examples/c/socp_robust.c.

using Printf
using Libdl
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const st = Ref{UInt32}(16180339)
# Deterministic pseudo-random draw in [0,1] from the fixed seed.
function rnd()
    st[] = st[] * 0x41c64e6d + 0x00003039           # 1103515245, 12345 (mod 2^32)
    return Float64((st[] >> 16) & 0x7fff) / 32767.0
end

# Solve M x = rhs (d x d, row-major) in place with the library's dense LU
# (DMat is {int m, n; double **v}); returns the C return code.
function dmat_solve_lu!(M::Vector{Float64}, rhs::Vector{Float64}, d::Int)
    lib = Primal._lib()
    p = ccall(Libdl.dlsym(lib, :dmat_new), Ptr{Cvoid}, (Cint, Cint), d, d)
    rows = unsafe_load(Ptr{Ptr{Ptr{Cdouble}}}(p + 2 * sizeof(Cint)))
    for i in 1:d
        row = unsafe_load(rows, i)
        for j in 1:d
            unsafe_store!(row, M[(i-1)*d+j], j)
        end
    end
    rc = ccall(Libdl.dlsym(lib, :dmat_solve_lu), Cint, (Ptr{Cvoid}, Ptr{Cdouble}, Cint), p, rhs, d)
    ccall(Libdl.dlsym(lib, :dmat_free), Cvoid, (Ptr{Cvoid},), p)
    return Int(rc)
end

function main()
    m = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 300
    d = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 40
    (m < 1 || d < 1) && return 2
    A = [rnd() * 2.0 - 1.0 for _ in 1:m*d]          # row-major, as the C example
    b = [rnd() * 2.0 - 1.0 for _ in 1:m]

    X0, R0, T = 0, d, d + m                         # vars: x (d), r (m), t (1)
    t = Model(0, 0)
    try
        add_vars(t, d + m + 1)
        for j in 0:d-1
            var_bounds(t, X0 + j, BK_FR; lo=-Inf, up=Inf)
        end
        for i in 0:m-1
            var_bounds(t, R0 + i, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(t, T, BK_LO; lo=0.0, up=Inf)
        cj(t, T, 1.0)
        add_cons(t, m)
        for i in 0:m-1                              # r_i - A_i x = -b_i
            a_row(t, i, [collect(X0:X0+d-1); R0 + i], [-A[i*d+1:i*d+d]; 1.0])
            con_bounds(t, i, BK_FX; lo=-b[i+1], up=-b[i+1])
        end
        cone(t, CT_QUAD, [T; collect(R0:R0+m-1)])

        t0 = time()
        r = solve(t)
        sec = time() - t0
        ok = r.rc == 0
        if ok
            obj = r.objective
            M = zeros(d * d)                        # reference: A'A x = A'b with dense LU
            rhs = zeros(d)
            for i in 0:d-1, j in 0:d-1
                s = 0.0
                for k in 0:m-1
                    s += A[k*d+i+1] * A[k*d+j+1]
                end
                M[i*d+j+1] = s
            end
            for i in 0:d-1
                s = 0.0
                for k in 0:m-1
                    s += A[k*d+i+1] * b[k+1]
                end
                rhs[i+1] = s
            end
            tref = 0.0
            if dmat_solve_lu!(M, rhs, d) == 0
                ss = 0.0
                for k in 0:m-1
                    res = -b[k+1]
                    for j in 0:d-1
                        res += A[k*d+j+1] * rhs[j+1]
                    end
                    ss += res * res
                end
                tref = sqrt(ss)
            end
            ok = abs(obj - tref) < 1e-4 * (1.0 + abs(tref))
            @printf("socp_ls    m=%d d=%d cone=%d obj=%.6f ref=%.6f t=%.4fs %s\n",
                    m, d, m + 1, obj, tref, sec, ok ? "OK" : "FAIL")
        else
            @printf("socp_ls    m=%d d=%d rc=%d FAIL\n", m, d, r.rc)
        end
        return ok ? 0 : 1
    finally
        free(t)
    end
end

exit(main())
