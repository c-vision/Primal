#!/usr/bin/env julia
# binary_quadratic.jl — binary quadratic problem via its Shor SDP relaxation
# (MOSEK/Tutorials, binary-quadratic):
#
#   minimize x'Qx + c'x + R  over x in {0,1}^n
#
# Lifting X = x x' to Z = [X x; x' 1] PSD with diag(X) = x and Z[n,n] = 1 gives
# the lower bound min R + c'x + <Q, X>, checked against brute force over 2^n.
# Instance n = 4: Q = tridiag(-2, 2, -2), c = 0, optimum -4 at x = (1,1,1,1).
# The PSD check uses the library's Jacobi eigensolver, as the C sample.
# Port of examples/c/binary_quadratic.c.

using Printf
using Libdl
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const N = 4
const Q = [2.0 -2 0 0; -2 2 -2 0; 0 -2 2 -2; 0 0 -2 2]
const C = [0.0, 0, 0, 0]

# Objective value of the binary quadratic at the binary point x.
function fval(x)
    s = 0.0
    for i in 1:N
        for j in 1:N
            s += Q[i, j] * x[i] * x[j]
        end
        s += C[i] * x[i]
    end
    return s
end

# Shor SDP relaxation; returns (ok, value, Z as (N+1)^2 row-major).
function shor()
    dim = N + 1
    m = Model(0, 0)
    try
        barvars(m, [dim])
        add_cons(m, dim)                            # N diag = last column, 1 last diag = 1
        # objective symmat: Q on the top-left block, c/2 on the (i,n) entries
        terms = [(i, i, Q[i+1, i+1]) for i in 0:N-1]
        append!(terms, [(i, j, Q[i+1, j+1]) for i in 0:N-1 for j in i+1:N-1])
        append!(terms, [(i, N, 0.5 * C[i+1]) for i in 0:N-1])
        idx = sparsesymmat(m, dim, terms)
        barcj(m, 0, [(idx, 1.0)])
        for i in 0:N-1                              # Z_ii - Z_iN = 0
            m1 = sparsesymmat(m, dim, [(i, i, 1.0)])
            m2 = sparsesymmat(m, dim, [(i, N, 0.5)])
            baraij(m, i, 0, [(m1, 1.0), (m2, -1.0)])
            con_bounds(m, i, BK_FX; lo=0.0, up=0.0)
        end
        mN = sparsesymmat(m, dim, [(N, N, 1.0)])
        baraij(m, N, 0, [(mN, 1.0)])
        con_bounds(m, N, BK_FX; lo=1.0, up=1.0)
        sense(m, SENSE_MIN)
        r = solve(m)
        r.rc == 0 || return false, 0.0, zeros(dim * dim)
        return true, r.objective, getbarxj(m, 0)
    finally
        free(m)
    end
end

function main()
    ok, sdp, Z = shor()
    best, bx = 1e300, zeros(Int, N)                 # exact optimum by brute force over 2^N
    for mask in 0:(1<<N)-1
        x = [(mask >> i) & 1 for i in 0:N-1]
        f = fval(x)
        f < best && ((best, bx) = (f, x))
    end
    ev, evec = zeros(N + 1), zeros((N + 1) * (N + 1))
    ccall(Libdl.dlsym(Primal._lib(), :dmat_eig_jacobi), Cvoid,
          (Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), N + 1, Z, ev, evec)
    mineig = minimum(ev)
    okall = ok && sdp <= best + 1e-6 && mineig > -1e-7
    @printf("binary_quadratic  n=%d\n", N)
    @printf("  Shor SDP lower bound = %.8f   (Z PSD, min_eig=%.2e)\n", sdp, mineig)
    @printf("  exact optimum        = %.8f   at x=(%d,%d,%d,%d)\n", best, bx[1], bx[2], bx[3], bx[4])
    @printf("  gap = %.8f\n", best - sdp)
    @printf("%s\n", okall ? "OK" : "FAIL")
    return okall ? 0 : 1
end

exit(main())
