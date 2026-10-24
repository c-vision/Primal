#!/usr/bin/env julia
# sparsecholesky.jl — MOSEK port "sparsecholesky", linear SDP version: a
# symmetric A admits a Cholesky factor iff it is PSD. A (3x3 tridiagonal,
# the 1-D Laplacian) is fixed as a bar variable X by equality rows and
# min 0 s.t. X PSD is solved: feasible <=> Cholesky exists. An explicit 3x3
# Cholesky cross-checks it. Port of examples/c/sparsecholesky.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const D = 3
const A = [2.0 -1 0; -1 2 -1; 0 -1 2]               # SPD tridiagonal (1-D Laplacian)

# Explicit 3x3 Cholesky (independent check); returns (ok, L).
function chol3(Mx)
    L = zeros(D, D)
    for i in 1:D, j in 1:i
        s = Mx[i, j]
        for k in 1:j-1
            s -= L[i, k] * L[j, k]
        end
        if i == j
            s <= 0 && return false, L
            L[i, i] = sqrt(s)
        else
            L[i, j] = s / L[j, j]
        end
    end
    return true, L
end

function main()
    m = Model(0, 0)
    sta = 0
    try
        add_vars(m, 1)
        var_bounds(m, 0, BK_FX; lo=0.0, up=0.0)     # dummy
        barvars(m, [D])
        mE = [sparsesymmat(m, D, [(i, i, 1.0)]) for i in 0:D-1]
        mO = [sparsesymmat(m, D, [(i, j, 1.0)]) for i in 0:D-1 for j in i+1:D-1]
        add_cons(m, 6)                              # fix X = A on every entry of the triangle
        row = 0
        for i in 0:D-1
            baraij(m, row, 0, [(mE[i+1], 1.0)])
            con_bounds(m, row, BK_FX; lo=A[i+1, i+1], up=A[i+1, i+1]); row += 1
        end
        oc = 1
        for i in 0:D-1, j in i+1:D-1                # <E_ij, X> = 2 X_ij = 2 A_ij
            baraij(m, row, 0, [(mO[oc], 1.0)])
            con_bounds(m, row, BK_FX; lo=2.0 * A[i+1, j+1], up=2.0 * A[i+1, j+1]); row += 1
            oc += 1
        end
        r = solve(m)
        if r.rc != 0
            @printf("optimize rc=%d\n", r.rc)
            return 1
        end
        sta = r.solsta
    finally
        free(m)
    end
    @printf("SDP feasibility of A: solsta = %d (%s)\n", sta, sta == SOLSTA_OPTIMAL ? "PSD" : "NOT PSD")

    ok_chol, L = chol3(A)                           # independent check: explicit Cholesky
    @printf("Direct Cholesky: %s\n", ok_chol ? "exists (A SPD)" : "does not exist")
    if ok_chol
        println("L =")
        for i in 1:D
            print("   ")
            for j in 1:D
                @printf(" %8.4f", L[i, j])
            end
            println()
        end
        err = 0.0                                   # A = L L' check
        for i in 1:D, j in 1:D
            s = 0.0
            for k in 1:D
                s += L[i, k] * L[j, k]
            end
            err += abs(s - A[i, j])
        end
        @printf("||L L' - A||_1 = %.3e\n", err)
        err > 1e-9 && (ok_chol = false)
    end
    ok = sta == SOLSTA_OPTIMAL && ok_chol
    @printf("%s (A PSD via SDP = Cholesky exists)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
