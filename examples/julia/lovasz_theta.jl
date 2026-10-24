#!/usr/bin/env julia
# lovasz_theta.jl — maximum-clique upper bound via the Lovasz theta SDP
# (Rudolfovoorg/Improving_Upper_Bounds_of_MCP_using_Reduction_Rules):
#
#   theta(Gbar) = max <J, X>  s.t.  tr(X) = 1,  X_ij = 0 on every non-edge of G,  X PSD
#
# omega(G) <= theta(Gbar) <= chi(G). Hand instances: C5 -> sqrt(5), K3 -> 3,
# E5 (no edges) -> 1. The primal X is checked (trace, PSD via the library's
# Jacobi eigensolver, zeros on non-edges). Port of examples/c/lovasz_theta.c.

using Printf
using Libdl
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# theta(Gbar) for a graph given by its symmetric adjacency; returns (ok, theta, X).
function theta_sdp(adj)
    n = size(adj, 1)
    m = Model(0, 0)
    try
        barvars(m, [n])
        ne = count(adj[i, j] == 0 for i in 1:n for j in i+1:n)
        add_cons(m, 1 + ne)
        # objective matrix J (all ones), lower triangle
        idx = sparsesymmat(m, n, [t for i in 0:n-1 for t in [(i, i, 1.0); [(i, j, 1.0) for j in 0:i-1]]])
        barcj(m, 0, [(idx, 1.0)])
        idx = sparsesymmat(m, n, [(i, i, 1.0) for i in 0:n-1])   # row 0: tr(X) = 1
        baraij(m, 0, 0, [(idx, 1.0)])
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        row = 1
        for i in 0:n-1, j in i+1:n-1
            adj[i+1, j+1] != 0 && continue
            idx = sparsesymmat(m, n, [(j, i, 0.5)])                # <E_ij,X> = X_ij
            baraij(m, row, 0, [(idx, 1.0)])
            con_bounds(m, row, BK_FX; lo=0.0, up=0.0)
            row += 1
        end
        sense(m, SENSE_MAX)
        r = solve(m)
        r.rc == 0 || return false, 0.0, zeros(n * n)
        return true, r.objective, getbarxj(m, 0)
    finally
        free(m)
    end
end

# min eigenvalue of the n x n solution matrix (PSD check)
function min_eig(n, X)
    ev, evec = zeros(n), zeros(n * n)
    ccall(Libdl.dlsym(Primal._lib(), :dmat_eig_jacobi), Cvoid,
          (Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), n, X, ev, evec)
    return minimum(ev)
end

# verify tr(X) = 1, X PSD, X_ij = 0 on non-edges
function check_X(adj, X, tol)
    n = size(adj, 1)
    tr = 0.0
    for i in 0:n-1
        tr += X[i*n+i+1]
    end
    abs(tr - 1.0) > 1e-6 && return false
    min_eig(n, X) < -tol && return false
    for i in 0:n-1, j in i+1:n-1
        adj[i+1, j+1] == 0 && abs(X[i*n+j+1]) > tol && return false
    end
    return true
end

function main()
    C5 = zeros(Int, 5, 5)
    for i in 0:4
        C5[i+1, (i+1)%5+1] = 1; C5[(i+1)%5+1, i+1] = 1
    end
    K3 = [i != j ? 1 : 0 for i in 1:3, j in 1:3]
    E5 = zeros(Int, 5, 5)

    ok5, th5, X = theta_sdp(C5); x5 = ok5 && check_X(C5, X, 1e-6)
    ok3, th3, X = theta_sdp(K3); x3 = ok3 && check_X(K3, X, 1e-6)
    oke, the, X = theta_sdp(E5); xe = oke && check_X(E5, X, 1e-6)
    fl(v) = floor(Int, v + 1e-6)
    ok = x5 && x3 && xe && abs(th5 - sqrt(5.0)) < 1e-6 && abs(th3 - 3.0) < 1e-6 && abs(the - 1.0) < 1e-6 &&
         fl(th5) == 2 && fl(th3) == 3 && fl(the) == 1
    println("lovasz_theta  max-clique upper bound (theta of the complement)")
    @printf("  C5 (omega=2): theta=%.10f   expected sqrt(5)=%.10f   X ok=%d\n", th5, sqrt(5.0), x5)
    @printf("  K3 (omega=3): theta=%.10f   expected 3              X ok=%d\n", th3, x3)
    @printf("  E5 (omega=1): theta=%.10f   expected 1              X ok=%d\n", the, xe)
    @printf("  integer bound floor(theta): C5->%d, K3->%d, E5->%d\n", fl(th5), fl(th3), fl(the))
    @printf("%s\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
