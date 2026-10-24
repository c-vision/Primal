#!/usr/bin/env julia
# quantum_separability.jl — PPT and CCNR criteria for bipartite entanglement
# (y1-zhu/quantum-correlations, RobustnessToPPT / RobustnessToCCNR).
#
#   PPT: max t  s.t.  (1/d^2) I + t C PSD,  C = rho^G - (1/d^2) I,
#        with closed form t* = 1/(1 - d^2 lambda_min(rho^G)).
#   CCNR: the white-noise robustness where ||R(t rho + (1-t) I/d^2)||_1 = 1.
#
# Instance: the qutrit isotropic state rho_p = p |Phi+><Phi+| + (1-p) I/9: both
# robustnesses are 1/(4p), exactly 1 at p = 1/4. The SDP uses one 9x9 bar
# variable; every number is cross-checked with closed forms and the library's
# Jacobi eigensolver. Port of examples/c/quantum_separability.c.

using Printf
using Libdl
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const D = 3
const N = D * D                                     # 9
idx(a, b, c, d) = (a * D + b) * N + (c * D + d) + 1 # entry ((a,b),(c,d)) of a row-major N x N

# rho_p and its partial transpose on B: rho^G[(i,k),(j,l)] = rho[(i,l),(j,k)]
function build_state(p)
    rho, rhot = zeros(N * N), zeros(N * N)
    for a in 0:D-1, b in 0:D-1, c in 0:D-1, d in 0:D-1
        v = (a == c && b == d) ? (1.0 - p) / Float64(D * D) : 0.0
        (a == b && c == d) && (v += p / Float64(D))
        rho[idx(a, b, c, d)] = v
    end
    for a in 0:D-1, k in 0:D-1, j in 0:D-1, l in 0:D-1
        rhot[idx(a, k, j, l)] = rho[idx(a, l, j, k)]
    end
    return rho, rhot
end

# eigenvalues of a real symmetric N x N matrix with the library's Jacobi routine
function eigs(A)
    ev, evec = zeros(N), zeros(N * N)
    ccall(Libdl.dlsym(Primal._lib(), :dmat_eig_jacobi), Cvoid,
          (Cint, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), N, A, ev, evec)
    return ev
end
lambda_min(A) = minimum(eigs(A))

# nuclear norm of a real N x N matrix, via the eigenvalues of M M'
function nuclear_norm(A)
    M = zeros(N * N)
    for i in 0:N-1, j in 0:N-1
        s = 0.0
        for k in 0:N-1
            s += A[i*N+k+1] * A[j*N+k+1]
        end
        M[i*N+j+1] = s
    end
    s = 0.0
    for e in eigs(M)
        s += sqrt(e > 0.0 ? e : 0.0)
    end
    return s
end

# CCNR: realigned state R[(i,j),(k,l)] = rho[(i,k),(j,l)]
function realign(rho)
    R = zeros(N * N)
    for i in 0:D-1, j in 0:D-1, k in 0:D-1, l in 0:D-1
        R[idx(i, j, k, l)] = rho[idx(i, k, j, l)]
    end
    return R
end

# SDP: max t s.t. S = (1/9) I + t C PSD, one row per lower-triangle entry of S - t C = (1/9) I
function ppt_robustness_sdp(rhot)
    m = Model(0, 0)
    r = try
        add_vars(m, 1)                              # scalar t
        var_bounds(m, 0, BK_RA; lo=0.0, up=1.0)
        barvars(m, [N])
        add_cons(m, N * (N + 1) ÷ 2)
        row = 0
        for L1 in 0:N-1, L2 in 0:L1
            C = rhot[L1*N+L2+1] - (L1 == L2 ? 1.0 / Float64(D * D) : 0.0)
            id = sparsesymmat(m, N, [(L1, L2, L1 == L2 ? 1.0 : 0.5)])   # <m,S> = S_L1L2
            baraij(m, row, 0, [(id, 1.0)])
            a_row(m, row, [0], [-C])
            rhs = L1 == L2 ? 1.0 / Float64(D * D) : 0.0
            con_bounds(m, row, BK_FX; lo=rhs, up=rhs)
            row += 1
        end
        cj(m, 0, 1.0)
        sense(m, SENSE_MAX)
        solve(m)
    finally
        free(m)
    end
    return r.rc == 0, (r.rc == 0 ? r.x[1] : 0.0)
end

function main()
    P, PCT = 0.5, 0.25                              # entangled state, PPT separability threshold
    rho, rhot = build_state(P)
    lmin = lambda_min(rhot)                         # eigenvalues of rho^G: closed form vs Jacobi
    lmin_cf = (1.0 - 4.0 * P) / Float64(D * D)
    ok_sdp, t_sdp = ppt_robustness_sdp(rhot)        # PPT robustness: SDP vs closed form
    t_cf = 1.0 / (1.0 - Float64(D * D) * lmin)
    Rmat = realign(rho)                             # CCNR: realignment norm and robustness
    ccnr = nuclear_norm(Rmat)
    ccnr_cf = (1.0 + 8.0 * P) / 3.0
    # R(t rho + (1-t) I/9) = t Rmat + (1-t)/9 e e', e e'[(i,j),(k,l)] = delta_ij delta_kl
    eeT = zeros(N * N)
    for i in 0:D-1, j in 0:D-1, k in 0:D-1, l in 0:D-1
        eeT[idx(i, j, k, l)] = (i == j && k == l) ? 1.0 : 0.0
    end
    t_ccnr = 1.0
    for k in 0:10000
        t = k * 1e-4
        RM = [t * Rmat[q] + (1.0 - t) * (1.0 / Float64(D * D)) * eeT[q] for q in 1:N*N]
        nuclear_norm(RM) > 1.0 && break
        t_ccnr = t
    end
    # witness W = I - |Phi~><Phi~|: W^G = I - SWAP PSD, tr(W rho) = 2/3 - 8p/3 < 0
    W = zeros(N * N)
    for a in 0:D-1, b in 0:D-1, c in 0:D-1, d in 0:D-1
        W[idx(a, b, c, d)] = ((a == c && b == d) ? 1.0 : 0.0) - ((a == b && c == d) ? 1.0 : 0.0)
    end
    WG = zeros(N * N)
    for a in 0:D-1, k in 0:D-1, j in 0:D-1, l in 0:D-1
        WG[idx(a, k, j, l)] = W[idx(a, l, j, k)]
    end
    wmin = lambda_min(WG)
    wtr = 0.0
    for a in 0:N-1, b in 0:N-1
        wtr += W[a*N+b+1] * rho[b*N+a+1]
    end
    wtr_cf = 2.0 / 3.0 - 8.0 * P / 3.0
    rho_t, rhot_t = build_state(PCT)                # thresholds: at p = 1/4 both criteria are 1
    t_thr = 1.0 / (1.0 - Float64(D * D) * lambda_min(rhot_t))
    ccnr_t = nuclear_norm(realign(rho_t))

    ok = ok_sdp && abs(t_sdp - t_cf) < 1e-5 && abs(lmin - lmin_cf) < 1e-9 && abs(ccnr - ccnr_cf) < 1e-8 &&
         abs(t_ccnr - t_cf) < 1e-3 && wmin > -1e-9 && wtr < -1e-6 && abs(wtr - wtr_cf) < 1e-9 &&
         abs(t_thr - 1.0) < 1e-8 && abs(ccnr_t - 1.0) < 1e-8
    @printf("quantum_separability  qutrit isotropic  p=%.2f  (p_thr=%.2f)\n", P, PCT)
    @printf("  lambda_min(rho^T) = %.10f   (closed form %.10f)\n", lmin, lmin_cf)
    @printf("  PPT robustness: SDP t*=%.10f   closed form 1/(4p)=%.10f   ok=%d\n", t_sdp, t_cf, ok_sdp)
    @printf("  CCNR ||R(rho)||_1 = %.10f   (closed form (1+8p)/3=%.10f)\n", ccnr, ccnr_cf)
    @printf("  CCNR robustness t*=%.10f   PPT robustness t*=%.10f\n", t_ccnr, t_cf)
    @printf("  witness W=I-|Phi~><Phi~|: lambda_min(W^T)=%.3e, tr(W rho)=%.10f (closed form %.10f)\n", wmin, wtr, wtr_cf)
    @printf("  threshold p=1/4: PPT t*=%.10f, CCNR ||R||=%.10f (both = 1)\n", t_thr, ccnr_t)
    @printf("%s\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
