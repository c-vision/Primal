#!/usr/bin/env julia
# wasserstein.jl — Wasserstein barycenter of discrete distributions (LP)
# (MOSEK/Tutorials, wasserstein-bary). For distributions nu_1..nu_K on a common
# support {x_1..x_n} and cost d_ij = |x_i - x_j|:
#
#   minimize sum_k sum_ij d_ij pi^k_ij
#   s.t.  sum_j pi^k_ij = mu_i,  sum_i pi^k_ij = nu^k_j,  pi >= 0,  mu >= 0
#
# Instance x = (0, 1, 2), nu_1 = (1/2, 1/2, 0), nu_2 = (0, 1/2, 1/2): the
# optimal value is 1 (the barycenter is not unique). Port of examples/c/wasserstein.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NB = 3                                        # support points
const NK = 2                                        # distributions
const XS = [0.0, 1.0, 2.0]
const NU = [0.5 0.5 0.0; 0.0 0.5 0.5]

function main()
    npi = NB * NB
    VMU, VPI, NV = 0, NB, NB + NK * npi
    m = Model(0, 0)
    r = try
        add_vars(m, NV)
        add_cons(m, NK * (NB + NB))                 # row + col marginals per k
        for i in 0:NB-1
            var_bounds(m, VMU + i, BK_LO; lo=0.0, up=Inf)
        end
        for q in 0:NK*npi-1
            var_bounds(m, VPI + q, BK_LO; lo=0.0, up=Inf)
        end
        row = 0
        for k in 0:NK-1
            base = VPI + k * npi
            for i in 0:NB-1                         # sum_j pi^k_ij - mu_i = 0
                a_row(m, row, [[base + i * NB + j for j in 0:NB-1]; VMU + i], [ones(NB); -1.0])
                con_bounds(m, row, BK_FX; lo=0.0, up=0.0); row += 1
            end
            for j in 0:NB-1                         # sum_i pi^k_ij = nu^k_j
                a_row(m, row, [base + i * NB + j for i in 0:NB-1], ones(NB))
                con_bounds(m, row, BK_FX; lo=NU[k+1, j+1], up=NU[k+1, j+1]); row += 1
            end
        end
        for k in 0:NK-1, i in 0:NB-1, j in 0:NB-1  # objective: sum d_ij pi^k_ij
            cj(m, VPI + k * npi + i * NB + j, abs(XS[i+1] - XS[j+1]))
        end
        sense(m, SENSE_MIN)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    obj, mu = 0.0, zeros(NB)
    if ok
        x = r.x
        mu = [x[VMU+i+1] for i in 0:NB-1]
        for k in 0:NK-1, i in 0:NB-1, j in 0:NB-1
            obj += abs(XS[i+1] - XS[j+1]) * x[VPI+k*npi+i*NB+j+1]
        end
    end
    msum = mu[1] + mu[2] + mu[3]
    okall = ok && abs(obj - 1.0) < 1e-8 && abs(msum - 1.0) < 1e-7 &&
            mu[1] >= -1e-9 && mu[2] >= -1e-9 && mu[3] >= -1e-9
    @printf("wasserstein  n=%d K=%d\n", NB, NK)
    @printf("  total cost = %.10f (expected 1.0)\n", obj)
    @printf("  barycenter mu = (%.6f, %.6f, %.6f)  sum=%.8f\n", mu[1], mu[2], mu[3], msum)
    @printf("%s\n", okall ? "OK" : "FAIL")
    return okall ? 0 : 1
end

exit(main())
