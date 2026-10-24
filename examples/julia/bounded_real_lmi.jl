#!/usr/bin/env julia
# bounded_real_lmi.jl — the Bounded Real Lemma (H-infinity norm) as an SDP
# (S-Salavati/Convex-Optimization-with-LMIs): the LMI M(x) <= 0 is a PSD bar
# variable R = -M(x) equated entrywise to M(x).
#
# For a stable LTI system (A,B,C,D), ||G||_inf < gamma iff there is P > 0 with
#   [A'P + PA + C'C, PB + C'D; B'P + D'C, D'D - gamma^2 I] <= 0.
# The model minimises gamma^2. Case: SISO A = -1, B = 1, C = 1, D = 0, whose
# H-infinity norm is |CB|/|A| = 1 (g >= P^2/(2P-1), minimised at P = 1).
# Port of examples/c/bounded_real_lmi.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    B, C, D = 1.0, 1.0, 0.0                         # SISO, A = -1, |CB|/|A| = 1
    m = Model(0, 0)
    ok, g2, P = false, 0.0, zeros(4)
    try
        r00 = sparsesymmat(m, 2, [(0, 0, 1.0)])
        r01 = sparsesymmat(m, 2, [(0, 1, 1.0)])
        r11 = sparsesymmat(m, 2, [(1, 1, 1.0)])
        p00 = sparsesymmat(m, 1, [(0, 0, 1.0)])
        barvars(m, [1])                             # barvar 0 : P (1x1 PSD)
        barvars(m, [2])                             # barvar 1 : R (2x2 PSD) = -M
        G = 0
        add_vars(m, 1)
        var_bounds(m, G, BK_LO; lo=0.0, up=Inf)     # g = gamma^2 >= 0
        sense(m, SENSE_MIN)
        cj(m, G, 1.0)
        add_cons(m, 3)
        # M = [-2P + C^2, P*B + C*D; P*B + D*C, D^2 - g]  (A = -1)
        baraij(m, 0, 1, [(r00, 1.0)])               # row 0 : R00 - 2P = -C^2
        baraij(m, 0, 0, [(p00, -2.0)])
        con_bounds(m, 0, BK_FX; lo=-C * C, up=-C * C)
        baraij(m, 1, 1, [(r01, 0.5)])               # row 1 : R01 + P*B = -C*D  (<E01,R>/2 = R01)
        baraij(m, 1, 0, [(p00, B)])
        con_bounds(m, 1, BK_FX; lo=-C * D, up=-C * D)
        baraij(m, 2, 1, [(r11, 1.0)])               # row 2 : R11 - g = -D^2
        a_row(m, 2, [G], [-1.0])
        con_bounds(m, 2, BK_FX; lo=-D * D, up=-D * D)
        r = solve(m)
        ok = r.rc == 0
        if ok
            g2 = r.x[G+1]
            P = getbarxj(m, 0)
        end
    finally
        free(m)
    end
    # independent check: the LMI at the returned point must be <= 0 (2x2), and
    # gamma = sqrt(g2) must equal |CB|/|A| = 1
    m00, m01, m11 = -2 * P[1] + C * C, P[1] * B + C * D, D * D - g2
    tr, det = m00 + m11, m00 * m11 - m01 * m01
    lmax = tr / 2 + sqrt(max(0.0, tr * tr / 4 - det))
    gamma = sqrt(g2)
    good = ok && abs(gamma - 1.0) < 1e-6 && lmax <= 1e-6
    @printf("bounded_real_lmi  gamma^2*=%.8f  gamma*=%.8f (expected 1)  P=%.6f  lmax=%.2e  %s\n",
            g2, gamma, P[1], lmax, good ? "OK" : "FAIL")
    return good ? 0 : 1
end

exit(main())
