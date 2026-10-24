#!/usr/bin/env julia
# factor_model.jl — minimum-variance portfolio under a factor risk model.
#
# Sigma = F F' + diag(D), so w'Sigma w = ||F'w||^2 + sum_j D_j w_j^2. The QP is
# built with the explicit Sigma and cross-checked against a hand value.
# Hand case: N = 3, F = (1,1,0), D = (0,0,1): Sigma = [1 1 0; 1 1 0; 0 0 1].
# With e'w = 1, w'Sigma w = (1-w2)^2 + w2^2, minimised at w2 = 1/2, value 1/2.
# Port of examples/c/factor_model.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

function main()
    F = [1.0, 1.0, 0.0]
    D = [0.0, 0.0, 1.0]
    Sigma = [F[i] * F[j] + (i == j ? D[i] : 0.0) for i in 1:3, j in 1:3]

    m = Model(0, 0)
    r = try
        add_vars(m, 3)
        add_cons(m, 1)
        for j in 0:2
            var_bounds(m, j, BK_FR; lo=-Inf, up=Inf)
        end
        a_row(m, 0, [0, 1, 2], [1.0, 1.0, 1.0])    # e'w = 1
        con_bounds(m, 0, BK_FX; lo=1.0, up=1.0)
        for i in 0:2, j in i:2                      # w'Sigma w: 1/2 w'Qw with Q = 2 Sigma
            qobj(m, [(i, j, 2.0 * Sigma[i+1, j+1])])
        end
        solve(m)
    finally
        free(m)
    end
    if r.rc != 0
        @printf("optimize rc=%d\n", r.rc)
        return 1
    end
    obj, w = r.objective, r.x
    quad = 0.0                                      # the operator, rebuilt from the published w
    for i in 1:3, j in 1:3
        quad += w[i] * Sigma[i, j] * w[j]
    end
    ok = abs(obj - 0.5) < 1e-7 && abs(w[3] - 0.5) < 1e-6 && abs(w[1] + w[2] - 0.5) < 1e-6 &&
         abs(w[1] + w[2] + w[3] - 1.0) < 1e-6 && abs(quad - obj) < 1e-7
    @printf("w = (%.4f, %.4f, %.4f), obj = %.6f (expected 0.5)\n", w[1], w[2], w[3], obj)
    @printf("w'Sigma w reconstructed = %.6f, e'w = %.2e\n", quad, w[1] + w[2] + w[3])
    @printf("%s (Sigma = F F' + diag(D), minimum at w2 = 1/2)\n", ok ? "OK" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
