#!/usr/bin/env julia
# mpc_linear.jl — finite-horizon linear MPC as a QP (the linearized branch of
# jonaylton/mpc-missile). Discretized double integrator p'' = u, h = 1:
#   x_{k+1} = A x_k + B u_k,  A = [1 1; 0 1],  B = (1/2, 1).
# Minimum-energy regulation to a terminal state over N steps:
#   minimize sum_k u_k^2  s.t.  x_0 = x_ini,  x_{k+1} = A x_k + B u_k,  x_N = x_term.
#
# Hand instance: N = 2, x_ini = (0,0), x_term = (1,0) -> u = (1,-1), sum u^2 = 2.
# Port of examples/c/mpc_linear.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NSTEP = 2
const NST = NSTEP + 1
P(k) = 2k                                           # state k occupies (p_k, v_k)
V(k) = 2k + 1
U(k) = 2NST + k                                     # controls after the states

function main()
    NV = 2NST + NSTEP
    NROW = 2 + 2NSTEP + 2
    A = [1.0 1.0; 0.0 1.0]
    B = [0.5, 1.0]
    xini = [0.0, 0.0]
    m = Model(NROW, NV)
    r = try
        for v in 0:NV-1
            var_bounds(m, v, BK_FR; lo=-Inf, up=Inf)
        end
        sense(m, SENSE_MIN)
        for k in 0:NSTEP-1
            qobj_ij(m, U(k), U(k), 2.0)             # u_k^2
        end
        row = 0
        for c in 0:1                                # x_0 = x_ini
            a_row(m, row, [c], [1.0])
            con_bounds(m, row, BK_FX; lo=xini[c+1], up=xini[c+1]); row += 1
        end
        for k in 0:NSTEP-1, c in 0:1                # x_{k+1} - A x_k - B u_k = 0
            sub = [c == 0 ? P(k + 1) : V(k + 1)]
            val = [1.0]
            for d in 0:1
                if A[c+1, d+1] != 0.0
                    push!(sub, d == 0 ? P(k) : V(k)); push!(val, -A[c+1, d+1])
                end
            end
            if B[c+1] != 0.0
                push!(sub, U(k)); push!(val, -B[c+1])
            end
            a_row(m, row, sub, val)
            con_bounds(m, row, BK_FX; lo=0.0, up=0.0); row += 1
        end
        a_row(m, row, [P(NSTEP)], [1.0])            # p_N = 1
        con_bounds(m, row, BK_FX; lo=1.0, up=1.0); row += 1
        a_row(m, row, [V(NSTEP)], [1.0])            # v_N = 0
        con_bounds(m, row, BK_FX; lo=0.0, up=0.0); row += 1
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    obj, u, x = 0.0, zeros(NSTEP), zeros(16)
    if ok
        obj, x = r.objective, r.x
        u = [x[U(k)+1] for k in 0:NSTEP-1]
    end
    good = ok && abs(obj - 2.0) < 1e-7 && abs(u[1] - 1.0) < 1e-6 && abs(u[2] + 1.0) < 1e-6 &&
           abs(x[P(NSTEP)+1] - 1.0) < 1e-6 && abs(x[V(NSTEP)+1]) < 1e-6
    @printf("mpc_linear: obj=%.8f  u=(%.6f, %.6f)  x_N=(%.6f, %.6f)  (expected 2, u=(1,-1), x_N=(1,0))  %s\n",
            obj, u[1], u[2], x[P(NSTEP)+1], x[V(NSTEP)+1], good ? "OK" : "FAIL")
    return good ? 0 : 1
end

exit(main())
