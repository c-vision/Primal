#!/usr/bin/env julia
# facility_location.jl — smallest k balls covering a point set (MISOCP)
# (MOSEK/Tutorials, facility-location/small_disks):
#
#   minimize t
#   s.t.  ||p_i - x_j||_2 <= r_j + M (1 - s_ij),  sum_j s_ij = 1,  s binary,  t >= r_j >= 0
#
# the Big-M form of "point i is assigned to ball j". Instance: points 0, 1, 2, 3
# on a line, k = 2: neighbours pair up, t = 0.5. Port of examples/c/facility_location.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NP = 4                                        # points
const NK = 2                                        # balls
const DIM = 2
const BIGM = 10.0

function main()
    P = [0.0 0; 1 0; 2 0; 3 0]
    # vars: T | R[NK] | X[NK*DIM] | S[NP*NK] | W[NP*NK] | D[NP*NK*DIM]
    VT = 0
    VR = VT + 1
    VX = VR + NK
    VS = VX + NK * DIM
    VW = VS + NP * NK
    VD = VW + NP * NK
    NV = VD + NP * NK * DIM
    nS, nW = NP * NK, NP * NK
    m = Model(0, 0)
    r = try
        add_vars(m, NV)
        add_cons(m, nS + NP * NK * DIM + NP + NK)   # W + D + assign + t>=R
        var_bounds(m, VT, BK_LO; lo=0.0, up=Inf)
        for j in 0:NK-1
            var_bounds(m, VR + j, BK_LO; lo=0.0, up=Inf)
        end
        for q in 0:NK*DIM-1
            var_bounds(m, VX + q, BK_FR; lo=-Inf, up=Inf)
        end
        for q in 0:nS-1
            var_bounds(m, VS + q, BK_RA; lo=0.0, up=1.0)
            var_type(m, VS + q, VT_INT_BIN)
        end
        for q in 0:nW-1
            var_bounds(m, VW + q, BK_FR; lo=-Inf, up=Inf)
        end
        for q in 0:NP*NK*DIM-1
            var_bounds(m, VD + q, BK_FR; lo=-Inf, up=Inf)
        end
        row = 0
        for i in 0:NP-1, j in 0:NK-1                # W_ij - R_j + M S_ij = M
            vs, vw = VS + i * NK + j, VW + i * NK + j
            a_row(m, row, [vw, VR + j, vs], [1.0, -1.0, BIGM])
            con_bounds(m, row, BK_FX; lo=BIGM, up=BIGM); row += 1
        end
        for i in 0:NP-1, j in 0:NK-1, q in 0:DIM-1  # D_ijq + X_jq = P_iq
            vd, vx = VD + (i * NK + j) * DIM + q, VX + j * DIM + q
            a_row(m, row, [vd, vx], [1.0, 1.0])
            con_bounds(m, row, BK_FX; lo=P[i+1, q+1], up=P[i+1, q+1]); row += 1
        end
        for i in 0:NP-1                             # sum_j s_ij = 1
            a_row(m, row, [VS + i * NK + j for j in 0:NK-1], ones(NK))
            con_bounds(m, row, BK_FX; lo=1.0, up=1.0); row += 1
        end
        for j in 0:NK-1                             # t >= R_j
            a_row(m, row, [VT, VR + j], [1.0, -1.0])
            con_bounds(m, row, BK_LO; lo=0.0, up=Inf); row += 1
        end
        for i in 0:NP-1, j in 0:NK-1                # W_ij >= ||D_ij||
            vw, vd = VW + i * NK + j, VD + (i * NK + j) * DIM
            cone(m, CT_QUAD, [vw, vd, vd + 1])
        end
        cj(m, VT, 1.0)
        sense(m, SENSE_MIN)
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    obj, R, X = 0.0, zeros(NK), zeros(NK * DIM)
    if ok
        x = r.x
        obj = x[VT+1]
        R = x[VR+1:VR+NK]
        X = x[VX+1:VX+NK*DIM]
    end
    covered = true                                  # every point within R[j] of its ball center
    for i in 1:NP
        ok || break
        best = 1e300
        for j in 0:NK-1
            dx, dy = P[i, 1] - X[2j+1], P[i, 2] - X[2j+2]
            dist = sqrt(dx * dx + dy * dy)
            dist - R[j+1] < best && (best = dist - R[j+1])
        end
        best > 1e-6 && (covered = false)
    end
    okall = ok && covered && abs(obj - 0.5) < 1e-6
    @printf("facility_location  n=%d k=%d\n", NP, NK)
    @printf("  t* = %.8f (expected 0.5)   radii=(%.4f,%.4f) coverage ok=%d\n", obj, R[1], R[2], covered)
    @printf("%s\n", okall ? "OK" : "FAIL")
    return okall ? 0 : 1
end

exit(main())
