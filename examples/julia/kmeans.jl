#!/usr/bin/env julia
# kmeans.jl — k-means clustering as a mixed-integer conic program with a
# disjunction (MOSEK Tutorials, kmeans-clustering):
#
#   minimize sum_i d_i
#   s.t.  dAux_ij >= ||p_i - c_j||^2  (rotated cone per i,j)
#         d_i >= dAux_ij AND y_i == j  for SOME j  (DJC per i)
#         y_i in {0..K-1},  c_0 <= c_1
#
# Instance: points 0, 1, 2, 3 and K = 2: centroids 0.5 and 2.5, inertia 1.
# The centroids are bounded to the data range [0, 3] (declared deviation: the
# notebook leaves them free). Port of examples/c/kmeans.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NPTS = 4
const NK = 2
const PT = [0.0, 1.0, 2.0, 3.0]
const DOM_RZERO = 1
const DOM_RPLUS = 2

function main()
    C, DIST, AUX = 0, NK, NK + NPTS
    D3 = NK + NPTS + NPTS * NK
    Y = NK + NPTS + 2 * NPTS * NK
    HALF = Y + NPTS
    NV = HALF + 1
    nAux = NPTS * NK
    nAfe = nAux + NPTS                              # one per (i,j) distance + one label per i
    m = Model(0, 0)
    r = try
        add_vars(m, NV)
        add_cons(m, nAux + 1)                       # D3 links + centroid ordering
        for j in 0:NK-1
            var_bounds(m, C + j, BK_RA; lo=0.0, up=3.0)
        end
        for i in 0:NPTS-1
            var_bounds(m, DIST + i, BK_LO; lo=0.0, up=Inf)
        end
        for q in 0:nAux-1
            var_bounds(m, AUX + q, BK_LO; lo=0.0, up=Inf)
        end
        for q in 0:nAux-1
            var_bounds(m, D3 + q, BK_FR; lo=-Inf, up=Inf)
        end
        for i in 0:NPTS-1
            var_bounds(m, Y + i, BK_RA; lo=0.0, up=Float64(NK - 1))
            var_type(m, Y + i, VT_INT)
        end
        var_bounds(m, HALF, BK_FX; lo=0.5, up=0.5)
        row = 0
        for i in 0:NPTS-1, j in 0:NK-1              # D3_ij - C_j = -point_i
            q = i * NK + j
            a_row(m, row, [D3 + q, C + j], [1.0, -1.0])
            con_bounds(m, row, BK_FX; lo=-PT[i+1], up=-PT[i+1]); row += 1
        end
        a_row(m, row, [C, C + 1], [1.0, -1.0])     # c0 <= c1
        con_bounds(m, row, BK_UP; lo=-Inf, up=0.0); row += 1
        for i in 0:NPTS-1, j in 0:NK-1              # dAux_ij >= D3_ij^2
            q = i * NK + j
            cone(m, CT_RQUAD, [AUX + q, HALF, D3 + q])
        end
        for i in 0:NPTS-1
            cj(m, DIST + i, 1.0)
        end
        sense(m, SENSE_MIN)

        # disjunction: for each point, d_i >= dAux_ij AND y_i == j, for SOME j
        afes(m, nAfe)
        for i in 0:NPTS-1, j in 0:NK-1              # afe iK+j = d_i - dAux_ij
            a = i * NK + j
            afe_entry(m, a, DIST + i, 1.0)
            afe_entry(m, a, AUX + a, -1.0)
        end
        for i in 0:NPTS-1                           # afe nAux+i = y_i
            afe_entry(m, nAux + i, Y + i, 1.0)
        end
        rplus = domain(m, DOM_RPLUS; n=1)
        rzero = domain(m, DOM_RZERO; n=1)
        djcs(m, NPTS)
        for i in 0:NPTS-1
            dom, afe, bb = Int[], Int[], Float64[]
            for j in 0:NK-1
                append!(dom, [rplus, rzero]); append!(afe, [i * NK + j, nAux + i]); append!(bb, [0.0, Float64(j)])
            end
            djc(m, i, dom, afe, bb, fill(2, NK))
        end
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    obj, cen = 0.0, zeros(NK)
    if ok
        x = r.x
        cen = [x[C+1], x[C+2]]
        for i in 0:NPTS-1
            obj += x[DIST+i+1]
        end
    end
    okall = ok && abs(obj - 1.0) < 1e-6
    @printf("kmeans  n=%d K=%d\n", NPTS, NK)
    @printf("  objective (inertia) = %.8f (expected 1.0)   centers = (%.4f, %.4f)\n", obj, cen[1], cen[2])
    @printf("%s\n", okall ? "OK" : "FAIL")
    return okall ? 0 : 1
end

exit(main())
