#!/usr/bin/env julia
# rank_one.jl — best-subset selection by rank-one convexification (MOSEK
# Tutorials, rank-one-regression), with a disjunction selecting the support:
#
#   minimize y'y - 2 y'X b + t + mu sum_i u_i
#   s.t.  t >= b'(X'X + lambda I) b  (rotated cone),  sum z <= k,  z binary,
#         |b_i| <= u_i,  (z_i == 1) OR (b_i == 0)  (one DJC per i)
#
# Instance X = [1 0; 1 1; 0 1], y = (1, 2, 1), k = 1, lambda = mu = 0: the best
# one-variable fit is b = (3/2, 0) (or (0, 3/2)), objective 3/2.
# Port of examples/c/rank_one.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const NPTS = 3
const NF = 2
const MU = 0.0
const KCAP = 1
const XP = [1.0 0; 1 1; 0 1; 0 0; 0 0]
const Y = [1.0, 2, 1]
const DOM_RZERO = 1

function main()
    B, U, TT, Z, W = 0, NF, 2NF, 2NF + 1, 3NF + 1
    HT = 3NF + 1 + (NPTS + NF)
    ONE = HT + 1
    NV = HT + 2
    m = Model(0, 0)
    r = try
        add_vars(m, NV)
        add_cons(m, (NPTS + NF) + 1 + 1 + 2NF)      # W links + HT + card + |b|<=u
        for i in 0:NF-1
            var_bounds(m, B + i, BK_FR; lo=-Inf, up=Inf)
            var_bounds(m, U + i, BK_LO; lo=0.0, up=Inf)
            var_bounds(m, Z + i, BK_RA; lo=0.0, up=1.0)
            var_type(m, Z + i, VT_INT_BIN)
        end
        var_bounds(m, TT, BK_LO; lo=0.0, up=Inf)
        for k in 0:NPTS+NF-1
            var_bounds(m, W + k, BK_FR; lo=-Inf, up=Inf)
        end
        var_bounds(m, HT, BK_LO; lo=0.0, up=Inf)
        var_bounds(m, ONE, BK_FX; lo=1.0, up=1.0)
        row = 0
        for k in 0:NPTS+NF-1                        # W_k - sum_j Xp[k][j] B_j = 0
            sub, val = [W + k], [1.0]
            for j in 0:NF-1
                XP[k+1, j+1] != 0.0 && (push!(sub, B + j); push!(val, -XP[k+1, j+1]))
            end
            a_row(m, row, sub, val)
            con_bounds(m, row, BK_FX; lo=0.0, up=0.0); row += 1
        end
        a_row(m, row, [HT, TT], [1.0, -0.5])
        con_bounds(m, row, BK_FX; lo=0.0, up=0.0); row += 1
        a_row(m, row, [Z + i for i in 0:NF-1], ones(NF))
        con_bounds(m, row, BK_UP; lo=-Inf, up=KCAP); row += 1
        for i in 0:NF-1                             # |B_i| <= U_i
            a_row(m, row, [B + i, U + i], [1.0, -1.0])
            con_bounds(m, row, BK_UP; lo=-Inf, up=0.0); row += 1
            a_row(m, row, [B + i, U + i], [-1.0, -1.0])
            con_bounds(m, row, BK_UP; lo=-Inf, up=0.0); row += 1
        end
        cone(m, CT_RQUAD, [ONE; HT; collect(W:W+NPTS+NF-1)])
        coeff = zeros(NF)
        for j in 1:NF, k in 1:NPTS
            coeff[j] += XP[k, j] * Y[k]
        end
        for j in 0:NF-1
            cj(m, B + j, -2.0 * coeff[j+1])
        end
        cj(m, TT, 1.0)
        for i in 0:NF-1
            cj(m, U + i, MU)
        end
        yy = 0.0
        for k in 1:NPTS
            yy += Y[k] * Y[k]
        end
        cfix(m, yy)
        sense(m, SENSE_MIN)
        afes(m, 2NF)                                # disjunction (z_i == 1) OR (b_i == 0)
        for i in 0:NF-1
            afe_entry(m, i, Z + i, 1.0)
            afe_entry(m, NF + i, B + i, 1.0)
        end
        rzero = domain(m, DOM_RZERO; n=1)
        djcs(m, NF)
        for i in 0:NF-1
            djc(m, i, [rzero, rzero], [i, NF + i], [1.0, 0.0], [1, 1])
        end
        solve(m)
    finally
        free(m)
    end
    ok = r.rc == 0
    obj, b0, b1 = ok ? (r.objective, r.x[B+1], r.x[B+2]) : (0.0, 0.0, 0.0)
    onezero = abs(b0) < 1e-6 || abs(b1) < 1e-6
    okall = ok && abs(obj - 1.5) < 1e-6 && onezero && (abs(abs(b0) - 1.5) < 1e-5 || abs(abs(b1) - 1.5) < 1e-5)
    @printf("rank_one  k=%d\n", KCAP)
    @printf("  b = (%.8f, %.8f)   objective = %.10f (expected 1.5)\n", b0, b1, obj)
    @printf("%s\n", okall ? "OK" : "FAIL")
    return okall ? 0 : 1
end

exit(main())
