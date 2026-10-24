#!/usr/bin/env julia
# surface_cycles.jl — shortest cycle in a homology class (MOSEK Tutorials,
# surfacecycles). For a triangulated surface and a 1-chain C the shortest
# homologous chain is the LP
#
#   min sum_e w_e |x_e|  s.t.  x - C = D2 y
#
# with D2 the boundary operator (edges x triangles), |x_e| linearized by
# xabs_e >= +-x_e. A triangulated square cylinder (betti1 = 1) is built in
# code and C = (top ring) + D2 (one triangle) is shortened to a ring of 4 unit
# edges. Port of examples/c/surface_cycles.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

const N = 4                                         # sides of the cylinder
const NV = 2N                                       # 8 vertices
const NT = 2N                                       # 8 triangles
const NE = 4N                                       # 2 rings + verticals + diagonals = 16

function main()
    sq = [0.5 0.5; -0.5 0.5; -0.5 -0.5; 0.5 -0.5]
    xyz = zeros(NV, 3)
    for i in 1:N
        xyz[i, :] = [sq[i, 1], sq[i, 2], 0.0]       # bottom ring
        xyz[N+i, :] = [sq[i, 1], sq[i, 2], 1.0]     # top ring
    end
    F = Vector{NTuple{3,Int}}()                     # each side quad -> 2 triangles
    for i in 0:N-1
        j = (i + 1) % N
        push!(F, (i, j, N + j)); push!(F, (i, N + j, N + i))
    end
    nf = length(F)
    E = Vector{Tuple{Int,Int}}()                    # unique edges (min, max) in face order
    edge_id(a, b) = (e = findfirst(==((min(a, b), max(a, b))), E); e === nothing ? -1 : e - 1)
    for f in F, q in 0:2
        a, b = f[q+1], f[(q+1)%3+1]
        edge_id(a, b) < 0 && push!(E, (min(a, b), max(a, b)))
    end
    if length(E) != NE
        @printf("surface_cycles  edge count %d != %d FAIL\n", length(E), NE)
        return 1
    end
    # signed edge: +1 if a->b matches the stored orientation (min,max)
    edge_sgn(a, b) = (edge_id(a, b), a <= b ? 1.0 : -1.0)
    W = zeros(NE)
    for e in 1:NE
        a, b = E[e][1] + 1, E[e][2] + 1
        W[e] = sqrt((xyz[a, 1] - xyz[b, 1]) * (xyz[a, 1] - xyz[b, 1]) +
                    (xyz[a, 2] - xyz[b, 2]) * (xyz[a, 2] - xyz[b, 2]) +
                    (xyz[a, 3] - xyz[b, 3]) * (xyz[a, 3] - xyz[b, 3]))
    end
    D1 = zeros(Int, NV, NE)                         # boundary of edge (a,b) = b - a
    for e in 1:NE
        D1[E[e][1]+1, e] = -1; D1[E[e][2]+1, e] = 1
    end
    D2 = zeros(Int, NE, nf)                         # boundary of (i,j,k) = (j,k) - (i,k) + (i,j)
    for t in 1:nf
        e0, s0 = edge_sgn(F[t][2], F[t][3])
        e1, s1 = edge_sgn(F[t][1], F[t][3])
        e2, s2 = edge_sgn(F[t][1], F[t][2])
        D2[e0+1, t] += Int(s0); D2[e1+1, t] -= Int(s1); D2[e2+1, t] += Int(s2)
    end
    Cring = zeros(NE)                               # C = top ring + D2 * (one triangle)
    for i in 0:N-1
        e, sg = edge_sgn(N + i, N + (i + 1) % N)
        Cring[e+1] += sg
    end
    y0 = zeros(nf); y0[1] = 1.0
    C = zeros(NE)
    for e in 1:NE
        C[e] = Cring[e]
        for t in 1:nf
            C[e] += D2[e, t] * y0[t]
        end
    end
    rv, rval = 0.0, 0.0                             # the ring (x = Cring, y = -y0) is feasible: value 4
    for e in 1:NE
        s = Cring[e] - C[e]
        for t in 1:nf
            s -= D2[e, t] * (-y0[t])
        end
        rv = max(rv, abs(s)); rval += W[e] * abs(Cring[e])
    end
    @printf("  ring candidate: value=%.6f  residual=%.2e\n", rval, rv)

    XB, AB, YB = 0, NE, 2NE                         # LP: vars x(E), xabs(E), y(T)
    NVAR = 2NE + nf
    m = Model(0, 0)
    r = try
        add_vars(m, NVAR)
        for i in 0:NVAR-1
            var_bounds(m, i, BK_FR; lo=-Inf, up=Inf)
        end
        add_cons(m, 3NE)
        row = 0
        for e in 0:NE-1                             # xabs >= x
            a_row(m, row, [XB + e, AB + e], [1.0, -1.0])
            con_bounds(m, row, BK_UP; lo=-Inf, up=0.0); row += 1
        end
        for e in 0:NE-1                             # xabs >= -x
            a_row(m, row, [XB + e, AB + e], [-1.0, -1.0])
            con_bounds(m, row, BK_UP; lo=-Inf, up=0.0); row += 1
        end
        for e in 0:NE-1                             # x - D2 y = C
            ts = [t for t in 0:nf-1 if D2[e+1, t+1] != 0]
            a_row(m, row, [XB + e; [YB + t for t in ts]], [1.0; [-Float64(D2[e+1, t+1]) for t in ts]])
            con_bounds(m, row, BK_FX; lo=C[e+1], up=C[e+1]); row += 1
        end
        for e in 0:NE-1
            cj(m, AB + e, W[e+1])
        end
        sense(m, SENSE_MIN)
        solve(m)
    finally
        free(m)
    end
    @printf("  rc=%d prosta=%d solsta=%d\n", r.rc, r.prosta, r.solsta)
    ok = r.rc == 0
    x, xabs, y = zeros(NE), zeros(NE), zeros(nf)
    if ok
        x, xabs, y = r.x[XB+1:XB+NE], r.x[AB+1:AB+NE], r.x[YB+1:YB+nf]
    end
    obj = 0.0
    for e in 1:NE
        obj += W[e] * xabs[e]
    end
    cyc = 0.0                                       # D1 x = 0 : x is a cycle
    for i in 1:NV
        s = 0.0
        for e in 1:NE
            s += D1[i, e] * x[e]
        end
        cyc = max(cyc, abs(s))
    end
    hom = 0.0                                       # x - C - D2 y = 0 : same class
    for e in 1:NE
        s = x[e] - C[e]
        for t in 1:nf
            s -= D2[e, t] * y[t]
        end
        hom = max(hom, abs(s))
    end
    absok = maximum(abs(xabs[e] - abs(x[e])) for e in 1:NE)   # xabs = |x|
    okall = ok && cyc < 1e-8 && hom < 1e-8 && absok < 1e-8 && abs(obj - 4.0) < 1e-6
    @printf("surface_cycles  e=%d t=%d  shortest cycle = %.6f\n", NE, nf, obj)
    @printf("  |D1 x|max = %.2e  |x - C - D2 y|max = %.2e  |xabs-|x||max = %.2e\n", cyc, hom, absok)
    print("  support = {")
    for e in 0:NE-1
        abs(x[e+1]) > 1e-6 && print(e, e == NE - 1 ? "" : ",")
    end
    println("}")
    @printf("%s\n", okall ? "OK" : "FAIL")
    return okall ? 0 : 1
end

exit(main())
