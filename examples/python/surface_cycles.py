#!/usr/bin/env python3
"""Shortest cycle in a homology class (MOSEK Tutorials, surfacecycles).

Given a triangulated surface and a 1-chain C, the shortest homologous chain is the LP
     min  sum_e w_e |x_e|   s.t.  x - C = D2 y
where D2 is the boundary operator (edges x triangles) and y a 2-chain; |x_e| is
linearized with xabs_e >= +-x_e. The sample builds a triangulated square cylinder
(betti1 = 1) in code and shortens C = (top ring) + D2*(one triangle), homologous to
the ring but not minimal; the shortest representative is a ring of 4 unit edges.
Port of examples/c/surface_cycles.c.
"""

import math

from primalsolver import Model, BK, SENSE


def csum(values, start=0.0):
    """Left-to-right sum, as the C loop (sum() is compensated since Python 3.12)."""
    s = start
    for v in values:
        s += v
    return s


N = 4                                               # sides of the cylinder
NV = 2 * N                                          # 8 vertices
NE = 4 * N                                          # 2 rings + verticals + diagonals = 16


def main() -> int:
    sq = [(0.5, 0.5), (-0.5, 0.5), (-0.5, -0.5), (0.5, -0.5)]
    xyz = [(sq[i][0], sq[i][1], 0.0) for i in range(N)] + [(sq[i][0], sq[i][1], 1.0) for i in range(N)]
    faces = []                                      # each side quad -> 2 triangles
    for i in range(N):
        j = (i + 1) % N
        faces.append((i, j, N + j))
        faces.append((i, N + j, N + i))
    nf = len(faces)
    edges = []                                      # unique edges (min, max) in face order
    for f in faces:
        for q in range(3):
            e = tuple(sorted((f[q], f[(q + 1) % 3])))
            if e not in edges:
                edges.append(e)
    if len(edges) != NE:
        print(f"surface_cycles  edge count {len(edges)} != {NE} FAIL")
        return 1

    def edge_sgn(a, b):
        """Edge id and sign: +1 if a->b matches the stored orientation (min,max)."""
        return edges.index(tuple(sorted((a, b)))), (1.0 if a <= b else -1.0)

    w = [math.sqrt((xyz[a][0] - xyz[b][0]) ** 2 + (xyz[a][1] - xyz[b][1]) ** 2 + (xyz[a][2] - xyz[b][2]) ** 2)
         for a, b in edges]
    d1 = [[0] * NE for _ in range(NV)]              # vertices x edges: boundary of (a,b) = b - a
    for e, (a, b) in enumerate(edges):
        d1[a][e] = -1
        d1[b][e] = +1
    d2 = [[0] * nf for _ in range(NE)]              # edges x triangles: (j,k) - (i,k) + (i,j)
    for t, (fi, fj, fk) in enumerate(faces):
        e0, s0 = edge_sgn(fj, fk)
        e1, s1 = edge_sgn(fi, fk)
        e2, s2 = edge_sgn(fi, fj)
        d2[e0][t] += int(s0)
        d2[e1][t] -= int(s1)
        d2[e2][t] += int(s2)
    cring = [0.0] * NE                              # C = top ring + D2 * (one triangle)
    for i in range(N):
        e, sg = edge_sgn(N + i, N + (i + 1) % N)
        cring[e] += sg
    y0 = [1.0] + [0.0] * (nf - 1)
    c = [csum((d2[e][t] * y0[t] for t in range(nf)), cring[e]) for e in range(NE)]
    rv = rval = 0.0                                 # the ring (x = Cring, y = -y0) is feasible: value 4
    for e in range(NE):
        s = cring[e] - c[e]
        for t in range(nf):
            s -= d2[e][t] * (-y0[t])
        rv = max(rv, abs(s))
        rval += w[e] * abs(cring[e])
    print(f"  ring candidate: value={rval:.6f}  residual={rv:.2e}")

    xb, ab, yb = 0, NE, 2 * NE                      # LP: vars x(E), xabs(E), y(T)
    nvar = 2 * NE + nf
    with Model() as m:
        m.add_vars(nvar)
        for i in range(nvar):
            m.var_bounds(i, BK.FR, -math.inf, math.inf)
        m.add_cons(3 * NE)
        r = 0
        for e in range(NE):                         # xabs >= x
            m.a_row(r, [xb + e, ab + e], [1.0, -1.0])
            m.con_bounds(r, BK.UP, -math.inf, 0.0)
            r += 1
        for e in range(NE):                         # xabs >= -x
            m.a_row(r, [xb + e, ab + e], [-1.0, -1.0])
            m.con_bounds(r, BK.UP, -math.inf, 0.0)
            r += 1
        for e in range(NE):                         # x - D2 y = C
            ts = [t for t in range(nf) if d2[e][t]]
            m.a_row(r, [xb + e] + [yb + t for t in ts], [1.0] + [-float(d2[e][t]) for t in ts])
            m.con_bounds(r, BK.FX, c[e], c[e])
            r += 1
        for e in range(NE):
            m.cj(ab + e, w[e])
        m.sense(SENSE.MIN)
        res = m.solve()
    print(f"  rc={res.rc} prosta={res.prosta} solsta={res.solsta}")
    ok = res.rc == 0
    x, xabs, y = [0.0] * NE, [0.0] * NE, [0.0] * nf
    if ok:
        x = res.x[xb:xb + NE]
        xabs = res.x[ab:ab + NE]
        y = res.x[yb:yb + nf]
    obj = 0.0
    for e in range(NE):
        obj += w[e] * xabs[e]
    cyc = max(abs(csum(d1[i][e] * x[e] for e in range(NE))) for i in range(NV))   # D1 x = 0
    hom = 0.0                                       # x - C - D2 y = 0
    for e in range(NE):
        s = x[e] - c[e]
        for t in range(nf):
            s -= d2[e][t] * y[t]
        hom = max(hom, abs(s))
    absok = max(abs(xabs[e] - abs(x[e])) for e in range(NE))   # xabs = |x|
    okall = ok and cyc < 1e-8 and hom < 1e-8 and absok < 1e-8 and abs(obj - 4.0) < 1e-6
    print(f"surface_cycles  e={NE} t={nf}  shortest cycle = {obj:.6f}")
    print(f"  |D1 x|max = {cyc:.2e}  |x - C - D2 y|max = {hom:.2e}  |xabs-|x||max = {absok:.2e}")
    print("  support = {" + "".join(f"{e}{'' if e == NE - 1 else ','}" for e in range(NE) if abs(x[e]) > 1e-6) + "}")
    print("OK" if okall else "FAIL")
    return 0 if okall else 1


if __name__ == "__main__":
    raise SystemExit(main())
