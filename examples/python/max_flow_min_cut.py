#!/usr/bin/env python3
"""Maximum flow / minimum cut as a linear program (LightConvex example list).

    maximize    sum_{e out of s} f_e
    subject to  0 <= f_e <= u_e                                    (capacity)
                sum_{e into v} f_e - sum_{e out of v} f_e = 0       (conservation, v != s,t)

The dual is the minimum cut; strong duality gives max flow = min cut, so agreement
is a genuine independent check. Instance (hand-derived max flow = min cut = 5):
arcs 0->1 (3), 0->2 (2), 1->3 (2), 2->3 (3), 1->2 (1), source 0, sink 3. The sample
recomputes conservation and capacities from the published flow and enumerates ALL
cuts. Port of examples/c/max_flow_min_cut.c.
"""

import math

from primalsolver import Model, BK

NARC = 5
ARCF = [0, 0, 1, 2, 1]                              # arc table: from, to, capacity
ARCT = [1, 2, 3, 3, 2]
CAP = [3.0, 2.0, 2.0, 3.0, 1.0]
NNODE = 4
SRC, SNK = 0, 3


def min_cut_bruteforce():
    """Minimum cut over the partitions with source in S and sink in T."""
    best = 1e30
    for mask in range(1 << (NNODE - 2)):
        in_s = {SRC: 1, SNK: 0}
        for v in range(NNODE):
            if v not in (SRC, SNK):
                bit = v if v < SRC else v - 1
                in_s[v] = (mask >> bit) & 1
        cut = 0.0
        for e in range(NARC):
            if in_s[ARCF[e]] and not in_s[ARCT[e]]:
                cut += CAP[e]
        best = min(best, cut)
    return best


def main() -> int:
    with Model(maxcon=NARC + (NNODE - 2), maxvar=NARC) as m:   # 5 capacity + 2 conservation rows
        for e in range(NARC):
            m.var_bounds(e, BK.LO, 0.0, math.inf)
        m.cj(0, -1.0)                               # maximize f0 + f1: min -(f0 + f1)
        m.cj(1, -1.0)
        row = 0
        for e in range(NARC):                       # capacity: f_e <= u_e
            m.a_row(row, [e], [1.0])
            m.con_bounds(row, BK.UP, -math.inf, CAP[e])
            row += 1
        for v in range(1, NNODE - 1):               # conservation: in - out = 0
            sub, val = [], []
            for e in range(NARC):
                c = (1.0 if ARCT[e] == v else 0.0) - (1.0 if ARCF[e] == v else 0.0)
                if c != 0.0:
                    sub.append(e)
                    val.append(c)
            m.a_row(row, sub, val)
            m.con_bounds(row, BK.FX, 0.0, 0.0)
            row += 1
        r = m.solve()
    if r.rc != 0:
        print(f"max_flow_min_cut  rc={r.rc} FAIL")
        return 1
    x = r.x
    flow = x[0] + x[1]
    maxcapviol = maxconsviol = 0.0                  # recompute from the published flow
    for e in range(NARC):
        if x[e] > CAP[e] + 1e-7:
            maxcapviol = max(maxcapviol, x[e] - CAP[e])
    for v in range(1, NNODE - 1):
        bal = 0.0
        for e in range(NARC):
            if ARCT[e] == v:
                bal += x[e]
            if ARCF[e] == v:
                bal -= x[e]
        maxconsviol = max(maxconsviol, abs(bal))
    cut = min_cut_bruteforce()
    ok = abs(flow - 5.0) < 1e-6 and maxcapviol < 1e-7 and maxconsviol < 1e-7 and abs(cut - flow) < 1e-9
    print(f"max_flow_min_cut  flow={flow:.6f} (expected 5)  min_cut={cut:.6f}  "
          f"max_cap_viol={maxcapviol:.2e} max_cons_viol={maxconsviol:.2e}  "
          f"f=({x[0]:.2f},{x[1]:.2f},{x[2]:.2f},{x[3]:.2f},{x[4]:.2f}) {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
