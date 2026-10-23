#!/usr/bin/env python3
"""Min-cost flow on a random graph (LP, scaling test).

    min sum_e c_e f_e
      s.t. sum_{e out of v} f_e - sum_{e into v} f_e = b_v   (balance)
           0 <= f_e <= u_e

The balances come from a random reference flow, so the problem is feasible by
construction. Verified: node balance, capacity, strong duality. The data come from
the same 32-bit linear congruential generator as the C example.

    python network_flow.py [nodes] [edges]   (default 200 800)

Port of examples/c/network_flow.c.
"""

import math
import sys
import time

from primalsolver import Model, BK

_state = 27182818


def rnd():
    """Deterministic pseudo-random draw in [0,1] from the fixed seed."""
    global _state
    _state = (_state * 1103515245 + 12345) & 0xFFFFFFFF
    return ((_state >> 16) & 0x7FFF) / 32767.0


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    e_cnt = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    if n < 2 or e_cnt < 1:
        return 2
    eu, ev, cost, cap = [], [], [], []
    b = [0.0] * n
    for _ in range(e_cnt):
        u = int(rnd() * n)
        v = int(rnd() * n)
        if v == u:
            v = (v + 1) % n
        eu.append(u); ev.append(v)
        cost.append(0.5 + rnd() * 10.0)
        cap.append(5.0 + rnd() * 20.0)
        flow0 = rnd() * cap[-1] * 0.5               # reference flow
        b[u] += flow0                               # node u sends flow0 out
        b[v] -= flow0                               # node v receives flow0 in
    with Model() as m:
        m.add_vars(e_cnt)
        m.add_cons(n)
        for e in range(e_cnt):
            m.cj(e, cost[e])
            m.var_bounds(e, BK.RA, 0.0, cap[e])
        for v in range(n):
            sub, val = [], []
            for e in range(e_cnt):
                if eu[e] == v:
                    sub.append(e); val.append(1.0)
                if ev[e] == v:
                    sub.append(e); val.append(-1.0)
            if not sub:
                sub, val = [0], [0.0]
            m.a_row(v, sub, val)
            m.con_bounds(v, BK.FX, b[v], b[v])
        t0 = time.perf_counter()
        r = m.solve()
        sec = time.perf_counter() - t0
    obj = dobj = 0.0
    ok = r.rc == 0
    if ok:
        obj, dobj, f = r.objective, r.dual, r.x
        bal = [0.0] * n
        maxcap = minf = 0.0
        for e in range(e_cnt):
            bal[eu[e]] += f[e]
            bal[ev[e]] -= f[e]
            maxcap = max(maxcap, f[e] - cap[e])
            minf = min(minf, f[e])
        maxbal = max(abs(bal[v] - b[v]) for v in range(n))
        ok = maxbal < 1e-5 and maxcap < 1e-5 and minf > -1e-6 and abs(obj - dobj) < 1e-5 * (1.0 + abs(obj))
    print(f"netflow    N={n} E={e_cnt} vars={e_cnt} cons={n} obj={obj:.6f} dobj={dobj:.6f} t={sec:.4f}s "
          f"{'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
