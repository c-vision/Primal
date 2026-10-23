#!/usr/bin/env python3
"""L1 (least absolute deviations) regression as an LP (scaling test: many rows).

    min ||A x - b||_1   <=>   min sum_i t_i
      s.t.  (A x - b)_i <= t_i,  -(A x - b)_i <= t_i,  t_i >= 0

Verified: feasibility and strong duality pobj ~ dobj. The data come from the same
32-bit linear congruential generator as the C example, so the model is identical.

    python cvx_regression.py [m] [d]   (default 400 30)

Port of examples/c/cvx_regression.c.
"""

import math
import sys
import time

from primalsolver import Model, BK

_state = 14142135


def rnd():
    """Deterministic pseudo-random draw in [0,1] from the fixed seed."""
    global _state
    _state = (_state * 1103515245 + 12345) & 0xFFFFFFFF
    return ((_state >> 16) & 0x7FFF) / 32767.0


def main() -> int:
    m_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    d = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    if m_rows < 1 or d < 1:
        return 2
    a = [rnd() * 2.0 - 1.0 for _ in range(m_rows * d)]
    b = [rnd() * 2.0 - 1.0 for _ in range(m_rows)]
    with Model() as m:
        m.add_vars(d + m_rows)                      # x (d) free, t (m) >= 0
        m.add_cons(2 * m_rows)
        for j in range(d):
            m.var_bounds(j, BK.FR, -math.inf, math.inf)
        for i in range(m_rows):
            m.var_bounds(d + i, BK.LO, 0.0, math.inf)
            m.cj(d + i, 1.0)
        cols = list(range(d))
        for i in range(m_rows):
            ai = a[i * d:(i + 1) * d]
            m.a_row(i, cols + [d + i], ai + [-1.0])                 # A_i x - t_i <= b_i
            m.con_bounds(i, BK.UP, -math.inf, b[i])
            m.a_row(m_rows + i, cols + [d + i], [-v for v in ai] + [-1.0])   # -A_i x - t_i <= -b_i
            m.con_bounds(m_rows + i, BK.UP, -math.inf, -b[i])
        t0 = time.perf_counter()
        r = m.solve()
        sec = time.perf_counter() - t0
    obj = dobj = 0.0
    ok = r.rc == 0
    if ok:
        obj, dobj, x = r.objective, r.dual, r.x
        maxviol = mint = 0.0
        for i in range(m_rows):
            res = -b[i] + sum(a[i * d + j] * x[j] for j in range(d))
            maxviol = max(maxviol, abs(res) - x[d + i])
            mint = min(mint, x[d + i])
        ok = maxviol < 1e-5 and mint > -1e-6 and abs(obj - dobj) < 1e-5 * (1.0 + abs(obj))
    print(f"cvx_reg    m={m_rows} d={d} vars={d + m_rows} cons={2 * m_rows} obj={obj:.6f} dobj={dobj:.6f} "
          f"t={sec:.4f}s {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
