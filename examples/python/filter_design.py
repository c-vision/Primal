#!/usr/bin/env python3
"""FIR filter design as a trigonometric-polynomial SDP (MOSEK/Tutorials, filterdesign).

A real filter response H(w) = x_0 + 2 x_1 cos(w) + ... + 2 x_n cos(n w) is
nonnegative on an interval iff its coefficient vector lies in a cone characterized
by PSD Toeplitz matrices (<T_i^m, Y> = sum_k Y_{k,k+|i|}, the i-th diagonal sum):
  [0,pi] :  x_i = <T_i^{n+1}, X>
  [0,a]  :  x_i = <T_i^{n+1},X1> + <T_{i+1}^n,X2> + <T_{i-1}^n,X2> - 2 cos(a) <T_i^n, X2>
  [a,pi] :  x_i = <T_i^{n+1},X1> - <T_{i+1}^n,X2> - <T_{i-1}^n,X2> + 2 cos(a) <T_i^n, X2>
The design minimizes the stopband level t subject to H >= 0 (global),
1-d <= H <= 1+d (passband [0,wp]) and H <= t (stopband [ws,pi]). The returned
response is checked on a dense grid (DECLARED DEVIATION: the specifications are
verified by sampling, not by a second solver). Port of examples/c/filter_design.c.
"""

import math

from primalsolver import Model, BK, SENSE

NF = 3                                              # polynomial degree n
X, TT = 0, NF + 1
NV = NF + 2


def add_t(m, row, bar, dim, i, coef):
    """Add coef * <T_i^dim, X_bar> to the row (one symmetric matrix per diagonal entry)."""
    ii = abs(i)
    if ii >= dim:
        return
    for k in range(dim - ii):
        mid = m.sparsesymmat(dim, [(k, k + ii, 1.0 if ii == 0 else 0.5)])   # <m,X> = X_{k,k+ii}
        m.baraij(row, bar, [(mid, coef)])


def main() -> int:
    delta, wp, ws = 0.05, math.pi / 4.0, math.pi / 4.0 + math.pi / 8.0
    with Model() as m:
        m.add_vars(NV)                              # scalar vars: x[0..NF], t
        for i in range(NF + 1):
            m.var_bounds(X + i, BK.FR, -math.inf, math.inf)
        m.var_bounds(TT, BK.LO, 0.0, math.inf)
        # bars: 0 global X0(NF+1); 1,2 passband upper; 3,4 passband lower; 5,6 stopband
        for dim in (NF + 1, NF + 1, NF, NF + 1, NF, NF + 1, NF):
            m.barvars([dim])
        m.add_cons(4 * (NF + 1))
        row = 0
        for i in range(NF + 1):                     # global nonnegativity: <T_i,X0> - x_i = 0
            add_t(m, row, 0, NF + 1, i, 1.0)
            m.a_row(row, [X + i], [-1.0])
            m.con_bounds(row, BK.FX, 0.0, 0.0)
            row += 1
        for i in range(NF + 1):                     # passband upper (level 1+delta)
            add_t(m, row, 1, NF + 1, i, 1.0)
            add_t(m, row, 2, NF, i + 1, 1.0)
            add_t(m, row, 2, NF, i - 1, 1.0)
            add_t(m, row, 2, NF, i, -2.0 * math.cos(wp))
            rhs = 1.0 + delta if i == 0 else 0.0
            m.a_row(row, [X + i], [1.0])
            m.con_bounds(row, BK.FX, rhs, rhs)
            row += 1
        for i in range(NF + 1):                     # passband lower (level 1-delta)
            add_t(m, row, 3, NF + 1, i, 1.0)
            add_t(m, row, 4, NF, i + 1, 1.0)
            add_t(m, row, 4, NF, i - 1, 1.0)
            add_t(m, row, 4, NF, i, -2.0 * math.cos(wp))
            rhs = -(1.0 - delta) if i == 0 else 0.0
            m.a_row(row, [X + i], [-1.0])
            m.con_bounds(row, BK.FX, rhs, rhs)
            row += 1
        for i in range(NF + 1):                     # stopband (level t)
            add_t(m, row, 5, NF + 1, i, 1.0)
            add_t(m, row, 6, NF, i + 1, -1.0)
            add_t(m, row, 6, NF, i - 1, -1.0)
            add_t(m, row, 6, NF, i, 2.0 * math.cos(ws))
            if i == 0:
                m.a_row(row, [X, TT], [1.0, -1.0])
            else:
                m.a_row(row, [X + i], [1.0])
            m.con_bounds(row, BK.FX, 0.0, 0.0)
            row += 1
        m.cj(TT, 1.0)
        m.sense(SENSE.MIN)
        r = m.solve()
    ok = r.rc == 0
    x = r.x[:NV] if ok else [0.0] * NV
    topt = x[TT] if ok else 0.0

    neg = pa_hi = pa_lo = sb = 0                    # verify the specs on a dense grid
    for q in range(4001):
        w = math.pi * q / 4000.0
        h = x[X]
        for i in range(1, NF + 1):
            h += 2.0 * x[X + i] * math.cos(i * w)
        if h < -1e-5:
            neg = 1
        if w <= wp + 1e-9:
            if h > 1.0 + delta + 1e-4:
                pa_hi = 1
            if h < 1.0 - delta - 1e-4:
                pa_lo = 1
        if w >= ws - 1e-9 and h > topt + 1e-4:
            sb = 1
    okall = ok and not neg and not pa_hi and not pa_lo and not sb and topt < 1.0
    print(f"filter_design  n={NF}  wp=pi/4  ws=3pi/8  delta={delta:.2f}")
    print(f"  t* (stopband) = {topt:.6f}   H>=0:{int(not neg)}  passband ok:{int(not (pa_hi or pa_lo))}  "
          f"stopband ok:{int(not sb)}")
    print("  x = (" + ", ".join(f"{x[X + i]:.5f}" for i in range(NF + 1)) + ")")
    print("OK" if okall else "FAIL")
    return 0 if okall else 1


if __name__ == "__main__":
    raise SystemExit(main())
