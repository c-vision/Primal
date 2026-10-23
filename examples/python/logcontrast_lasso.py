#!/usr/bin/env python3
"""Log-contrast LASSO logistic regression (audreyolmsted/admm-portfolio, "Log-Contrast
LASSO Logistic Regression via ADMM", benchmarked against MOSEK/CVXR).

    minimize  (1/n) sum_i log(1 + exp(-y_i eta_i))  +  lambda * sum_j s_j |b_j|
    s.t.      eta = [1 X] beta ,   sum_j b_j = 0         (log-contrast)

with CLR-transformed features X, labels y in {+1,-1} and column scales s_j. Two
independent solvers run on the same deterministic instance:
  1. PrimalSolver: the softplus t_i >= log(1+exp(z_i)) is two exponential cones per
     sample, the L1 term a linear epigraph, the log-contrast one equality;
  2. ADMM, the repo's first-order method (Newton beta-step, closed-form joint
     proximal z-step).
Both are checked against a dense brute force over the reduced space
(intercept, b_1, b_2). Instance: n=16, p=3, lambda=0.3; the LASSO sets feature 0 to
exactly zero. The data come from the same 32-bit generator as the C example.
Port of examples/c/logcontrast_lasso.c.
"""

import math

from primalsolver import Model, BK, CT

NF = 3                                              # features
NS = 16                                             # samples
NL = 4                                              # p + 1 unknowns
LAM = 0.3                                           # LASSO penalty
WTRUE = [-0.5, -0.5, 1.0]                           # true log-contrast signal (sum zero)
_state = 20260924


def rnd():
    """Deterministic pseudo-random draw in [0,1] from the fixed seed."""
    global _state
    _state = (_state * 1103515245 + 12345) & 0xFFFFFFFF
    return ((_state >> 16) & 0x7FFF) / 32767.0


def make_instance():
    """The deterministic CLR-transformed instance, its labels and column scales."""
    x = []
    for _ in range(NS):
        raw = [rnd() * 2.0 - 1.0 for _ in range(NF)]
        m = 0.0
        for v in raw:
            m += v
        m /= NF
        x.append([v - m for v in raw])              # CLR: row mean 0
    y = []
    for i in range(NS):
        eta = 0.0
        for j in range(NF):
            eta += x[i][j] * WTRUE[j]
        eta += rnd() - 0.5                          # label noise
        y.append(1.0 if eta > 0.0 else -1.0)
    sig = []
    for j in range(NF):
        mu = v = 0.0
        for i in range(NS):
            mu += x[i][j]
        mu /= NS
        for i in range(NS):
            v += (x[i][j] - mu) * (x[i][j] - mu)
        s = math.sqrt(((NS - 1) / NS) * (v / (NS - 1)))
        sig.append(s if s != 0.0 else 1.0)
    return x, y, sig


X, Y, SIG = make_instance()


def softplus(z):
    return z + math.log1p(math.exp(-z)) if z >= 0.0 else math.log1p(math.exp(z))


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0.0 else math.exp(z) / (1.0 + math.exp(z))


def loss(b):
    """Mean logistic loss at beta (intercept first)."""
    s = 0.0
    for xi, yi in zip(X, Y):
        eta = b[0]
        for j in range(NF):
            eta += xi[j] * b[1 + j]
        s += softplus(-yi * eta)
    return s / NS


def objective(b):
    s = loss(b)
    for j in range(NF):
        s += LAM * SIG[j] * abs(b[1 + j])
    return s


# 1. conic model: vars b0 | b_j | g_j | per sample (t,m,s,a,b) | one
def BJ(j): return 1 + j                             # noqa: E704
def GJ(j): return 1 + NF + j                        # noqa: E704
def SB(i): return 1 + 2 * NF + 5 * i                # noqa: E704 - t,m,s,a,b


ONE = 1 + 2 * NF + 5 * NS
NVAR = ONE + 1
NR = 1 + 2 * NF + 3 * NS


def conic_solve():
    """Solve the log-contrast LASSO through the PEXP conic model; returns (ok, beta, obj)."""
    with Model(maxcon=NR, maxvar=NVAR) as m:
        for j in range(NVAR):
            m.var_bounds(j, BK.FR, -math.inf, math.inf)
        for j in range(NF):
            m.var_bounds(GJ(j), BK.LO, 0.0, math.inf)
        for i in range(NS):
            m.var_bounds(SB(i) + 3, BK.LO, 0.0, math.inf)
            m.var_bounds(SB(i) + 4, BK.LO, 0.0, math.inf)
        m.var_bounds(ONE, BK.FX, 1.0, 1.0)
        m.a_row(0, [BJ(j) for j in range(NF)], [1.0] * NF)   # log-contrast: sum_j b_j = 0
        m.con_bounds(0, BK.FX, 0.0, 0.0)
        row = 1
        for j in range(NF):                         # g_j >= |b_j|
            m.a_row(row, [GJ(j), BJ(j)], [1.0, -1.0]); m.con_bounds(row, BK.LO, 0.0, math.inf); row += 1
            m.a_row(row, [GJ(j), BJ(j)], [1.0, 1.0]);  m.con_bounds(row, BK.LO, 0.0, math.inf); row += 1
        for i in range(NS):
            t, mm, s, a, b = (SB(i) + k for k in range(5))
            m.a_row(row, [mm, t], [1.0, 1.0])       # m = -t
            m.con_bounds(row, BK.FX, 0.0, 0.0)
            row += 1
            # s + t + y_i eta_i = 0, eta_i = b0 + sum_j X_ij b_j
            m.a_row(row, [s, t, 0] + [BJ(j) for j in range(NF)], [1.0, 1.0, Y[i]] + [Y[i] * X[i][j] for j in range(NF)])
            m.con_bounds(row, BK.FX, 0.0, 0.0)
            row += 1
            m.a_row(row, [a, b], [1.0, 1.0])        # a + b <= 1
            m.con_bounds(row, BK.UP, -math.inf, 1.0)
            row += 1
        for i in range(NS):
            m.cone(CT.PEXP, [SB(i) + 3, ONE, SB(i) + 1])
            m.cone(CT.PEXP, [SB(i) + 4, ONE, SB(i) + 2])
        for i in range(NS):
            m.cj(SB(i), 1.0 / NS)
        for j in range(NF):
            m.cj(GJ(j), LAM * SIG[j])
        r = m.solve()
    if r.rc != 0:
        return False, None, 0.0
    beta = [r.x[0]] + [r.x[BJ(j)] for j in range(NF)]
    return True, beta, objective(beta)


def admm_solve():
    """2. ADMM (the repo's method): returns (beta, iterations, converged)."""
    rho, abs_tol, rel_tol, max_iter = 1.0, 1e-6, 1e-4, 5000
    beta, z, u = [0.0] * NL, [0.0] * NF, [0.0] * NF
    conv, it = False, 0
    while it < max_iter:
        zold = z[:]
        target = [z[j] - u[j] for j in range(NF)]   # beta-step: Newton with backtracking
        for _ in range(20):
            g = [0.0] * NL
            h = [[0.0] * NL for _ in range(NL)]
            for i in range(NS):
                eta = beta[0]
                xi = [1.0] + X[i]
                for j in range(NF):
                    eta += X[i][j] * beta[1 + j]
                p = sigmoid(Y[i] * eta)             # P(correct)
                gi = -Y[i] * (1.0 - p) / NS         # d loss / d eta
                hi = p * (1.0 - p) / NS
                for a in range(NL):
                    g[a] += gi * xi[a]
                    for b in range(NL):
                        h[a][b] += hi * xi[a] * xi[b]
            for j in range(NF):                     # penalty gradient/Hessian
                g[1 + j] += rho * (beta[1 + j] - target[j])
                h[1 + j][1 + j] += rho
            aug = [h[a][:] + [-g[a]] for a in range(NL)]   # solve H d = -g (Gauss, pivoting)
            d = [0.0] * NL
            for c in range(NL):
                piv = c
                for r in range(c + 1, NL):
                    if abs(aug[r][c]) > abs(aug[piv][c]):
                        piv = r
                aug[c], aug[piv] = aug[piv], aug[c]
                if abs(aug[c][c]) < 1e-14:
                    d[c] = 0.0
                    continue
                for r in range(c + 1, NL):
                    f = aug[r][c] / aug[c][c]
                    for b in range(c, NL + 1):
                        aug[r][b] -= f * aug[c][b]
            for c in range(NL - 1, -1, -1):
                s = aug[c][NL]
                for b in range(c + 1, NL):
                    s -= aug[c][b] * d[b]
                d[c] = 0.0 if abs(aug[c][c]) < 1e-14 else s / aug[c][c]
            dn = 0.0
            for a in range(NL):
                dn += d[a] * d[a]
            if dn < 1e-18:
                break

            def penalized(bv):
                f = loss(bv)
                for j in range(NF):
                    r = bv[1 + j] - target[j]
                    f += 0.5 * rho * r * r
                return f

            step, f0 = 1.0, penalized(beta)         # backtracking on the objective
            for _ls in range(30):
                bn = [beta[a] + step * d[a] for a in range(NL)]
                if penalized(bn) <= f0 - 1e-12 * step * step:
                    beta = bn
                    break
                step *= 0.5
            if step < 1e-9:
                break
        w = beta[1:]
        # z-step: soft-threshold, shifted so that sum(z) = 0 (monotone root by bisection)
        thr = [LAM * SIG[j] / rho for j in range(NF)]
        v = [w[j] + u[j] for j in range(NF)]
        lo = hi = v[0]
        for j in range(NF):
            lo = min(lo, v[j] - thr[j], v[j] + thr[j])
            hi = max(hi, v[j] + thr[j], v[j] - thr[j])
        c = 0.0
        for _ in range(80):
            c = 0.5 * (lo + hi)
            s = 0.0
            for j in range(NF):
                zz = v[j] - c
                if zz > thr[j]:
                    s += zz - thr[j]
                elif zz < -thr[j]:
                    s += zz + thr[j]
            if s > 0.0:
                lo = c
            else:
                hi = c
        for j in range(NF):
            zz = v[j] - c
            z[j] = zz - thr[j] if zz > thr[j] else (zz + thr[j] if zz < -thr[j] else 0.0)
        for j in range(NF):                         # dual update
            u[j] += w[j] - z[j]
        rn = sn = wn = zn = un = 0.0                # convergence
        for j in range(NF):
            rn += (w[j] - z[j]) * (w[j] - z[j])
            sn += (z[j] - zold[j]) * (z[j] - zold[j])
            wn += w[j] * w[j]
            zn += z[j] * z[j]
            un += u[j] * u[j]
        rn, sn = math.sqrt(rn), rho * math.sqrt(sn)
        eps_pri = math.sqrt(NF) * abs_tol + rel_tol * max(math.sqrt(wn), math.sqrt(zn))
        eps_dual = math.sqrt(NF) * abs_tol + rel_tol * rho * math.sqrt(un)
        it += 1
        if rn < eps_pri and sn < eps_dual:
            conv = True
            break
    return [beta[0]] + z, it, conv


def brute_force():
    """Dense brute force over (intercept, b_1, b_2) with b_3 = -b_1 - b_2."""
    span, ng = 6.0, 120
    best, bb = 1e300, [0.0] * NL
    grid = [-span + 2 * span * k / ng for k in range(ng + 1)]
    for b0 in grid:
        for t1 in grid:
            for t2 in grid:
                b = [b0, t1, t2, -t1 - t2]
                f = objective(b)
                if f < best:
                    best, bb = f, b
    return best, bb


def main() -> int:
    ok_c, bc, oc = conic_solve()
    if not ok_c:
        print("logcontrast_lasso  conic rc!=OK FAIL")
        return 1
    ba, iters, conv = admm_solve()
    oa = objective(ba)
    ob, bb = brute_force()
    sumc = suma = dbmax = 0.0
    for j in range(NF):
        sumc += bc[1 + j]
        suma += ba[1 + j]
        dbmax = max(dbmax, abs(ba[1 + j] - bc[1 + j]))
    ok_conic = abs(oc - ob) < 1e-3 * (1.0 + abs(ob))
    ok_admm = conv and abs(oa - oc) < 1e-4 * (1.0 + abs(oc)) and dbmax < 1e-3
    ok_sum = abs(sumc) < 1e-6 and abs(suma) < 1e-6
    ok_sparse = abs(bc[1]) < 1e-6 and abs(ba[1]) < 1e-6   # feature 0 penalised to 0
    ok = ok_conic and ok_admm and ok_sum and ok_sparse
    yes = lambda f: "OK" if f else "FAIL"           # noqa: E731
    print(f"logcontrast_lasso  n={NS} p={NF} lambda={LAM:.3f}")
    print(f"  conic : obj={oc:.8f}  beta=({bc[0]:.5f}, {bc[1]:.5f}, {bc[2]:.5f}, {bc[3]:.5f})  sum={sumc:.2e}")
    print(f"  admm  : obj={oa:.8f}  beta=({ba[0]:.5f}, {ba[1]:.5f}, {ba[2]:.5f}, {ba[3]:.5f})  sum={suma:.2e}  "
          f"iters={iters} {'converged' if conv else 'NOT-converged'}")
    print(f"  brute : obj={ob:.8f}  beta=({bb[0]:.5f}, {bb[1]:.5f}, {bb[2]:.5f}, {bb[3]:.5f})")
    print(f"  [conic vs brute {yes(ok_conic)}] [admm vs conic: obj {yes(ok_admm)}, max|db| {dbmax:.2e}] "
          f"[zero-sum {yes(ok_sum)}] [sparse {yes(ok_sparse)}]")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
