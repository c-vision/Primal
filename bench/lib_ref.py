#!/usr/bin/env python3
#
# PrimalSolver - a convex optimization solver in C99 (LP/QP/SOCP/SDP/exp-power/MIP).
# Copyright 2026 Gaetano Minardi
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  A copy of the License
# is in the repository root (LICENSE) and at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations
# under the License.

"""The reference solvers on the library instances (bench/fetch_instances.py):
each takes a file and returns (status, seconds, objective), status one of
'ok', 'infeasible' (certified), 'timeout', 'n/a' (not solved) or
'unsupported' (the solver has no such class).

  - CBF: Clarabel and SCS through a conic conversion (continuous models only;
    PSD variables become svec variables in each solver's triangle order);
    SCIP (SCIP-SDP for PSD models) on the model built through its C API,
    bench/scip_cbf.c.
  - MPS/QPS: HiGHS reads the file; its matrices feed Clarabel, SCS and, for a
    QP, SCIP on the factored objective (see bench_all.solve_scip_qp); SCIP
    reads an LP/MILP file itself.

Times cover setup and solve (Clarabel/SCS construction, SCIP's factorization),
not reading or model building.
"""
import math
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy import sparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cbfio
import conic_ref
import run_bench

SQ2 = math.sqrt(2.0)


# ---------------------------------------------------------------- conic form
class Conic:
    """min q'z + const  s.t.  A z + s = b,  s in K  (K in SCS order:
    zero, nonnegative, second-order, PSD, exponential, power)."""

    def __init__(self, nz):
        self.nz = nz
        self.q = np.zeros(nz)
        self.P = None
        self.const = 0.0
        self.sign = 1.0                       # -1 when the model maximizes
        self.blocks = {"z": [], "l": [], "q": [], "s": [], "ep": [], "p": []}

    def assemble(self):
        """(A, b, cone list) with cones as (kind, size-or-param)."""
        rows, cones = [], []
        for kind in ("z", "l"):
            if self.blocks[kind]:
                rows += self.blocks[kind]
                cones.append((kind, len(self.blocks[kind])))
        for kind in ("q", "s", "ep", "p"):
            for par, rs in self.blocks[kind]:
                rows += rs
                cones.append((kind, par))
        A = sparse.lil_matrix((len(rows), self.nz))
        b = np.zeros(len(rows))
        for i, (coef, c0) in enumerate(rows):
            for j, v in coef.items():
                A[i, j] = -v                  # s = u = coef.z + c0
            b[i] = c0
        return A.tocsc(), b, cones


def _add(u, v, a=1.0):
    out = dict(u[0])
    for j, x in v[0].items():
        out[j] = out.get(j, 0.0) + a * x
    return out, u[1] + a * v[1]


def _scale(u, a):
    return {j: a * x for j, x in u[0].items()}, a * u[1]


def _put_cone(cf, name, mem, pows):
    """Append the CBF cone `name` on the member expressions mem."""
    if name == "F":
        return
    if name == "L+":
        cf.blocks["l"] += mem
    elif name == "L-":
        cf.blocks["l"] += [_scale(u, -1.0) for u in mem]
    elif name == "L=":
        cf.blocks["z"] += mem
    elif name == "Q":
        cf.blocks["q"].append((len(mem), mem))
    elif name == "QR":                        # ||(u0-u1, sqrt2 w)|| <= u0+u1
        rs = [_add(mem[0], mem[1]), _add(mem[0], mem[1], -1.0)] + [_scale(u, SQ2) for u in mem[2:]]
        cf.blocks["q"].append((len(rs), rs))
    elif name == "EXP":                       # CBF u0 >= u1 exp(u2/u1); SCS/Clarabel (x,y,z)
        cf.blocks["ep"].append((None, [mem[2], mem[1], mem[0]]))
    elif name.startswith("@") and name.endswith(":POW") and len(mem) == 3:
        al = pows[int(name[1:name.index(":")])]
        if len(al) != 2:
            raise cbfio.Unsupported("POW alphas")
        cf.blocks["p"].append((al[0] / (al[0] + al[1]), mem))
    else:
        raise cbfio.Unsupported("cone %s" % name)


def svec_pairs(n, order):
    """(p, q), p >= q, in the solver's triangle order: Clarabel stacks the
    upper triangle by columns (= the lower one by rows), SCS the lower
    triangle by columns."""
    if order == "clarabel":
        return [(p, q) for p in range(n) for q in range(p + 1)]
    return [(p, q) for q in range(n) for p in range(q, n)]


def cbf_conic(path, order):
    d = cbfio.read_cbf(path)
    if d["ints"]:
        raise cbfio.Unsupported("integer variables")
    nv, vgroups = d["var"]
    # svec variables of the PSD variables, after the scalar ones
    pos, nz = [], nv
    for n in d["psdvar"]:
        pos.append({pq: nz + k for k, pq in enumerate(svec_pairs(n, order))})
        nz += n * (n + 1) // 2
    cf = Conic(nz)

    def fterm(coef, k, p, q, v):
        p, q = max(p, q), min(p, q)
        j = pos[k][(p, q)]
        coef[j] = coef.get(j, 0.0) + (v if p == q else SQ2 * v)

    for j, v in d["obja"].items():
        cf.q[j] += v
    obj = {}
    for k, p, q, v in d["objf"]:
        fterm(obj, k, p, q, v)
    for j, v in obj.items():
        cf.q[j] += v
    cf.const = d["objb"]
    if d["sense"] == "MAX":
        cf.sign = -1.0
        cf.q, cf.const = -cf.q, -cf.const

    start = 0
    for name, size in vgroups:
        _put_cone(cf, name, [({j: 1.0}, 0.0) for j in range(start, start + size)], d["pows"])
        start += size
    ncon, cgroups = d["con"]
    rows = [({}, d["b"].get(r, 0.0)) for r in range(ncon)]
    for r, j, v in d["a"]:
        rows[r][0][j] = rows[r][0].get(j, 0.0) + v
    for r, k, p, q, v in d["f"]:
        fterm(rows[r][0], k, p, q, v)
    start = 0
    for name, size in cgroups:
        _put_cone(cf, name, rows[start:start + size], d["pows"])
        start += size
    for k, n in enumerate(d["psdvar"]):       # X_k in S_+
        cf.blocks["s"].append((n, [({pos[k][pq]: 1.0}, 0.0) for pq in svec_pairs(n, order)]))
    for c, n in enumerate(d["psdcon"]):       # svec(sum_j H_j x_j + D) in S_+
        ent = {pq: ({}, 0.0) for pq in svec_pairs(n, order)}
        for cc, j, p, q, v in d["h"]:
            if cc == c:
                pq = (max(p, q), min(p, q))
                ent[pq][0][j] = ent[pq][0].get(j, 0.0) + v
        for cc, p, q, v in d["d"]:
            if cc == c:
                pq = (max(p, q), min(p, q))
                ent[pq] = (ent[pq][0], ent[pq][1] + v)
        cf.blocks["s"].append((n, [ent[pq] if pq[0] == pq[1] else _scale(ent[pq], SQ2)
                                   for pq in svec_pairs(n, order)]))
    return cf


# ------------------------------------------------------------ MPS/QPS via HiGHS
def highs_model(path):
    """(c, A csr, row lo, row up, col lo, col up, Q full or None, integer?, sense, offset)."""
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(path)
    model = h.getModel()
    lp = model.lp_
    n, m = lp.num_col_, lp.num_row_
    a = lp.a_matrix_
    A = sparse.csc_matrix((a.value_, a.index_, a.start_), shape=(m, n)).tocsr()
    Q = None
    hs = model.hessian_
    if hs.dim_ > 0 and len(hs.value_):
        T = sparse.csc_matrix((hs.value_, hs.index_, hs.start_), shape=(n, n))
        Q = (T + T.T - sparse.diags(T.diagonal())).tocsc()   # lower triangle -> full
    integ = any(int(v) != 0 for v in lp.integrality_) if len(lp.integrality_) else False
    sense = -1.0 if str(lp.sense_).endswith("kMaximize") else 1.0
    return (np.array(lp.col_cost_), A, np.array(lp.row_lower_), np.array(lp.row_upper_),
            np.array(lp.col_lower_), np.array(lp.col_upper_), Q, integ, sense, lp.offset_)


def mps_conic(path):
    c, A, rl, ru, lo, up, Q, integ, sense, offset = highs_model(path)
    if integ:
        raise cbfio.Unsupported("integer variables")
    n = len(c)
    cf = Conic(n)
    cf.q = sense * c
    cf.const = sense * offset
    cf.sign = sense
    if Q is not None:
        cf.P = sparse.triu(sense * Q).tocsc()
    big = 1e20
    for i in range(A.shape[0]):
        row = A.getrow(i)
        coef = dict(zip(row.indices, row.data))
        if rl[i] == ru[i]:
            cf.blocks["z"].append((_scale((coef, 0.0), -1.0)[0], ru[i]))
            continue
        if ru[i] < big:                       # s = ru - a x
            cf.blocks["l"].append((_scale((coef, 0.0), -1.0)[0], ru[i]))
        if rl[i] > -big:                      # s = a x - rl
            cf.blocks["l"].append((coef, -rl[i]))
    for j in range(n):
        if lo[j] == up[j]:
            cf.blocks["z"].append(({j: -1.0}, up[j]))
            continue
        if up[j] < big:
            cf.blocks["l"].append(({j: -1.0}, up[j]))
        if lo[j] > -big:
            cf.blocks["l"].append(({j: 1.0}, -lo[j]))
    return cf


# ---------------------------------------------------------------- solvers
def _conic(path, order):
    return cbf_conic(path, order) if path.endswith(".cbf") else mps_conic(path)


def solve_clarabel(path, timeout):
    try:
        import clarabel
    except Exception:
        return "unsupported", None, None
    try:
        cf = _conic(path, "clarabel")
    except cbfio.Unsupported:
        return "unsupported", None, None
    A, b, cones = cf.assemble()
    K = []
    for kind, par in cones:
        K.append({"z": lambda p: clarabel.ZeroConeT(p), "l": lambda p: clarabel.NonnegativeConeT(p),
                  "q": lambda p: clarabel.SecondOrderConeT(p), "s": lambda p: clarabel.PSDTriangleConeT(p),
                  "ep": lambda p: clarabel.ExponentialConeT(), "p": lambda p: clarabel.PowerConeT(p)}[kind](par))
    P = cf.P if cf.P is not None else sparse.csc_matrix((cf.nz, cf.nz))
    st = conic_ref.clarabel_settings(time_limit=timeout)
    t = time.perf_counter()  # setup counts: scaling, KKT solver setup
    solver = clarabel.DefaultSolver(P, cf.q, A, b, K, st)
    sol = solver.solve()
    dt = time.perf_counter() - t
    # the Almost* statuses are Clarabel's reduced-accuracy verdicts: not an answer
    s = str(sol.status).split(".")[-1]
    if s == "PrimalInfeasible":
        return "infeasible", dt, None
    if s == "MaxTime":
        return "timeout", None, None
    if s != "Solved":
        return "n/a", None, None
    return "ok", dt, cf.sign * (sol.obj_val + cf.const)


def solve_scs(path, timeout):
    try:
        import scs
    except Exception:
        return "unsupported", None, None
    try:
        cf = _conic(path, "scs")
    except cbfio.Unsupported:
        return "unsupported", None, None
    A, b, cones = cf.assemble()
    K = {}
    for kind, par in cones:
        if kind in ("z", "l"):
            K[kind] = par
        elif kind == "ep":
            K["ep"] = K.get("ep", 0) + 1
        else:
            K.setdefault(kind, []).append(par)
    data = dict(A=A, b=b, c=cf.q)
    if cf.P is not None:
        data["P"] = cf.P
    t = time.perf_counter()  # setup counts: scaling, KKT solver setup
    solver = scs.SCS(data, K, time_limit_secs=timeout, **conic_ref.SCS_SETTINGS)
    sol = solver.solve()
    dt = time.perf_counter() - t
    s = sol["info"]["status"]
    if s == "infeasible":
        return "infeasible", dt, None
    if s != "solved":
        return "timeout" if dt >= timeout else "n/a", None, None
    return "ok", dt, cf.sign * (sol["info"]["pobj"] + cf.const)


def solve_scip_mps(path, cls, timeout):
    """SCIP on an MPS/QPS file: the file itself for LP/MILP, the factored
    objective for a QP (its time includes the factorization)."""
    if cls != "qp":
        st, dt, obj = run_bench.solve_scip(path, timeout)
        return st, dt, obj
    from pyscipopt import Model, quicksum
    c, A, rl, ru, lo, up, Q, integ, sense, offset = highs_model(path)
    n = len(c)
    t = time.perf_counter()
    lam, V = np.linalg.eigh(Q.toarray())
    keep = lam > 1e-12 * max(1.0, lam.max())
    F = (V[:, keep] * np.sqrt(lam[keep])).T  # 0.5 x'Qx = 0.5 ||F x||^2
    t_fact = time.perf_counter() - t
    m = Model()
    m.hideOutput()
    m.setParam("limits/time", timeout)
    big = 1e20
    x = [m.addVar(lb=None if lo[j] <= -big else lo[j], ub=None if up[j] >= big else up[j])
         for j in range(n)]
    for i in range(A.shape[0]):
        row = A.getrow(i)
        e = quicksum(v * x[j] for j, v in zip(row.indices, row.data))
        if rl[i] == ru[i]:
            m.addCons(e == ru[i])
        else:
            if ru[i] < big:
                m.addCons(e <= ru[i])
            if rl[i] > -big:
                m.addCons(e >= rl[i])
    y = [m.addVar(lb=None) for _ in range(F.shape[0])]
    for k in range(F.shape[0]):
        m.addCons(y[k] == quicksum(F[k, j] * x[j] for j in range(n) if F[k, j] != 0.0))
    z = m.addVar(lb=None)
    m.addCons(0.5 * quicksum(v * v for v in y) <= z)
    m.setObjective(quicksum(c[j] * x[j] for j in range(n) if c[j]) + z)
    t = time.perf_counter()
    m.optimize()
    dt = t_fact + time.perf_counter() - t
    if m.getStatus() == "timelimit":
        return "timeout", None, None
    if m.getStatus() == "infeasible":
        return "infeasible", dt, None
    if m.getStatus() != "optimal":
        return "n/a", None, None
    return "ok", dt, m.getObjVal() + offset


def solve_scip_cbf(path, timeout):
    """SCIP on a CBF model built through its C API (out/bench/scip_cbf, `make
    bench-scip`): SCIP-SDP when the model has PSD parts, SCIP otherwise.  The
    time is SCIP's solving time, model building excluded."""
    exe = os.path.join(os.path.dirname(HERE), "out", "bench", "scip_cbf")
    if not os.path.exists(exe):
        return "unsupported", None, None
    try:
        p = subprocess.run([exe, path, str(timeout)], capture_output=True, text=True,
                           timeout=timeout + 120)
    except subprocess.TimeoutExpired:
        return "timeout", None, None
    # DSDP prints progress on stdout: the result is the last line
    last = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "n/a,,"
    st, sec, obj = (last.split(",") + ["", "", ""])[:3]
    if st not in ("ok", "infeasible", "timeout", "unsupported"):
        return "n/a", None, None
    return st, float(sec) if sec else None, float(obj) if obj else None


def solve_scip(path, cls, timeout):
    if path.endswith(".cbf"):
        return solve_scip_cbf(path, timeout)
    return solve_scip_mps(path, cls, timeout)


def solve_highs(path, cls, timeout):
    if path.endswith(".cbf"):
        return "unsupported", None, None
    return run_bench.solve_highs(path, timeout)


def julia_exe():
    """$JULIA, else julia on PATH, else juliaup's default install."""
    for exe in (os.environ.get("JULIA"), shutil.which("julia"),
                os.path.expanduser("~/.juliaup/bin/julia")):
        if exe and os.path.exists(exe):
            return exe
    return None


def solve_julia_batch(paths, timeout):
    """Pajarito (integer models) / Hypatia (continuous) on every file, in one
    Julia process (bench/julia/ref.jl): {path: (status, seconds, objective)}.
    Julia's compile time is not charged (ref.jl reports a warm second run)."""
    exe = julia_exe()
    if not exe or not paths:
        return {p: ("unsupported", None, None) for p in paths}
    import json
    jl = os.path.join(HERE, "julia")
    p = subprocess.run([exe, "--project=" + jl, os.path.join(jl, "ref.jl")],
                       input="".join("%s\t%g\n" % (q, timeout) for q in paths),
                       capture_output=True, text=True, timeout=len(paths) * (2 * timeout + 60) + 600)
    out = {}
    for ln in p.stdout.splitlines():
        if not ln.startswith("{"):            # Pajarito prints progress to stdout
            continue
        r = json.loads(ln)
        st = r["status"]
        out[r["path"]] = ("timeout" if st == "timeout" else st,
                          r["seconds"] if st in ("ok", "infeasible") else None, r["obj"])
    return {q: out.get(q, ("n/a", None, None)) for q in paths}

