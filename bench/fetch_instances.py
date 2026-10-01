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

"""Fetch the public benchmark instances of bench_all.py's library table into
bench/instances/lib/ (gitignored) and write its index.csv.

A few small instances from each set:
  - CBLIB (cblib.zib.de), CBF: the families that combine PSD, exponential and
    power cones with integer variables, plus infeasible SDPs;
  - SDPLIB (github.com/vsdp/SDPLIB), SDPA sparse, converted to CBF;
  - Maros-Meszaros convex QPs (github.com/qpsolvers/maros_meszaros_qpbenchmark),
    .mat, converted to QPS in .mps files (the constant term r is dropped);
  - Netlib LP (bench/fetch_netlib.py);
  - MIPLIB 2017 (miplib.zib.de), MPS.

index.csv: name,set,class,path,ref -- ref is the published optimal value where
the set publishes one (SDPLIB README, Netlib readme, MIPLIB .solu),
"infeasible" for a certified infeasible instance, empty otherwise (the
solvers are then cross-checked against each other).

Usage:  python3 bench/fetch_instances.py
Needs numpy/scipy (the .mat reader).
"""
import csv
import gzip
import io
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "instances", "lib")

CBLIB = "https://cblib.zib.de/download/all/%s.cbf.gz"
SDPLIB = "https://raw.githubusercontent.com/vsdp/SDPLIB/master/data/%s.dat-s"
MAROS = "https://raw.githubusercontent.com/qpsolvers/maros_meszaros_qpbenchmark/main/data/%s.mat"
MIPLIB = "https://miplib.zib.de/WebData/instances/%s.mps.gz"

# (name, class): class names the combination the instance exercises
CBLIB_SET = [
    ("port_12_9_3_a_1", "soc+exp+psd+int"), ("port_12_9_3_b_1", "soc+exp+psd+int"),
    ("port_16_12_4_b_1", "soc+exp+psd+int"),
    ("expdesign_D_8_4", "exp+psd+int"), ("expdesign_D_12_6", "exp+psd+int"),
    ("expdesign_D_16_8", "exp+psd+int"),
    ("kpart_diw.15.4.29", "psd+int"), ("rt_2x4_3bars", "psd+int"),
    ("clsq_random_32_2_a", "psd+int"), ("expdesign_A_8_4", "psd+int"),
    ("expdesign_E_8_4", "psd+int"),
    ("syn05m", "exp+int"), ("syn10m", "exp+int"), ("rsyn0805m", "exp+int"),
    ("batchdes", "exp+int"), ("ex1223a", "exp+int"), ("synthes2", "exp+int"),
    ("beck751", "exp"), ("demb761", "exp"), ("rijc781", "exp"), ("gp_dave_1", "exp"),
    ("LogExpCR-n20-m400", "exp"),
    ("HMCR-n20-m400", "pow"), ("HMCR-n20-m800", "pow"),
    ("infeas_clean_10_10_1", "sdp-infeas"), ("infeas_clean_10_10_3", "sdp-infeas"),
    ("weak_clean_10_10_3", "sdp-infeas"),
]
# the Pataki instances are infeasible by construction (weak_*: weakly so)
CBLIB_REF = {n: "infeasible" for n, c in CBLIB_SET if c == "sdp-infeas"}
# Not published: SCIP-SDP's optima, whose points were checked against the CBF
# data here (exp cones within 5e-6, PSD and integrality exact); Pajarito
# reports OPTIMAL at worse values (0.110795, 3.38664) on both.
CBLIB_REF.update({"expdesign_D_12_6": "-0.5202525", "expdesign_D_16_8": "2.931324"})

# SDPLIB optimal values from its README, as printed there: their digits are
# the precision bench_all checks them to
SDPLIB_SET = {"control1": "1.778463e+01", "hinf1": "2.0326e+00", "theta1": "2.300000e+01",
              "truss1": "-8.999996e+00", "qap5": "-4.360e+02", "mcp100": "2.261574e+02"}

MAROS_SET = ["HS21", "HS118", "QAFIRO", "DUAL1", "PRIMAL1", "CVXQP1_S"]

# Netlib optimal values from https://www.netlib.org/lp/data/readme
NETLIB_SET = {"afiro": -464.75314286, "adlittle": 225494.96316, "blend": -30.812149846,
              "sc50a": -64.575077059, "share2b": -415.73224074, "stocfor1": -41131.976219}

# MIPLIB 2017 optimal values from miplib2017-v31.solu
MIPLIB_SET = {"flugpl": 1201500.0, "p0201": 7615.0, "gt2": 21166.0, "pk1": 11.0}


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def sdpa_to_cbf(text):
    """SDPA sparse  min c'x  s.t.  sum_i F_i x_i - F_0 in S_+  as CBF: one PSDCON
    per matrix block (H = F_i, D = -F_0), one L+ row per diagonal-block entry."""
    toks = [ln.split('"')[0].split("*")[0] for ln in text.splitlines()]
    toks = [ln for ln in toks if ln.strip()]
    it = iter(toks)
    m = int(next(it).split()[0])
    nb = int(next(it).split()[0])
    blocks = [int(float(v)) for v in next(it).replace(",", " ").replace("{", " ")
              .replace("}", " ").replace("(", " ").replace(")", " ").split()[:nb]]
    c = [float(v) for v in next(it).replace(",", " ").replace("{", " ").replace("}", " ").split()[:m]]
    psd = {b: k for k, b in enumerate(i for i in range(nb) if blocks[i] > 0)}
    lin_off, off = {}, 0
    for b in range(nb):
        if blocks[b] < 0:
            lin_off[b] = off
            off += -blocks[b]
    nlin = off
    h, d, a, bc = [], [], [], []
    for ln in it:
        f = ln.split()
        if len(f) < 5:
            continue
        mat, blk, i, j, v = int(f[0]), int(f[1]) - 1, int(f[2]) - 1, int(f[3]) - 1, float(f[4])
        if blocks[blk] > 0:
            p, q = max(i, j), min(i, j)          # CBF: lower triangle
            if mat == 0:
                d.append("%d %d %d %.17g" % (psd[blk], p, q, -v))
            else:
                h.append("%d %d %d %d %.17g" % (psd[blk], mat - 1, p, q, v))
        else:
            r = lin_off[blk] + i
            if mat == 0:
                bc.append("%d %.17g" % (r, -v))
            else:
                a.append("%d %d %.17g" % (r, mat - 1, v))
    L = ["VER", "3", "OBJSENSE", "MIN", "VAR", "%d 1" % m, "F %d" % m]
    if nlin:
        L += ["CON", "%d 1" % nlin, "L+ %d" % nlin]
    L += ["PSDCON", str(len(psd))] + [str(blocks[b]) for b in sorted(psd, key=psd.get)]
    obj = ["%d %.17g" % (i, v) for i, v in enumerate(c) if v != 0.0]
    L += ["OBJACOORD", str(len(obj))] + obj
    for key, rows in (("ACOORD", a), ("BCOORD", bc), ("HCOORD", h), ("DCOORD", d)):
        if rows:
            L += [key, str(len(rows))] + rows
    return "\n".join(L) + "\n"


def mat_to_qps(raw, name):
    """A Maros-Meszaros .mat (qpbenchmark's sif2mat layout: A = [C; I], l/u for
    both, 1e20 as infinity) as a QPS file: min 0.5 x'Px + q'x, ranged rows,
    box bounds, QUADOBJ lower triangle."""
    import numpy as np
    import scipy.io as spio
    from scipy import sparse
    md = spio.loadmat(io.BytesIO(raw))
    P = sparse.csc_matrix(md["P"].astype(float))
    q = md["q"].T.flatten().astype(float)
    A = sparse.csr_matrix(md["A"].astype(float))
    lo = md["l"].T.flatten().astype(float)
    up = md["u"].T.flatten().astype(float)
    n = int(md["n"].T.flatten()[0])
    inf = 9e19
    C, lc, uc, lb, ub = A[:-n], lo[:-n], up[:-n], lo[-n:], up[-n:]
    rows, rng = [], {}
    for i in range(C.shape[0]):
        l, u = lc[i], uc[i]
        if l > -inf and u < inf:
            if l == u:
                rows.append(("E", l))
            else:
                rows.append(("L", u))
                rng[i] = u - l
        elif u < inf:
            rows.append(("L", u))
        elif l > -inf:
            rows.append(("G", l))
        else:
            rows.append(("N", 0.0))
    L = ["NAME " + name, "ROWS", " N obj"]
    L += [" %s r%d" % (t, i) for i, (t, _) in enumerate(rows)]
    L.append("COLUMNS")
    Cc = C.tocsc()
    for j in range(n):
        # every column is declared here, even one that only the quadratic
        # term uses: a name that first appears in BOUNDS is not a column
        if q[j] != 0.0 or Cc.indptr[j] == Cc.indptr[j + 1]:
            L.append("    x%d obj %.17g" % (j, q[j]))
        for k in range(Cc.indptr[j], Cc.indptr[j + 1]):
            if Cc.data[k] != 0.0:
                L.append("    x%d r%d %.17g" % (j, Cc.indices[k], Cc.data[k]))
    L.append("RHS")
    L += ["    rhs r%d %.17g" % (i, v) for i, (t, v) in enumerate(rows) if t != "N" and v != 0.0]
    if rng:
        L.append("RANGES")
        L += ["    rng r%d %.17g" % (i, v) for i, v in sorted(rng.items())]
    L.append("BOUNDS")
    for j in range(n):
        l, u = lb[j], ub[j]
        if l <= -inf and u >= inf:
            L.append(" FR bnd x%d" % j)
        elif l == u:
            L.append(" FX bnd x%d %.17g" % (j, l))
        else:
            L.append(" MI bnd x%d" % j if l <= -inf else " LO bnd x%d %.17g" % (j, l))
            if u < inf:
                L.append(" UP bnd x%d %.17g" % (j, u))
    L.append("QUADOBJ")
    Pl = sparse.tril(P).tocoo()
    for i, j, v in sorted(zip(Pl.row, Pl.col, Pl.data), key=lambda t: (t[1], t[0])):
        if v != 0.0:
            L.append("    x%d x%d %.17g" % (j, i, v))
    L.append("ENDATA")
    return "\n".join(L) + "\n"


def main():
    os.makedirs(DEST, exist_ok=True)
    index = []

    def save(name, data, ext):
        path = os.path.join(DEST, name + ext)
        with open(path, "wb" if isinstance(data, bytes) else "w") as fh:
            fh.write(data)
        print("  %-24s -> %s" % (name, os.path.relpath(path, HERE)))
        return os.path.relpath(path, HERE)

    for name, cls in CBLIB_SET:
        p = save(name, gzip.decompress(get(CBLIB % name)), ".cbf")
        index.append((name, "cblib", cls, p, CBLIB_REF.get(name, "")))
    for name, ref in SDPLIB_SET.items():
        p = save(name, sdpa_to_cbf(get(SDPLIB % name).decode()), ".cbf")
        index.append((name, "sdplib", "sdp", p, ref))
    for name in MAROS_SET:
        p = save(name, mat_to_qps(get(MAROS % name), name), ".mps")  # QPS; HiGHS goes by extension
        index.append((name, "maros-meszaros", "qp", p, ""))
    netlib_dir = os.path.join(HERE, "instances")
    if not all(os.path.exists(os.path.join(netlib_dir, n + ".mps")) for n in NETLIB_SET):
        import fetch_netlib
        fetch_netlib.main()
    for name, ref in NETLIB_SET.items():
        index.append((name, "netlib", "lp", os.path.join("instances", name + ".mps"), repr(ref)))
    for name, ref in MIPLIB_SET.items():
        p = save(name, gzip.decompress(get(MIPLIB % name)), ".mps")
        index.append((name, "miplib", "milp", p, repr(ref)))

    with open(os.path.join(DEST, "index.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "set", "class", "path", "ref"])
        w.writerows(index)
    print("%d instances; index in %s" % (len(index), os.path.relpath(os.path.join(DEST, "index.csv"), HERE)))


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.exit(main())
