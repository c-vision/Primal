#!/usr/bin/env python3
"""The SOC / exp / mixed measurement matrix: one row per cell, medians.

bench/pexp_matrix.c builds the instance; this script runs it and reads the
GMB_DBG trace for what the public API does not expose — the KKT order, the
structural nonzeros, whether the route is dense or sparse, and the iteration
count. Those four numbers are what tells a *cost* from a *trajectory* problem,
and they are the reason this runner exists.

    python3 bench/pexp_matrix.py                 # the three experiments
    python3 bench/pexp_matrix.py --runs 9
    python3 bench/pexp_matrix.py --only family

Experiments (one knob at a time):

  family    soc | exp | mix   over the number of blocks, at d=8, rho=0.5
  density   soc | exp | mix   over the link density rho, at n=40, d=12
  degen     exp | mix         over rep (identical rows and cones), at n=40

Columns: blocks = cone blocks, rows = linking rows, KKT n = augmented order,
nnz = structural nonzeros of the sparse form, route = dense/sparse, it =
iterations, ms = median wall time of the solve, ms/it = the two divided.
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "out" / "bench" / "pexp_matrix"

KKT = re.compile(r"\[kkt\] pass=0 n=(\d+) reuse=0(.*)")
ITER = re.compile(r"^\s*(?:socp(?:_sp)? )?it=(\d+)")
ROUTE = re.compile(r"\[route\] status=(\d)")
RETRY = re.compile(r"\[retry\]")


def run(fam: str, n: int, d: int, rho: float, rep: int):
    env = dict(os.environ, GMB_DBG="1")
    p = subprocess.run([str(BIN), fam, str(n), str(d), f"{rho:.2f}", str(rep)],
                       capture_output=True, text=True, env=env, timeout=600)
    out = [l for l in p.stdout.splitlines() if l.startswith("matrix,")]
    if not out:
        raise SystemExit(f"{fam} {n} {d} {rho} {rep}: no matrix line\n{p.stderr[-800:]}")
    f = out[-1].split(",")
    secs, pobj, dobj = float(f[8]), float(f[9]), float(f[10])
    pinf, dinf = float(f[11]), float(f[12])
    relgap = abs(pobj - dobj) / (1.0 + abs(pobj) + abs(dobj))
    err = p.stderr
    order = nnz = None
    sparse = None
    for line in err.splitlines():
        m = KKT.search(line)
        if m:
            order = int(m.group(1))
            rest = m.group(2)
            sparse = "sparse" in rest
            mm = re.search(r"nnz=(\d+)", rest)
            nnz = int(mm.group(1)) if mm else None
            break
    its = 0
    for line in err.splitlines():
        m = ITER.match(line)
        if m:
            its = max(its, int(m.group(1)) + 1)
    route = None
    m = ROUTE.search(err)
    if m:
        route = "native" if m.group(1) == "0" else "cuts"
    if "socp it=" in err or "socp_sp it=" in err:
        route = "socp"
    retries = len(RETRY.findall(err))
    return dict(secs=secs, pobj=pobj, pinf=pinf, dinf=dinf, relgap=relgap, order=order,
                nnz=nnz, sparse=sparse, its=its, route=route, retries=retries)


def cell(fam, n, d, rho, rep, runs):
    vals = [run(fam, n, d, rho, rep) for _ in range(runs)]
    med = {k: statistics.median([v[k] for v in vals])
           for k in ("secs", "pobj", "pinf", "dinf", "relgap")}
    first = vals[0]
    med.update(order=first["order"], nnz=first["nnz"], sparse=first["sparse"],
               its=first["its"], route=first["route"], retries=first["retries"])
    return med


def table(title, cases, runs):
    print(f"\n### {title}\n")
    print("| case | blocks | rows | KKT n | nnz | route | it | ms | ms/it | rel_gap | retry |")
    print("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    for fam, n, d, rho, rep in cases:
        c = cell(fam, n, d, rho, rep, runs)
        blocks, rows = n * rep, 0 if fam == "soc" else n * rep
        gap = c["relgap"]
        route = "dense" if c["sparse"] is False else "sparse" if c["sparse"] else (c["route"] or "-")
        order = c["order"] if c["order"] is not None else "-"
        nnz = c["nnz"] if c["nnz"] is not None else "-"
        ms = c["secs"] * 1e3
        per = ms / c["its"] if c["its"] else 0.0
        label = f"{fam} n={n} rho={rho:.2f} rep={rep}"
        print(f"| {label} | {blocks} | {rows} | {order} | {nnz} | {route} | {c['its']} | "
              f"{ms:.2f} | {per:.3f} | {gap:.1e} | {c['retries']} |")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--only", default="family,density,degen")
    a = ap.parse_args()
    if not BIN.exists():
        raise SystemExit(f"missing {BIN}: run `make bench` first")

    only = set(a.only.split(","))
    if "family" in only:
        cases = [(f, n, 8, 0.5, 1) for f in ("soc", "exp", "mix")
                 for n in (10, 20, 40, 80)]
        table("family and blocks (d=8, rho=0.50, rep=1)", cases, a.runs)
    if "density" in only:
        cases = [(f, 40, 12, r, 1) for f in ("soc", "exp", "mix")
                 for r in (0.25, 0.50, 1.00)]
        table("link density (n=40, d=12, rep=1)", cases, a.runs)
    if "degen" in only:
        cases = [(f, 40, 8, 0.5, r) for f in ("exp", "mix") for r in (1, 2, 4)]
        table("degeneracy by replication (n=40, d=8, rho=0.50)", cases, a.runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
