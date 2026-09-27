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
# Estratto della licenza (Apache License 2.0, §2 "Grant of Copyright License"):
#   "Subject to the terms and conditions of this License, each Contributor
#    hereby grants to You a perpetual, worldwide, non-exclusive, no-charge,
#    royalty-free, irrevocable copyright license to reproduce, prepare
#    Derivative Works of, publicly display, publicly perform, sublicense, and
#    distribute the Work and such Derivative Works in Source or Object form."
# 
# Esonero di responsabilita' e assenza di garanzia (Apache License 2.0, §7-§8):
#   [§7] Unless required by applicable law or agreed to in writing, Licensor
#   provides the Work (and each Contributor provides its Contributions) on an
#   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#   or implied, including, without limitation, any warranties or conditions of
#   TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR
#   PURPOSE.  You are solely responsible for determining the appropriateness of
#   using or redistributing the Work.
#   [§8] In no event and under no legal theory, whether in tort (including
#   negligence), contract, or otherwise, unless required by applicable law or
#   agreed to in writing, shall any Contributor be liable to You for damages,
#   including any direct, indirect, special, incidental, or consequential
#   damages arising as a result of this License or out of the use or inability
#   to use the Work.  This software is provided without any guarantee that it
#   will operate correctly or be free of defects.

"""Fetch the netlib LP test set into bench/instances/ and add its rows to
index.csv, so `run_bench.py` includes them.

The netlib files are in the compressed `emps` format; this downloads the
expander (`emps.c`), compiles it, expands each instance to MPS and records
(name, class=netlib, vars, cons, nnz) in index.csv.

Usage:  python3 bench/fetch_netlib.py
Requires: curl (or urllib), a C compiler (`cc`).
The .mps files land in bench/instances/ (gitignored, like the generated ones).
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "instances")
BASE = "https://www.netlib.org/lp/data/"
# Set PICCOLO e veloce: le istanze netlib grandi (e226, israel, bandm, beaconfd)
# restano fuori per non rendere il benchmark esageratamente lungo.
NAMES = ["afiro", "adlittle", "blend", "share2b", "sc50a", "sc50b", "sc105",
         "sc205", "stocfor1", "kb2"]


def dims(path):
    """(vars, cons, nnz) of an MPS file: distinct columns, ROWS minus the N row,
    and the (row,value) pairs in COLUMNS."""
    sec = None
    cols = set()
    nrow = 0
    nnz = 0
    hasobj = False
    for ln in open(path):
        if not ln.strip() or ln[0] == '*':
            continue
        f = ln.split()
        if f and f[0] in ("NAME", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS",
                          "ENDATA", "OBJSENSE", "QSECTION", "QUADOBJ"):
            sec = f[0]
            continue
        if sec == "ROWS" and len(f) >= 2:
            nrow += 1
            if f[0] == 'N':
                hasobj = True
        elif sec == "COLUMNS":
            cols.add(f[0])
            nnz += (len(f) - 1) // 2
    return len(cols), nrow - (1 if hasobj else 0), nnz


def get(url, path):
    with urllib.request.urlopen(url, timeout=60) as r, open(path, "wb") as fh:
        shutil.copyfileobj(r, fh)


def main():
    os.makedirs(DEST, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="netlib-")
    try:
        emps_c = os.path.join(tmp, "emps.c")
        get(BASE + "emps.c", emps_c)
        emps = os.path.join(tmp, "emps")
        subprocess.run(["cc", "-O2", "-o", emps, emps_c], check=True)

        for n in NAMES:
            raw = os.path.join(tmp, n)
            get(BASE + n, raw)
            mps = os.path.join(DEST, n + ".mps")
            with open(mps, "w") as out:
                subprocess.run([emps, raw], check=True, stdout=out)
            print("  %-12s -> %s" % (n, mps))

        idx = os.path.join(DEST, "index.csv")
        rows = list(csv.DictReader(open(idx))) if os.path.exists(idx) else []
        have = set(r["name"] for r in rows)
        fields = ["name", "class", "vars", "cons", "nnz"]
        with open(idx, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
            for n in NAMES:
                if n in have:
                    continue
                v, c, z = dims(os.path.join(DEST, n + ".mps"))
                w.writerow({"name": n, "class": "netlib", "vars": v, "cons": c, "nnz": z})
        print("index.csv updated with the netlib rows")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
