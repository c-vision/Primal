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

"""A CBF file as plain Python data (read_cbf), for the conic conversions of
bench/lib_ref.py and the table dimensions of bench_all.py.  SCIP builds its
models from CBF in C: bench/scip_cbf.c.
"""
import gzip


class Unsupported(Exception):
    pass


def read_cbf(path):
    """The sections of a CBF file as plain Python data."""
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        toks = [ln.strip() for ln in f]
    toks = [t for t in toks if t and not t.startswith("#")]
    d = dict(sense="MIN", var=(0, []), con=(0, []), ints=[], pows=[],
             obja={}, objb=0.0, a=[], b={}, psdvar=[], psdcon=[],
             objf=[], f=[], h=[], d=[])
    i = 0

    def take():
        nonlocal i
        i += 1
        return toks[i - 1]

    def groups():
        n, k = map(int, take().split())
        gs = []
        for _ in range(k):
            name, size = take().split()
            gs.append((name, int(size)))
        return n, gs

    while i < len(toks):
        kw = take()
        if kw == "VER":
            take()
        elif kw == "OBJSENSE":
            d["sense"] = take()
        elif kw == "POWCONES":
            n, _ = map(int, take().split())
            for _ in range(n):
                k = int(take())
                d["pows"].append([float(take()) for _ in range(k)])
        elif kw == "VAR":
            d["var"] = groups()
        elif kw == "CON":
            d["con"] = groups()
        elif kw == "INT":
            d["ints"] = [int(take()) for _ in range(int(take()))]
        elif kw == "OBJACOORD":
            for _ in range(int(take())):
                j, v = take().split()
                d["obja"][int(j)] = d["obja"].get(int(j), 0.0) + float(v)
        elif kw == "OBJBCOORD":
            d["objb"] = float(take())
        elif kw == "ACOORD":
            for _ in range(int(take())):
                r, j, v = take().split()
                d["a"].append((int(r), int(j), float(v)))
        elif kw == "BCOORD":
            for _ in range(int(take())):
                r, v = take().split()
                d["b"][int(r)] = d["b"].get(int(r), 0.0) + float(v)
        elif kw in ("PSDVAR", "PSDCON"):
            d[kw.lower()] = [int(take()) for _ in range(int(take()))]
        elif kw in ("OBJFCOORD", "FCOORD", "HCOORD", "DCOORD"):
            key = {"OBJFCOORD": "objf", "FCOORD": "f", "HCOORD": "h", "DCOORD": "d"}[kw]
            for _ in range(int(take())):
                f = take().split()
                d[key].append(tuple(int(v) for v in f[:-1]) + (float(f[-1]),))
        else:
            raise Unsupported("CBF section %s" % kw)
    return d
