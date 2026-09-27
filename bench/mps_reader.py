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

"""Minimal MPS reader for the benchmark.

Enough for the generated LP/QP-less/MILP instances (bench/gen_instances.py) and
the small netlib LPs (bench/fetch_netlib.py).  Returns the model in a form the
SciPy HiGHS front-ends accept (linprog / milp).

read_mps(path) -> dict with:
    sense   'MIN' | 'MAX'
    c       list[float]          objective (as written)
    rows    list[(type, rhs, {j: aij})]   type in 'L','E','G'
    lo, up  list[float]          variable bounds (up may be math.inf)
    integ   list[int]            0/1 per variable (1 = integer)

RANGES and QUADOBJ/QSECTION are rejected: the QP class is not comparable with
linprog/milp, and the generated/ netlib sets do not use them.
"""
import math

_INF = math.inf


def read_mps(path):
    rows_type = {}          # name -> 'L'/'E'/'G'
    row_order = []
    objrow = None
    cols = {}               # name -> {rowname: val}
    col_order = []
    rhs = {}                # rowname -> val
    lo = {}
    up = {}
    integ = {}
    sense = "MIN"
    section = None
    integ_mode = False
    prev_col = None
    saw_quad = False

    def col(name):
        if name not in cols:
            cols[name] = {}
            col_order.append(name)
        return cols[name]

    with open(path) as fh:
        for ln in fh:
            if not ln.strip() or ln[0] == "*":
                continue
            f = ln.split()
            head = f[0]
            # A section keyword starts at column 1; data lines are indented.
            # (The default RHS-set name is literally "RHS", so a name cannot
            # disambiguate a header from a data line.)
            if ln[0] != " " and head in (
                    "NAME", "ROWS", "COLUMNS", "RHS", "RANGES", "BOUNDS",
                    "ENDATA", "OBJSENSE", "OBJNAME", "QUADOBJ", "QSECTION",
                    "QMATRIX"):
                if head in ("QUADOBJ", "QSECTION", "QMATRIX"):
                    saw_quad = True
                if head == "OBJSENSE":
                    section = "OBJSENSE"
                    continue
                section = head
                continue
            if section == "OBJSENSE":
                if f[0].upper() in ("MAX", "MAXIMIZE", "MAXIMUM"):
                    sense = "MAX"
                section = None
                continue
            if section == "ROWS":
                ty, name = f[0], f[1]
                if ty == "N":
                    objrow = name
                else:
                    rows_type[name] = ty
                    row_order.append(name)
            elif section == "COLUMNS":
                if len(f) >= 3 and f[1] == "'MARKER'":
                    integ_mode = (f[2] == "'INTORG'")
                    continue
                # the column-name field lives in fixed columns 5-12: if it is
                # blank the line continues the previous column.  (Positional
                # rule: namespaces of rows and columns can overlap, as in
                # netlib `blend`, so names cannot disambiguate.)
                if len(ln) > 4 and ln[4] != " ":
                    cname, start = f[0], 1
                    prev_col = cname
                else:
                    cname, start = prev_col, 0
                d = col(cname)
                if cname not in integ:
                    integ[cname] = 0
                if integ_mode:
                    integ[cname] = 1
                for k in range(start, len(f) - 1, 2):
                    rn, val = f[k], float(f[k + 1])
                    if objrow is not None and rn == objrow:
                        d["__obj__"] = d.get("__obj__", 0.0) + val
                    else:
                        d[rn] = d.get(rn, 0.0) + val
            elif section == "RHS":
                # same fixed field: blank RHS-name -> f[0] is a row name.
                start = 1 if (len(ln) > 4 and ln[4] != " ") else 0
                for k in range(start, len(f) - 1, 2):
                    rn, val = f[k], float(f[k + 1])
                    if objrow is not None and rn == objrow:
                        continue
                    rhs[rn] = rhs.get(rn, 0.0) + val
            elif section == "RANGES":
                raise ValueError("MPS RANGES not supported (%s)" % path)
            elif section == "BOUNDS":
                tys = ("LO", "UP", "FX", "FR", "MI", "PL", "BV", "LI", "UI")
                i = 0
                while i < len(f) and f[i] not in tys:
                    i += 1
                if i >= len(f):
                    continue
                ty = f[i]
                # the column name is the first token after the type that is a
                # known column: `[name] type col val` (netlib) and
                # `type name col val` (bench/gen_instances.py) both resolve.
                j = i + 1
                while j < len(f) and f[j] not in cols:
                    j += 1
                if j >= len(f):
                    continue
                cname = f[j]
                val = float(f[j + 1]) if len(f) > j + 1 else 0.0
                if ty == "LO":
                    lo[cname] = val
                elif ty == "UP":
                    up[cname] = val
                elif ty == "FX":
                    lo[cname] = up[cname] = val
                elif ty == "FR":
                    lo[cname] = -_INF
                    up[cname] = _INF
                elif ty == "MI":
                    lo[cname] = -_INF
                elif ty == "PL":
                    up[cname] = _INF
                elif ty == "BV":
                    lo[cname] = 0.0
                    up[cname] = 1.0
                    integ[cname] = 1
                elif ty == "LI":
                    lo[cname] = val
                    integ[cname] = 1
                elif ty == "UI":
                    up[cname] = val
                    integ[cname] = 1
                else:
                    raise ValueError("MPS bound type %r not supported" % ty)
    if saw_quad:
        raise ValueError("QUADOBJ/QSECTION not supported (QP class)")

    c = [cols[n].get("__obj__", 0.0) for n in col_order]
    n = len(col_order)
    lo_v = [lo.get(nm, 0.0) for nm in col_order]
    up_v = [up.get(nm, _INF) for nm in col_order]
    integ_v = [1 if integ.get(nm, 0) else 0 for nm in col_order]
    idx = {nm: j for j, nm in enumerate(col_order)}
    rows = []
    for rn in row_order:
        coef = {}
        for nm in col_order:
            if rn in cols[nm]:
                coef[idx[nm]] = cols[nm][rn]
        rows.append((rows_type[rn], rhs.get(rn, 0.0), coef))
    return {"sense": sense, "c": c, "rows": rows, "lo": lo_v, "up": up_v,
            "integ": integ_v, "nvar": n}
