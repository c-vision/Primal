#!/usr/bin/env python3
#
# PrimalSolver - a convex optimization solver in C99 (LP/QP/SOCP/SDP/exp-power/MIP).
# Copyright 2026 Gaetano Minardi
# SPDX-License-Identifier: Apache-2.0
#
# Repeated-run aggregation for the canonical benchmark table.
#
# WHY THIS EXISTS.  bench/bench_all.py takes ONE run per cell, so the table it
# prints is a single-run snapshot.  That is not enough to resolve the effect of a
# change: re-running it on the SAME build moves Clarabel by -2.0% and SCS by -1.5%,
# and both are binaries a change to this repository cannot affect.  That ~2% is the
# noise floor, and it is only knowable because two independent columns moved.  So a
# percentage claimed from a single run is not a result.
#
# WHAT IT DOES.  Runs bench_all.py N times, takes the MEDIAN of every cell, checks
# that every objective is identical across all N runs, and prints the table.  It
# also compares two aggregated tables per family and reports which cases moved
# beyond the noise floor.  It deliberately does NOT re-implement the measurement:
# bench_all.py stays the single source of truth for how a cell is measured.
#
#   python3 tools/bench_repeat.py --runs 5                 # table, medians of 5
#   python3 tools/bench_repeat.py --runs 5 --compare old.md
#   python3 tools/bench_repeat.py --check-rows-only        # parser self-test
#
# Requires the same interpreter as the README procedure, i.e. one that has numpy,
# scipy and the optional references (clarabel, scs, highspy, pyscipopt).

import argparse
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_ALL = os.path.join(ROOT, "bench", "bench_all.py")

# Columns of the table that hold a TIMING and are therefore medianed.  The
# objective is always the LAST column and is never medianed: it must be stable.
# "instance | class | vars x cons | nnz | Primal | HiGHS | Clarabel | SCS | SCIP | obj"
TIME_COLS = range(4, 9)
NOISE_PCT = 2.0


def _cells(line):
    return [x.strip() for x in line.strip().strip("|").split("|")]


def parse_table(text):
    """Parse a markdown table into {instance: [cells]}.

    The separator row is detected structurally (every field blank/dash/colon),
    NOT by looking for a dash inside the content.  An earlier ad-hoc parser used
    `set(''.join(cells)) > set('-')`, which silently dropped every row that did not
    contain a literal dash -- 11 of 33 rows, i.e. all of LP, QP and MILP, because
    only SOCP/SDP carry a `-` in the nnz column.  See README guide section 6.2.
    """
    rows = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        c = _cells(line)
        if not c or not c[0]:
            continue
        if c[0].lower() == "instance":
            continue
        if all(set(x) <= set("-: ") for x in c):
            continue
        rows[c[0]] = c
    return rows


def _num(v):
    try:
        return float(v)
    except ValueError:
        return None


def run_once(python, timeout):
    r = subprocess.run([python, BENCH_ALL], capture_output=True, text=True,
                       timeout=timeout)
    if r.returncode != 0:
        sys.exit("bench_all.py failed (exit %d):\n%s" % (r.returncode, r.stderr[-2000:]))
    return parse_table(r.stdout)


def aggregate(python, runs, timeout):
    tabs = [run_once(python, timeout) for _ in range(runs)]
    base = tabs[0]
    for i, t in enumerate(tabs[1:], start=2):
        if set(t) != set(base):
            sys.exit("run %d has a different instance set than run 1; "
                     "the build changed under us" % i)
    out = {}
    for k, cells in base.items():
        row = []
        for i, v in enumerate(cells):
            if i in TIME_COLS and i < len(cells):
                vals = [_num(t[k][i]) for t in tabs]
                if any(x is None for x in vals):
                    row.append(v)                      # e.g. "N/A" in every run
                else:
                    row.append("%.4f" % statistics.median(vals))
            else:
                row.append(v)                          # objective: never medianed
        out[k] = row
    # objectives must be identical across runs; that is the correctness signal
    unstable = [k for k in base if len({t[k][-1] for t in tabs}) > 1]
    return out, unstable


def family_totals(rows):
    fam = {}
    for k, c in rows.items():
        t = _num(c[4])
        if t is None:
            continue
        a = fam.setdefault(c[1], [0.0, 0, 0])
        a[0] += t
        a[1] += 1
        if t < 0.001:
            a[2] += 1
    return fam


def compare(old, new):
    fo, fn = family_totals(old), family_totals(new)
    names = sorted(set(fo) | set(fn))
    w = max([len(n) for n in names] + [6])
    print("\nper family, PrimalSolver column (negative = faster)")
    print("%-*s %5s %11s %11s %9s" % (w, "family", "cases", "base(s)", "new(s)", "delta"))
    tb = tn = 0.0
    for n in names:
        a, b = fo.get(n, [0.0, 0]), fn.get(n, [0.0, 0])
        tb += a[0]
        tn += b[0]
        d = (100 * (b[0] - a[0]) / a[0]) if a[0] else 0.0
        print("%-*s %5d %11.4f %11.4f %+8.2f%%" % (w, n, a[1], a[0], b[0], d))
    print("%-*s %5s %11.4f %11.4f %+8.2f%%"
          % (w, "TOTAL", "", tb, tn, (100 * (tn - tb) / tb) if tb else 0.0))

    thr = 1.0 + NOISE_PCT / 100.0
    worse, better = [], []
    for k in old:
        if k not in new:
            continue
        a, b = _num(old[k][4]), _num(new[k][4])
        if not a or not b:
            continue
        if b > a * thr:
            worse.append((k, a, b))
        elif b < a / thr:
            better.append((k, a, b))
    print("\nbeyond the ~%.0f%% noise floor: %d faster, %d slower"
          % (NOISE_PCT, len(better), len(worse)))
    for label, lst in (("slower", worse), ("faster", better)):
        for k, a, b in sorted(lst, key=lambda r: -abs(r[2] - r[1]) / r[1]):
            print("  %-8s %-12s %.4f -> %.4f  %+.1f%%"
                  % (label, k, a, b, 100 * (b - a) / a))
    if worse:
        print("\nA change confined to one code path CANNOT move a family it does not "
              "call.\nCheck which families the change actually reaches before treating "
              "this as a regression.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=5,
                    help="runs of bench_all.py to median (default 5)")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter with numpy/scipy and the references")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--out", help="write the aggregated table here")
    ap.add_argument("--compare", metavar="OLD_MD",
                    help="compare against a previously saved table")
    ap.add_argument("--check-rows-only", action="store_true",
                    help="parse the committed README table and report row count")
    args = ap.parse_args()

    if args.check_rows_only:
        txt = open(os.path.join(ROOT, "README.md")).read()
        # Isolate ONLY the canonical benchmark table: it starts at its own header
        # row and ends at the closing </div>.  Parsing the whole README picks up
        # the methods table and the Class/Route table as well.
        start = txt.index("| instance | class | vars x cons |")
        end = txt.index("</div>", start)
        rows = parse_table(txt[start:end])
        fams = sorted({c[1] for c in rows.values()})
        print("README benchmark table: %d data rows parsed" % len(rows))
        print("families:", fams)
        ok = set(fams) == {"lp", "qp", "milp", "socp", "sdp"}
        # 32 instances: lp 4 + qp 3 + milp 3 + socp 3 + sdp 19
        good = ok and len(rows) == 32
        print("self-test:", "OK" if good
              else "UNEXPECTED (want 32 rows, families lp/qp/milp/socp/sdp)")
        return 0 if good else 1

    if args.runs < 1:
        sys.exit("--runs must be >= 1")

    rows, unstable = aggregate(args.python, args.runs, args.timeout)
    if not rows:
        sys.exit("no rows parsed from bench_all.py output")
    print("aggregated %d instances over %d runs" % (len(rows), args.runs))
    if unstable:
        sys.exit("OBJECTIVES UNSTABLE across runs: %s" % unstable)
    print("objectives identical in all %d runs" % args.runs)

    text = "\n".join("| " + " | ".join(c) + " |" for c in rows.values()) + "\n"
    if args.out:
        open(args.out, "w").write(text)
        print("wrote %s" % args.out)
    else:
        print(text)

    if args.compare:
        compare(parse_table(open(args.compare).read()), rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())