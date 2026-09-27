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

"""bench/expcone_metric_grid.py - route/accuracy grid of the native exp/power IPM.

The unified conic IPM writes each exp/power cone block with one of two metrics:

  nt : the Hessian-NT chain  Hs^{-1/2} (Hs^{1/2} Hz Hs^{1/2})^{1/2} Hs^{-1/2},
       which exists only when the dual direction s/scale lies in int K* (that is
       what gives the scaling point W, hence Hs), or
  hz : the Newton row of the barrier itself, grad^2 f(z).

Existence is the whole criterion.  A conditioning gate on top of it was measured
and dropped: at every finite value of cond(Hs^{1/2} Hz Hs^{1/2}) the reference
oracles PEXP(u=1,v=1), PEXP(u=2,v=1) and PPOW(0.3) fell back to the tangent
cuts, and the chain never answered a case worse than grad^2 f(z) did.

The nt-rows column is a reading aid, not an assertion: 0 is not a fault.  Measured
on the polish tail, PPOW(0.3) and the CBF case T47 run ENTIRELY on the degraded
grad^2 f(z) row (the chain does not exist there, and condz/conds/rel/wn print -1
meaning "not computed") and both still answer natively, at 1.9e-13 and KKT 7.08e-6
respectively.  The column says which row carried the point, not which row should
have.

This script is how that was measured and how it stays measured: it rebuilds the
route probe from the current sources, runs every reference case natively and with
the cuts forced (GMB_NO_EXP_IPM), and reports per case the route that answered,
its accuracy, and the [expm] metric trace of the polish tail (gap, condz, conds,
rel, margins).  The build is admitted when every case is answered inside the
tolerance the public tests check it at (1e-4, the documented oracle tolerance
and 4 orders above the 1e-8 relative primal/dual/gap triple that the route gate
is declared to satisfy).  The tangent-cut accuracy is printed next to it as the
reference, and a case where the cuts are better is listed: that is a difference
to know, not a contract to keep, since the cuts carry no error bound at all.

The second section checks the *selector*, on the exp/power samples rather than on
the probe cases.  The polish snapshots one point and the gate judges it, so the
snapshot has to be chosen by the same quantity the gate thresholds:

    viol = max(rel_pri/tol_pri, rel_dual/tol_dual, rel_gap/tol_gap)

The invariant is that the delivered point is the best-violation point the trace
contains.  This is not a style preference: the selector ranked by gap alone, and
on market_impact that delivered viol=1.82 (rel_pri 1.8e-8, outside the declared
1e-8, so the cuts answered) when the very same run had visited viol=0.367 --
inside all three tolerances.  Both the trace and the decision are read from the
solver's own GMB_DBG output, so the check cannot be satisfied by editing the
harness.

  usage:  python3 bench/expcone_metric_grid.py [--tail 1e-6]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys

SRC = ["linalg", "stdform", "simplex", "ipm", "socp", "sdp", "expcone",
       "mpsio", "cbf", "scaling", "presolve",
       "primal_core", "primal_put", "primal_qcon", "primal_vartype",
       "primal_get", "primal_names", "primal_meta", "primal_solution",
       "primal_afe", "primal_djc", "primal_bar", "primal_sdptask",
       "primal_mip", "primal_mip_opt", "primal_conicopt", "primal_quad",
       "primal_solio", "primal_std", "primal_verdict", "primal_optimize",
       "primal_misc", "primal_info"]
METRICS = ["gap", "condz", "conds", "rel", "mz", "ms", "wn"]
EXPM = re.compile(
    r"\[expm\] mu=(\S+) sig=(\S+) blk=(\S+) kind=(\S+) "
    + " ".join(f"{m}=(\\S+)" for m in METRICS) + r" row=(\w+)")
CASE = re.compile(r"^(PEXP|PPOW|RPOW|T36|T47)\b.*?\b(NATV|fall|cuts)\s+rc=")
# the polish's own per-iterate triple, and the decision taken on it
ITRA = re.compile(r"^it=\d+ \S+=\S+ \S+=\S+ \S+=\S+ \| "
                  r"rel_pri=(\S+) rel_dual=(\S+) rel_gap=(\S+)")
# "near" is the reference's acceptance factor (INTPNT_CO_TOL_NEAR_REL) the gate
# judged the point against; a solve that never restored a point prints no triple
# at all, and there the selector has nothing to be checked on.
ROUT = re.compile(r"\[route\] status=(\d+) rel_pri=(\S+) rel_dual=(\S+) "
                  r"rel_gap=(\S+) \(tol (\S+)/(\S+)/(\S+)(?: near (\S+))?\)")
NOMEAS = re.compile(r"\[route\] status=(\d+) triple not measured")
REFU = re.compile(r"\[route\] native (?:conic IPM|exp/power) refused the model")
# exp/power cases of the sample set: enough blocks to exercise the selector
SAMPLES = ["mosek_comparison/pow1", "mosek_comparison/gp1",
           "finance/market_impact", "logistic_large",
           "mosek_comparison/logistic", "mosek_comparison/ceo1"]
# accuracy the probe prints for each case family
ERR = [re.compile(r"\|t-want\|=(\S+)"),
       re.compile(r"radial KKT=(-?\S+)"),
       re.compile(r"\bc=(-?\S+)")]
ORACLE_TOL = 1e-4        # the tolerance T79/T81 check these cases at


def build(out, repo):
    """Compile the route probe from the current sources (in a /tmp copy)."""
    tmp = "/tmp/expgrid"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    for f in sorted(os.listdir(repo)):
        if f.endswith((".c", ".h")) and f != "test_primal.c":
            shutil.copy(os.path.join(repo, f), tmp)
    src = open(os.path.join(tmp, "expcone.c")).read()
    assert re.search(r"if \(M\.have_s &&", src), "the row-selection anchor moved"
    cmd = (["gcc", "-std=c99", "-Wall", "-Wextra", "-pedantic", "-O2",
            "-I", tmp, os.path.join(repo, "bench/expcone_route_probe.c")]
           + [os.path.join(tmp, f"{s}.c") for s in SRC] + ["-lm", "-o", out])
    subprocess.run(cmd, check=True)
    return out


def run(probe):
    env = dict(os.environ, GMB_DBG="1")
    env.pop("GMB_NO_EXP_IPM", None)
    # the case labels go to stdout, the [expm] trace to stderr; both streams are
    # unbuffered, so merging them keeps the records grouped under their case
    return subprocess.run([probe], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=600, env=env).stdout


def errof(line, t36=False):
    for rx in ERR[:2]:
        m = rx.search(line)
        if m:
            return abs(float(m.group(1)))
    if t36:
        m = ERR[2].search(line)
        if m:
            return abs(float(m.group(1)))
    return None


def parse(log):
    """-> [(case, route, err, [records])] where a record is a metric dict."""
    lines = [l.strip() for l in log.splitlines()]
    cases, records = [], []
    for i, s in enumerate(lines):
        m = EXPM.match(s)
        if m:
            g = m.groups()
            rec = {"mu": float(g[0]), "sig": float(g[1]), "blk": g[2], "kind": g[3],
                   "row": g[4 + len(METRICS)]}
            for k, v in zip(METRICS, g[4:4 + len(METRICS)]):
                rec[k] = float(v)
            records.append(rec)
            continue
        c = CASE.search(s)
        if c:
            t36 = c.group(1) == "T36"
            err = errof(s, t36)
            if err is None:                      # T47 prints its KKT on line 2
                err = errof(" ".join(lines[i + 1:i + 3]), t36)
            # label = inside the match, everything before the route token
            cases.append((s[:c.start(2)].strip(), c.group(2), err, records))
            records = []
    return cases


def clean(label):
    """'PEXP cuts u=1 v=1' / 'PPOW cuts(0.3) u=1 v=2' -> the native-run label.
    T47 prints the same label for both runs, so the route token is what tells
    the reference apart; this only normalises the name."""
    return re.sub(r"\s+cuts\b", "", label).strip()


def tail(rel):
    v = [r["rel"] for r in rel if r["rel"] > 0]
    return (min(v), max(v)) if v else None


def report(cases, refs, cut):
    """Admit the build when every case is answered inside the tolerance the
    public tests check it at.  A case where the cuts answer better than the
    native blocks is listed but not a fault (a `fall` answers WITH that
    reference, so the route column is what says who replied)."""
    faults, behind = [], []
    for lab in sorted(cases):
        route, err, recs = cases[lab]
        t = [r for r in recs if 0 < r["gap"] <= cut]
        ref = refs.get(lab)
        if err is None or err > ORACLE_TOL:
            faults.append(f"{lab}: {fmtv(err)} outside {ORACLE_TOL:g}")
        elif ref is not None and err > ref:
            behind.append(f"{lab} {err:g} > cuts {ref:g}")
        nt = sum(1 for r in t if r["row"] == "nt")
        print(f"  {lab:24s} {route:5s} err={fmtv(err)} ref={fmtv(ref)}"
              f" nt-rows {nt}/{len(t):3d} rel={fmtspan(tail(t))}")
    if behind:
        print("  cuts better (recorded, not a fault): " + "; ".join(behind))
    print("\n" + (f"ADMITTED: every case inside {ORACLE_TOL:g}"
                  if not faults else "REJECTED: " + "; ".join(faults)))
    return 0 if not faults else 1


def fmtv(v):
    return f"{v:<9.3g}" if v is not None else "-        "


def fmtspan(rng):
    return f"{rng[0]:.2g}..{rng[1]:.2g}".ljust(14) if rng else "-"


def sample_bin(spec, tmp, repo):
    """One sample, compiled against the same /tmp source copy the probe used."""
    out = os.path.join(tmp, os.path.basename(spec))
    subprocess.run(["gcc", "-std=c99", "-Wall", "-Wextra", "-pedantic", "-O2",
                    "-I", tmp, os.path.join(repo, "samples", spec + ".c")]
                   + [os.path.join(tmp, f"{s}.c") for s in SRC]
                   + ["-lm", "-o", out], check=True)
    return out


def violate(triple, tols):
    """The dimensionless number the gate thresholds: worst criterion first."""
    return max(r / t for r, t in zip(triple, tols))


def selector_report(repo, tmp):
    """The delivered point must be the best-violation point the trace contains."""
    faults, env = [], dict(os.environ, GMB_DBG="1")
    env.pop("GMB_NO_EXP_IPM", None)
    print("\n  selector contract (is the delivered point the best in the trace?)")
    for spec in SAMPLES:
        name = os.path.basename(spec)
        log = subprocess.run([sample_bin(spec, tmp, repo)], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, timeout=900,
                             env=env).stdout
        # A failed exp/power run is retried in sdp_ipm's secant mode, which
        # prints its own trace and [route] after a "[retry]" line.  The point
        # delivered is the retry's when its [route] says status=0, and the
        # first run's otherwise (sdp_ipm restores it): judge that run alone.
        if "[retry]" in log:
            first, second = log.split("[retry]", 1)
            r2 = next((x for x in (ROUT.search(l) for l in second.splitlines()) if x), None)
            log = second if r2 is not None and int(r2.group(1)) == 0 else first
        tr = [tuple(map(float, m.groups())) for m in
              (ITRA.match(l.strip()) for l in log.splitlines()) if m]
        rm = next((x for x in (ROUT.search(l) for l in log.splitlines()) if x), None)
        if rm is None and REFU.search(log):
            print(f"  {name:16s} refused by the native conversion (ranged var or QP"
                  f" objective) -- the cuts answer, the selector never ran")
            continue
        nm = next((x for x in (NOMEAS.search(l) for l in log.splitlines()) if x), None)
        if nm is not None:
            print(f"  {name:16s} status={int(nm.group(1))} no point restored -- nothing"
                  f" to judge, the selector never ran")
            continue
        if not tr or rm is None:
            faults.append(f"{name}: no it= trace or [route] line")
            continue
        g = rm.groups()
        status, deliv, tols = int(g[0]), tuple(map(float, g[1:4])), tuple(map(float, g[4:7]))
        best, got = min(violate(x, tols) for x in tr), violate(deliv, tols)
        if best < 1.0 and status != 0:
            faults.append(f"{name}: a point inside the tolerances (viol {best:.3g}) "
                          f"was seen and not delivered (status={status})")
        if any(deliv) and got > 1.02 * best:
            faults.append(f"{name}: delivered viol {got:.3g} over the trace's "
                          f"best {best:.3g}")
        print(f"  {name:16s} status={status} viol={got:<8.3g} best-in-trace={best:<8.3g}"
              f" tol={tols[0]:g} over {len(tr)} iterates")
    return faults


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tail", type=float, default=1e-6, help="gap cutoff of the polish tail")
    args = ap.parse_args()
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(repo, "out/bench"), exist_ok=True)
    tmp = "/tmp/expgrid"
    exe = build(os.path.join(repo, "out/bench/route_probe"), repo)
    cases, refs = {}, {}
    for lab, route, err, recs in parse(run(exe)):
        lab = clean(lab)
        if route == "cuts":
            refs[lab] = err
        else:
            cases[lab] = (route, err, recs)
    rc = report(cases, refs, args.tail)
    faults = selector_report(repo, tmp)
    print("\n" + ("SELECTOR OK" if not faults else "SELECTOR FAULTS: " + "; ".join(faults)))
    sys.exit(rc or (1 if faults else 0))


if __name__ == "__main__":
    main()
