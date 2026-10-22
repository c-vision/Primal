#!/usr/bin/env python3
"""Interleaved SOCP comparison; all solvers run within one session.

Run using the benchmark Python environment with clarabel/numpy/scipy.
The main boundary is PRIMAL_optimize versus Clarabel setup+solve.
Model assembly is excluded from both. Raw observations are saved as JSON.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

import clarabel
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "bench"))
from conic_ref import build_socp

CASES = [(40,0),(120,0),(200,0),(120,1),(200,1),(120,2),(200,2)]


def summarize(rows):
    result = {}
    for n, variant in CASES:
        key = f"{n}/{variant}"
        result[key] = {}
        for mode in sorted({r["mode"] for r in rows}):
            subset = [r for r in rows if r["n"] == n and r["variant"] == variant and r["mode"] == mode]
            if not subset:
                continue
            values = [r["seconds"] for r in subset]
            q = statistics.quantiles(values, n=4, method="inclusive") if len(values)>1 else values*3
            result[key][mode] = dict(count=len(values), median=statistics.median(values),
                                     p25=q[0], p75=q[2],
                                     quality_max={k: max(r[k] for r in subset) for k in ("primal","dual","gap")})
            if mode == "clarabel":
                result[key][mode]["solve_median"] = statistics.median(r["solve_seconds"] for r in subset)
    return result


def reference_data(n, variant):
    N, q, A, b, _, dims = build_socp(n)
    K = n//2
    if variant == 1:
        eq = sparse.lil_matrix((2, N))
        eq[0, :n:2] = 1
        eq[1, 1:n:2] = 1
        A = sparse.vstack([eq, A[1:]], format="csc")
        b = np.r_[0.6, 0.4, np.zeros(3*K)]
    elif variant == 2:
        A = sparse.lil_matrix((K+1+3*K, N))
        b = np.zeros(K+1+3*K)
        A[0, K:n] = 1
        b[0] = 1
        for k in range(K):
            A[1+k, k] = 1
            b[1+k] = 0.5
            r = K+1+3*k
            A[r, n+k] = A[r, k] = -1/math.sqrt(2)
            A[r+1, n+k] = -1/math.sqrt(2)
            A[r+1, k] = 1/math.sqrt(2)
            A[r+2, K+k] = -1
        A = A.tocsc()
    if variant:
        q[n:] = 1+0.05*(np.arange(K) % 7)
    zeros = 2 if variant == 1 else (K+1 if variant == 2 else 1)
    cones = [clarabel.ZeroConeT(zeros)] + [clarabel.SecondOrderConeT(d) for d in dims]
    return sparse.csc_matrix((N, N)), q, A, b, cones


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=("canonical", "rotated"), default="canonical",
                    help="predeclared acceptance profile; canonical compares to pre-corrector baseline")
    ap.add_argument("--small-fixed", action="store_true", help="also guard small dense rotated models")
    ap.add_argument("--runs", type=int, default=31)
    ap.add_argument("--out", type=Path, default=ROOT/"out/socp_compare")
    ap.add_argument("--baseline-worker", type=Path,
                    help="preserved worker linked to the unchanged baseline")
    ap.add_argument("--baseline-archive", type=Path,
                    help="preserved libprimal.a; compare isolated libraries in one process")
    ap.add_argument("--sham", action="store_true",
                    help="calibrate identical libraries from the baseline archive")
    args = ap.parse_args()
    if args.baseline_worker and args.baseline_archive:
        ap.error("select either a baseline worker or an archive")
    if args.sham and not args.baseline_archive:
        ap.error("--sham requires --baseline-archive")
    baseline_input = args.baseline_archive or args.baseline_worker
    if args.small_fixed:
        CASES.extend([(2,2),(10,2),(40,2)])
    if args.runs < 1:
        ap.error("--runs must be positive")
    args.out = args.out.resolve()
    if not args.out.is_relative_to(ROOT/"out"):
        ap.error("build/output directory must be under out/")
    args.out.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted([*ROOT.glob("*.c"), *ROOT.glob("*.h"), ROOT/"Makefile",
                                      ROOT/"bench/socp_compare.c", Path(__file__).resolve()])}
    build = args.out/"build"
    with (args.out/"build.log").open("w") as log:
        subprocess.run(["make", "-s", "OUT="+str(build), "lib"], cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        binary = build/"socp_compare"
        subprocess.run(["gcc", "-std=c99", "-Wall", "-Wextra", "-pedantic", "-O2",
                        "-I"+str(ROOT), str(ROOT/"bench/socp_compare.c"),
                        str(build/"libprimal.a"), "-lm", "-o", str(binary)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
        if args.baseline_archive:
            archives = {"baseline": args.baseline_archive.resolve(),
                        "candidate": args.baseline_archive.resolve() if args.sham else build/"libprimal.a"}
            for label, archive in archives.items():
                shared = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-Wl,-Bsymbolic"]
                subprocess.run(["gcc", "-std=c99", "-Wall", "-Wextra", "-pedantic", "-O2",
                                *shared, "-DSOCP_COMPARE_LIBRARY", "-I"+str(ROOT),
                                str(ROOT/"bench/socp_compare.c"), str(archive), "-lm",
                                "-o", str(build/(label+".dylib"))],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
    workers = {}
    libraries = {}
    for mode in (["baseline"] if baseline_input else []) + ["default", "ldl"]:
        if args.baseline_archive:
            label = "baseline" if mode == "baseline" else "candidate"
            lib = ctypes.CDLL(str(build/(label+".dylib")), mode=os.RTLD_LOCAL|os.RTLD_NOW)
            fn = lib.socp_compare_json
            fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
            fn.restype = ctypes.c_int
            libraries[mode] = lib
            workers[mode] = fn
            continue
        env = {k: v for k, v in os.environ.items() if not k.startswith("GMB_")}
        if mode == "ldl":
            env["GMB_SOCP_LDL"] = "1"
        worker_binary = args.baseline_worker.resolve() if mode == "baseline" else binary
        workers[mode] = subprocess.Popen([str(worker_binary)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, env=env)
    observations = []
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    # Same declared 1e-8 feasibility/gap tolerances; no reduced accuracy mode.
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-8
    try:
        for n, variant in CASES:
            data = reference_data(n, variant)
            for run in range(-3, args.runs):
                order = list(workers) + ["clarabel"]
                if run % 2:
                    order.reverse()
                for mode in order:
                    if mode == "clarabel":
                        if args.baseline_archive:
                            clarabel.DefaultSolver(*data, settings).solve()
                        start = time.perf_counter()
                        solver = clarabel.DefaultSolver(*data, settings)
                        ready = time.perf_counter()
                        sol = solver.solve()
                        end = time.perf_counter()
                        row = dict(seconds=end-start, solve_seconds=end-ready,
                                   setup_seconds=ready-start, objective=float(sol.obj_val),
                                   status=str(sol.status), primal=sol.r_prim, dual=sol.r_dual,
                                   gap=abs(sol.obj_val-sol.obj_val_dual)/(1+abs(sol.obj_val)+abs(sol.obj_val_dual)))
                        if str(sol.status) != "Solved":
                            raise RuntimeError(f"Clarabel did not solve {n}/{variant}: {row}")
                    else:
                        if args.baseline_archive:
                            for key in list(os.environ):
                                if key.startswith("GMB_"): del os.environ[key]
                            if mode == "ldl": os.environ["GMB_SOCP_LDL"] = "1"
                            buffer = ctypes.create_string_buffer(2048)
                            # Both images get an immediate untimed warmup;
                            # candidate/default otherwise shares hot code with LDL.
                            for _ in range(2):
                                if workers[mode](n, variant, buffer, len(buffer)):
                                    raise RuntimeError("C benchmark adapter failed")
                            row = json.loads(buffer.value)
                        else:
                            p = workers[mode]
                            p.stdin.write(f"{n} {variant}\n")
                            p.stdin.flush()
                            row = json.loads(p.stdout.readline())
                        if not row["valid"] or row["status"] != 1:
                            raise RuntimeError(f"Primal did not solve {mode}/{n}/{variant}: {row}")
                    expected = (1/math.sqrt(2) if variant == 0 else
                                math.sqrt(0.52) if variant == 1 else
                                1/sum(1/(1+0.05*(k%7)) for k in range(n//2)))
                    if abs(row["objective"]-expected) > 1e-6*(1+abs(expected)):
                        raise RuntimeError(f"objective mismatch {mode}/{n}/{variant}: {row}")
                    if any(not math.isfinite(row[k]) for k in ("seconds", "primal", "dual", "gap")):
                        raise RuntimeError(f"nonfinite result: {row}")
                    if max(row["primal"], row["dual"]) > 1e-6 or row["gap"] > 1e-8:
                        raise RuntimeError(f"quality gate failed {mode}/{n}/{variant}: {row}")
                    if run >= 0:
                        observations.append({**row, "n": n, "variant": variant,
                                             "run": run, "mode": mode})
            med = {m: statistics.median(r["seconds"] for r in observations
                                       if r["n"] == n and r["variant"] == variant and r["mode"] == m)
                   for m in workers.keys() | {"clarabel"}}
            print(n, variant, {k: round(v*1000, 4) for k, v in med.items()}, flush=True)
    finally:
        if not args.baseline_archive:
            for p in workers.values():
                p.stdin.close()
                p.wait()
        expected = args.runs*len(CASES)*(len(workers)+1)
        complete = len(observations) == expected
        summary = summarize(observations)
        gates = None
        if complete and baseline_input:
            def ratio(n, v=0):
                r=summary[f"{n}/{v}"]
                return r["default"]["median"]/r["baseline"]["median"]
            r200=summary["200/0"]
            gates = dict(socp200_gain=ratio(200)<=0.6,
                         socp200_vs_clarabel=r200["default"]["median"]<=1.5*r200["clarabel"]["median"],
                         socp120_gain=ratio(120)<=0.8,
                         socp40_regression=ratio(40)<=1.05,
                         controls_no_regression=all(ratio(n,v)<=1.05 for n,v in CASES if v))
            if args.target == "rotated":
                rr=summary["200/2"]
                gates=dict(rquad200_gain=ratio(200,2)<=0.7,
                           rquad200_vs_clarabel=rr["default"]["median"]<=3*rr["clarabel"]["median"],
                           other_cases_no_regression=all(ratio(n,v)<=1.05 for n,v in CASES if (n,v)!=(200,2)))
            if args.sham:
                gates = dict(identical_libraries=all(1/1.05<=ratio(n,v)<=1.05 for n,v in CASES))
            gates["all"] = all(gates.values())
        (args.out/"results.json").write_text(json.dumps(dict(
            revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            source_hashes=source_hashes,
            baseline_worker=str(args.baseline_worker) if args.baseline_worker else None,
            baseline_sha256=hashlib.sha256(args.baseline_worker.resolve().read_bytes()).hexdigest() if args.baseline_worker else None,
            baseline_archive=str(args.baseline_archive) if args.baseline_archive else None,
            archive_sha256=hashlib.sha256(args.baseline_archive.resolve().read_bytes()).hexdigest() if args.baseline_archive else None,
            measurement="same_process" if args.baseline_archive else "separate_processes", sham=args.sham,
            immediate_warmup=bool(args.baseline_archive),
            clarabel=clarabel.__version__, runs=args.runs, target=args.target, cases=CASES, complete=complete,
            observations=observations, summary=summary, gates=gates), indent=2))
    changed = [p for p, digest in source_hashes.items()
               if hashlib.sha256((ROOT/p).read_bytes()).hexdigest() != digest]
    if changed:
        raise RuntimeError(f"sources changed during measurement; discard results: {changed}")
    if gates is not None and not gates["all"]:
        raise SystemExit("Predeclared performance gates failed; see results.json")


if __name__ == "__main__":
    main()
