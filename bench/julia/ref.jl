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

# Pajarito (integer models: HiGHS for the MIP outer approximation, Clarabel
# without chordal decomposition for the conic subproblems) and Hypatia
# (continuous models) on CBF / MPS files,
# for bench/lib_ref.py.  Reads "path<TAB>timeout" lines on stdin and prints one
# JSON object per line: {"path", "status", "seconds", "obj"}, status one of
# ok, infeasible, timeout, n/a, unsupported.
#
# Julia compiles each method on first use, so a model that finishes is solved
# a second time and that run is the one reported: the time is the solve's, not
# the compiler's.  It covers optimize! (copying the model into the solver,
# bridging, solving), not reading the file.
#
# Usage:  julia --project=bench/julia bench/julia/ref.jl < list.tsv
#         (once: julia --project=bench/julia -e 'using Pkg; Pkg.instantiate()')

using JuMP
import MathOptInterface as MOI
import HiGHS, Hypatia, Pajarito, Clarabel, JSON

# Pajarito's conic subproblems go to Clarabel: with Hypatia there, the cuts
# built from its subproblem solutions cut off the optimum of CBLIB
# expdesign_D_12_6 / _16_8 (OPTIMAL at 0.110795 / 3.38664, where feasible
# points reach -0.520253 / 2.931324); Clarabel's chordal decomposition fails
# on several of those subproblems, so it is off.

function has_integers(m)
    for (F, S) in list_of_constraint_types(m)
        (S <: MOI.Integer || S <: MOI.ZeroOne) && return true
    end
    return false
end

function attach!(m, timeout)
    if has_integers(m)
        set_optimizer(m, optimizer_with_attributes(Pajarito.Optimizer,
            "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer, MOI.Silent() => true),
            "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer, MOI.Silent() => true,
                                                        "chordal_decomposition_enable" => false)))
        return "pajarito"
    end
    set_optimizer(m, Hypatia.Optimizer)
    return "hypatia"
end

function solve_once(path, timeout)
    m = read_from_file(path)
    which = attach!(m, timeout)
    set_silent(m)
    set_time_limit_sec(m, timeout)
    t = @elapsed optimize!(m)
    st = termination_status(m)
    if st == MOI.OPTIMAL && primal_status(m) == MOI.FEASIBLE_POINT
        return ("ok", t, objective_value(m), which)
    elseif st == MOI.INFEASIBLE
        return ("infeasible", t, nothing, which)
    elseif st == MOI.TIME_LIMIT
        return ("timeout", t, nothing, which)
    end
    return ("n/a ($(st))", t, nothing, which)
end

for line in eachline(stdin)
    isempty(strip(line)) && continue
    path, tl = String.(split(line, '\t'))
    timeout = parse(Float64, tl)
    res = try
        r = solve_once(path, timeout)
        # warm run: the first one paid for compilation
        (r[1] in ("ok", "infeasible") && r[2] < timeout) ? solve_once(path, timeout) : r
    catch err
        msg = sprint(showerror, err)
        println(stderr, path, ": ", first(msg, 300))
        (occursin("not supported", msg) || occursin("Unsupported", msg) ?
            "unsupported" : "n/a (error)", nothing, nothing, "")
    end
    println(JSON.json(Dict("path" => path, "status" => res[1], "seconds" => res[2],
                           "obj" => res[3], "solver" => res[4])))
    flush(stdout)
end
