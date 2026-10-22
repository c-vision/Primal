#!/usr/bin/env julia
# deps/build.jl — Compile the PrimalSolver C99 shared library for the Julia binding.
#
# The C sources are read-only input (never modified). They are resolved from:
#   1. $PRIMAL_C_SRC   (explicit checkout of the C repository)
#   2. ../../../       (live C repo, i.e. the parent of this package)
#
# The shared library (libprimal.dylib / .so / .dll) is written into deps/ so the
# Bindings module can dlopen it at runtime.
#
# Usage:
#   julia deps/build.jl
#   PRIMAL_C_SRC=/path/to/Primal julia deps/build.jl

const PRIMAL_C_SRC = get(ENV, "PRIMAL_C_SRC", nothing)
const HERE = @__DIR__
const PKG_ROOT = dirname(HERE)                       # .../PrimalSolver/julia

# The package is julia/ at the root of the PrimalSolver repository:
#   HERE       = .../PrimalSolver/julia/deps
#   PKG_ROOT   = .../PrimalSolver/julia
#   REPO_ROOT  = .../PrimalSolver   (= dirname(PKG_ROOT))
# (It used to live one level deeper, in julia/Primal/; the examples
# reorganisation moved it and this path was left pointing above the repo.)
const REPO_ROOT = dirname(PKG_ROOT)
const SRC_DIR = something(PRIMAL_C_SRC, REPO_ROOT)

isfile(joinpath(SRC_DIR, "primal.h")) ||
    error("primal.h not found in $SRC_DIR (set PRIMAL_C_SRC)")

# Gather the 33 C translation units that make up the library (LIBSRCS from the
# C Makefile): the 11 base files plus all primal_*.c TU's.
BASE_SRCS = [
    "linalg.c", "stdform.c", "simplex.c", "ipm.c", "socp.c", "sdp.c",
    "expcone.c", "mpsio.c", "cbf.c", "scaling.c", "presolve.c",
]

srcs = String[]
for f in BASE_SRCS
    p = joinpath(SRC_DIR, f)
    isfile(p) || error("missing C source: $p")
    push!(srcs, p)
end
for f in readdir(SRC_DIR)
    if startswith(f, "primal_") && endswith(f, ".c")
        push!(srcs, joinpath(SRC_DIR, f))
    end
end
sort!(srcs)

# Output library name and flags
libname = Sys.iswindows() ? "libprimal.dll" :
          Sys.isapple()   ? "libprimal.dylib" : "libprimal.so"
out = joinpath(HERE, libname)

cc = get(ENV, "CC", "gcc")
cc_cmd = split(cc)

base_flags = String["-std=c99", "-O2", "-fPIC", "-I", SRC_DIR]

if Sys.iswindows()
    cmd = [cc_cmd..., "-shared", "-static", "-o", out, srcs..., "-lm"]
elseif Sys.isapple()
    arch = occursin(r"-arch", join(cc_cmd)) ? String[] : ["-arch", "arm64", "-arch", "x86_64"]
    cmd = [cc_cmd..., base_flags..., arch..., "-dynamiclib",
           "-install_name", "@rpath/libprimal.dylib", "-o", out, srcs..., "-lm"]
else
    cmd = [cc_cmd..., base_flags..., "-shared", "-o", out, srcs..., "-lm"]
end

println("[Primal.jl] cc -> $out ($(length(srcs)) sources from $SRC_DIR)")
run(Cmd(cmd))

println("[Primal.jl] shared library written to $out")
