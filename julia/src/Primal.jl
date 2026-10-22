module Primal

using Libdl

# ---- Low-level ccall bindings (auto-generated from primal.h) ---------------
include("bindings.jl")
import .Bindings
const _lib = Bindings._lib

# ---- Public enums, numbered as the C header (and as the vendor reference) --

"""
    BK

Bound key (PRIMALboundkeye), numbered as the reference:
LO=0, UP=1, FX=2, FR=3, RA=4.
"""
const BK = Dict{Symbol,Int}(
    :LO => 0, :UP => 1, :FX => 2, :FR => 3, :RA => 4,
)
BK_LO = 0; BK_UP = 1; BK_FX = 2; BK_FR = 3; BK_RA = 4

"""
    SENSE

Objective sense (PRIMALobjsensee): MIN=0, MAX=1.
"""
const SENSE = Dict{Symbol,Int}(:MIN => 0, :MAX => 1)
SENSE_MIN = 0; SENSE_MAX = 1

"""
    CT

Conic type (PRIMALconetypee): QUAD=0, RQUAD=1, PEXP=2, DEXP=3, PPOW=4, RPOW=7.
"""
const CT = Dict{Symbol,Int}(
    :QUAD => 0, :RQUAD => 1, :PEXP => 2, :DEXP => 3, :PPOW => 4, :RPOW => 7,
)
CT_QUAD = 0; CT_RQUAD = 1; CT_PEXP = 2; CT_DEXP = 3; CT_PPOW = 4; CT_RPOW = 7

"""
    VAR_TYPE

Variable type (PRIMALvariabletypee): CONT=0, INT=1, INT_BIN=2, SEMI_CONT=3, SEMI_INT=4.
"""
const VAR_TYPE = Dict{Symbol,Int}(
    :CONT => 0, :INT => 1, :INT_BIN => 2, :SEMI_CONT => 3, :SEMI_INT => 4,
)
VT_CONT = 0; VT_INT = 1; VT_INT_BIN = 2; VT_SEMI_CONT = 3; VT_SEMI_INT = 4

"""
    SOL

Solution key (PRIMALsolt): ITR=0, BAS=1, ITG=2.
"""
const SOL = Dict{Symbol,Int}(:ITR => 0, :BAS => 1, :ITG => 2)
SOL_ITR = 0; SOL_BAS = 1; SOL_ITG = 2

"""
    SOLSTA

Solution status (PRIMALsolstae): UNKNOWN=0, OPTIMAL=1, PRIM_FEAS=2,
PRIM_INFEAS_CER=5, DUAL_INFEAS_CER=6, INTEGER_OPTIMAL=9.
"""
const SOLSTA = Dict{Symbol,Int}(
    :UNKNOWN => 0, :OPTIMAL => 1, :PRIM_FEAS => 2,
    :PRIM_INFEAS_CER => 5, :DUAL_INFEAS_CER => 6, :INTEGER_OPTIMAL => 9,
)
SOLSTA_UNKNOWN = 0; SOLSTA_OPTIMAL = 1; SOLSTA_PRIM_FEAS = 2
SOLSTA_PRIM_INFEAS_CER = 5; SOLSTA_DUAL_INFEAS_CER = 6; SOLSTA_INTEGER_OPTIMAL = 9

"""
    RC

Response code (PRIMALrescodee): OK=0, ERR_ARG=1001, ERR_INFEASIBLE=1002,
ERR_UNBOUNDED=1003, ERR_ALLOC=1004, ERR_FILE=1005, ERR_NULL=1006,
TRM_MAX_ITER=1007, TRM_MAX_TIME=1008, TRM_OBJECTIVE_RANGE=1009.
"""
const RC = Dict{Symbol,Int}(
    :OK => 0, :ERR_ARG => 1001, :ERR_INFEASIBLE => 1002, :ERR_UNBOUNDED => 1003,
    :ERR_ALLOC => 1004, :ERR_FILE => 1005, :ERR_NULL => 1006,
    :TRM_MAX_ITER => 1007, :TRM_MAX_TIME => 1008, :TRM_OBJECTIVE_RANGE => 1009,
)
RC_OK = 0

const INF = Inf  # PRIMAL_isinfinity uses IEEE INF; C INFINITY == Inf

# ---- Result type -----------------------------------------------------------
"""
    Result(x, y, objective, dual, solsta, prosta, pinf, rc)

Outcome of a solve, mirroring the Python wrapper's `Result`.
"""
struct Result
    x::Vector{Float64}
    y::Vector{Float64}
    objective::Float64
    dual::Float64
    solsta::Int
    prosta::Int
    pinf::Float64
    rc::Int
end

"""
    Primal.version() -> VersionNumber

Version of the loaded C library (PRIMAL_getversion).
"""
function version()
    major, minor, rev = Ref{Cint}(0), Ref{Cint}(0), Ref{Cint}(0)
    rc = ccall(Libdl.dlsym(_lib(), "PRIMAL_getversion"), Cint,
               (Ptr{Cint}, Ptr{Cint}, Ptr{Cint}), major, minor, rev)
    rc == 0 || error("PRIMAL_getversion failed (rc=$rc)")
    return VersionNumber(major[], minor[], rev[])
end

# ---- High-level Model (convenience wrapper, like the Python Model) ---------
include("Model.jl")

# Export all public names so `using .Primal` brings them into scope.
_export_names = String[]
for _n in names(Primal, all=true)
    _s = String(_n)
    if !startswith(_s, "_") && !startswith(_s, "PRIMAL_") && !contains(_s, "#") &&
       _s ∉ ("Primal", "eval", "include", "Bindings", "Libdl", "Result", "Model")
        push!(_export_names, _s)
    end
end
# Always export the main types.  Model is NOT exported: JuMP exports its own,
# and `using JuMP, Primal` must leave `Model(Primal.Optimizer)` unambiguous --
# the low-level wrapper is used qualified, as Primal.Model.
push!(_export_names, "Result", "BK", "SENSE", "CT", "VAR_TYPE", "SOL", "SOLSTA", "RC", "INF")
if !isempty(_export_names)
    @eval export $(Symbol.(_export_names)...)
end

# ---- MathOptInterface wrapper: Model(Primal.Optimizer) in JuMP ---------------
# Included after the export block on purpose: Optimizer is used qualified,
# as Primal.Optimizer, like every other MOI solver.
include("MOI_wrapper.jl")

end  # module Primal
