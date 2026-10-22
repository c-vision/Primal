# MOI_wrapper.jl — MathOptInterface wrapper, so Primal works from JuMP:
#
#     using JuMP, Primal
#     model = Model(Primal.Optimizer)
#
# The wrapper is a copy_to optimizer: JuMP keeps the model in its cache and
# hands it over complete at optimize!. Scalar affine constraints become rows
# (constant moved into the bounds); VariableIndex constraints become variable
# bounds and integrality; vector constraints become affine conic constraints
# (AFE rows + one ACC per constraint) on the matching domain. Constraints MOI
# does not support directly reach the solver through MOI's bridges.

import MathOptInterface as MOI

const _SAF = MOI.ScalarAffineFunction{Float64}
const _SQF = MOI.ScalarQuadraticFunction{Float64}
const _VAF = MOI.VectorAffineFunction{Float64}
const _VOV = MOI.VectorOfVariables
const _SCALAR_SETS = Union{MOI.GreaterThan{Float64},MOI.LessThan{Float64},
                           MOI.EqualTo{Float64},MOI.Interval{Float64}}
const _CONE_SETS = Union{MOI.Zeros,MOI.Nonnegatives,MOI.Nonpositives,
                         MOI.SecondOrderCone,MOI.RotatedSecondOrderCone,
                         MOI.ExponentialCone,MOI.DualExponentialCone,
                         MOI.PowerCone{Float64},MOI.DualPowerCone{Float64}}

# ---- C entry points used here ------------------------------------------------
_fn(name) = Libdl.dlsym(_lib(), name)
function _ok(rc, what)
    Int(rc) == 0 || error("Primal: $what failed (rc=$(Int(rc)))")
    return nothing
end

const _IPAR_LOG = 4
const _DPAR_OPTIMIZER_MAX_TIME = 22
const _SOL_ITR = Cint(0)
const _SOL_ITG = Cint(2)

# Kinds of constraints, for the result getters.
const _KIND_BOUND = 1   # VariableIndex in a scalar set or ZeroOne/Integer
const _KIND_ROW = 2     # ScalarAffineFunction in a scalar set
const _KIND_ACC = 3     # vector function in a cone

struct _ConInfo
    kind::Int
    index::Int                 # variable (bound), row or ACC index, 0-based
    func::Any                  # the MOI function, for ConstraintPrimal
    set::Any
end

"""
    Primal.Optimizer()

MathOptInterface optimizer for PrimalSolver: `Model(Primal.Optimizer)` in
JuMP. Supports linear and convex quadratic objectives, linear constraints,
integer and binary variables, and second-order, rotated, exponential and
power cones (and their duals); other convex constraints reach it through
MOI's bridges.
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    env::Ref{Ptr{Cvoid}}
    task::Ref{Ptr{Cvoid}}
    silent::Bool
    time_limit::Union{Nothing,Float64}
    raw::Dict{String,Any}
    # model
    nvar::Int
    sense::MOI.OptimizationSense
    integer::Bool
    cons::Dict{MOI.ConstraintIndex,_ConInfo}
    acc_map::Dict{Int,Tuple{Vector{Int},Vector{Float64}}}  # ACC index -> (perm, scale), see _cone_perm
    # result
    optimized::Bool
    rc::Int
    solve_time::Float64
    x::Vector{Float64}
    function Optimizer()
        model = new(Ref{Ptr{Cvoid}}(C_NULL), Ref{Ptr{Cvoid}}(C_NULL), false,
                    nothing, Dict{String,Any}(), 0, MOI.FEASIBILITY_SENSE,
                    false, Dict{MOI.ConstraintIndex,_ConInfo}(),
                    Dict{Int,Tuple{Vector{Int},Vector{Float64}}}(), false, 0, 0.0, Float64[])
        _new_task!(model)
        finalizer(_free!, model)
        return model
    end
end

function _new_task!(model::Optimizer)
    _ok(ccall(_fn("PRIMAL_makeenv"), Cint, (Ptr{Ptr{Cvoid}}, Ptr{Cvoid}),
              model.env, C_NULL), "makeenv")
    _ok(ccall(_fn("PRIMAL_maketask"), Cint,
              (Ptr{Cvoid}, Cint, Cint, Ptr{Ptr{Cvoid}}),
              model.env[], 0, 0, model.task), "maketask")
    return nothing
end

function _free!(model::Optimizer)
    if model.task[] != C_NULL
        ccall(_fn("PRIMAL_deletetask"), Cint, (Ptr{Ptr{Cvoid}},), model.task)
        model.task[] = C_NULL
    end
    if model.env[] != C_NULL
        ccall(_fn("PRIMAL_deleteenv"), Cint, (Ptr{Ptr{Cvoid}},), model.env)
        model.env[] = C_NULL
    end
    return nothing
end

_task(model::Optimizer) = model.task[]

# Sizes read from the task itself. An affine conic constraint makes the C
# layer append one auxiliary variable per AFE row (and may add rows), so the
# task holds MORE variables than the MOI model: every buffer a getter fills is
# sized from these counts, never from the model's own -- a getxx into an
# nvar-long buffer overran it and corrupted Julia's heap.
function _numvar(t::Ptr{Cvoid})
    n = Ref{Cint}(0)
    _ok(ccall(_fn("PRIMAL_getnumvar"), Cint, (Ptr{Cvoid}, Ptr{Cint}), t, n), "getnumvar")
    return Int(n[])
end
function _numcon(t::Ptr{Cvoid})
    n = Ref{Cint}(0)
    _ok(ccall(_fn("PRIMAL_getnumcon"), Cint, (Ptr{Cvoid}, Ptr{Cint}), t, n), "getnumcon")
    return Int(n[])
end

# ---- basic MOI interface ------------------------------------------------------
MOI.get(::Optimizer, ::MOI.SolverName) = "Primal"

function MOI.get(::Optimizer, ::MOI.SolverVersion)
    v = version()
    return string(v)
end

function MOI.is_empty(model::Optimizer)
    return model.nvar == 0 && isempty(model.cons) &&
           model.sense == MOI.FEASIBILITY_SENSE && !model.optimized
end

function MOI.empty!(model::Optimizer)
    _free!(model)
    _new_task!(model)
    model.nvar = 0
    model.sense = MOI.FEASIBILITY_SENSE
    model.integer = false
    empty!(model.cons)
    empty!(model.acc_map)
    model.optimized = false
    model.rc = 0
    model.solve_time = 0.0
    model.x = Float64[]
    return nothing
end

MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.get(model::Optimizer, ::MOI.Silent) = model.silent
MOI.set(model::Optimizer, ::MOI.Silent, v::Bool) = (model.silent = v; nothing)

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
MOI.get(model::Optimizer, ::MOI.TimeLimitSec) = model.time_limit
function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, v::Union{Nothing,Real})
    model.time_limit = v === nothing ? nothing : Float64(v)
    return nothing
end

MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true
function MOI.get(model::Optimizer, p::MOI.RawOptimizerAttribute)
    haskey(model.raw, p.name) || throw(MOI.GetAttributeNotAllowed(p))
    return model.raw[p.name]
end
function MOI.set(model::Optimizer, p::MOI.RawOptimizerAttribute, v)
    model.raw[p.name] = v
    return nothing
end

# ---- what copy_to accepts -------------------------------------------------------
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{<:Union{MOI.VariableIndex,_SAF,_SQF}}) = true

MOI.supports_constraint(::Optimizer, ::Type{MOI.VariableIndex},
    ::Type{<:Union{_SCALAR_SETS,MOI.ZeroOne,MOI.Integer}}) = true
MOI.supports_constraint(::Optimizer, ::Type{_SAF}, ::Type{<:_SCALAR_SETS}) = true
MOI.supports_constraint(::Optimizer, ::Type{<:Union{_VAF,_VOV}}, ::Type{<:_CONE_SETS}) = true

MOI.supports_incremental_interface(::Optimizer) = false

# ---- copy_to ----------------------------------------------------------------------
_lower(s::MOI.GreaterThan) = s.lower
_lower(s::MOI.EqualTo) = s.value
_lower(s::MOI.Interval) = s.lower
_lower(::MOI.LessThan) = -Inf
_upper(s::MOI.LessThan) = s.upper
_upper(s::MOI.EqualTo) = s.value
_upper(s::MOI.Interval) = s.upper
_upper(::MOI.GreaterThan) = Inf

# Bound key for a (lower, upper) pair: LO=0, UP=1, FX=2, FR=3, RA=4.
function _boundkey(l::Float64, u::Float64)
    if l == -Inf && u == Inf
        return Cint(3)
    elseif u == Inf
        return Cint(0)
    elseif l == -Inf
        return Cint(1)
    elseif l == u
        return Cint(2)
    end
    return Cint(4)
end

# MOI orders the exponential cones (x, y, z) with y e^{x/y} <= z; the C API, like
# the reference it follows, orders them (z, y, x). Every other cone shares MOI's
# order. perm[k] = MOI row placed at cone position k.
# How a MOI cone is written in a C domain: cone position k holds scale[k] times
# MOI row perm[k]. The exponential cones are reversed (MOI (x,y,z) with
# y*exp(x/y) <= z is the C PEXP (m0,m1,m2) = (z,y,x)). The dual cones are
# written on the primal C domains -- the C DEXP is -PEXP, not the dual cone, and
# a dual power domain is not implemented:
#   DualExponentialCone (u,v,w)  <=>  (e*w, -u, -v) in PEXP
#   DualPowerCone(a)    (u,v,w)  <=>  (u/a, v/(1-a), w) in PowerCone(a)
# A dual is mapped back by the adjoint: MOI row perm[k] gets scale[k]*doty[k].
_cone_perm(::MOI.ExponentialCone, d) = [3, 2, 1]
_cone_perm(::MOI.DualExponentialCone, d) = [3, 1, 2]
_cone_perm(::Any, d) = collect(1:d)
_cone_scale(::MOI.DualExponentialCone, d) = [exp(1.0), -1.0, -1.0]
_cone_scale(s::MOI.DualPowerCone, d) = [1 / s.exponent, 1 / (1 - s.exponent), 1.0]
_cone_scale(::Any, d) = ones(d)

function _domain!(model::Optimizer, s, d::Int)
    idx = Ref{Int64}(0)
    t = _task(model)
    if s isa MOI.Zeros
        _ok(ccall(_fn("PRIMAL_appendrzerodomain"), Cint, (Ptr{Cvoid}, Int64, Ptr{Int64}), t, d, idx), "appendrzerodomain")
    elseif s isa MOI.Nonnegatives
        _ok(ccall(_fn("PRIMAL_appendrplusdomain"), Cint, (Ptr{Cvoid}, Int64, Ptr{Int64}), t, d, idx), "appendrplusdomain")
    elseif s isa MOI.Nonpositives
        _ok(ccall(_fn("PRIMAL_appendrminusdomain"), Cint, (Ptr{Cvoid}, Int64, Ptr{Int64}), t, d, idx), "appendrminusdomain")
    elseif s isa MOI.SecondOrderCone
        _ok(ccall(_fn("PRIMAL_appendquadraticconedomain"), Cint, (Ptr{Cvoid}, Int64, Ptr{Int64}), t, d, idx), "appendquadraticconedomain")
    elseif s isa MOI.RotatedSecondOrderCone
        _ok(ccall(_fn("PRIMAL_appendrquadraticconedomain"), Cint, (Ptr{Cvoid}, Int64, Ptr{Int64}), t, d, idx), "appendrquadraticconedomain")
    elseif s isa Union{MOI.ExponentialCone,MOI.DualExponentialCone}
        _ok(ccall(_fn("PRIMAL_appendprimalexpconedomain"), Cint, (Ptr{Cvoid}, Ptr{Int64}), t, idx), "appendprimalexpconedomain")
    elseif s isa Union{MOI.PowerCone,MOI.DualPowerCone}
        _ok(ccall(_fn("PRIMAL_appendprimalpowerconedomain"), Cint, (Ptr{Cvoid}, Int64, Cdouble, Ptr{Int64}), t, d, s.exponent, idx), "appendprimalpowerconedomain")
    else
        throw(MOI.UnsupportedConstraint{_VAF,typeof(s)}())
    end
    return idx[]
end

# Attributes copy_to may meet in the source. Names live in the cache and starting
# points are hints an interior-point method cannot use, so both are dropped;
# anything else this optimizer does not support is an error, as MOI requires.
const _IGNORED_ATTRIBUTES = (MOI.Name, MOI.VariableName, MOI.ConstraintName,
    MOI.VariablePrimalStart, MOI.ConstraintPrimalStart, MOI.ConstraintDualStart)

function _check_attributes(dest::Optimizer, src::MOI.ModelLike)
    for attr in MOI.get(src, MOI.ListOfModelAttributesSet())
        attr isa Union{_IGNORED_ATTRIBUTES...,MOI.ObjectiveSense,MOI.ObjectiveFunction} && continue
        MOI.supports(dest, attr) || throw(MOI.UnsupportedAttribute(attr))
    end
    for attr in MOI.get(src, MOI.ListOfVariableAttributesSet())
        attr isa Union{_IGNORED_ATTRIBUTES...} && continue
        MOI.supports(dest, attr, MOI.VariableIndex) || throw(MOI.UnsupportedAttribute(attr))
    end
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        for attr in MOI.get(src, MOI.ListOfConstraintAttributesSet{F,S}())
            attr isa Union{_IGNORED_ATTRIBUTES...} && continue
            MOI.supports(dest, attr, MOI.ConstraintIndex{F,S}) || throw(MOI.UnsupportedAttribute(attr))
        end
    end
    return
end

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    MOI.empty!(dest)
    _check_attributes(dest, src)
    t = _task(dest)
    map = MOI.Utilities.IndexMap()
    # variables, all free until their bound constraints are read
    vis = MOI.get(src, MOI.ListOfVariableIndices())
    n = length(vis)
    col = Dict{MOI.VariableIndex,Int}()
    for (j, vi) in enumerate(vis)
        map[vi] = MOI.VariableIndex(j)
        col[vi] = j - 1
    end
    dest.nvar = n
    if n > 0
        _ok(ccall(_fn("PRIMAL_appendvars"), Cint, (Ptr{Cvoid}, Cint), t, n), "appendvars")
    end
    lb = fill(-Inf, n)
    ub = fill(Inf, n)
    isint = falses(n)
    nacc = 0
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        MOI.supports_constraint(dest, F, S) || throw(MOI.UnsupportedConstraint{F,S}())
        for (k, ci) in enumerate(MOI.get(src, MOI.ListOfConstraintIndices{F,S}()))
            f = MOI.get(src, MOI.ConstraintFunction(), ci)
            s = MOI.get(src, MOI.ConstraintSet(), ci)
            dci = MOI.ConstraintIndex{F,S}(k)
            map[ci] = dci
            if F == MOI.VariableIndex
                j = col[f]
                if s isa Union{MOI.ZeroOne,MOI.Integer}
                    isint[j + 1] = true
                    if s isa MOI.ZeroOne
                        lb[j + 1] = max(lb[j + 1], 0.0)
                        ub[j + 1] = min(ub[j + 1], 1.0)
                    end
                else
                    lb[j + 1] = max(lb[j + 1], _lower(s))
                    ub[j + 1] = min(ub[j + 1], _upper(s))
                end
                dest.cons[dci] = _ConInfo(_KIND_BOUND, j, MOI.VariableIndex(j + 1), s)
            elseif F == _SAF
                fc = MOI.Utilities.canonical(f)
                # The row index is read from the task: an affine conic
                # constraint may add internal rows, so a local counter drifts.
                nrow = _numcon(t)
                _ok(ccall(_fn("PRIMAL_appendcons"), Cint, (Ptr{Cvoid}, Cint), t, 1), "appendcons")
                sub = Cint[col[term.variable] for term in fc.terms]
                val = Float64[term.coefficient for term in fc.terms]
                _ok(ccall(_fn("PRIMAL_putarow"), Cint, (Ptr{Cvoid}, Cint, Cint, Ptr{Cint}, Ptr{Cdouble}),
                          t, nrow, length(sub), sub, val), "putarow")
                l = _lower(s) - fc.constant
                u = _upper(s) - fc.constant
                _ok(ccall(_fn("PRIMAL_putconbound"), Cint, (Ptr{Cvoid}, Cint, Cint, Cdouble, Cdouble),
                          t, nrow, _boundkey(l, u), l, u), "putconbound")
                dest.cons[dci] = _ConInfo(_KIND_ROW, nrow, MOI.Utilities.map_indices(map, fc), s)
            else
                g = MOI.Utilities.canonical(convert(_VAF, f))
                d = MOI.output_dimension(g)
                perm = _cone_perm(s, d)
                scale = _cone_scale(s, d)
                inv = invperm(perm)          # MOI row r sits at cone position inv[r]
                base = Ref{Int64}(0)
                _ok(ccall(_fn("PRIMAL_getnumafe"), Cint, (Ptr{Cvoid}, Ptr{Int64}), t, base), "getnumafe")
                _ok(ccall(_fn("PRIMAL_appendafes"), Cint, (Ptr{Cvoid}, Int64), t, d), "appendafes")
                afe = Int64[]; vidx = Cint[]; vv = Float64[]
                for term in g.terms
                    push!(afe, base[] + inv[term.output_index] - 1)
                    push!(vidx, col[term.scalar_term.variable])
                    push!(vv, scale[inv[term.output_index]] * term.scalar_term.coefficient)
                end
                if !isempty(afe)
                    _ok(ccall(_fn("PRIMAL_putafefentrylist"), Cint,
                              (Ptr{Cvoid}, Int64, Ptr{Int64}, Ptr{Cint}, Ptr{Cdouble}),
                              t, length(afe), afe, vidx, vv), "putafefentrylist")
                end
                for r in 1:d
                    if g.constants[r] != 0.0
                        _ok(ccall(_fn("PRIMAL_putafeg"), Cint, (Ptr{Cvoid}, Int64, Cdouble),
                                  t, base[] + inv[r] - 1, scale[inv[r]] * g.constants[r]), "putafeg")
                    end
                end
                dom = _domain!(dest, s, d)
                afelist = Int64[base[] + k2 - 1 for k2 in 1:d]
                _ok(ccall(_fn("PRIMAL_appendacc"), Cint,
                          (Ptr{Cvoid}, Int64, Int64, Ptr{Int64}, Ptr{Cdouble}),
                          t, dom, d, afelist, C_NULL), "appendacc")
                dest.acc_map[nacc] = (perm, scale)
                dest.cons[dci] = _ConInfo(_KIND_ACC, nacc, MOI.Utilities.map_indices(map, g), s)
                nacc += 1
            end
        end
    end
    for j in 0:(n - 1)
        l, u = lb[j + 1], ub[j + 1]
        _ok(ccall(_fn("PRIMAL_putvarbound"), Cint, (Ptr{Cvoid}, Cint, Cint, Cdouble, Cdouble),
                  t, j, _boundkey(l, u), l, u), "putvarbound")
        if isint[j + 1]
            _ok(ccall(_fn("PRIMAL_putvartype"), Cint, (Ptr{Cvoid}, Cint, Cint), t, j, 1), "putvartype")
        end
    end
    dest.integer = any(isint)
    # objective
    dest.sense = MOI.get(src, MOI.ObjectiveSense())
    _ok(ccall(_fn("PRIMAL_putobjsense"), Cint, (Ptr{Cvoid}, Cint), t,
              dest.sense == MOI.MAX_SENSE ? 1 : 0), "putobjsense")
    if dest.sense != MOI.FEASIBILITY_SENSE
        F = MOI.get(src, MOI.ObjectiveFunctionType())
        obj = MOI.get(src, MOI.ObjectiveFunction{F}())
        q = obj isa _SQF ? MOI.Utilities.canonical(obj) :
            MOI.Utilities.canonical(convert(_SAF, obj))
        aff = q isa _SQF ? q.affine_terms : q.terms
        for term in aff
            _ok(ccall(_fn("PRIMAL_putcj"), Cint, (Ptr{Cvoid}, Cint, Cdouble),
                      t, col[term.variable], term.coefficient), "putcj")
        end
        _ok(ccall(_fn("PRIMAL_putcfix"), Cint, (Ptr{Cvoid}, Cdouble), t, q.constant), "putcfix")
        if q isa _SQF && !isempty(q.quadratic_terms)
            # MOI: coefficient c on x_i*x_j (i != j) is the term c x_i x_j; on x_i*x_i
            # it is (c/2) x_i^2. The C API takes 1/2 x'Qx from the lower triangle,
            # which is the same coefficient in both cases.
            qi = Cint[]; qj = Cint[]; qv = Float64[]
            for term in q.quadratic_terms
                a, b = col[term.variable_1], col[term.variable_2]
                push!(qi, max(a, b)); push!(qj, min(a, b)); push!(qv, term.coefficient)
            end
            _ok(ccall(_fn("PRIMAL_putqobj"), Cint, (Ptr{Cvoid}, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}),
                      t, length(qi), qi, qj, qv), "putqobj")
        end
    end
    return map
end

# ---- optimize! ------------------------------------------------------------------------
function MOI.optimize!(model::Optimizer)
    t = _task(model)
    _ok(ccall(_fn("PRIMAL_putintparam"), Cint, (Ptr{Cvoid}, Cint, Cint), t,
              _IPAR_LOG, model.silent ? 0 : 1), "putintparam(LOG)")
    _ok(ccall(_fn("PRIMAL_putdouparam"), Cint, (Ptr{Cvoid}, Cint, Cdouble), t,
              _DPAR_OPTIMIZER_MAX_TIME,
              model.time_limit === nothing ? -1.0 : model.time_limit), "putdouparam(MAX_TIME)")
    for (name, value) in model.raw
        rc = if value isa Integer
            ccall(_fn("PRIMAL_putnaintparam"), Cint, (Ptr{Cvoid}, Cstring, Cint), t, name, value)
        elseif value isa Real
            ccall(_fn("PRIMAL_putnadouparam"), Cint, (Ptr{Cvoid}, Cstring, Cdouble), t, name, value)
        else
            error("Primal: parameter $name must be an integer or a real number")
        end
        rc == 0 || error("Primal: cannot set parameter $name = $value (rc=$rc): expected a ",
                         "PRIMAL_IPAR_* name with an integer or a PRIMAL_DPAR_* name with a ",
                         "real value, within its range")
    end
    start = time_ns()
    model.rc = Int(ccall(_fn("PRIMAL_optimize"), Cint, (Ptr{Cvoid},), t))
    model.solve_time = (time_ns() - start) / 1e9
    model.optimized = true
    model.x = fill(NaN, model.nvar)
    if _has_solution(model) && model.nvar > 0
        xx = fill(NaN, _numvar(t))
        _ok(ccall(_fn("PRIMAL_getxx"), Cint, (Ptr{Cvoid}, Cint, Ptr{Cdouble}), t,
                  _soltype(model), xx), "getxx")
        copyto!(model.x, 1, xx, 1, model.nvar)
    end
    return nothing
end

_soltype(model::Optimizer) = model.integer ? _SOL_ITG : _SOL_ITR

function _has_solution(model::Optimizer)
    model.optimized || return false
    d = Ref{Cint}(0)
    rc = ccall(_fn("PRIMAL_solutiondef"), Cint, (Ptr{Cvoid}, Cint, Ptr{Cint}),
               _task(model), _soltype(model), d)
    return rc == 0 && d[] != 0
end

function _solsta(model::Optimizer)
    _has_solution(model) || return 0
    s = Ref{Cint}(0)
    ccall(_fn("PRIMAL_getsolsta"), Cint, (Ptr{Cvoid}, Cint, Ptr{Cint}), _task(model), _soltype(model), s)
    return Int(s[])
end

function _prosta(model::Optimizer)
    _has_solution(model) || return 0
    s = Ref{Cint}(0)
    ccall(_fn("PRIMAL_getprosta"), Cint, (Ptr{Cvoid}, Cint, Ptr{Cint}), _task(model), _soltype(model), s)
    return Int(s[])
end

# ---- status -----------------------------------------------------------------------------
function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    model.optimized || return MOI.OPTIMIZE_NOT_CALLED
    rc = model.rc
    sol, pro = _solsta(model), _prosta(model)
    if sol == 1 || sol == 9
        return MOI.OPTIMAL
    elseif sol == 5 || pro == 4
        return MOI.INFEASIBLE
    elseif sol == 6 || pro == 5
        return MOI.DUAL_INFEASIBLE
    elseif pro == 8
        return MOI.INFEASIBLE_OR_UNBOUNDED
    elseif rc == 1002
        return MOI.INFEASIBLE
    elseif rc == 1003
        return MOI.DUAL_INFEASIBLE
    elseif rc == 1007
        return MOI.ITERATION_LIMIT
    elseif rc == 1008
        return MOI.TIME_LIMIT
    elseif rc == 1009
        return MOI.OBJECTIVE_LIMIT
    elseif rc == 1004
        return MOI.MEMORY_LIMIT
    elseif sol == 2
        return MOI.OTHER_LIMIT
    end
    return MOI.OTHER_ERROR
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    model.optimized || return "optimize not called"
    return "rc=$(model.rc) prosta=$(_prosta(model)) solsta=$(_solsta(model))"
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    attr.result_index == 1 || return MOI.NO_SOLUTION
    sol = _solsta(model)
    if sol in (1, 2, 9)
        return MOI.FEASIBLE_POINT
    elseif sol == 6
        return MOI.INFEASIBILITY_CERTIFICATE
    end
    return MOI.NO_SOLUTION
end

function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    attr.result_index == 1 || return MOI.NO_SOLUTION
    model.integer && return MOI.NO_SOLUTION
    sol = _solsta(model)
    if sol == 1
        return MOI.FEASIBLE_POINT
    elseif sol == 5
        return MOI.INFEASIBILITY_CERTIFICATE
    end
    return MOI.NO_SOLUTION
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return MOI.get(model, MOI.PrimalStatus()) == MOI.NO_SOLUTION &&
           MOI.get(model, MOI.DualStatus()) == MOI.NO_SOLUTION ? 0 : 1
end

MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time

# ---- values ---------------------------------------------------------------------------
function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    v = Ref{Cdouble}(0.0)
    _ok(ccall(_fn("PRIMAL_getprimalobj"), Cint, (Ptr{Cvoid}, Cint, Ptr{Cdouble}),
              _task(model), _soltype(model), v), "getprimalobj")
    return v[]
end

function MOI.get(model::Optimizer, attr::MOI.DualObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    v = Ref{Cdouble}(0.0)
    _ok(ccall(_fn("PRIMAL_getdualobj"), Cint, (Ptr{Cvoid}, Cint, Ptr{Cdouble}),
              _task(model), _soltype(model), v), "getdualobj")
    return v[]
end

# Best bound: the dual objective of a continuous solve, the branch-and-bound
# bound (DINF_MIO_OBJ_BOUND) of an integer one.
const _DINF_MIO_OBJ_BOUND = 36
function MOI.get(model::Optimizer, ::MOI.ObjectiveBound)
    v = Ref{Cdouble}(0.0)
    _ok(ccall(_fn("PRIMAL_getdouinf"), Cint, (Ptr{Cvoid}, Cint, Ptr{Cdouble}),
              _task(model), _DINF_MIO_OBJ_BOUND, v), "getdouinf(MIO_OBJ_BOUND)")
    return v[]
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    return model.x[vi.value]
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex)
    MOI.check_result_index_bounds(model, attr)
    info = model.cons[ci]
    return MOI.Utilities.eval_variables(vi -> model.x[vi.value], info.func)
end

# Duals are read in the C API's convention and given MOI's: a >= constraint has a
# nonnegative dual and a <= one a nonpositive dual whatever the sense, a cone's
# dual lies in the dual cone, and a maximisation's duals are those of minimising
# the negated objective. The C layer's convention, measured on small problems
# with known duals (see test/MOI_wrapper.jl), differs by sense and by kind:
#   rows, Zeros/Nonnegatives/Nonpositives  MOI = -y      (MIN)   +y      (MAX)
#   nonlinear cones (SOC, RSOC, exp, pow)  MOI = +doty   (MIN)   -doty   (MAX)
#   variable bounds                        sl = -slx, su = -sux  (MIN)
#                                          sl = +sux, su = +slx  (MAX)
_dual_sign(model::Optimizer) = model.sense == MOI.MAX_SENSE ? 1.0 : -1.0

_linear_cone(s) = s isa Union{MOI.Zeros,MOI.Nonnegatives,MOI.Nonpositives}

function _getvec(model::Optimizer, name::String, len::Int)
    v = zeros(len)
    len == 0 && return v
    _ok(ccall(_fn(name), Cint, (Ptr{Cvoid}, Cint, Ptr{Cdouble}),
              _task(model), _soltype(model), v), name)
    return v
end

function MOI.get(model::Optimizer, attr::MOI.ConstraintDual, ci::MOI.ConstraintIndex)
    MOI.check_result_index_bounds(model, attr)
    info = model.cons[ci]
    sgn = _dual_sign(model)
    if info.kind == _KIND_BOUND
        j = info.index + 1
        slx = _getvec(model, "PRIMAL_getslx", _numvar(_task(model)))[j]
        sux = _getvec(model, "PRIMAL_getsux", _numvar(_task(model)))[j]
        sl, su = model.sense == MOI.MAX_SENSE ? (sux, slx) : (-slx, -sux)
        s = info.set
        if s isa MOI.GreaterThan
            return sl
        elseif s isa MOI.LessThan
            return su
        end
        return sl + su
    elseif info.kind == _KIND_ROW
        y = _getvec(model, "PRIMAL_gety", _numcon(_task(model)))
        return sgn * y[info.index + 1]
    end
    perm, scale = model.acc_map[info.index]
    d = length(perm)
    doty = zeros(d)
    _ok(ccall(_fn("PRIMAL_getaccdoty"), Cint, (Ptr{Cvoid}, Cint, Int64, Ptr{Cdouble}),
              _task(model), _soltype(model), info.index, doty), "getaccdoty")
    csgn = _linear_cone(info.set) ? sgn : -sgn
    out = zeros(d)
    for k in 1:d
        out[perm[k]] = csgn * scale[k] * doty[k]
    end
    return out
end
