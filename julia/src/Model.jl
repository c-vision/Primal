# Model.jl — High-level Model wrapper (mirrors the Python primalsolver.Model).
# Uses direct ccall + dlsym for full control over C types.

mutable struct Model
    env::Ref{Ptr{Cvoid}}
    task::Ref{Ptr{Cvoid}}
    closed::Bool
end

"""
    Model(maxcon=0, maxvar=0)

Create a solver environment and a task with preallocated capacity hints.
Use as `with(Model()) do m; ...; end` or call `free(m)` manually.
"""
function Model(maxcon::Int=0, maxvar::Int=0)
    env = Ref{Ptr{Cvoid}}(C_NULL)
    task = Ref{Ptr{Cvoid}}(C_NULL)
    rc = ccall(Libdl.dlsym(_lib(), "PRIMAL_makeenv"), Cint,
               (Ptr{Ptr{Cvoid}}, Ptr{Cvoid}), env, C_NULL)
    rc == RC_OK || error("PRIMAL_makeenv failed (rc=$rc)")
    rc = ccall(Libdl.dlsym(_lib(), "PRIMAL_maketask"), Cint,
               (Ptr{Cvoid}, Cint, Cint, Ptr{Ptr{Cvoid}}),
               env[], Int32(maxcon), Int32(maxvar), task)
    rc == RC_OK || error("PRIMAL_maketask failed (rc=$rc)")
    return Model(env, task, false)
end

function free(m::Model)
    if !m.closed
        if m.task[] !== C_NULL
            ccall(Libdl.dlsym(_lib(), "PRIMAL_deletetask"), Cint,
                  (Ptr{Ptr{Cvoid}},), m.task)
            m.task[] = C_NULL
        end
        if m.env[] !== C_NULL
            ccall(Libdl.dlsym(_lib(), "PRIMAL_deleteenv"), Cint,
                  (Ptr{Ptr{Cvoid}},), m.env)
            m.env[] = C_NULL
        end
        m.closed = true
    end
    return nothing
end

# convenience: `with(Model()) do m ... end`
macro with_model(fcall)
    return esc(quote
        local _m = Model(0, 0)
        try
            $fcall
        finally
            free(_m)
        end
    end)
end

# ---- helpers ----------------------------------------------------------------
const _Cint = Cint
const _Cdouble = Cdouble

_ivec(seq::Vector{Int}) = Int32[Int32(i) for i in seq]
_dvec(seq) = Float64[Float64(v) for v in seq]

_check(rc::Integer, what::String = "") = Int(rc) == RC_OK || error("PRIMAL $what failed (rc=$(Int(rc)))")

_numcon(m::Model) = begin
    n = Ref{Cint}(0)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_getnumcon"), Cint,
                 (Ptr{Cvoid}, Ptr{Cint}), m.task[], n), "getnumcon")
    Int(n[])
end
_numvar(m::Model) = begin
    n = Ref{Cint}(0)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_getnumvar"), Cint,
                 (Ptr{Cvoid}, Ptr{Cint}), m.task[], n), "getnumvar")
    Int(n[])
end

# ---- model building ---------------------------------------------------------
add_cons(m::Model, n::Int) = _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_appendcons"), Cint,
                                          (Ptr{Cvoid}, Cint), m.task[], Int32(n)), "appendcons")
add_vars(m::Model, n::Int) = _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_appendvars"), Cint,
                                          (Ptr{Cvoid}, Cint), m.task[], Int32(n)), "appendvars")

function obj(m::Model, idx, vals)
    idx = collect(Int, idx); vals = collect(Float64, vals)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putclist"), Cint,
                 (Ptr{Cvoid}, Cint, Ptr{Cint}, Ptr{Cdouble}),
                 m.task[], Int32(length(idx)), _ivec(idx), _dvec(vals)), "putclist")
    return nothing
end

function a_ij(m::Model, i::Int, j::Int, value::Real)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putaij"), Cint,
                 (Ptr{Cvoid}, Cint, Cint, Cdouble),
                 m.task[], Int32(i), Int32(j), Float64(value)), "putaij")
    return nothing
end

function a_row(m::Model, i::Int, cols, vals)
    cols = collect(Int, cols); vals = collect(Float64, vals)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putarow"), Cint,
                 (Ptr{Cvoid}, Cint, Cint, Ptr{Cint}, Ptr{Cdouble}),
                 m.task[], Int32(i), Int32(length(cols)), _ivec(cols), _dvec(vals)), "putarow")
    return nothing
end

function cj(m::Model, j::Int, value::Real)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putcj"), Cint,
                 (Ptr{Cvoid}, Cint, Cdouble),
                 m.task[], Int32(j), Float64(value)), "putcj")
    return nothing
end

function qobj(m::Model, terms)
    terms = collect(terms)
    qi = _ivec([Int(t[1]) for t in terms])
    qj = _ivec([Int(t[2]) for t in terms])
    qv = _dvec([Float64(t[3]) for t in terms])
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putqobj"), Cint,
                 (Ptr{Cvoid}, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}),
                 m.task[], Int32(length(terms)), qi, qj, qv), "putqobj")
    return nothing
end

function cone(m::Model, ctype::Int, members; param::Real=0.0)
    members = collect(Int, members)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_appendcone"), Cint,
                 (Ptr{Cvoid}, Cint, Cdouble, Cint, Ptr{Cint}),
                 m.task[], Int32(ctype), Float64(param), Int32(length(members)), _ivec(members)), "appendcone")
    return nothing
end

function a_col(m::Model, j::Int, rows, vals)
    rows = collect(Int, rows); vals = collect(Float64, vals)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putacol"), Cint,
                 (Ptr{Cvoid}, Cint, Cint, Ptr{Cint}, Ptr{Cdouble}),
                 m.task[], Int32(j), Int32(length(rows)), _ivec(rows), _dvec(vals)), "putacol")
    return nothing
end

function qconk(m::Model, k::Int, terms)
    terms = collect(terms)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putqconk"), Cint,
                 (Ptr{Cvoid}, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}),
                 m.task[], Int32(k), Int32(length(terms)),
                 _ivec([Int(t[1]) for t in terms]),
                 _ivec([Int(t[2]) for t in terms]),
                 _dvec([Float64(t[3]) for t in terms])), "putqconk")
    return nothing
end

function qobj_ij(m::Model, i::Int, j::Int, value::Real)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putqobjij"), Cint,
                 (Ptr{Cvoid}, Cint, Cint, Cdouble),
                 m.task[], Int32(i), Int32(j), Float64(value)), "putqobjij")
    return nothing
end

function var_type(m::Model, j::Int, vt::Int)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putvartype"), Cint,
                 (Ptr{Cvoid}, Cint, Cint),
                 m.task[], Int32(j), Int32(vt)), "putvartype")
    return nothing
end

function var_types(m::Model, cols, types)
    cols = collect(Int, cols); types = collect(Int, types)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putvartypelist"), Cint,
                 (Ptr{Cvoid}, Cint, Ptr{Cint}, Ptr{Cint}),
                 m.task[], Int32(length(cols)), _ivec(cols), Int32[Int32(t) for t in types]), "putvartypelist")
    return nothing
end

function cfix(m::Model, value::Real)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putcfix"), Cint,
                 (Ptr{Cvoid}, Cdouble),
                 m.task[], Float64(value)), "putcfix")
    return nothing
end

function int_param(m::Model, param::Int, value::Int)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putintparam"), Cint,
                 (Ptr{Cvoid}, Cint, Cint),
                 m.task[], Int32(param), Int32(value)), "putintparam")
    return nothing
end

function dou_param(m::Model, param::Int, value::Real)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putdouparam"), Cint,
                 (Ptr{Cvoid}, Cint, Cdouble),
                 m.task[], Int32(param), Float64(value)), "putdouparam")
    return nothing
end

function warm_start(m::Model; x=nothing, y=nothing)
    if x !== nothing
        _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putxx"), Cint,
                     (Ptr{Cvoid}, Cint, Ptr{Cdouble}),
                     m.task[], SOL_ITR, _dvec(collect(Float64, x))), "putxx")
    end
    if y !== nothing
        _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_puty"), Cint,
                     (Ptr{Cvoid}, Cint, Ptr{Cdouble}),
                     m.task[], SOL_ITR, _dvec(collect(Float64, y))), "puty")
    end
    return nothing
end

function sos(m::Model, kind::Int, members; weights=nothing)
    members = collect(Int, members)
    w = weights === nothing ? Float64[Float64(i + 1) for i in 1:length(members)] : collect(Float64, weights)
    fn = kind == 1 ? "PRIMAL_appendsos1" : "PRIMAL_appendsos2"
    _check(ccall(Libdl.dlsym(_lib(), fn), Cint,
                 (Ptr{Cvoid}, Cint, Ptr{Cint}, Ptr{Cdouble}),
                 m.task[], Int32(length(members)),
                 convert(Vector{Cint}, members),
                 convert(Vector{Cdouble}, w)), fn)
    return nothing
end

# ---- SDP (semi-definite bar variables) -------------------------------------
function barvars(m::Model, dims)
    dims = collect(Int, dims)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_appendbarvars"), Cint,
                 (Ptr{Cvoid}, Cint, Ptr{Cint}),
                 m.task[], Int32(length(dims)), _ivec(dims)), "appendbarvars")
    return nothing
end

function sparsesymmat(m::Model, dim::Int, terms)
    terms = collect(terms)
    idx = Ref{Cint}(0)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_appendsparsesymmat"), Cint,
                 (Ptr{Cvoid}, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cint}),
                 m.task[], Int32(dim), Int32(length(terms)),
                 _ivec([Int(t[1]) for t in terms]),
                 _ivec([Int(t[2]) for t in terms]),
                 _dvec([Float64(t[3]) for t in terms]), idx), "appendsparsesymmat")
    return Int(idx[])
end

function baraij(m::Model, i::Int, j::Int, terms)
    terms = collect(terms)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putbaraij"), Cint,
                 (Ptr{Cvoid}, Cint, Cint, Cint, Ptr{Cint}, Ptr{Cdouble}),
                 m.task[], Int32(i), Int32(j), Int32(length(terms)),
                 _ivec([Int(t[1]) for t in terms]),
                 _dvec([Float64(t[2]) for t in terms])), "putbaraij")
    return nothing
end

function barcj(m::Model, j::Int, terms)
    terms = collect(terms)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putbarcj"), Cint,
                 (Ptr{Cvoid}, Cint, Cint, Ptr{Cint}, Ptr{Cdouble}),
                 m.task[], Int32(j), Int32(length(terms)),
                 _ivec([Int(t[1]) for t in terms]),
                 _dvec([Float64(t[2]) for t in terms])), "putbarcj")
    return nothing
end

# ---- AFE / ACC / DJC --------------------------------------------------------
function afes(m::Model, num::Int)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_appendafes"), Cint,
                 (Ptr{Cvoid}, Int64), m.task[], Int64(num)), "appendafes")
    return nothing
end

function afe_entry(m::Model, i::Int, j::Int, value::Real)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putafefentry"), Cint,
                 (Ptr{Cvoid}, Int64, Cint, Cdouble),
                 m.task[], Int64(i), Int32(j), Float64(value)), "putafefentry")
    return nothing
end

function afe_g(m::Model, i::Int, value::Real)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putafeg"), Cint,
                 (Ptr{Cvoid}, Int64, Cdouble),
                 m.task[], Int64(i), Float64(value)), "putafeg")
    return nothing
end

function domain(m::Model, kind::Int; n::Int=1)
    fn = kind == 0 ? "PRIMAL_appendrdomain" :
         kind == 1 ? "PRIMAL_appendrzerodomain" :
         kind == 2 ? "PRIMAL_appendrplusdomain" :
         kind == 3 ? "PRIMAL_appendrminusdomain" : error("unknown domain kind $kind")
    idx = Ref{Int64}(0)
    _check(ccall(Libdl.dlsym(_lib(), fn), Cint,
                 (Ptr{Cvoid}, Int64, Ptr{Int64}),
                 m.task[], Int64(n), idx), fn)
    return Int(idx[])
end

function acc(m::Model, domidx::Int, afeidx; b=nothing)
    afeidx = collect(Int, afeidx)
    bb = b === nothing ? Float64[0.0 for _ in afeidx] : collect(Float64, b)
    a64 = Int64[Int64(a) for a in afeidx]
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_appendacc"), Cint,
                 (Ptr{Cvoid}, Int64, Int64, Ptr{Int64}, Ptr{Cdouble}),
                 m.task[], Int64(domidx), Int64(length(afeidx)), a64, bb), "appendacc")
    return nothing
end

function djcs(m::Model, num::Int)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_appenddjcs"), Cint,
                 (Ptr{Cvoid}, Int64), m.task[], Int64(num)), "appenddjcs")
    return nothing
end

function djc(m::Model, idx::Int, domidxlist, afeidxlist, b, termsizelist)
    dom = Int64[Int64(d) for d in domidxlist]
    afe = Int64[Int64(a) for a in afeidxlist]
    bv = collect(Float64, b)
    ts = Int64[Int64(t) for t in termsizelist]
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putdjc"), Cint,
                 (Ptr{Cvoid}, Int64, Int64, Ptr{Int64}, Int64, Ptr{Int64}, Ptr{Cdouble}, Int64, Ptr{Int64}),
                 m.task[], Int64(idx), Int64(length(domidxlist)), dom,
                 Int64(length(afeidxlist)), afe, bv, Int64(length(termsizelist)), ts), "putdjc")
    return nothing
end

# ---- data getters -----------------------------------------------------------
getcj(m::Model, j::Int) = begin
    v = Ref{Cdouble}(0.0)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_getcj"), Cint,
                 (Ptr{Cvoid}, Cint, Ptr{Cdouble}),
                 m.task[], Int32(j), v), "getcj")
    return Float64(v[])
end

getaij(m::Model, i::Int, j::Int) = begin
    v = Ref{Cdouble}(0.0)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_getaij"), Cint,
                 (Ptr{Cvoid}, Cint, Cint, Ptr{Cdouble}),
                 m.task[], Int32(i), Int32(j), v), "getaij")
    return Float64(v[])
end

function getvarbound(m::Model, j::Int)
    bk = Ref{Cint}(0); bl = Ref{Cdouble}(0.0); bu = Ref{Cdouble}(0.0)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_getvarbound"), Cint,
                 (Ptr{Cvoid}, Cint, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}),
                 m.task[], Int32(j), bk, bl, bu))
    return Int(bk[]), Float64(bl[]), Float64(bu[])
end

function getconbound(m::Model, i::Int)
    bk = Ref{Cint}(0); bl = Ref{Cdouble}(0.0); bu = Ref{Cdouble}(0.0)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_getconbound"), Cint,
                 (Ptr{Cvoid}, Cint, Ptr{Cint}, Ptr{Cdouble}, Ptr{Cdouble}),
                 m.task[], Int32(i), bk, bl, bu))
    return Int(bk[]), Float64(bl[]), Float64(bu[])
end

getcfix(m::Model) = begin
    v = Ref{Cdouble}(0.0)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_getcfix"), Cint,
                 (Ptr{Cvoid}, Ptr{Cdouble}), m.task[], v), "getcfix")
    return Float64(v[])
end

getdualinfeas(m::Model, which::Int=SOL_ITR) = begin
    v = Ref{Cdouble}(0.0)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_getdualinfeas"), Cint,
                 (Ptr{Cvoid}, Cint, Ptr{Cdouble}),
                 m.task[], Int32(which), v), "getdualinfeas")
    return Float64(v[])
end

function getreducedcosts(m::Model, first::Int=0, last::Int=-1)
    if last == -1
        last = _numvar(m)
    end
    buf = Vector{Float64}(undef, last - first)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_getreducedcosts"), Cint,
                 (Ptr{Cvoid}, Cint, Cint, Cint, Ptr{Cdouble}),
                 m.task[], SOL_ITR, Int32(first), Int32(last), buf), "getreducedcosts")
    return buf
end

getintparam(m::Model, param::Int) = begin
    v = Ref{Cint}(0)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_getintparam"), Cint,
                 (Ptr{Cvoid}, Cint, Ptr{Cint}),
                 m.task[], Int32(param), v), "getintparam")
    return Int(v[])
end

getdouparam(m::Model, param::Int) = begin
    v = Ref{Cdouble}(0.0)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_getdouparam"), Cint,
                 (Ptr{Cvoid}, Cint, Ptr{Cdouble}),
                 m.task[], Int32(param), v), "getdouparam")
    return Float64(v[])
end

numvar(m::Model) = _numvar(m)
numcon(m::Model) = _numcon(m)

function getbarxj(m::Model, j::Int=0)
    dim = Ref{Cint}(0)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_getdimbarvarj"), Cint,
                 (Ptr{Cvoid}, Cint, Ptr{Cint}), m.task[], Int32(j), dim), "getdimbarvarj")
    buf = Vector{Float64}(undef, Int(dim[])^2)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_getbarxj"), Cint,
                 (Ptr{Cvoid}, Cint, Cint, Ptr{Cdouble}),
                 m.task[], SOL_ITR, Int32(j), buf), "getbarxj")
    return buf
end

function getbarsj(m::Model, j::Int=0)
    dim = Ref{Cint}(0)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_getdimbarvarj"), Cint,
                 (Ptr{Cvoid}, Cint, Ptr{Cint}), m.task[], Int32(j), dim), "getdimbarvarj")
    buf = Vector{Float64}(undef, Int(dim[])^2)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_getbarsj"), Cint,
                 (Ptr{Cvoid}, Cint, Cint, Ptr{Cdouble}),
                 m.task[], SOL_ITR, Int32(j), buf), "getbarsj")
    return buf
end

function var_bounds(m::Model, j::Int, bk::Int; lo::Real=0.0, up::Real=0.0)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putvarbound"), Cint,
                 (Ptr{Cvoid}, Cint, Cint, Cdouble, Cdouble),
                 m.task[], Int32(j), Int32(bk), Float64(lo), Float64(up)), "putvarbound")
    return nothing
end

function con_bounds(m::Model, i::Int, bk::Int; lo::Real=0.0, up::Real=0.0)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putconbound"), Cint,
                 (Ptr{Cvoid}, Cint, Cint, Cdouble, Cdouble),
                 m.task[], Int32(i), Int32(bk), Float64(lo), Float64(up)), "putconbound")
    return nothing
end

function sense(m::Model, sense::Int)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_putobjsense"), Cint,
                 (Ptr{Cvoid}, Cint),
                 m.task[], Int32(sense)), "putobjsense")
    return nothing
end

# ---- file I/O ---------------------------------------------------------------
function read(m::Model, path::AbstractString)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_readdataautoformat"), Cint,
                 (Ptr{Cvoid}, Cstring),
                 m.task[], String(path)), "readdataautoformat")
    return nothing
end

function write(m::Model, path::AbstractString)
    _check(ccall(Libdl.dlsym(_lib(), "PRIMAL_writedata"), Cint,
                 (Ptr{Cvoid}, Cstring),
                 m.task[], String(path)), "writedata")
    return nothing
end

# ---- solve ------------------------------------------------------------------
function solve(m::Model, which::Int=SOL_ITR)
    rc = Int(ccall(Libdl.dlsym(_lib(), "PRIMAL_optimize"), Cint, (Ptr{Cvoid},), m.task[]))
    nv = _numvar(m)
    nc = _numcon(m)
    x = nv > 0 ? Vector{Float64}(undef, nv) : Float64[]
    y = nc > 0 ? Vector{Float64}(undef, nc) : Float64[]
    obj = Ref{Cdouble}(0.0)
    dual = Ref{Cdouble}(0.0)
    pinf = Ref{Cdouble}(0.0)
    sta = Ref{Cint}(0)
    pro = Ref{Cint}(0)

    if nv > 0
        ccall(Libdl.dlsym(_lib(), "PRIMAL_getxx"), Cint,
              (Ptr{Cvoid}, Cint, Ptr{Cdouble}), m.task[], Int32(which), x)
    end
    if nc > 0
        ccall(Libdl.dlsym(_lib(), "PRIMAL_gety"), Cint,
              (Ptr{Cvoid}, Cint, Ptr{Cdouble}), m.task[], Int32(which), y)
    end
    ccall(Libdl.dlsym(_lib(), "PRIMAL_getprimalobj"), Cint,
          (Ptr{Cvoid}, Cint, Ptr{Cdouble}), m.task[], Int32(which), obj)
    ccall(Libdl.dlsym(_lib(), "PRIMAL_getdualobj"), Cint,
          (Ptr{Cvoid}, Cint, Ptr{Cdouble}), m.task[], Int32(which), dual)
    ccall(Libdl.dlsym(_lib(), "PRIMAL_getprimalinfeas"), Cint,
          (Ptr{Cvoid}, Cint, Ptr{Cdouble}), m.task[], Int32(which), pinf)
    ccall(Libdl.dlsym(_lib(), "PRIMAL_getsolsta"), Cint,
          (Ptr{Cvoid}, Cint, Ptr{Cint}), m.task[], Int32(which), sta)
    ccall(Libdl.dlsym(_lib(), "PRIMAL_getprosta"), Cint,
          (Ptr{Cvoid}, Cint, Ptr{Cint}), m.task[], Int32(which), pro)

    return Result(Float64[x for x in x], Float64[y for y in y],
                  Float64(obj[]), Float64(dual[]),
                  Int(sta[]), Int(pro[]), Float64(pinf[]), rc)
end
