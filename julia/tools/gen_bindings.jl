#!/usr/bin/env julia
# gen_bindings.jl — Generate src/bindings.jl from the C public header primal.h.
#
# Parses every PRIMAL_* prototype and emits a 1:1 ccall binding for the whole
# public surface, exactly like python/tools/gen_bindings.py does for ctypes.
#
# Usage:
#   julia tools/gen_bindings.jl                 # reads ../primal.h (live C repo)
#   PRIMAL_C_SRC=/path/to/Primal julia tools/gen_bindings.jl

using Printf

const PRIMAL_C_SRC = get(ENV, "PRIMAL_C_SRC", nothing)
const HERE = @__DIR__
const ROOT = dirname(dirname(HERE))          # .../PrimalSolver/julia/tools -> .../PrimalSolver
# PRIMAL_C_SRC names the C source directory, as in deps/build.jl and the usage above.
const HEADER = joinpath(something(PRIMAL_C_SRC, ROOT), "primal.h")
const PKG_ROOT = dirname(HERE)               # .../PrimalSolver/julia
const OUT = joinpath(PKG_ROOT, "src", "bindings.jl")

isfile(HEADER) || error("primal.h not found at $HEADER (set PRIMAL_C_SRC)")

# ---- C type -> Julia ccall type ---------------------------------------------
const CTYPE_MAP = Dict{String,String}(
    "void"                => "Cvoid",
    "int"                 => "Cint",
    "double"              => "Cdouble",
    "float"               => "Cfloat",
    "char"                => "Cchar",
    "size_t"              => "Csize_t",
    "unsigned"            => "Cuint",
    "long long"           => "Clonglong",
    "PRIMALenv_t"         => "Ptr{Cvoid}",
    "PRIMALtask_t"        => "Ptr{Cvoid}",
    "PRIMALrealt"         => "Cdouble",
    "PRIMALint64t"        => "Clonglong",
    "PRIMALint32t"        => "Cint",
    "PRIMALrescodee"      => "Cint",
    "PRIMALsolstae"       => "Cint",
    "PRIMALprostae"       => "Cint",
    "PRIMALboundkeye"     => "Cint",
    "PRIMALobjsensee"     => "Cint",
    "PRIMALconetypee"     => "Cint",
    "PRIMALvariabletypee" => "Cint",
    "PRIMALsolt"          => "Cint",
    "PRIMALoptimizer"     => "Cint",
    "PRIMALdataformate"   => "Cint",
    "PRIMALcompresstypee" => "Cint",
    "PRIMALnametypee"     => "Cint",
    "PRIMALinftypee"      => "Cint",
    "PRIMALstreamtypee"   => "Cint",
    "PRIMALproblemtypee"  => "Cint",
    "PRIMALstreamfunc"    => "Ptr{Cvoid}",
    "PRIMALexitfunc"      => "Ptr{Cvoid}",
)

const VARIADIC = Set(["PRIMAL_echotask", "PRIMAL_echoenv"])

function jl_type(ctype::String)::String
    t = strip(ctype)
    t = replace(t, r"\b(const|volatile|struct)\b" => "")
    t = replace(t, r"\s+" => " ")
    t = strip(t)

    stars = count('*', t)
    base = strip(replace(t, "*" => ""))
    base = replace(base, r"\s+" => " ")
    base = strip(base)

    inner = get(CTYPE_MAP, base, base)
    # Catch-all: any C enum type (ends in 'e') or unknown type -> Cint/Cvoid pointer
    if inner == base && !haskey(CTYPE_MAP, base)
        if endswith(base, "e")
            inner = "Cint"
        else
            inner = "Ptr{Cvoid}"
        end
    end
    if base == "char"
        return stars == 0 ? "Cchar" : (stars == 1 ? "Cstring" : "Ptr{Ptr{Cchar}}")
    end
    if stars == 0
        return inner
    elseif stars == 1
        return "Ptr{$inner}"
    else
        inner_jl = "Ptr{$inner}"
        for _ in 2:stars
            inner_jl = "Ptr{$inner_jl}"
        end
        return inner_jl
    end
end

function strip_param_name(p::String)::String
    p = strip(p)
    m = match(r"^(.*?[\s\*])([A-Za-z_]\w*)$", p)
    return m !== nothing ? strip(m.captures[1]) : p
end

function main()
    raw = read(HEADER, String)
    no_comments = replace(raw, r"/\*.*?\*/"s => " ")
    no_comments = replace(no_comments, r"//[^\n]*" => " ")
    no_comments = replace(no_comments, r"^\s*#.*$"m => " ")
    no_comments = replace(no_comments, r"extern\s+\"C\"" => " ")

    n = length(no_comments)

    # ---- Collect prototype statements ---------------------------------------
    # Match PRIMAL_name(  (name immediately followed by "(") — excludes typedefs.
    protos = Tuple{String,String,Vector{String},Bool}[]

    for m in eachmatch(r"PRIMAL_[A-Za-z0-9_]*\s*\(", no_comments)
        # Extract the pure function name (strip trailing "(")
        name_match = match(r"PRIMAL_[A-Za-z0-9_]+", m.match)
        name_match === nothing && continue
        name = name_match.match
        m_start = m.offset

        # Return type: text between previous ";" or "}" and the function name
        search_from = max(1, m_start - 300)
        preceding = no_comments[search_from:m_start-1]
        seps = findall(r"[;}]", preceding)
        if isempty(seps)
            ret_start = search_from
        else
            last_sep = last(seps)
            ret_start = search_from + last(last_sep)  # skip past the ; or }
        end
        ret_raw = strip(no_comments[ret_start:m_start-1])
        ret_raw = replace(ret_raw, r"^(PRIMALAPI|static|inline|extern)\s+" => "")
        ret_raw = strip(ret_raw)

        # Find opening paren after the name
        paren_search = match(r"\s*\(", no_comments[m_start:end])
        paren_search === nothing && continue
        paren_open = m_start + paren_search.offset - 1

        # Scan to matching ")" then ";"
        depth = 1
        i = paren_open + 1
        close_paren = 0
        while i <= n
            c = no_comments[i]
            if c == '('
                depth += 1
            elseif c == ')'
                depth -= 1
                if depth == 0
                    close_paren = i
                    break
                end
            end
            i += 1
        end
        close_paren == 0 && continue

        j = close_paren + 1
        while j <= n && no_comments[j] in " \t\n\r"
            j += 1
        end
        if j > n || no_comments[j] != ';'
            continue
        end

        params_raw = strip(no_comments[paren_open+1:close_paren-1])
        variadic = occursin(r"\.\.\.", params_raw)
        params_raw = replace(params_raw, r"\.\.\." => "")
        params_raw = strip(params_raw, [' ', ','])

        args = String[]
        if !isempty(params_raw) && params_raw != "void"
            depth = 0
            cur = IOBuffer()
            for c in params_raw
                if c == '('
                    depth += 1
                    write(cur, c)
                elseif c == ')'
                    depth -= 1
                    write(cur, c)
                elseif c == ',' && depth == 0
                    push!(args, strip(String(take!(cur))))
                else
                    write(cur, c)
                end
            end
            last_arg = strip(String(take!(cur)))
            if !isempty(last_arg)
                push!(args, last_arg)
            end
        end

        # A prototype the header repeats (legal C, e.g. PRIMAL_putcfix) is
        # bound once: Julia >= 1.12 refuses a method overwritten during
        # precompilation, so a duplicate made the whole package unloadable.
        any(p -> p[1] == name, protos) && continue
        push!(protos, (name, ret_raw, args, variadic))
    end

    # ---- Generate bindings.jl -----------------------------------------------
    lines = String[]
    push!(lines, "# AUTO-GENERATED by tools/gen_bindings.jl from primal.h -- DO NOT EDIT.")
    push!(lines, "# Regenerate after any change to the C public header.")
    push!(lines, "")
    push!(lines, "module Bindings")
    push!(lines, "")
    push!(lines, "using Libdl")
    push!(lines, "")
    push!(lines, "###############################################################################")
    push!(lines, "# Low-level ccall binding to libprimal (the PrimalSolver C99 shared library).")
    push!(lines, "# Every PRIMAL_* function is bound here with its exact C signature.")
    push!(lines, "# Opaque handles (PRIMALenv_t / PRIMALtask_t) are Ptr{Cvoid}.")
    push!(lines, "###############################################################################")
    push!(lines, "")
    push!(lines, "const PRIMAL_LIBNAME = Sys.iswindows() ? \"libprimal.dll\" :")
    push!(lines, "                 Sys.isapple()   ? \"libprimal.dylib\" : \"libprimal.so\"")
    push!(lines, "")
    push!(lines, "const _PKG_DIR = dirname(dirname(@__FILE__))       # .../PrimalSolver/julia")
    push!(lines, "")
    push!(lines, "function _primaldl()")
    push!(lines, "    # Look for the library next to this module (deps/), then in the C repo's")
    push!(lines, "    # out/ directory, then on the system loader path.")
    push!(lines, "    candidates = [")
    push!(lines, "        joinpath(_PKG_DIR, \"deps\", PRIMAL_LIBNAME),")
    push!(lines, "        joinpath(dirname(_PKG_DIR), \"out\", PRIMAL_LIBNAME),")
    push!(lines, "        PRIMAL_LIBNAME,")
    push!(lines, "    ]")
    push!(lines, "    for c in candidates")
    push!(lines, "        isfile(c) && return c")
    push!(lines, "    end")
    push!(lines, "    # Not built yet: Pkg.add runs deps/build.jl, Pkg.develop does not, so the")
    push!(lines, "    # first use compiles it (once, in its own process: a C compiler is needed).")
    push!(lines, "    build = joinpath(_PKG_DIR, \"deps\", \"build.jl\")")
    push!(lines, "    if isfile(build)")
    push!(lines, "        @info \"Primal: compiling the C library (first use)\"")
    push!(lines, "        run(`\$(Base.julia_cmd()) --startup-file=no \$build`)")
    push!(lines, "        isfile(candidates[1]) && return candidates[1]")
    push!(lines, "    end")
    push!(lines, "    error(\"libprimal shared library not found; run deps/build.jl or set PRIMAL_C_SRC\")")
    push!(lines, "end")
    push!(lines, "")
    push!(lines, "const _LIB = Ref{Any}(nothing)")
    push!(lines, "const _FUNCS = Dict{String,Any}()")
    push!(lines, "")
    push!(lines, "function _lib()")
    push!(lines, "    if _LIB[] === nothing")
    push!(lines, "        _LIB[] = Libdl.dlopen(_primaldl())")
    push!(lines, "    end")
    push!(lines, "    return _LIB[]")
    push!(lines, "end")
    push!(lines, "")
    push!(lines, "function _func(name::String)")
    push!(lines, "    if !haskey(_FUNCS, name)")
    push!(lines, "        _FUNCS[name] = Libdl.dlsym(_lib(), name)")
    push!(lines, "    end")
    push!(lines, "    return _FUNCS[name]")
    push!(lines, "end")
    push!(lines, "")
    push!(lines, "###############################################################################")
    push!(lines, "# Function bindings")
    push!(lines, "###############################################################################")
    push!(lines, "")

    missed = String[]
    bound = 0

    for (name, ret_raw, args, variadic) in protos
        if variadic
            push!(missed, name)
            continue
        end

        jl_ret = jl_type(ret_raw)
        jl_args = String[]
        for a in args
            ctype = strip_param_name(a)
            push!(jl_args, jl_type(ctype))
        end
        argstr = isempty(jl_args) ? "()" : (length(jl_args) == 1 ? "($(jl_args[1]),)" : "($(join(jl_args, ", ")))")

        cname = name
        jl_name = Symbol(replace(name, r"^PRIMAL_" => ""))

        arg_syms = [Symbol("a$i") for i in 1:length(args)]
        push!(lines, "function $jl_name($(join(arg_syms, ", ")))")
        push!(lines, "    _fn = Libdl.dlsym(_lib(), \"$cname\")")
        push!(lines, "    ccall(_fn, $jl_ret, $argstr,")
        if !isempty(arg_syms)
            push!(lines, "        $(join(arg_syms, ", "))")
        end
        push!(lines, "    )")
        push!(lines, "end")
        push!(lines, "")
        bound += 1
    end

    push!(lines, "###############################################################################")
    push!(lines, "# End of bindings ($bound functions bound, $(length(missed)) variadic skipped)")
    push!(lines, "###############################################################################")
    push!(lines, "")
    push!(lines, "# Export all public (non-underscore) names so `using .Bindings` brings them into scope.")
    push!(lines, "for _n in names(Bindings, all=true)")
    push!(lines, "    _s = String(_n)")
    push!(lines, "    if !startswith(_s, \"_\") && !contains(_s, \"#\") && _s != \"Bindings\" && _s != \"eval\" && _s != \"include\"")
    push!(lines, "        @eval export \$(Symbol(_s))")
    push!(lines, "    end")
    push!(lines, "end")
    push!(lines, "")
    push!(lines, "end  # module Bindings")
    push!(lines, "")

    write(OUT, join(lines, "\n"))

    println("wrote $OUT: $bound functions bound, $(length(missed)) variadic skipped")
    if !isempty(missed)
        println("variadic (skipped): ", join(missed, ", "))
    end
end

main()
