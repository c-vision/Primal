#!/usr/bin/env julia
# poly_date.jl — "poly-date": the shortest homogeneous three-variable
# polynomial equation with all coefficients 1 satisfied by a date (d, m, y)
# (aszek/poly-date). For each degree k, two DISJOINT subsets of the monomials
# d^a m^b y^c (a+b+c = k) with equal value sums give an equation P_s1 = P_s2:
#
#   minimize sum (s1 + s2)  s.t.  sum (s1 - s2) val = 0,  s1 + s2 <= 1,  sum s1 >= 1,  binary
#
# The coefficients are verified in exact integer arithmetic. Known case:
# 8/11/15 first works at degree 3 (4 terms = 5 terms, both sides 6730).
#
#   julia poly_date.jl [d m y maxDeg]   (default 8 11 15 5)
#
# Port of examples/c/poly_date.c.

using Printf
using Primal
using Primal: Model   # not exported: JuMP exports its own Model

# Enumerate the degree-k monomial values of (d, m, y).
function gen_monomials(d, m, y, k)
    vals = Int[]
    for a in 0:k, b in 0:k-a
        c = k - a - b
        push!(vals, d^a * m^b * y^c)
    end
    return vals
end

# MILP for one degree; returns (feasible, nmono, n1, n2, L, R).
function solve_degree(d, mm, y, k)
    vals = gen_monomials(d, mm, y, k)
    n = length(vals)
    S1, S2, nv = 0, n, 2n
    m = Model(0, 0)
    r = try
        add_vars(m, nv)
        sense(m, SENSE_MIN)
        for i in 0:nv-1
            var_type(m, i, VT_INT)
            var_bounds(m, i, BK_RA; lo=0.0, up=1.0)
            cj(m, i, 1.0)
        end
        add_cons(m, 1 + n + 1)
        row = 0
        a_row(m, row, collect(0:nv-1), [Float64.(vals); -Float64.(vals)])
        con_bounds(m, row, BK_FX; lo=0.0, up=0.0); row += 1
        for i in 0:n-1                              # disjoint
            a_row(m, row, [S1 + i, S2 + i], [1.0, 1.0])
            con_bounds(m, row, BK_UP; lo=-Inf, up=1.0); row += 1
        end
        a_row(m, row, collect(S1:S1+n-1), ones(n)) # an equation, not 0 = 0
        con_bounds(m, row, BK_LO; lo=1.0, up=Inf); row += 1
        solve(m)
    finally
        free(m)
    end
    ok, n1, n2, L, R = false, 0, 0, 0, 0
    if r.rc == 0
        x = r.x
        for i in 1:n
            x[S1+i] > 0.5 && (L += vals[i]; n1 += 1)
            x[S2+i] > 0.5 && (R += vals[i]; n2 += 1)
        end
        ok = n1 > 0 && L == R
    end
    return ok, n, n1, n2, L, R
end

function main()
    d = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 8
    m = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 11
    y = length(ARGS) > 2 ? parse(Int, ARGS[3]) : 15
    maxDeg = length(ARGS) > 3 ? parse(Int, ARGS[4]) : 5
    @printf("poly_date  d=%d m=%d y=%d maxDeg=%d\n", d, m, y, maxDeg)
    allok, found = true, false
    for k in 1:maxDeg
        found && break
        ok, n, n1, n2, L, R = solve_degree(d, m, y, k)
        if ok
            good = L == R && L > 0 && n1 > 0
            @printf("  degree %d: %d monomials, %d + %d terms, LHS=RHS=%d  %s\n", k, n, n1, n2, L, good ? "OK" : "FAIL")
            allok &= good
            found = true
        else
            @printf("  degree %d: no equation\n", k)
        end
    end
    if !found
        @printf("  no equation up to degree %d\n", maxDeg)
        allok = false
    end
    @printf("%s\n", allok ? "OK" : "FAIL")
    return allok ? 0 : 1
end

exit(main())
