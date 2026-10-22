# jump.jl — Primal.Optimizer through JuMP, on small problems with known answers.
#
#     julia --project=julia/test julia/test/jump.jl

module TestPrimalJuMP

using Test
using JuMP
import Primal

const TOL = 1e-6

function runtests()
    @testset "JuMP" begin
        @testset "solver attributes" begin
            m = Model(Primal.Optimizer)
            @test solver_name(m) == "Primal"
            set_silent(m)
            set_time_limit_sec(m, 10.0)
            @test time_limit_sec(m) == 10.0
        end

        @testset "LP" begin
            # max 3x + 2y  s.t.  x + y <= 4, x + 3y <= 6, x <= 3.5, x, y >= 0: optimum
            # (3.5, 0.5) with c1 and the bound on x active, c2 slack (unique duals)
            m = Model(Primal.Optimizer); set_silent(m)
            @variable(m, 0 <= x <= 3.5)
            @variable(m, y >= 0)
            @constraint(m, c1, x + y <= 4)
            @constraint(m, c2, x + 3y <= 6)
            @objective(m, Max, 3x + 2y)
            optimize!(m)
            @test termination_status(m) == OPTIMAL
            @test primal_status(m) == FEASIBLE_POINT
            @test dual_status(m) == FEASIBLE_POINT
            @test objective_value(m) ≈ 11.5 atol = TOL
            @test value(x) ≈ 3.5 atol = TOL
            @test value(y) ≈ 0.5 atol = TOL
            @test shadow_price(c1) ≈ 2 atol = TOL
            @test shadow_price(c2) ≈ 0 atol = TOL
            @test reduced_cost(x) ≈ 1 atol = TOL
        end

        @testset "MILP" begin
            # knapsack: max 5a + 4b + 3c  s.t.  2a + 3b + c <= 5, binary
            m = Model(Primal.Optimizer); set_silent(m)
            @variable(m, z[1:3], Bin)
            @constraint(m, 2z[1] + 3z[2] + z[3] <= 5)
            @objective(m, Max, 5z[1] + 4z[2] + 3z[3])
            optimize!(m)
            @test termination_status(m) == OPTIMAL
            @test objective_value(m) ≈ 9 atol = TOL
            @test round.(Int, value.(z)) == [1, 1, 0]
            @test dual_status(m) == NO_SOLUTION
        end

        @testset "QP" begin
            # min (x-1)^2 + (y-2)^2  s.t.  x + y == 1
            m = Model(Primal.Optimizer); set_silent(m)
            @variable(m, x); @variable(m, y)
            @constraint(m, c, x + y == 1)
            @objective(m, Min, (x - 1)^2 + (y - 2)^2)
            optimize!(m)
            @test termination_status(m) == OPTIMAL
            @test value(x) ≈ 0 atol = 1e-5
            @test value(y) ≈ 1 atol = 1e-5
            @test objective_value(m) ≈ 2 atol = 1e-5
        end

        @testset "SOC" begin
            # min t  s.t.  ||(x - 1, y - 2)|| <= t, x + y == 0
            m = Model(Primal.Optimizer); set_silent(m)
            @variable(m, t); @variable(m, x); @variable(m, y)
            @constraint(m, [t; x - 1; y - 2] in SecondOrderCone())
            @constraint(m, x + y == 0)
            @objective(m, Min, t)
            optimize!(m)
            @test termination_status(m) == OPTIMAL
            @test objective_value(m) ≈ 3 / sqrt(2) atol = 1e-5
            @test value(x) ≈ -0.5 atol = 1e-5
            @test value(y) ≈ 0.5 atol = 1e-5
        end

        @testset "exponential cone" begin
            # max x  s.t.  exp(x) <= 2   i.e. (x, 1, 2) in ExponentialCone
            m = Model(Primal.Optimizer); set_silent(m)
            @variable(m, x)
            @constraint(m, [x, 1, 2] in MOI.ExponentialCone())
            @objective(m, Max, x)
            optimize!(m)
            @test termination_status(m) == OPTIMAL
            @test value(x) ≈ log(2) atol = 1e-5
        end

        @testset "infeasible LP" begin
            m = Model(Primal.Optimizer); set_silent(m)
            @variable(m, x >= 1)
            @constraint(m, x <= 0)
            @objective(m, Min, x)
            optimize!(m)
            @test termination_status(m) in (INFEASIBLE, INFEASIBLE_OR_UNBOUNDED)
            @test primal_status(m) != FEASIBLE_POINT
        end
    end
end

end  # module

TestPrimalJuMP.runtests()
