# CasADi functions as the functions of SciML problems, with their derivatives.
include("ode_helpers.jl")
import SciMLBase
using OrdinaryDiffEqVerner: Vern9
using SimpleNonlinearSolve: SimpleNewtonRaphson
using SpecialFunctions: erf

@testset "ODEFunction from a CasADi function" begin
    dae = two_body()
    t = SX.sym("t")
    rhs = CasADiCodegen.Function("rhs", [dae["x"], dae["p"], t], [dae["ode"] * (1 + 0 * t)])
    f = SciMLBase.ODEFunction(rhs)
    @test SciMLBase.isinplace(f)
    du, J, P = zeros(4), zeros(4, 4), zeros(4, 1)
    f(du, ODE_X0, ODE_MU, 0.0)
    r3 = (ODE_X0[1]^2 + ODE_X0[2]^2)^1.5
    @test du ≈ [ODE_X0[3], ODE_X0[4], -ODE_X0[1] / r3, -ODE_X0[2] / r3]
    f.jac(J, ODE_X0, ODE_MU, 0.0)
    @test J ≈ ForwardDiff.jacobian(u -> (d = zeros(eltype(u), 4); f(d, u, ODE_MU, 0.0); d), ODE_X0)
    f.paramjac(P, ODE_X0, ODE_MU, 0.0)
    @test vec(P) ≈ [0.0, 0.0, -ODE_X0[1] / r3, -ODE_X0[2] / r3]
    @test (@allocated f(du, ODE_X0, ODE_MU, 0.0)) == 0 && (@allocated f.jac(J, ODE_X0, ODE_MU, 0.0)) == 0
    # A solver takes it, and gives the state of the integrator of this package.
    problem = SciMLBase.ODEProblem(f, ODE_X0, (0.0, 3.0), ODE_MU)
    solution = SciMLBase.solve(problem, Vern9(); abstol = 1e-13, reltol = 1e-13)
    @test closeto([solution.u[end]], casadi_eval(two_body_integrator(), ODE_X0, ODE_MU), 1e-10)
    # Functions of the state alone, and of the state and the parameters; other keywords go to ODEFunction.
    x = SX.sym("x", 2)
    g = SciMLBase.ODEFunction(CasADiCodegen.Function("g", [x], [vertcat(x[2], -x[1])]); jac_prototype = zeros(2, 2))
    @test g.jac_prototype == zeros(2, 2) && g.paramjac === nothing
    d2 = zeros(2)
    g(d2, [1.0, 2.0], nothing, 0.0)
    @test d2 == [2.0, -1.0]
    k = SX.sym("k")
    h = SciMLBase.ODEFunction(CasADiCodegen.Function("h", [x, k], [x * k]); jac = false)
    @test h.jac === nothing
    @test_throws ArgumentError SciMLBase.ODEFunction(CasADiCodegen.Function("w", [x], [x[1]]))                 # not the size of the state
    @test_throws ArgumentError SciMLBase.ODEFunction(CasADiCodegen.Function("w", [x, k, SX.sym("t"), SX.sym("s")], [x]))           # too many arguments
end

@testset "NonlinearFunction from a CasADi function" begin
    z, b = SX.sym("z", 2), SX.sym("b", 2)
    f = SciMLBase.NonlinearFunction(CasADiCodegen.Function("r", [z, b], [vertcat(z[1]^3 + z[2], z[2] * 2) - b]))
    r, J = zeros(2), zeros(2, 2)
    f(r, [1.0, 2.0], [0.5, 1.0])
    @test r == [2.5, 3.0]
    f.jac(J, [1.0, 2.0], [0.5, 1.0])
    @test J == [3.0 1.0; 0.0 2.0]
    solution = SciMLBase.solve(SciMLBase.NonlinearProblem(f, [1.0, 1.0], [9.0, 2.0]), SimpleNewtonRaphson(); abstol = 1e-12)
    @test solution.u ≈ [2.0, 1.0]
    @test_throws ArgumentError SciMLBase.NonlinearFunction(CasADiCodegen.Function("w", [z], [z[1]]))
end

@testset "OptimizationFunction from CasADi functions" begin
    x, p = SX.sym("x", 3), SX.sym("p", 2)
    objective = CasADiCodegen.Function("cost", [x, p], [p[1] * x[1]^2 * x[2] + sin(x[3]) * p[2]])
    constraints = CasADiCodegen.Function("cons", [x, p], [vertcat(x[1] * x[2] * x[3] - p[1], x[1]^2 + x[3]^2)])
    f = SciMLBase.OptimizationFunction(objective, constraints)
    xv, pv = [1.0, 2.0, 0.5], [3.0, 4.0]
    cost(u) = pv[1] * u[1]^2 * u[2] + sin(u[3]) * pv[2]
    cons(u) = [u[1] * u[2] * u[3] - pv[1], u[1]^2 + u[3]^2]
    @test f.f(xv, pv) ≈ cost(xv)
    G, H, c, J, L = zeros(3), zeros(3, 3), zeros(2), zeros(2, 3), zeros(3, 3)
    f.grad(G, xv, pv)
    @test G ≈ ForwardDiff.gradient(cost, xv)
    f.hess(H, xv, pv)
    @test H ≈ ForwardDiff.hessian(cost, xv)
    f.cons(c, xv, pv)
    @test c ≈ cons(xv)
    f.cons_j(J, xv, pv)
    @test J ≈ ForwardDiff.jacobian(cons, xv)
    each = [zeros(3, 3), zeros(3, 3)]
    f.cons_h(each, xv, pv)
    @test each[1] ≈ ForwardDiff.hessian(u -> cons(u)[1], xv) && each[2] ≈ ForwardDiff.hessian(u -> cons(u)[2], xv)
    @test (@allocated f.f(xv, pv)) == 0 && (@allocated f.grad(G, xv, pv)) == 0 && (@allocated f.cons_h(each, xv, pv)) == 0
    # OptimizationBase.jl cannot call a Hessian of the Lagrange function that the user gives, so there is none.
    @test f.lag_h === nothing
    # No constraints, and no parameters.
    alone = SciMLBase.OptimizationFunction(CasADiCodegen.Function("c", [x], [sum(x[i]^2 for i in 1:3)]))
    @test alone.f(xv, nothing) ≈ sum(abs2, xv) && alone.cons === nothing && alone.cons_h === nothing
    alone.hess(L, xv, nothing)
    @test L ≈ 2.0 * [1 0 0; 0 1 0; 0 0 1]
    @test_throws ArgumentError SciMLBase.OptimizationFunction(CasADiCodegen.Function("w", [x], [x]))                        # not a scalar
    @test_throws ArgumentError SciMLBase.OptimizationFunction(objective, CasADiCodegen.Function("w", [SX.sym("y", 2), p], [p]))  # another unknown
end

# A solver object of CasADi holds a problem: its equations and its data. The
# constructors of the SciML problems take the object, so that the user solves
# the problem with a solver of the SciML ecosystem.
using OrdinaryDiffEqRosenbrock: Rodas5P
const CC = CasADiCodegen.casadi

@testset "ODEProblem from an integrator of CasADi" begin
    x, p, t, c = MX.sym("x", 2), MX.sym("p"), MX.sym("t"), MX.sym("c")
    dae = Dict("x" => x, "p" => p, "t" => t, "ode" => -p * x * t, "quad" => x[1])
    exact(x0, rate, time) = x0 .* exp(-rate * time^2 / 2)
    @testset "$plugin" for plugin in ("cvodes", "rk")
        rule = CC.integrator("rule", plugin, dae, 0.5, [1.0, 2.0])
        problem = SciMLBase.ODEProblem(rule, [0.4, -0.3], [1.7])
        @test problem.tspan == (0.5, 2.0) && problem.u0 == [0.4, -0.3, 0.0] && problem.p == [1.7]
        @test problem.kwargs[:saveat] == [1.0, 2.0]
        # The tolerances of a solver of CasADi with a variable step are those of the problem.
        @test (plugin == "cvodes") == haskey(problem.kwargs, :abstol)
        # The problem is the same code for both plugins, so one solve shows the numbers.
        plugin == "cvodes" || continue
        @test problem.kwargs[:abstol] == 1e-8 && problem.kwargs[:reltol] == 1e-6
        solution = SciMLBase.solve(problem, Vern9(); abstol = 1e-13, reltol = 1e-13)
        @test length(solution.u) == 2
        shift(time) = exact([0.4, -0.3], 1.7, time) ./ exp(-1.7 * 0.5^2 / 2)
        @test solution.u[1][1:2] ≈ shift(1.0) rtol = 1e-10
        @test solution.u[2][1:2] ≈ shift(2.0) rtol = 1e-10
        # The last state is the quadrature, which CasADi returns as `qf`: the integral of the first state.
        integral = 0.4 * exp(1.7 / 8) * sqrt(pi / 3.4) * (erf(2 * sqrt(0.85)) - erf(0.5 * sqrt(0.85)))
        @test solution.u[2][3] ≈ integral rtol = 1e-10
        # The Jacobians of the right side are generated code.
        J = zeros(3, 3)
        problem.f.jac(J, [0.4, -0.3, 0.0], [1.7], 2.0)
        @test J == [-3.4 0 0; 0 -3.4 0; 1 0 0]
    end
    @testset "controls of one interval are parameters" begin
        rule = CC.integrator("rule", "cvodes", Dict("x" => x, "p" => p, "u" => c, "ode" => -p * x + c), 0.0, 1.0)
        problem = SciMLBase.ODEProblem(rule, [0.4, -0.3], [1.7]; u = [0.5], abstol = 1e-12)
        @test problem.p == [1.7, 0.5] && problem.kwargs[:abstol] == 1e-12
        @test_throws "the controls must have 1 number" SciMLBase.ODEProblem(rule, [0.4, -0.3], [1.7])
        @test_throws "the initial state must have 2 numbers" SciMLBase.ODEProblem(rule, [0.4], [1.7]; u = [0.5])
    end
    # One control per interval between the output times: the right side jumps at each output time.
    @testset "controls of several intervals change at the output times" begin
        grid = CC.integrator("rule", "cvodes", Dict("x" => x, "p" => p, "u" => c, "ode" => -p * x + c), 0.0, [1.0, 2.5])
        problem = SciMLBase.ODEProblem(grid, [0.4, -0.3], [1.7]; u = [0.5, -0.2])
        arc(start, control, span) = start .* exp(-1.7 * span) .+ control / 1.7 * (1 - exp(-1.7 * span))
        first_state = arc([0.4, -0.3], 0.5, 1.0)
        # The second solve starts from the first control again.
        for pass in 1:2
            solution = SciMLBase.solve(problem, Vern9(); abstol = 1e-13, reltol = 1e-13)
            @test length(solution.u) == 2
            @test solution.u[1] ≈ first_state rtol = 1e-10
            @test solution.u[2] ≈ arc(first_state, -0.2, 1.5) rtol = 1e-10
        end
        # A callback of the user runs after the change of the controls, so it sees the control of the new interval.
        seen = Float64[]
        record = SciMLBase.DiscreteCallback((_, t, _) -> t == 1.0, i -> push!(seen, i.p[2]); save_positions = (false, false))
        solution = SciMLBase.solve(SciMLBase.ODEProblem(grid, [0.4, -0.3], [1.7]; u = [0.5, -0.2], callback = record), Vern9(); abstol = 1e-13, reltol = 1e-13)
        @test seen == [-0.2]
        @test solution.u[2] ≈ arc(first_state, -0.2, 1.5) rtol = 1e-10
        @test_throws "the controls must have 2 numbers" SciMLBase.ODEProblem(grid, [0.4, -0.3], [1.7]; u = [0.5])
    end
    @testset "algebraic states give a mass matrix" begin
        z = MX.sym("z")
        # z = p x1 at each time, so that x1' = -z is a decay with the rate p.
        rule = CC.integrator("rule", "idas", Dict("x" => x, "z" => z, "p" => p, "ode" => CC.vertcat([-z, x[1]]), "alg" => z - p * x[1]), 0.0, 1.0)
        problem = SciMLBase.ODEProblem(rule, [0.4, -0.3], [1.7]; z0 = [0.68])
        @test problem.u0 == [0.4, -0.3, 0.68]
        @test problem.f.mass_matrix == [1 0 0; 0 1 0; 0 0 0]
        solution = SciMLBase.solve(problem, Rodas5P(); abstol = 1e-11, reltol = 1e-11)
        final = 0.4 * exp(-1.7)
        @test solution.u[end] ≈ [final, -0.3 + 0.4 * (1 - exp(-1.7)) / 1.7, 1.7 * final] rtol = 1e-8
        @test_throws "the algebraic states must have 1 number" SciMLBase.ODEProblem(rule, [0.4, -0.3], [1.7])
        # With an event: x doubles when z = r x falls to 0.55, and the algebraic state follows the jump.
        y = MX.sym("y")
        doubling = CC.Function("transition", Dict("x" => y, "post_x" => 2y), CC.event_in(), CC.event_out())
        refill = CC.integrator("refill", "idas", Dict("x" => y, "z" => z, "p" => p, "ode" => -z, "alg" => z - p * y, "zero" => z - 0.55), 0.0, [0.4, 1.0, 1.6],
                               Dict("transition" => doubling))
        solution = SciMLBase.solve(SciMLBase.ODEProblem(refill, [1.0], [1.1]; z0 = [1.1]), Rodas5P(); abstol = 1e-11, reltol = 1e-11)
        expected = [refilled(t, 1.0, 1.1) for t in (0.4, 1.0, 1.6)]
        @test reduce(vcat, solution.u) ≈ vec([expected 1.1 * expected]') rtol = 1e-7
    end
    @testset "an event function and its transition give a callback" begin
        y, e, g = MX.sym("y", 2), MX.sym("e"), MX.sym("g")
        jump = CC.Function("transition", Dict("x" => y, "p" => e, "post_x" => CC.vertcat([y[1], -e * y[2]])), CC.event_in(), CC.event_out())
        dae = Dict("x" => y, "p" => e, "u" => g, "ode" => CC.vertcat([y[2], -g]), "quad" => y[2]^2, "zero" => y[1])
        ball = CC.integrator("ball", "cvodes", dae, 0.0, [0.6, 1.4, 2.5], Dict("transition" => jump))
        problem = SciMLBase.ODEProblem(ball, [1.5, 2.0], [0.8]; u = [9.81, 5.0, 12.0])
        solution = SciMLBase.solve(problem, Vern9(); abstol = 1e-13, reltol = 1e-13)
        finals, sums = ball_reference([1.5, 2.0], 0.8, [9.81, 5.0, 12.0], [0.6, 1.4, 2.5])
        @test reduce(vcat, solution.u) ≈ vec([reshape(finals, 2, 3); sums']) rtol = 1e-9
    end
    @test_throws "is not an integrator of CasADi" SciMLBase.ODEProblem(CasADiCodegen.Function("f", [x], [2x]), [0.4, -0.3])
end

@testset "NonlinearProblem from a rootfinder of CasADi" begin
    a, x, b = MX.sym("a"), MX.sym("x", 2), MX.sym("b", 2)
    residual = CasADiCodegen.Function("r", [a, x, b], [x * a, x^2 - a * b])
    @testset "$plugin" for plugin in ("newton", "kinsol", "fast_newton")
        finder = CC.rootfinder("finder", plugin, residual, Dict("implicit_input" => true, "implicit_output" => true))
        # The parameters are the other arguments of the rootfinder, in their order.
        problem = SciMLBase.NonlinearProblem(finder, [3.0, 1.0], [2.0], [8.0, 0.5])
        @test problem.u0 == [3.0, 1.0] && problem.p == [2.0, 8.0, 0.5]
        solution = SciMLBase.solve(problem, SimpleNewtonRaphson(); abstol = 1e-13)
        @test solution.u ≈ [4.0, 1.0]
        @test solution.u ≈ Vector{Float64}(CC.nonzeros(finder(CC.DM(2.0), CC.DM([3.0, 1.0]), CC.DM([8.0, 0.5]))[2])) rtol = 1e-7
        J = zeros(2, 2)
        problem.f.jac(J, [3.0, 1.0], [2.0, 8.0, 0.5])
        @test J == [6.0 0; 0 2.0]
    end
    finder = CC.rootfinder("finder", "newton", residual, Dict("implicit_input" => true, "implicit_output" => true))
    @test_throws "the other arguments must have the sizes (1, 2)" SciMLBase.NonlinearProblem(finder, [3.0, 1.0], [2.0])
    @test_throws "is not a rootfinder of CasADi" SciMLBase.NonlinearProblem(residual, [3.0, 1.0])
end

@testset "OptimizationProblem from a solver of CasADi" begin
    v, q = MX.sym("v", 2), MX.sym("q")
    solver = CC.nlpsol("solver", "ipopt", Dict("x" => v, "p" => q, "f" => (v[1] - q)^2 + v[2]^2 + v[1] * v[2], "g" => v[1] + v[2]))
    problem = SciMLBase.OptimizationProblem(solver, [0.1, 0.2], [0.3]; lbx = [-5.0, -5.0], ubx = [5.0, 5.0], lbg = [1.0], ubg = [1.0])
    @test problem.u0 == [0.1, 0.2] && problem.p == [0.3]
    @test problem.lb == [-5.0, -5.0] && problem.ub == [5.0, 5.0] && problem.lcons == [1.0] && problem.ucons == [1.0]
    cost(u) = (u[1] - 0.3)^2 + u[2]^2 + u[1] * u[2]
    @test problem.f.f([0.1, 0.2], [0.3]) ≈ cost([0.1, 0.2])
    G, c, J = zeros(2), zeros(1), zeros(1, 2)
    problem.f.grad(G, [0.1, 0.2], [0.3])
    @test G ≈ ForwardDiff.gradient(cost, [0.1, 0.2])
    problem.f.cons(c, [0.1, 0.2], [0.3])
    problem.f.cons_j(J, [0.1, 0.2], [0.3])
    @test c ≈ [0.3] && J == [1.0 1.0]
    # At the solution the gradient of the objective is along that of the constraint: v = (0.8, 0.2) for q = 0.3.
    problem.f.grad(G, [0.8, 0.2], [0.3])
    @test G[1] ≈ G[2]
    # The bounds of CasADi when none are given: no bound on the unknowns, and none on the constraints.
    free = SciMLBase.OptimizationProblem(solver, [0.1, 0.2], [0.3])
    @test free.lb === nothing && free.ub === nothing && free.lcons == [-Inf] && free.ucons == [Inf]
    # No parameters and no constraints.
    plain = SciMLBase.OptimizationProblem(CC.nlpsol("plain", "ipopt", Dict("x" => v, "f" => CC.sumsqr(v - 1))), [0.1, 0.2])
    @test plain.f.f([1.0, 3.0], plain.p) ≈ 4.0 && plain.lcons === nothing && plain.f.cons === nothing
    @test_throws "the unknowns must have 2 numbers" SciMLBase.OptimizationProblem(solver, [0.1], [0.3])
    @test_throws "is not a solver of CasADi for optimization problems" SciMLBase.OptimizationProblem(CasADiCodegen.Function("f", [v], [2v]), [0.1, 0.2])
end
