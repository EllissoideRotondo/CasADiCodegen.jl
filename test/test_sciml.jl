# CasADi functions as the functions of SciML problems, with their derivatives.
include("ode_helpers.jl")
import SciMLBase
using OrdinaryDiffEqVerner: Vern9
using SimpleNonlinearSolve: SimpleNewtonRaphson
using SpecialFunctions: erf
using SparseArrays: SparseMatrixCSC, nnz, findnz
using LinearAlgebra: triu
using StaticArrays: SVector, SMatrix

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

@testset "sparse = true gives the Jacobian a sparse prototype" begin
    # A chain: each derivative depends on the state and its neighbors, so the Jacobian is tridiagonal.
    n = 6
    x, k, t = SX.sym("x", n), SX.sym("k"), SX.sym("t")
    field(s) = vertcat([k * (j > 1 ? x[j-1] : 0.0) - 2 * x[j]^2 + (j < n ? x[j+1] : 0.0) * s for j in 1:n]...)
    rhs = CasADiCodegen.Function("chain", [x, k, t], [field(t)])
    u, p, Jd = collect(1.0:n), [0.5], zeros(n, n)
    SciMLBase.ODEFunction(rhs).jac(Jd, u, p, 1.0)
    f = SciMLBase.ODEFunction(rhs; sparse = true)
    J = copy(f.jac_prototype)
    @test J isa SparseMatrixCSC && nnz(J) == 3n - 2
    f.jac(J, u, p, 1.0)
    @test J == Jd && (@allocated f.jac(J, u, p, 1.0)) == 0
    D = fill(NaN, n, n)
    f.jac(D, u, p, 1.0)
    @test D == Jd                               # a dense matrix gets the same values
    # The residual of a nonlinear problem, here the right side at t = 1.
    residual = CasADiCodegen.Function("r", [x, k], [field(1.0)])
    g = SciMLBase.NonlinearFunction(residual; sparse = true)
    Jg = copy(g.jac_prototype)
    g.jac(Jg, u, p)
    @test Jg isa SparseMatrixCSC && Jg == Jd
    @test_throws ArgumentError SciMLBase.ODEFunction(rhs; sparse = true, jac = false)
    @test_throws ArgumentError SciMLBase.NonlinearFunction(residual; sparse = true, jac_prototype = J)
end

@testset "inplace = false gives functions of static arrays" begin
    u, k, t = SX.sym("u", 2), SX.sym("k"), SX.sym("t")
    rhs = CasADiCodegen.Function("osc", [u, k, t], [vertcat(u[2], -k * u[1] + 0.1 * t)])
    f, g = SciMLBase.ODEFunction(rhs; inplace = false), SciMLBase.ODEFunction(rhs)
    us, ks = SVector(1.0, 0.5), SVector(4.0)
    du, J, P = zeros(2), zeros(2, 2), zeros(2, 1)
    g(du, us, ks, 0.3); g.jac(J, us, ks, 0.3); g.paramjac(P, us, ks, 0.3)
    @test !SciMLBase.isinplace(f) && f(us, ks, 0.3) isa SVector{2,Float64} && f(us, ks, 0.3) == du
    @test f.jac(us, ks, 0.3) isa SMatrix{2,2,Float64} && f.jac(us, ks, 0.3) == J
    @test f.paramjac(us, ks, 0.3) isa SMatrix{2,1,Float64} && f.paramjac(us, ks, 0.3) == P   # one parameter: still a matrix
    final(fun, u0) = SciMLBase.solve(SciMLBase.ODEProblem(fun, u0, (0.0, 2.0), ks), Vern9(); abstol = 1e-12, reltol = 1e-12).u[end]
    @test final(f, us) isa SVector{2,Float64} && isapprox(final(f, us), final(g, collect(us)); rtol = 1e-10)
    # One state: the Jacobian is a 1x1 matrix, as the implicit solvers need.
    x = SX.sym("x")
    decay = SciMLBase.ODEFunction(CasADiCodegen.Function("decay", [x, k], [-k * x^2]); inplace = false)
    @test decay.jac(SVector(2.0), ks, 0.0) === SMatrix{1,1}(-16.0)
    z = SX.sym("z", 2)
    r = SciMLBase.NonlinearFunction(CasADiCodegen.Function("r", [z, k], [vertcat(z[1]^2 - k, z[2] - z[1])]); inplace = false)
    @test SciMLBase.solve(SciMLBase.NonlinearProblem(r, SVector(1.0, 1.0), ks), SimpleNewtonRaphson(); abstol = 1e-12).u ≈ SVector(2.0, 2.0)
    @test_throws ArgumentError SciMLBase.ODEFunction(rhs; inplace = false, sparse = true)
    # A Jacobian of 65 by 65 states is above the limit of static arrays.
    v = SX.sym("v", 65)
    @test_throws ArgumentError SciMLBase.ODEFunction(CasADiCodegen.Function("big", [v], [-v]); inplace = false)
end

@testset "keywords of the code and of the specialization" begin
    u, k = SX.sym("u", 3), SX.sym("k")
    rhs = CasADiCodegen.Function("chain", [u, k], [vertcat(k * u[2] - u[1]^2, u[3] * u[1], sin(u[2]) - k)])
    reference, du, dv = SciMLBase.ODEFunction(rhs), zeros(3), zeros(3)
    reference(du, [1.0, 2.0, 3.0], [0.5], 0.0)
    @test_throws ArgumentError reference(zeros(2), [1.0, 2.0, 3.0], [0.5], 0.0)   # the inline code keeps its checks
    # Split code computes the same values.
    split = SciMLBase.ODEFunction(rhs; split_above = 1, chunk_size = 2, optlevel = 1)
    split(dv, [1.0, 2.0, 3.0], [0.5], 0.0)
    @test dv == du
    @test_throws ArgumentError SciMLBase.ODEFunction(rhs; optlevel = 4)
    @test_throws ArgumentError SciMLBase.NonlinearFunction(rhs; optlevel = 4)
    @test_throws ArgumentError SciMLBase.OptimizationFunction(CasADiCodegen.Function("c", [u], [u[1] + u[2]]); optlevel = 4)
    # With the cache on disk, the code is a package in the given directory.
    mktempdir() do directory
        cached = SciMLBase.ODEFunction(rhs; jac = false, paramjac = false, cache = directory)
        dv .= 0
        Base.invokelatest(cached, dv, [1.0, 2.0, 3.0], [0.5], 0.0)   # the code is newer than this test set
        @test dv == du && !isempty(readdir(directory))
    end
    auto = SciMLBase.ODEFunction(rhs; specialize = SciMLBase.AutoSpecialize)
    @test SciMLBase.specialization(auto) === SciMLBase.AutoSpecialize
    problem = SciMLBase.ODEProblem(auto, [1.0, 2.0, 3.0], (0.0, 0.1), [0.5])
    @test SciMLBase.solve(problem, Vern9(); abstol = 1e-12, reltol = 1e-12).u[end] ≈
          SciMLBase.solve(SciMLBase.remake(problem; f = reference), Vern9(); abstol = 1e-12, reltol = 1e-12).u[end]
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
    # The dense form has no Hessian of the Lagrange function.
    @test f.lag_h === nothing
    # No constraints, and no parameters.
    alone = SciMLBase.OptimizationFunction(CasADiCodegen.Function("c", [x], [sum(x[i]^2 for i in 1:3)]))
    @test alone.f(xv, nothing) ≈ sum(abs2, xv) && alone.cons === nothing && alone.cons_h === nothing
    alone.hess(L, xv, nothing)
    @test L ≈ 2.0 * [1 0 0; 0 1 0; 0 0 1]
    @test_throws ArgumentError SciMLBase.OptimizationFunction(CasADiCodegen.Function("w", [x], [x]))                        # not a scalar
    @test_throws ArgumentError SciMLBase.OptimizationFunction(objective, CasADiCodegen.Function("w", [SX.sym("y", 2), p], [p]))  # another unknown
    # hess = false: no derivative of the second order, for a solver of the first order.
    firstorder = SciMLBase.OptimizationFunction(objective, constraints; hess = false)
    @test firstorder.hess === nothing && firstorder.cons_h === nothing
    firstorder.grad(G, xv, pv)
    @test G ≈ ForwardDiff.gradient(cost, xv)
    @testset "sparse" begin
        sparse = SciMLBase.OptimizationFunction(objective, constraints; sparse = true)
        # The prototypes have the patterns of CasADi, and the functions write their stored values.
        Hs, Js = copy(sparse.hess_prototype), copy(sparse.cons_jac_prototype)
        @test Hs isa SparseMatrixCSC && nnz(Hs) == 4 && nnz(Js) == 5
        sparse.hess(Hs, xv, pv)
        sparse.cons_j(Js, xv, pv)
        @test Hs ≈ ForwardDiff.hessian(cost, xv) && Js ≈ ForwardDiff.jacobian(cons, xv)
        # lag_h writes the upper triangle of σ∇²f + Σμ∇²g, in the order of the stored values of lag_hess_prototype.
        σ, μ = 0.5, [2.0, -3.0]
        lagrange = ForwardDiff.hessian(u -> σ * cost(u) + μ' * cons(u), xv)
        P = sparse.lag_hess_prototype
        @test P isa SparseMatrixCSC && all(i <= j for (i, j, _) in zip(findnz(P)...))
        h = zeros(nnz(P))
        sparse.lag_h(h, xv, σ, μ, pv)
        @test SparseMatrixCSC(P.m, P.n, P.colptr, P.rowval, h) ≈ triu(lagrange)
        @test (@allocated sparse.lag_h(h, xv, σ, μ, pv)) == 0 && (@allocated sparse.cons_j(Js, xv, pv)) == 0
        # With lag_h there are no Hessians of the constraints; without it they are sparse.
        @test sparse.cons_h === nothing
        each = SciMLBase.OptimizationFunction(objective, constraints; sparse = true, lag_h = false)
        Hc = copy.(each.cons_hess_prototype)
        each.cons_h(Hc, xv, pv)
        @test each.lag_h === nothing && Hc[2] ≈ ForwardDiff.hessian(u -> cons(u)[2], xv) && nnz(Hc[2]) == 2
        @test_throws "lag_h = true needs sparse = true and hess = true" SciMLBase.OptimizationFunction(objective, constraints; lag_h = true)
        # A prototype of the user would replace the pattern into which the code writes.
        @test_throws "makes the prototypes itself" SciMLBase.OptimizationFunction(objective, constraints; sparse = true, hess_prototype = Hs)
    end
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

@testset "ODEProblem from a DaeBuilder" begin
    # x'' = -w + z + u with w = k x and 0 = z - 2x, that is x'' = -(k - 2) x + u.
    dae = CC.DaeBuilder("spring")
    x, v, z = (CC.add(dae, n, "local", "continuous") for n in ("x", "v", "z"))
    k, u = CC.add(dae, "k", "parameter", "tunable"), CC.add(dae, "u", "input", "continuous")
    w = CC.add(dae, "w", "local", "continuous")
    CC.eq(dae, w, k * x)
    CC.eq(dae, CC.der(dae, x), v)
    CC.eq(dae, CC.der(dae, v), -w + z + u)
    CC.eq(dae, MX(0.0), z - 2x)
    foreach(((n, value),) -> CC.set_start(dae, n, value), ("x" => 1.0, "z" => 2.0, "k" => 3.0, "u" => 0.5))
    problem = SciMLBase.ODEProblem(dae, (0.0, 1.0))
    @test problem.u0 == [1.0, 0.0, 2.0] && problem.p == [3.0, 0.5]
    @test problem.f.mass_matrix == [1 0 0; 0 1 0; 0 0 0]
    solution = SciMLBase.solve(problem, Rodas5P(); abstol = 1e-11, reltol = 1e-11)
    @test solution[:x][end] ≈ 0.5 + 0.5 * cos(1.0) rtol = 1e-8
    @test solution[:z][end] ≈ 2 * solution[:x][end] rtol = 1e-8
    # The time span of the DaeBuilder is the default; CasADi leaves it unset.
    @test_throws "set the start and stop times" SciMLBase.ODEProblem(dae)
    CC.set_start_time(dae, 0.0)
    CC.set_stop_time(dae, 2.0)
    @test SciMLBase.ODEProblem(dae).tspan == (0.0, 2.0)
    # Regression: a constant without a value made the right side NaN with no error.
    decay = CC.DaeBuilder("decay")
    y, c = CC.add(decay, "y", "local", "continuous"), CC.add(decay, "c", "local", "constant")
    CC.eq(decay, CC.der(decay, y), c - y)
    @test_throws "give each constant its value" SciMLBase.ODEProblem(decay, (0.0, 1.0))
    CC.when(decay, y > 0.5, String[])
    @test_throws "when equations" SciMLBase.ODEProblem(decay, (0.0, 1.0))
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
    # sparse = true reaches the SciML function.
    sparse = SciMLBase.OptimizationProblem(solver, [0.1, 0.2], [0.3]; sparse = true)
    @test sparse.f.cons_jac_prototype isa SparseMatrixCSC && nnz(sparse.f.lag_hess_prototype) == 3
    @test_throws "the unknowns must have 2 numbers" SciMLBase.OptimizationProblem(solver, [0.1], [0.3])
    @test_throws "is not a solver of CasADi for optimization problems" SciMLBase.OptimizationProblem(CasADiCodegen.Function("f", [v], [2v]), [0.1, 0.2])
end

@testset "OptimizationProblem from an Opti object" begin
    opti = CC.Opti()
    z, a = CC.variable(opti, 2), CC.parameter(opti)
    CC.minimize(opti, (z[1] - a)^2 + z[2]^2)
    CC.subject_to(opti, z[1] + z[2] >= 1)
    CC.subject_to(opti, z[1] <= 3)
    CC.set_value(opti, a, CC.DM(0.25))
    CC.set_initial(opti, z, CC.DM([2.0, 1.5]))
    problem = SciMLBase.OptimizationProblem(opti; sparse = true)
    # The initial values, the values of the parameters and the bounds of the constraints come from the Opti object.
    @test (problem.u0, problem.p, problem.lcons, problem.ucons) == ([2.0, 1.5], [0.25], [1.0, -Inf], [Inf, 3.0])
    c = zeros(2)
    problem.f.cons(c, [2.0, 1.5], [0.25])
    @test problem.f.f([2.0, 1.5], [0.25]) ≈ 1.75^2 + 1.5^2 && c == [3.5, 2.0]
    unset = CC.Opti()
    w, b = CC.variable(unset), CC.parameter(unset)
    CC.minimize(unset, (w - b)^2)
    @test_throws "give each parameter its value" SciMLBase.OptimizationProblem(unset)
end

"The right side of an ODE on a matrix state, written in place for numbers of any type."
function heat!(du, u, p, t)
    n = size(u, 1)
    for j in axes(u, 2), i in 1:n
        left = i > 1 ? u[i-1, j] : zero(eltype(u))
        right = i < n ? u[i+1, j] : zero(eltype(u))
        du[i, j] = p[1] * (left - 2u[i, j] + right) + p[2] * sin(t) * u[i, j]^2
    end
end

@testset "traced_function of a SciML problem" begin
    u0, p = reshape(collect(1.0:6.0) ./ 7, 3, 2), [0.5, 2.0]
    original = SciMLBase.ODEProblem(heat!, u0, (0.0, 1.0), p)
    f = SciMLBase.ODEFunction(CasADiCodegen.traced_function("heat", original); jac = false, paramjac = false)
    du, expected = zeros(6), similar(u0)
    f(du, vec(u0), p, 0.3)
    heat!(expected, u0, p, 0.3)
    @test du ≈ vec(expected)
    # A nonlinear problem out of place has no time.
    nonlinear = SciMLBase.NonlinearProblem{false}((z, q) -> z .^ 2 .- q, [1.0, 1.0], [4.0, 9.0])
    residual = CasADiCodegen.traced_function("square", nonlinear)
    @test CC.n_in(residual) == 2 && build_function(residual; expression = Val{false})[1]([2.0, 1.0], [4.0, 9.0]) ≈ [0.0, -8.0]
end

@testset "names of the CasADi symbols in the solution" begin
    x, v, k, τ = SX.sym("x"), SX.sym("v"), SX.sym("k"), SX.sym("τ")
    rhs = CasADiCodegen.Function("oscillator", [vertcat(x, v), k, τ], [vertcat(v, -k * x), (v^2 + k * x^2) / 2], ["u", "k", "τ"], ["du", "energy"])
    problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(rhs), [1.0, 0.0], (0.0, 1.0), [4.0])
    solution = SciMLBase.solve(problem, Vern9(); abstol = 1e-12, reltol = 1e-12)
    @test solution[:v] == solution[2, :]
    # A further result of the function is an observed variable with the name of that result.
    @test solution[:energy] ≈ fill(2.0, length(solution.t)) rtol = 1e-9
    @test solution(0.5; idxs = :energy) ≈ 2.0 rtol = 1e-9
    remade = SciMLBase.remake(problem; p = [:k => 9.0], u0 = [:v => 1.0])
    @test (remade.u0, remade.p) == ([1.0, 1.0], [9.0])
    # The elements of an MX symbol have the name of the symbol and their index from 0, as CasADi names those of an SX symbol.
    z, q = MX.sym("z", 2), MX.sym("q")
    root = SciMLBase.solve(SciMLBase.NonlinearProblem(SciMLBase.NonlinearFunction(CasADiCodegen.Function("r", [z, q], [z^2 - q])), [1.0, 1.0], [4.0]), SimpleNewtonRaphson())
    @test root[:z_1] ≈ 2.0
    # Names that occur twice give no names; a system of the user is kept.
    twice = CasADiCodegen.Function("twice", [vertcat(x, v), SX.sym("x")], [vertcat(v, -x)])
    @test SciMLBase.ODEFunction(twice).sys === nothing
    observed_twice = CasADiCodegen.Function("observed_twice", [vertcat(x, v), SX.sym("x")], [vertcat(v, -x), x^2])
    @test_throws "need distinct names" SciMLBase.ODEFunction(observed_twice)
    @test SciMLBase.ODEFunction(rhs; sys = SciMLBase.SymbolCache([:a, :b], [:c], :s)).sys isa SciMLBase.SymbolCache
end
