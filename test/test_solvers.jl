# A solver of CasADi in the graph of a function, replaced in the generated
# code by a solver of the SciML ecosystem that the user selects with
# `CasADiCodegen.Solvers`. CasADi with tight tolerances and exact solutions are the oracles.
include("ode_helpers.jl")
import ForwardDiff
import SciMLBase
using OrdinaryDiffEqVerner: Vern9, Vern7
using OrdinaryDiffEqRosenbrock: Rodas5P
using SimpleNonlinearSolve: SimpleNewtonRaphson
const K = CasADiCodegen.Codegen
using CasADiCodegen: Solvers, Solver

native(f; kwargs...) = build_function(f; nonzeros = true, expression = Val{false}, expand = false, kwargs...)
evaluated(g, f, nz) = g([fill(NaN, n) for n in nnz_out(f)], Tuple(nz))
agree(a, e; rtol) = length(a) == length(e) && all(isapprox(x, y; rtol, atol = 1e-12) for (x, y) in zip(a, e))
all_agree(actual, expected; rtol) = all(agree(a, e; rtol) for (a, e) in zip(actual, expected))
# One compiled solver for all integrators of the group: the specialization is a matter of the integrator group.
const PRECISE = Solver(Vern9(); abstol = 1e-13, reltol = 1e-13, specialize = SciMLBase.AutoSpecialize)

@testset "the selection of solvers" begin
    x = MX.sym("x", 2)
    rule = C.integrator("rule", "cvodes", Dict("x" => x, "ode" => -x), 0.0, 1.0)
    one = Solver(rule, Vern7(); abstol = 1e-9)
    @test_throws "an override needs the function of CasADi that it replaces" Solvers(overrides = (PRECISE,))
    @test_throws "has a function of CasADi; give it in overrides" Solvers(ode = one)
end

@testset "integrator with a variable step" begin
    x, p, u, t = MX.sym("x", 2), MX.sym("p"), MX.sym("u", 1, 3), MX.sym("t")
    control = MX.sym("c")
    dae = Dict("x" => x, "p" => p, "u" => control, "t" => t, "ode" => C.vertcat([x[2], -p * C.sin(x[1]) + control * t]), "quad" => C.dot(x, x))
    rule = C.integrator("rule", "cvodes", dae, 0.0, [0.5, 1.0, 2.0])
    run(x0, parameter, controls) = rule(x0, MX(0, 1), parameter, controls, MX(), MX(), MX())
    final, quadrature = run(x, p, u)[1], run(x, p, u)[3]
    f = CasADiCodegen.Function("f", [x, p, u], [final, quadrature, C.jacobian(final, x), C.jacobian(quadrature, p), C.jacobian(final[:, 3], u)])
    values = [[0.4, -0.3], [1.7], [0.2, -0.1, 0.5]]
    # The oracle is CasADi with 2000 steps of rk, whose error of about h^4 = 1e-12 is below the tolerance of the checks.
    fine = C.integrator("fine", "rk", dae, 0.0, [0.5, 1.0, 2.0], Dict("number_of_finite_elements" => 2000))
    reached = fine(x, MX(0, 1), p, u, MX(), MX(), MX())
    cost(final, quadrature) = C.sumsqr(final) + quadrature[3]
    reference = CasADiCodegen.Function("reference", [x, p, u], [reached[1], reached[3], C.jacobian(reached[1], x), C.jacobian(reached[3], p),
                                                                C.jacobian(reached[1][:, 3], u), C.gradient(cost(reached[1], reached[3]), C.vertcat([x, p, C.vec(u)]))])
    expected = mx_casadi(reference, values)
    solvers = Solvers(ode = PRECISE)
    @test_throws "CasADiCodegen.Solvers(ode = ...)" K.algorithm(f; expand = false)
    g = native(f; solvers)
    @test all_agree(evaluated(g, f, values), expected[1:5]; rtol = 1e-9)
    # The function without derivatives keeps the compilation short.
    plain = CasADiCodegen.Function("plain", [x, p, u], [final])
    @testset "reverse mode" begin
        # The gradient of one number by all arguments, for which CasADi asks the integrator for adjoint sensitivities.
        h = CasADiCodegen.Function("h", [x, p, u], [C.gradient(cost(final, quadrature), C.vertcat([x, p, C.vec(u)]))])
        @test evaluated(native(h; solvers), h, values)[1] ≈ expected[6] rtol = 1e-8
    end
    # The tolerances of the integrator of CasADi are those of the solver when the user gives none.
    # No public result names the tolerances in use, so the size of the error shows them.
    loose = evaluated(native(plain; solvers = Solvers(ode = Solver(Vern9(); specialize = SciMLBase.AutoSpecialize))), plain, values)
    @test all_agree(loose, expected[1:1]; rtol = 1e-4)
    @test !all_agree(loose, expected[1:1]; rtol = 1e-11)
end

@testset "one solver for one function" begin
    x, p = MX.sym("x", 2), MX.sym("p")
    dae = Dict("x" => x, "p" => p, "ode" => -p * x)
    fixed = C.integrator("fixed", "rk", dae, 0.0, 1.0)
    free = C.integrator("free", "cvodes", dae, 0.0, 1.0)
    call(rule) = rule(x, MX(0, 1), p, MX(0, 1), MX(), MX(), MX())[1]
    f = CasADiCodegen.Function("f", [x, p], [call(fixed), call(free), C.jacobian(call(fixed), p)])
    values = [[0.4, -0.3], [1.7]]
    exact = [0.4, -0.3] .* exp(-1.7)
    # The integrator with a fixed step keeps the Julia code of its own method, unless an override names it.
    # No public result names the method in use, so the error of the fixed step shows it.
    plain = evaluated(native(f; solvers = Solvers(ode = PRECISE)), f, values)
    @test !isapprox(plain[1], exact; rtol = 1e-9) && isapprox(plain[1], exact; rtol = 1e-3) && isapprox(plain[2], exact; rtol = 1e-11)
    named = evaluated(native(f; solvers = Solvers(ode = PRECISE, overrides = (Solver(fixed, Vern9(); abstol = 1e-13, reltol = 1e-13, specialize = SciMLBase.AutoSpecialize),))), f, values)
    @test isapprox(named[1], exact; rtol = 1e-11) && isapprox(named[3], -exact; rtol = 1e-10)
    # The override is for the function with that name and its derivatives only.
    @test_throws "function \"free\" is the integrator \"cvodes\" of CasADi" K.algorithm(f; expand = false, solvers = Solvers(overrides = (Solver(fixed, Vern9()),)))
    @test K.algorithm(f; expand = false, solvers = Solvers(overrides = (Solver(fixed, Vern9()), Solver(free, Vern7())))) isa K.MXAlgorithm
end

@testset "rootfinder" begin
    a, x, b = MX.sym("a"), MX.sym("x", 2), MX.sym("b", C.Sparsity.diag(2))
    residual = CasADiCodegen.Function("r", [a, x, b], [x * a, x^3 + C.mtimes(b, x) - a])
    values = [[2.0], [1.0, 1.0], [0.5, 3.0]]
    @testset "$plugin" for plugin in ("kinsol", "fast_newton")
        finder = C.rootfinder("finder", plugin, residual, Dict("implicit_input" => true, "implicit_output" => true, "abstol" => 1e-13))
        results = collect(finder(a, x, b))
        f = CasADiCodegen.Function("f", [a, x, b], [results..., C.jacobian(results[2], a), C.jacobian(results[2], b)])
        solvers = Solvers(nonlinear = Solver(SimpleNewtonRaphson(); abstol = 1e-13))
        @test_throws "CasADiCodegen.Solvers(nonlinear = ...)" K.algorithm(f; expand = false)
        # The two plugins take one path once they are named, so one of them runs.
        plugin == "kinsol" && @test all_agree(evaluated(native(f; solvers), f, values), mx_casadi(f, values); rtol = 1e-7)
    end
end

@testset "reverse mode and derivatives of the second order through a replaced integrator" begin
    x, p = MX.sym("x", 2), MX.sym("p")
    fixed = C.integrator("fixed", "rk", Dict("x" => x, "p" => p, "ode" => -p * x), 0.0, 1.0)
    f = CasADiCodegen.Function("f", [x, p], [C.gradient(C.sumsqr(fixed(x, MX(0, 1), p, MX(0, 1), MX(), MX(), MX())[1]), p)])
    values = [[0.4, -0.3], [1.7]]
    exact = [-2 * 0.25 * exp(-3.4)]
    named = evaluated(native(f; solvers = Solvers(overrides = (Solver(fixed, Vern9(); abstol = 1e-13, reltol = 1e-13, specialize = SciMLBase.AutoSpecialize),))), f, values)
    @test named[1] ≈ exact rtol = 1e-10
    # The Hessian through an integrator with a variable step: CasADi asks for the forward derivative of the adjoint sensitivities.
    free = C.integrator("free", "cvodes", Dict("x" => x, "p" => p, "ode" => -p * x), 0.0, 1.0)
    cost = C.sumsqr(free(x, MX(0, 1), p, MX(0, 1), MX(), MX(), MX())[1])
    f = CasADiCodegen.Function("f", [x, p], [C.gradient(cost, p), C.densify(CasADiCodegen.hessian(cost, C.vertcat([x, p])))])
    slope, curve = evaluated(native(f; solvers = Solvers(ode = PRECISE)), f, values)
    decay = exp(-3.4)
    @test slope ≈ exact rtol = 1e-10
    @test reshape(curve, 3, 3) ≈ [2decay 0 -1.6decay; 0 2decay 1.2decay; -1.6decay 1.2decay 4 * 0.25 * decay] rtol = 1e-8 atol = 1e-12
end

@testset "integrator of CasADi with an event function" begin
    x, e, g = MX.sym("x", 2), MX.sym("e"), MX.sym("g")
    dae = Dict("x" => x, "p" => e, "u" => g, "ode" => C.vertcat([x[2], -g]), "quad" => x[2]^2, "zero" => x[1])
    transition = C.Function("transition", Dict("x" => x, "p" => e, "post_x" => C.vertcat([x[1], -e * x[2]])), C.event_in(), C.event_out())
    times = [0.6, 1.4, 2.5]
    ball = C.integrator("ball", "cvodes", dae, 0.0, times, Dict("transition" => transition))
    p, u = MX.sym("p"), MX.sym("u", 1, 3)
    results = ball(x, MX(0, 1), p, u, MX(), MX(), MX())
    final, quadrature = C.vec(results[1]), C.vec(results[3])
    # CasADi gives the Jacobian of an integrator with events the sparsity pattern of the model without them, so the
    # Jacobian is that of directional derivatives, one per argument.
    arguments = C.vertcat([x, p, C.vec(u)])
    slopes = C.jtimes(C.vertcat([final, quadrature]), arguments, MX(C.DM([Float64(i == j) for i in 1:6, j in 1:6])))
    f = CasADiCodegen.Function("f", [x, p, u], [final, quadrature, C.densify(slopes)])
    # Four impacts, two of them in the last interval.
    values = [[1.5, 2.0], [0.8], [9.81, 5.0, 12.0]]
    J = ForwardDiff.jacobian(v -> reduce(vcat, ball_reference(v[1:2], v[3], v[4:6], times)), reduce(vcat, values))
    expected = [ball_reference(values[1], values[2][1], values[3], times)..., vec(J)]
    solvers = Solvers(ode = PRECISE)
    @test_throws "CasADiCodegen.Solvers(ode = ...)" K.algorithm(f; expand = false)
    # The Jacobian is that of the forward sensitivities, whose states jump at each impact.
    @test all_agree(evaluated(native(f; solvers), f, values), expected; rtol = 1e-8)
    @testset "jacobian of a graph with the integrator is that of forward directions" begin
        slopes_jacobian = CasADiCodegen.Function("slopes_jacobian", [x, p, u], [C.densify(jacobian(C.vertcat([final, quadrature]), arguments))])
        @test all_agree(evaluated(native(slopes_jacobian; solvers), slopes_jacobian, values), expected[3:3]; rtol = 1e-8)
        @test_throws "jtimes" CasADiCodegen.Function("whole", [x, p, u], [final]).jacobian()
    end
    plain = CasADiCodegen.Function("plain", [x, p, u], [final])
    @testset "a written module has the solver" begin
        text = sprint(io -> generate_julia(plain, io; module_name = :Written, name = :run!, expand = false, solvers))
        src = Module()
        Base.include_string(src, text)
        written = Base.invokelatest((r, a) -> src.Written.run_nonzeros!(r, a), [fill(NaN, 6)], Tuple(values))
        @test all_agree(written, expected[1:1]; rtol = 1e-9)
    end
    # As the integrator of CasADi, a call stops after `max_events` events, 20 by default: an elastic ball under a strong
    # gravity hits the ground about 30 times.
    @test_throws "more than 20 events" evaluated(native(plain; solvers), plain, [values[1], [1.0], fill(2000.0, 3)])
    # An event exactly on an output time, where CasADi 3.8.1 misses it: the generated code applies it and returns
    # the state after it at that time, the exact solution of N' = 100 - N with N += 50 at t = 10.
    N, M, τ = MX.sym("N"), MX.sym("M"), MX.sym("t")
    injection = C.Function("transition", Dict("x" => N, "p" => M, "post_x" => N + M), C.event_in(), C.event_out())
    cells = C.integrator("cells", "cvodes", Dict("x" => N, "p" => M, "t" => τ, "ode" => 100 - N, "zero" => 10 - τ), 0.0, [5.0, 10.0, 15.0],
                         Dict("transition" => injection))
    injected = CasADiCodegen.Function("injected", [N, M], [cells(N, MX(0, 1), M, MX(1, 0), MX(), MX(), MX())[1]])
    after_ten = 100 * (1 - exp(-10.0)) + 50
    exact = [100 * (1 - exp(-5.0)), after_ten, 100 + (after_ten - 100) * exp(-5.0)]
    @test all_agree(evaluated(native(injected; solvers), injected, [[0.0], [50.0]]), [exact]; rtol = 1e-9)
    # A fixed step has no Julia code with events: only an override replaces it.
    fixed = C.integrator("fixed", "rk", dae, 0.0, times, Dict("transition" => transition))
    k = CasADiCodegen.Function("k", [x, p, u], [fixed(x, MX(0, 1), p, u, MX(), MX(), MX())[1]])
    @test_throws "overrides = [CasADiCodegen.Solver(I, alg)]" K.algorithm(k; expand = false, solvers)
    named = evaluated(native(k; solvers = Solvers(overrides = (Solver(fixed, Vern9(); abstol = 1e-13, reltol = 1e-13, specialize = SciMLBase.AutoSpecialize),))), k, values)
    @test all_agree(named, expected[1:1]; rtol = 1e-9)
end

# Algebraic states: the state z = p x1 makes x1' = -z a decay with the rate p, and x2 is the integral of x1.
const DAE_SOLVER = Solver(Rodas5P(); abstol = 1e-11, reltol = 1e-11, specialize = SciMLBase.AutoSpecialize)
dae_exact(x0, rate) = (final = x0[1] * exp(-rate); [final, x0[2] + x0[1] * (1 - exp(-rate)) / rate, rate * final])

# The set takes more than 15 s: its first evaluation compiles Rodas5P for a mass matrix, and the adjoint
# integrator and the event each compile another solver; each part is the one case of its behavior.
@testset "integrator of this package with algebraic states" begin
    y, p = MX.sym("y", 3), MX.sym("p")
    model = Dict("x" => y, "p" => p, "ode" => C.vertcat([-y[3], y[1], y[3] - p * y[1]]))
    rule = CasADiCodegen.integrator("stiff", Rodas5P(), model, 0.0, 1.0; mass = [1.0, 1.0, 0.0], abstol = 1e-11, reltol = 1e-11, specialize = SciMLBase.AutoSpecialize)
    f = CasADiCodegen.Function("f", [y, p], [rule(y, p), C.jacobian(rule(y, p), y), C.jacobian(rule(y, p), p)])
    g = native(f)
    by_rate = ForwardDiff.derivative(r -> dae_exact([0.4, -0.3], r), 1.7)
    by_state = ForwardDiff.jacobian(v -> dae_exact(v, 1.7), [0.4, -0.3])
    # The third initial value is a guess of the algebraic state: the result does not depend on it.
    for guess in (0.68, 0.0)
        final, J, P = evaluated(g, f, [[0.4, -0.3, guess], [1.7]])
        @test final ≈ dae_exact([0.4, -0.3], 1.7) rtol = 1e-8
        @test reshape(J, 3, 3) ≈ [by_state zeros(3)] rtol = 1e-7 atol = 1e-9
        @test P ≈ by_rate rtol = 1e-7
    end
    @test_throws "one number per state" CasADiCodegen.integrator("stiff", Rodas5P(), model, 0.0, 1.0; mass = [1.0, 0.0])
    @test_throws "a vector of times" CasADiCodegen.integrator("grid", Rodas5P(), model, 0.0, [0.5, 1.0]; mass = [1.0, 1.0, 0.0])
    # The adjoint equations give the gradient of a weighted sum of the final state, the algebraic state included.
    @testset "adjoint equations" begin
        backward = CasADiCodegen.integrator("stiff_a", Rodas5P(), model, 0.0, 1.0; mass = [1.0, 1.0, 0.0], adjoint = true, abstol = 1e-11, reltol = 1e-11, specialize = SciMLBase.AutoSpecialize)
        weights = [0.7, -1.3, 2.1]
        h = CasADiCodegen.Function("h", [y, p], [C.gradient(C.dot(MX(C.DM(weights)), backward(y, p)), C.vertcat([y, p]))])
        gradient = evaluated(native(h), h, [[0.4, -0.3, 0.2], [1.7]])[1]
        # The guess of the algebraic state has no effect on the result.
        @test gradient ≈ [by_state' * weights; 0.0; sum(by_rate .* weights)] rtol = 1e-7 atol = 1e-9
        # With free times the gradient by the final time is the rate of the weighted sum, and that by the initial time its opposite.
        free = CasADiCodegen.integrator("stiff_f", Rodas5P(), model; mass = [1.0, 1.0, 0.0], adjoint = true, abstol = 1e-11, reltol = 1e-11, specialize = SciMLBase.AutoSpecialize)
        ta, tb = MX.sym("ta"), MX.sym("tb")
        k = CasADiCodegen.Function("k", [y, p, ta, tb], [C.gradient(C.dot(MX(C.DM(weights)), free(y, p, ta, tb)), C.vertcat([ta, tb]))])
        final = dae_exact([0.4, -0.3], 1.7)
        rate = sum([-final[3], final[1], -1.7 * final[3]] .* weights)
        @test evaluated(native(k), k, [[0.4, -0.3, 0.2], [1.7], [0.5], [1.5]])[1] ≈ [-rate, rate] rtol = 1e-7
        # Forward over reverse: the Hessian by the parameter differentiates the adjoint integrator.
        curve = CasADiCodegen.Function("curve", [y, p], [C.hessian(C.dot(MX(C.DM(weights)), backward(y, p)), p)[1]])
        bend = ForwardDiff.derivative(r -> ForwardDiff.derivative(s -> sum(dae_exact([0.4, -0.3], s) .* weights), r), 1.7)
        @test evaluated(native(curve), curve, [[0.4, -0.3, 0.2], [1.7]])[1] ≈ [bend] rtol = 1e-6
    end
    # An algebraic equation with no solution, z^2 + 1 = 0: the Newton method stops after its 50 steps.
    never = K.RightSide((res, arg) -> (res[1][1] = -arg[1][1]; res[1][2] = arg[1][2]^2 + 1; nothing), [1.0, 0.0], (res, arg) -> (res[1][1] = 2 * arg[1][2]; nothing))
    @test_throws "the algebraic equations of the model have no consistent initial values near the guess" K.consistent!([1.0, 0.3], never, Float64[], 0.0)
    # With free times the derivative of an algebraic state by the final time comes from the algebraic equation, not from its residual.
    arc = CasADiCodegen.integrator("stiff", Rodas5P(), model; mass = [1.0, 1.0, 0.0], abstol = 1e-11, reltol = 1e-11, specialize = SciMLBase.AutoSpecialize)
    t0, tf = MX.sym("t0"), MX.sym("tf")
    m = CasADiCodegen.Function("m", [y, p, t0, tf], [C.jacobian(arc(y, p, t0, tf), C.vertcat([y, t0, tf]))])
    final = dae_exact([0.4, -0.3], 1.7)
    rate = [-final[3], final[1], -1.7 * final[3]]
    @test reshape(evaluated(native(m), m, [[0.4, -0.3, 0.2], [1.7], [0.5], [1.5]])[1], 3, 5) ≈ [by_state zeros(3) -rate rate] rtol = 1e-7 atol = 1e-9
    # An event function of the algebraic state z = r x, with a jump of x, so that z no longer satisfies its equation after it.
    @testset "an event" begin
        w = MX.sym("w", 2)
        decay = Dict("x" => w, "p" => p, "ode" => C.vertcat([-w[2], w[2] - p * w[1]]), "zero" => w[2] - 0.55, "transition" => C.vertcat([2w[1], w[2]]))
        refill = CasADiCodegen.integrator("refill", Rodas5P(), decay; mass = [1.0, 0.0], events = 2, terminal = false, abstol = 1e-11, reltol = 1e-11,
                                          specialize = SciMLBase.AutoSpecialize)
        out = refill(w, p, MX(0.0), MX(1.6))
        n = CasADiCodegen.Function("n", [w, p], [out[1], out[2], C.densify(C.jacobian(C.vertcat([out[1], out[2]]), C.vertcat([w, p])))])
        onset(x0, rate) = log(rate * x0 / 0.55) / rate
        exact(v) = (x = refilled(1.6, v[1], v[3]); [x, v[3] * x, onset(v[1], v[3]), onset(v[1], v[3]) + log(2) / v[3]])
        at = [1.0, 0.3, 1.1]                      # the second number is a guess of the algebraic state
        @test all_agree(evaluated(native(n), n, [at[1:2], at[3:3]]), [exact(at)[1:2], exact(at)[3:4], vec(ForwardDiff.jacobian(exact, at))]; rtol = 1e-7)
        # Exhaustive tier, because the solver compiles again for dual numbers: dual numbers through the
        # generated code give the derivative of the first event time. The integrator group checks dual numbers through an event.
        if EXHAUSTIVE
            times = build_function(CasADiCodegen.Function("e", [w, p], [out[2]]); nonzeros = true, expression = Val{false})
            event(v) = (res = [zeros(eltype(v), 2)]; times(res, (v[1:2], v[3:3])); res[1][1])
            @test ForwardDiff.gradient(event, at) ≈ ForwardDiff.gradient(v -> onset(v[1], v[3]), at) rtol = 1e-7
        end
    end
end

@testset "integrator of CasADi with algebraic states" begin
    x, z, p = MX.sym("x", 2), MX.sym("z"), MX.sym("p")
    rule = C.integrator("rule", "idas", Dict("x" => x, "z" => z, "p" => p, "ode" => C.vertcat([-z, x[1]]), "alg" => z - p * x[1], "quad" => z), 0.0, [0.5, 1.0])
    results = rule(x, z, p, MX(0, 1), MX(), MX(), MX())
    f = CasADiCodegen.Function("f", [x, z, p], [results[1], results[2], results[3], C.jacobian(results[1][:, 2], p), C.jacobian(results[2][:, 2], x)])
    solvers = Solvers(dae = DAE_SOLVER)
    @test_throws "function \"rule\" is the integrator \"idas\" of CasADi" K.algorithm(f; expand = false)
    values = [[0.4, -0.3], [0.1], [1.7]]
    half = dae_exact([0.4, -0.3], 1.7 * 0.5) .* [1, 1, 2]       # the state at the time 0.5: a decay over half the time
    half[2] = -0.3 + 0.4 * (1 - exp(-0.85)) / 1.7
    full = dae_exact([0.4, -0.3], 1.7)
    by_rate = ForwardDiff.derivative(r -> dae_exact([0.4, -0.3], r), 1.7)
    by_state = ForwardDiff.jacobian(v -> dae_exact(v, 1.7), [0.4, -0.3])
    expected = [[half[1:2]; full[1:2]], [half[3], full[3]], [0.4 - half[1], 0.4 - full[1]], by_rate[1:2], by_state[3, 1:1]]       # CasADi knows that the algebraic state does not depend on the second state
    g = native(f; solvers)
    @test all_agree(evaluated(g, f, values), expected; rtol = 1e-7)
    # Reverse mode: the gradient by the parameter of one number of the states, the algebraic state and the quadrature.
    h = CasADiCodegen.Function("h", [x, z, p], [C.gradient(C.sumsqr(results[1][:, 2]) + results[2][2] + results[3][2], p)])
    @test evaluated(native(h; solvers), h, values)[1] ≈ [2 * full[1] * by_rate[1] + 2 * full[2] * by_rate[2] + by_rate[3] - by_rate[1]] rtol = 1e-7
    # The solver for ordinary differential equations is not that of a model with algebraic states.
    @test_throws "give one with the option solvers = CasADiCodegen.Solvers(dae = ...)" K.algorithm(f; expand = false, solvers = Solvers(ode = PRECISE))
    # An event: the decay x' = -z with z = r x doubles x when z falls to 0.55, so that the algebraic state follows the jump.
    # The event function depends on the algebraic state, so the event time depends on its rate.
    y, r = MX.sym("y"), MX.sym("r")
    doubling = C.Function("transition", Dict("x" => y, "post_x" => 2y), C.event_in(), C.event_out())
    refill = C.integrator("refill", "idas", Dict("x" => y, "z" => z, "p" => r, "ode" => -z, "alg" => z - r * y, "zero" => z - 0.55), 0.0, [0.4, 1.0, 1.6],
                          Dict("transition" => doubling))
    after = refill(y, z, r, MX(0, 1), MX(), MX(), MX())
    k = CasADiCodegen.Function("k", [y, z, r], [after[1], after[2], C.jtimes(after[1], r, MX(1.0)), C.jtimes(after[1], y, MX(1.0)), C.jtimes(after[2], r, MX(1.0))])
    states(x0, rate) = [refilled(t, x0, rate) for t in (0.4, 1.0, 1.6)]
    by_rate = ForwardDiff.derivative(rate -> states(1.0, rate), 1.1)
    expected = [states(1.0, 1.1), 1.1 * states(1.0, 1.1), by_rate, ForwardDiff.derivative(x0 -> states(x0, 1.1), 1.0), states(1.0, 1.1) + 1.1 * by_rate]
    @test all_agree(evaluated(native(k; solvers), k, [[1.0], [0.0], [1.1]]), expected; rtol = 1e-7)
end
