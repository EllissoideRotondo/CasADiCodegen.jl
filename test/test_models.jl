# Models written as equations with `Differential` and `~`, and compiled by `System`.
using CasADiCodegen.Models
import SciMLBase
using OrdinaryDiffEqRosenbrock: Rodas5P
using OrdinaryDiffEqVerner: Vern9

names_of(v) = [C.name(s) for s in v]
solve(problem) = SciMLBase.solve(problem, Rodas5P(); abstol = 1e-10, reltol = 1e-10)

@testset "derivatives of symbols and expressions" begin
    t, x, y = SX.sym("t"), SX.sym("x"), SX.sym("y")
    D = Differential(t)
    # One derivative symbol per symbol, whichever Differential of t makes it.
    @test C.is_equal(D(x), Differential(t)(x)) && C.name(D(D(x))) == "ddx"
    # The chain rule, with the explicit time: d/dt (x y + sin t) = dx y + x dy + cos t.
    expected = D(x) * y + x * D(y) + cos(t)
    f = CasadiFunction("f", [vcat(x, y, D(x), D(y), t)], [D(x * y + sin(t)) - expected])
    @test only(casadi_eval(f, [0.3, -1.2, 0.7, 2.1, 0.4])[1]) ≈ 0 atol = 1e-14
    # A vector symbol gives the vector of the derivatives of its elements.
    q = SX.sym("q", 2)
    @test names_of(C.vertsplit(D(q))) == ["dq_0", "dq_1"]
end

@testset "a second-order equation becomes two states" begin
    t, x, ω = SX.sym("t"), SX.sym("x"), SX.sym("ω")
    D = Differential(t)
    sys = System([D(D(x)) ~ -ω^2 * x], t; parameters = [ω], name = "oscillator")
    @test names_of(unknowns(sys)) == ["x", "dx"]
    problem = SciMLBase.ODEProblem(sys, [x => 1.0, D(x) => 0.0], (0.0, 2.0), [ω => 3.0])
    solution = SciMLBase.solve(problem, Vern9(); abstol = 1e-12, reltol = 1e-12)
    @test solution[:x][end] ≈ cos(6.0) rtol = 1e-9
    # The derivative of the highest order is an observed variable.
    @test solution[:ddx][end] ≈ -9 * cos(6.0) rtol = 1e-8
end

@testset "aliases of a circuit are eliminated into observed variables" begin
    t, R, Cap, V = SX.sym("t"), SX.sym("R"), SX.sym("C"), SX.sym("V")
    D = Differential(t)
    pin(name) = (v = SX.sym("$(name)_v"), i = SX.sym("$(name)_i"))
    function oneport(name)
        p, n, v, i = pin("$(name)_p"), pin("$(name)_n"), SX.sym("$(name)_v"), SX.sym("$(name)_i")
        (; p, n, v, i, equations = [v ~ p.v - n.v, 0 ~ p.i + n.i, i ~ p.i])
    end
    connect(a, b...) = [[a.v ~ q.v for q in b]; 0 ~ a.i + sum(q.i for q in b)]
    source, resistor, capacitor, ground = oneport("source"), oneport("resistor"), oneport("capacitor"), pin("ground")
    equations = [source.equations; resistor.equations; capacitor.equations
                 resistor.v ~ R * resistor.i
                 Cap * D(capacitor.v) ~ capacitor.i
                 source.v ~ V
                 ground.v ~ 0
                 connect(source.p, resistor.p)
                 connect(resistor.n, capacitor.p)
                 connect(capacitor.n, source.n, ground)]
    sys = System(equations, t; parameters = [R, Cap, V], name = "rc")
    # 20 unknowns and 20 equations, of which one is differential and every other is solved explicitly.
    @test names_of(unknowns(sys)) == ["capacitor_v"]
    @test length(observed(sys)) == 20    # 19 eliminated unknowns and the derivative of the state
    problem = SciMLBase.ODEProblem(sys, [:capacitor_v => 0.0], (0.0, 10.0), [:R => 2.0, :C => 1.0, :V => 1.0])
    solution = solve(problem)
    @test solution[:capacitor_v][end] ≈ 1 - exp(-5) rtol = 1e-8
    @test solution[:resistor_i][end] ≈ exp(-5) / 2 rtol = 1e-7
end

@testset "linearly coupled derivatives and a nonlinear implicit one" begin
    t, x, y, w = SX.sym("t"), SX.sym("x"), SX.sym("y"), SX.sym("w")
    D = Differential(t)
    # x' + y' = -x - y and x' - y' = x - y: x' = -y, y' = -x.
    coupled = System([D(x) + D(y) ~ -x - y, D(x) - D(y) ~ x - y], t)
    @test isempty(algebraic(coupled))
    solution = solve(SciMLBase.ODEProblem(coupled, [x => 1.0, y => 0.0], (0.0, 1.0)))
    @test [solution[:x][end], solution[:y][end]] ≈ [cosh(1.0), -sinh(1.0)] rtol = 1e-8
    # exp(w') = exp(-w) has no explicit form for the solver to find, so w' becomes an algebraic unknown.
    implicit = System([exp(D(w)) ~ exp(-w)], t)
    @test names_of(algebraic(implicit)) == ["dw"]
    solution = solve(SciMLBase.ODEProblem(implicit, [w => 1.0, D(w) => -1.0], (0.0, 1.0)))
    @test solution[:w][end] ≈ exp(-1.0) rtol = 1e-8
end

@testset "the index of a pendulum is reduced" begin
    t, x, y, T, g, L = (SX.sym(n) for n in ("t", "x", "y", "T", "g", "L"))
    D = Differential(t)
    sys = System([D(D(x)) ~ T * x, D(D(y)) ~ T * y - g, x^2 + y^2 ~ L^2], t; parameters = [g, L], name = "pendulum")
    @test differentiation_index(sys) == 3 && size(invariants(sys), 1) == 2
    # From rest at the angle 0.5 from the vertical, against the angle ODE θ'' = -(g / L) sin θ.
    θ0, tf = 0.5, 1.5
    u0 = [x => sin(θ0), y => -cos(θ0), D(x) => 0.0, D(y) => 0.0, T => -9.81 * cos(θ0)]
    solution = solve(SciMLBase.ODEProblem(sys, u0, (0.0, tf), [g => 9.81, L => 1.0]))
    θ, ω = SX.sym("θ"), SX.sym("ω")
    angle = C.integrator("A", "cvodes", Dict("x" => vcat(θ, ω), "ode" => vcat(ω, -9.81 * sin(θ))), 0.0, tf,
                         Dict("abstol" => 1e-12, "reltol" => 1e-12))
    θf = Float64(angle(Dict("x0" => C.DM([θ0, 0.0])))["xf"][1])
    @test [solution[:x][end], solution[:y][end]] ≈ [sin(θf), -cos(θf)] atol = 1e-6
    # The dictionary of the model goes to the integrators of CasADi.
    dae = Dict(sys)
    F = C.integrator("F", "idas", dae, 0.0, tf, Dict("abstol" => 1e-10, "reltol" => 1e-10))
    start = Dict(C.name(s) => v for (s, v) in u0)
    @test names_of(unknowns(sys)) == ["x", "dx", "y", "dy", "T"]
    xf = F(Dict("x0" => C.DM([start[n] for n in ("x", "dx", "y", "dy")]), "z0" => C.DM([start["T"]]), "p" => C.DM([9.81, 1.0])))["xf"]
    @test Matrix(xf)[[1, 3]] ≈ [sin(θf), -cos(θf)] atol = 1e-6
end

@testset "a time derivative of a parameter is zero" begin
    t, x, a = SX.sym("t"), SX.sym("x"), SX.sym("a")
    D = Differential(t)
    sys = System([D(x) ~ D(a * t)], t; parameters = [a])
    solution = solve(SciMLBase.ODEProblem(sys, [x => 0.0], (0.0, 2.0), [a => 1.5]))
    @test solution[:x][end] ≈ 3.0 rtol = 1e-10
end

@testset "models that System refuses" begin
    t, x, y, k = SX.sym("t"), SX.sym("x"), SX.sym("y"), SX.sym("k")
    D = Differential(t)
    refusal(f) = try f(); nothing catch e e end
    # More equations than unknowns.
    e = refusal(() -> System([D(x) ~ -x, x ~ 2 * k], t; parameters = [k]))
    @test e isa ArgumentError && occursin("2 equations", e.msg) && occursin("1 variable", e.msg)
    # An equation that the elimination of the others leaves constant.
    w = SX.sym("w")
    e = refusal(() -> System([D(x) ~ y + w, y ~ 1, y ~ 2], t))
    @test e isa ArgumentError && occursin("equation 3", e.msg)
    # Two symbols with one name.
    other = SX.sym("x")
    e = refusal(() -> System([D(x) ~ other, other ~ 1], t))
    @test e isa ArgumentError && occursin("\"x\"", e.msg)
    # A differential state without an initial value.
    sys = System([D(x) ~ -k * x], t; parameters = [k])
    e = refusal(() -> SciMLBase.ODEProblem(sys, Pair[], (0.0, 1.0), [k => 1.0]))
    @test e isa ArgumentError && occursin("x", e.msg)
    @test refusal(() -> Differential(MX.sym("t"))) isa MethodError
end
