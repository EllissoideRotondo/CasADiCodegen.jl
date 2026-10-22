# Ordinary differential equations integrated by a SciML solver inside CasADi
# graphs and inside the generated code.
include("ode_helpers.jl")

@testset "an integrator is a CasADi function" begin
    F = two_body_integrator(; specialize = SciMLBase.FullSpecialize)        # the default
    @test (C.n_in(F), C.n_out(F)) == (2, 1)
    ours = casadi_eval(F, ODE_X0, ODE_MU)
    reference = casadi_eval(with_derivatives(cvodes_call(two_body_cvodes()), 0), ODE_X0, ODE_MU)
    @test closeto(ours, reference, 1e-9)
    f = with_derivatives(F, 0)
    @test ode_emitted(f, (ODE_X0, ODE_MU)) == ours
    @test ode_emitted(f, (ODE_X0, ODE_MU); expand = false, chunk_size = 1) == ours
end

@testset "derivatives to order two agree with cvodes" begin
    f = with_derivatives(two_body_integrator(), 2)
    reference = casadi_eval(with_derivatives(cvodes_call(two_body_cvodes()), 2), ODE_X0, ODE_MU)
    @test closeto(casadi_eval(f, ODE_X0, ODE_MU), reference, 1e-8)
    @test closeto(ode_emitted(f, (ODE_X0, ODE_MU)), reference, 1e-8)
end

@testset "each order integrates the derivatives by the initial state and the parameters only" begin
    F = two_body_integrator()
    with_derivatives(F, 2)
    record(integration) = only(call for (call, _) in values(CasADiCodegen.Codegen.JULIA_CALLS) if call.f === integration)
    first_order = CasADiCodegen.Codegen.JULIA_CALLS[C.__hash__(F)][1].f.sensitivity[]
    second_order = first_order.sensitivity[]
    # 4 states and 1 parameter: 4 * (1 + 4 + 1) states, then 24 * (1 + 4 + 1).
    @test record(first_order).nnz_in == [4, 1] && record(first_order).nnz_out == [24]
    @test record(second_order).nnz_in == [4, 1] && record(second_order).nnz_out == [144]
end

@testset "the Jacobian integrator is built on demand" begin
    built = length(CasADiCodegen.Codegen.JULIA_CALLS)
    F = two_body_integrator()
    @test length(CasADiCodegen.Codegen.JULIA_CALLS) == built + 1
    x0, p = MX.sym("x0", 4), MX.sym("p", 1)
    J = jacobian(F(x0, p), x0)
    @test length(CasADiCodegen.Codegen.JULIA_CALLS) == built + 2
    jacobian(F(x0, p), p)                       # CasADi keeps the Jacobian function
    @test length(CasADiCodegen.Codegen.JULIA_CALLS) == built + 2
    jacobian(J, x0)
    @test length(CasADiCodegen.Codegen.JULIA_CALLS) == built + 3
end

@testset "time in the right side, no parameter, backward in time" begin
    x, p, t = SX.sym("x", 1), SX.sym("p", 1), SX.sym("t")
    F = integrator("T", Vern9(), Dict("x" => x, "p" => p, "t" => t, "ode" => t * p), 1.0, 3.0; TIGHT...)
    x0, q = MX.sym("x0", 1), MX.sym("q", 1)
    f = CasADiCodegen.Function("f", [x0, q], [F(x0, q), jacobian(F(x0, q), x0), jacobian(F(x0, q), q)])
    @test closeto(ode_emitted(f, ([0.5], [2.0])), [[8.5], [1.0], [4.0]], 1e-12)
    @test closeto(casadi_eval(f, [0.5], [2.0]), [[8.5], [1.0], [4.0]], 1e-12)

    y = SX.sym("y", 2)
    D = integrator("D", Vern9(), Dict("x" => y, "ode" => vertcat([-y[1], 2 * y[2]])), 0.0, 1.0; TIGHT...)
    @test C.n_in(D) == 1
    y0 = MX.sym("y0", 2)
    d = CasADiCodegen.Function("d", [y0], [D(y0), jacobian(D(y0), y0), jacobian(jacobian(D(y0), y0), y0)])
    expected = [[exp(-1) * 3, exp(2) * 0.5], [exp(-1), 0.0, 0.0, exp(2)], zeros(8)]
    @test closeto(ode_emitted(d, ([3.0, 0.5],)), expected, 1e-11)
    @test closeto(casadi_eval(d, [3.0, 0.5]), expected, 1e-11)

    B = integrator("B", Vern9(), Dict("x" => y, "ode" => vertcat([-y[1], 2 * y[2]])), 1.0, 0.0; TIGHT...)
    b = CasADiCodegen.Function("b", [y0], [B(D(y0))])
    @test closeto(ode_emitted(b, ([3.0, 0.5],)), [[3.0, 0.5]], 1e-11)
end

@testset "matrix right side that calls a function" begin
    a = SX.sym("a", 2)
    force = CasADiCodegen.Function("force", [a], [vertcat([a[2], -sin(a[1])])])
    x = MX.sym("x", 2)
    F = integrator("P", Vern9(), Dict("x" => x, "ode" => force(x)), 0.0, 2.0; TIGHT...)
    R = integrator("PR", "cvodes", Dict("x" => a, "ode" => force(a)), 0.0, 2.0, Dict("abstol" => 1e-12, "reltol" => 1e-12))
    x0 = MX.sym("x0", 2)
    f = CasADiCodegen.Function("f", [x0], [F(x0), jacobian(F(x0), x0)])
    xr = R(Dict("x0" => x0))["xf"]
    r = CasADiCodegen.Function("r", [x0], [xr, jacobian(xr, x0)])
    @test closeto(ode_emitted(f, ([0.4, 0.3],)), casadi_eval(r, [0.4, 0.3]), 1e-9)
end

@testset "arcs in a chain, scalar and matrix code" begin
    F = two_body_integrator()
    x0, p = MX.sym("x0", 4), MX.sym("p", 1)
    xf = F(F(x0, p) * 1.0, p)
    f = CasADiCodegen.Function("chain", [x0, p], [xf, jacobian(xf, x0)])
    R = cvodes_call(two_body_cvodes())
    xr = R(R(x0, p), p)
    reference = casadi_eval(CasADiCodegen.Function("r", [x0, p], [xr, jacobian(xr, x0)]), ODE_X0, ODE_MU)
    ours = ode_emitted(f, (ODE_X0, ODE_MU))
    @test closeto(ours, reference, 1e-7)
    # Matrix code runs the calls in another order, so that the state can come from another integration.
    @test closeto(ode_emitted(f, (ODE_X0, ODE_MU); expand = false), ours, 1e-10)
    @test closeto(ode_emitted(f, (ODE_X0, ODE_MU); expand = false, chunk_size = 1), ours, 1e-10)
end

@testset "the state comes from the integration of the sensitivities" begin
    F = two_body_integrator()
    run = CasADiCodegen.Codegen.JULIA_CALLS[C.__hash__(F)][1].f
    @test run.sensitivity[] === nothing
    f = with_derivatives(F, 1)
    sibling = run.sensitivity[]
    @test sibling isa CasADiCodegen.Codegen.Integration
    calls(integration) = task_local_storage()[(integration.solver, Float64)].stats.nf
    memo() = CasADiCodegen.Codegen.memo(sibling, Float64)
    res = [zeros(4), zeros(16), zeros(4)]
    state_only = build_function(with_derivatives(F, 0); nonzeros = true, expression = Val{false})
    alone = state_only([zeros(4)], (ODE_X0, ODE_MU))[1]
    sibling(ODE_X0, ODE_MU)                                        # so that the task has both solvers
    # Scalar code around the calls, and matrix code: one integration, that of the sensitivities.
    for options in ((;), (; expand = false))
        g = build_function(f; nonzeros = true, expression = Val{false}, options...)
        before = (calls(run), calls(sibling))
        g(res, (ODE_X0, ODE_MU))
        @test calls(run) == before[1]
        @test calls(sibling) > before[2]
        @test closeto([res[1]], [alone], 1e-11)
        @test closeto(res, casadi_eval(f, ODE_X0, ODE_MU), 1e-10)
        @test memo().head > memo().count                           # each record was used
    end
    # A function with the state alone runs the integration of the state.
    before = (calls(run), calls(sibling))
    @test state_only([zeros(4)], (ODE_X0, ODE_MU))[1] == alone
    @test calls(run) > before[1] && calls(sibling) == before[2]
    # A run that fails leaves no record.
    short = two_body_integrator(; maxiters = 200)
    h = build_function(with_derivatives(short, 1); nonzeros = true, expression = Val{false})
    h(res, (ODE_X0, ODE_MU))
    far = [1.0, 0.0, 0.0, 0.05]                                    # close pass: needs more steps
    redirect_stderr(devnull) do
        @test_throws ErrorException h(res, (far, ODE_MU))
    end
    short_memo = CasADiCodegen.Codegen.memo(CasADiCodegen.Codegen.JULIA_CALLS[C.__hash__(short)][1].f.sensitivity[], Float64)
    @test short_memo.head > short_memo.count
end

@testset "arcs in a map: one integration for each arc" begin
    F = two_body_integrator()
    run = CasADiCodegen.Codegen.JULIA_CALLS[C.__hash__(F)][1].f
    arcs = 3
    X, p = MX.sym("X", 4, arcs), MX.sym("p", 1)
    ends = C.map(F, arcs)(X, C.repmat(p, 1, arcs))
    f = CasADiCodegen.Function("shoot", [X, p], [ends, jacobian(ends, X)])
    g = build_function(f; nonzeros = true, expression = Val{false})
    starts = vec([ODE_X0[i] * (1 + 0.01k) for i in 1:4, k in 1:arcs])
    # Integrators with equal content share the state of the solver, so the calls are counted.
    calls() = (state = get(task_local_storage(), (run.solver, Float64), nothing); state === nothing ? 0 : state.stats.nf)
    before = calls()
    res = g([zeros(4 * arcs), zeros(C.nnz_out(f, 1))], (starts, ODE_MU))
    @test calls() == before                                        # the state alone was not integrated
    @test closeto(res, casadi_eval(f, reshape(starts, 4, arcs), ODE_MU), 1e-10)
    sibling = run.sensitivity[]
    @test CasADiCodegen.Codegen.memo(sibling, Float64).head > arcs
end

@testset "free initial and final times" begin
    F = integrator("G", Vern9(), two_body(); TIGHT...)
    @test C.n_in(F) == 4 && C.n_out(F) == 1
    @test [C.name_in(F, i) for i in 0:3] == ["x0", "p", "t0", "tf"]
    x0, p, t0, tf = MX.sym("x0", 4), MX.sym("p", 1), MX.sym("t0"), MX.sym("tf")
    xf = F(x0, p, t0, tf)
    f = CasADiCodegen.Function("f", [x0, p, t0, tf], [xf, jacobian(xf, x0), jacobian(xf, p), jacobian(xf, t0), jacobian(xf, tf)])
    g = build_function(f; nonzeros = true, expression = Val{false})
    run(a, b) = g([zeros(4), zeros(16), zeros(4), zeros(4), zeros(4)], (ODE_X0, ODE_MU, [a], [b]))
    # The times of the integrator with fixed times, and the same arc later: the model has no time in it.
    reference = ode_emitted(with_derivatives(two_body_integrator(), 1), (ODE_X0, ODE_MU))
    ours = run(0.0, 3.0)
    @test closeto(ours[1:3], reference, 1e-10)
    @test closeto(run(1.5, 4.5)[1:3], reference, 1e-10)
    @test closeto(casadi_eval(f, ODE_X0, ODE_MU, 0.0, 3.0), ours, 1e-10)
    # The derivative by the final time is the right side there; that by the initial time is its opposite here.
    r3 = (ours[1][1]^2 + ours[1][2]^2)^1.5
    rate = [ours[1][3], ours[1][4], -ours[1][1] / r3, -ours[1][2] / r3]
    @test closeto(ours[5:5], [rate], 1e-10)
    @test closeto(ours[4:4], [-rate], 1e-9)
    # Backward in time, and an arc of no length.
    back = g([zeros(4), zeros(16), zeros(4), zeros(4), zeros(4)], (ours[1], ODE_MU, [3.0], [0.0]))
    @test closeto(back[1:1], [ODE_X0], 1e-10)
    still = run(2.0, 2.0)
    @test still[1] == ODE_X0 && still[2] == [Float64(i == j) for i in 1:4 for j in 1:4] && still[3] == zeros(4)

    # Time in the right side and no parameter: x' = t x, so x(tf) = x0 exp((tf^2 - t0^2) / 2).
    y, t = SX.sym("y", 1), SX.sym("t")
    T = integrator("T2", Vern9(), Dict("x" => y, "t" => t, "ode" => t * y); TIGHT...)
    @test C.n_in(T) == 3
    y0 = MX.sym("y0", 1)
    yf = T(y0, t0, tf)
    first_order = [jacobian(yf, v) for v in (y0, t0, tf)]
    second_order = [jacobian(first_order[2], t0), jacobian(first_order[3], tf), jacobian(first_order[3], t0), jacobian(first_order[3], y0)]
    d = CasADiCodegen.Function("d", [y0, t0, tf], [yf, first_order..., second_order...])
    a, b, c = 0.7, 0.5, 1.5
    e = exp((c^2 - b^2) / 2)
    expected = [[a * e], [e], [-b * a * e], [c * a * e], [(b^2 - 1) * a * e], [(c^2 + 1) * a * e], [-b * c * a * e], [c * e]]
    @test closeto(ode_emitted(d, ([a], [b], [c])), expected, 1e-9)
    @test closeto(casadi_eval(d, [a], b, c), expected, 1e-9)

    # A first run with equal times, then a run with an arc.
    Z = integrator("Z", Vern9(), Dict("x" => y, "t" => t, "ode" => 2 * t * y); TIGHT...)
    z = build_function(CasADiCodegen.Function("z", [y0, t0, tf], [Z(y0, t0, tf)]); nonzeros = true, expression = Val{false})
    @test z([zeros(1)], ([a], [b], [b])) == [[a]]
    @test closeto(z([zeros(1)], ([a], [b], [c])), [[a * e^2]], 1e-11)

    # Arcs with their own times in a map.
    arcs = 3
    X, T0, T1 = MX.sym("X", 4, arcs), MX.sym("T0", 1, arcs), MX.sym("T1", 1, arcs)
    ends = C.map(F, arcs)(X, C.repmat(p, 1, arcs), T0, T1)
    shoot = CasADiCodegen.Function("shoot", [X, p, T0, T1], [ends, jacobian(ends, T1)])
    starts = repeat(ODE_X0, arcs)
    res = ode_emitted(shoot, (starts, ODE_MU, [0.0, 0.0, 1.5], [3.0, 1.0, 4.5]))
    @test closeto([res[1][1:4], res[1][9:12]], [ours[1], ours[1]], 1e-10)
    @test closeto(res, casadi_eval(shoot, reshape(starts, 4, arcs), ODE_MU, [0.0 0.0 1.5], [3.0 1.0 4.5]), 1e-9)
end

@testset "states on a grid of times" begin
    grid = [1.0, 3.0, 3.0, 2.0]
    F = integrator("K", Vern9(), two_body(), 0.0, grid; TIGHT...)
    @test (C.n_in(F), C.n_out(F)) == (2, 1)
    @test (C.size1(C.sparsity_out(F, 0)), C.size2(C.sparsity_out(F, 0))) == (4, 4)
    x0, p = MX.sym("x0", 4), MX.sym("p", 1)
    X = F(x0, p)
    f = CasADiCodegen.Function("f", [x0, p], [X, jacobian(X[:, 2], x0), jacobian(X[:, 2], p)])
    ours = ode_emitted(f, (ODE_X0, ODE_MU))
    reference = ode_emitted(with_derivatives(two_body_integrator(), 1), (ODE_X0, ODE_MU))      # at the time 3
    @test closeto([ours[1][5:8], ours[2], ours[3]], reference, 1e-9)
    @test ours[1][9:12] == ours[1][5:8]                            # the same time again
    one = integrator("K1", Vern9(), two_body(), 0.0, 1.0; TIGHT...)
    two = integrator("K2", Vern9(), two_body(), 0.0, 2.0; TIGHT...)
    @test closeto([ours[1][1:4], ours[1][13:16]], [casadi_eval(one, ODE_X0, ODE_MU)[1], casadi_eval(two, ODE_X0, ODE_MU)[1]], 1e-9)
    @test closeto(casadi_eval(f, ODE_X0, ODE_MU), ours, 1e-10)
    # No parameter, and wrong use.
    y = SX.sym("y", 1)
    D = integrator("KD", Vern9(), Dict("x" => y, "ode" => -y), 0.0, [1.0, 2.0]; TIGHT...)
    @test C.n_in(D) == 1
    @test closeto(casadi_eval(D, [3.0]), [[3 * exp(-1), 3 * exp(-2)]], 1e-11)
    # One integration gives the three distinct times after the initial time; the solver does not restart at each.
    record = only(call for (call, _) in values(CasADiCodegen.Codegen.JULIA_CALLS) if call.name == "K")
    @test record.nnz_out == [4 * 3]
    # Times before the initial time and at it, in any order.
    B = integrator("KB", Vern9(), Dict("x" => y, "ode" => -y), 0.0, [1.0, 0.0, -1.0, 2.0, -0.5]; TIGHT...)
    @test closeto(casadi_eval(B, [3.0]), [3 .* exp.(-[1.0, 0.0, -1.0, 2.0, -0.5])], 1e-11)
    @test_throws ArgumentError integrator("K", Vern9(), two_body(), 0.0, Float64[])
    @test_throws ArgumentError integrator("K", Vern9(), two_body(), 0.0, [1.0, Inf])
    @test_throws ArgumentError integrator("K", Vern9(), two_body(), NaN, [1.0])
    @test_throws ArgumentError integrator("K", Vern9(), two_body(), 0.0, [1.0]; saveat = 0.5)
end

@testset "quadratures" begin
    # x' = -p x with the quadrature of x: q(T) = x0 (1 - exp(-p T)) / p.
    x, p, t = SX.sym("x", 1), SX.sym("p", 1), SX.sym("t")
    dae = Dict("x" => x, "p" => p, "t" => t, "ode" => -p * x, "quad" => vertcat([x, 2 * t]))
    a, k = 3.0, 0.7
    decay(T) = exp(-k * T)
    area(T) = a * (1 - decay(T)) / k
    F = integrator("Q", Vern9(), dae, 0.0, 2.0; TIGHT...)
    @test (C.n_in(F), C.n_out(F)) == (2, 2)
    @test [C.name_out(F, j) for j in 0:1] == ["xf", "qf"]
    x0, q = MX.sym("x0", 1), MX.sym("q", 1)
    both = F(x0, q)
    f = CasADiCodegen.Function("f", [x0, q], [both[1], both[2], jacobian(both[2], x0), jacobian(both[2], q)])
    slope = (2 * a * decay(2.0) * k - a * (1 - decay(2.0))) / k^2          # d area / d k
    expected = [[a * decay(2.0)], [area(2.0), 4.0], [(1 - decay(2.0)) / k, 0.0], [slope, 0.0]]
    @test closeto(ode_emitted(f, ([a], [k])), expected, 1e-10)
    @test closeto(casadi_eval(f, [a], [k]), expected, 1e-10)
    # Free times, where the quadrature of 2 t gives tf^2 - t0^2, and a grid of times.
    G = integrator("QG", Vern9(), dae; TIGHT...)
    t0, tf = MX.sym("t0"), MX.sym("tf")
    free = G(x0, q, t0, tf)
    g = CasADiCodegen.Function("g", [x0, q, t0, tf], [free[1], free[2], jacobian(free[2], tf), jacobian(free[2], t0)])
    xT = a * decay(1.5)
    @test closeto(ode_emitted(g, ([a], [k], [0.5], [2.0])), [[xT], [area(1.5), 3.75], [xT, 4.0], [-xT, -1.0]], 1e-9)
    K = integrator("QK", Vern9(), dae, 0.0, [1.0, 2.0]; TIGHT...)
    @test C.n_out(K) == 2
    @test closeto(casadi_eval(K, [a], [k]), [[a * decay(1.0), a * decay(2.0)], [area(1.0), 1.0, area(2.0), 4.0]], 1e-10)
    @test_throws ArgumentError integrator("Q", Vern9(), merge(dae, Dict("quad" => SX.sym("m", 2, 2))), 0.0, 1.0)
end

@testset "an event ends the integration" begin
    # A body that falls: height h and speed v with h' = v, v' = -g. It reaches the ground at
    # te = t0 + (v0 + sqrt(v0^2 + 2 g h0)) / g, and the quadrature of v is then -h0.
    x, g, t = SX.sym("x", 2), SX.sym("g", 1), SX.sym("t")
    fall = Dict("x" => x, "p" => g, "ode" => vertcat([x[2], -g[1]]), "zero" => x[1], "quad" => x[2])
    F = integrator("fall", Vern9(), fall; TIGHT...)
    @test [C.name_in(F, j) for j in 0:3] == ["x0", "p", "t0", "tf"]
    @test [C.name_out(F, j) for j in 0:2] == ["xf", "qf", "te"]
    landing(v) = v[4] + (v[2] + sqrt(v[2]^2 + 2 * v[3] * v[1])) / v[3]          # v = (h0, v0, g, t0)
    final(v) = [0 * v[1], v[2] - v[3] * (landing(v) - v[4])]
    x0, q, t0, tf = MX.sym("x0", 2), MX.sym("q", 1), MX.sym("t0"), MX.sym("tf")
    out = F(x0, q, t0, tf)
    all_in = vertcat([x0, q, t0])
    f = CasADiCodegen.Function("f", [x0, q, t0, tf],
        [out[1], out[2], out[3], jacobian(out[3], all_in), jacobian(out[1], all_in), hessian(out[3], x0), jacobian(out[3], tf)])
    at = [10.0, 3.0, 9.81, 0.5]
    te = landing(at)
    expected = [[0.0, 3.0 - 9.81 * (te - 0.5)], [-10.0], [te], ForwardDiff.gradient(landing, at),
                vec(ForwardDiff.jacobian(final, at)), vec(ForwardDiff.hessian(landing, at)[1:2, 1:2]), [0.0]]
    arguments = ([10.0, 3.0], [9.81], [0.5], [20.0])
    ours = ode_emitted(f, arguments)
    @test closeto(ours[1:3], expected[1:3], 1e-10) && abs(ours[1][1]) < 1e-10
    @test closeto(ours[4:7], expected[4:7], 1e-8)
    @test closeto(casadi_eval(f, [10.0, 3.0], [9.81], 0.5, 20.0), expected, 1e-8)
    # No event before the final time: the error names the integrator.
    err = try ode_emitted(f, ([10.0, 3.0], [9.81], [0.5], [0.6])); nothing catch e; e end
    @test err isa ErrorException && occursin("integrator \"fall\"", err.msg) && occursin("no event", err.msg)

    # A start on the event surface: from one apsis of an orbit, where the radial speed is zero,
    # to the next one, half a period later. No parameter in the call when the model has none.
    y = SX.sym("y", 4)
    r3 = (y[1]^2 + y[2]^2)^1.5
    orbit = Dict("x" => y, "ode" => vertcat([y[3], y[4], -y[1] / r3, -y[2] / r3]), "zero" => y[1] * y[3] + y[2] * y[4])
    A = integrator("apsis", Vern9(), orbit; TIGHT...)
    @test C.n_in(A) == 3 && [C.name_out(A, j) for j in 0:1] == ["xf", "te"]
    half(v) = π * (1 / (2 / v[1] - v[4]^2))^1.5                                 # v = (x, y, vx, vy) at an apsis on the x axis
    s0 = MX.sym("s0", 4)
    reached = A(s0, MX(0.0), MX(100.0))
    h = CasADiCodegen.Function("h", [s0], [reached[1], reached[2], jacobian(reached[2], s0)[:, 1], jacobian(reached[2], s0)[:, 4]])
    start = [1.0, 0.0, 0.0, 1.1]
    slopes = ForwardDiff.gradient(half, start)
    a = 1 / (2 - 1.1^2)
    @test closeto(ode_emitted(h, (start,)), [[1.0 - 2a, 0.0, 0.0, -1.1 / (2a - 1)], [half(start)], slopes[1:1], slopes[4:4]], 1e-8)

    # The state at the event and the derivatives of the event time need the sensitivities at the
    # same arguments: the generated code integrates them once, as for an arc to the event time.
    Solver = Base.get_extension(CasADiCodegen, :CasADiCodegenSciMLBaseExt).Solver
    # A solver with 20 states integrates the 4 states with their sensitivities in the 4 directions.
    evaluations(name) = sum(state.stats.nf for (key, state) in task_local_storage()
                            if key isa Tuple && key[1] isa Solver && key[1].name == name && length(state.u) == 20; init = 0)
    plain = integrator("plain", Vern9(), delete!(copy(orbit), "zero"); TIGHT...)
    to = MX.sym("to")
    reference = CasADiCodegen.Function("reference", [s0, to], [plain(s0, MX(0.0), to), jacobian(plain(s0, MX(0.0), to), s0)])
    both = CasADiCodegen.Function("both", [s0], [reached[1], jacobian(reached[1], s0), jacobian(reached[2], s0)])
    count(f, arguments) = begin
        g = build_function(f; nonzeros = true, expression = Val{false})
        result = [zeros(C.nnz_out(f, j - 1)) for j in 1:C.n_out(f)]
        g(result, arguments)                                       # the first call makes the solvers
        before = evaluations("plain") + evaluations("apsis")
        g(result, arguments)
        evaluations("plain") + evaluations("apsis") - before
    end
    @test count(both, (start,)) == count(reference, (start, [half(start)])) > 0

    # The direction of the crossing, and an event function of the time and of a parameter.
    w, c = SX.sym("w", 2), SX.sym("c", 1)
    swing = Dict("x" => w, "p" => c, "t" => t, "ode" => vertcat([w[2], -w[1]]), "zero" => w[1] - c[1])
    time_of(; options...) = begin
        E = integrator("swing", Vern9(), swing; TIGHT..., options...)
        only(only(ode_emitted(CasADiCodegen.Function("e", [x0, q], [E(x0, q, MX(0.0), MX(6.0))[2]]), ([0.0, 1.0], [0.5]))))
    end
    @test isapprox(time_of(; direction = 1), π / 6; atol = 1e-10)
    # With no point between the ends of a step, the solver looks for a change of sign at the ends only.
    @test isapprox(time_of(; direction = -1, interp_points = 0), 5π / 6; atol = 1e-10)
    clock = Dict("x" => w, "p" => c, "t" => t, "ode" => vertcat([w[2], -w[1]]), "zero" => t - c[1])
    K = integrator("clock", Vern9(), clock; TIGHT...)
    tk = K(x0, q, t0, tf)[2]
    k = CasADiCodegen.Function("k", [x0, q, t0, tf], [tk, jacobian(tk, q), jacobian(tk, t0)])
    @test closeto(ode_emitted(k, ([0.0, 1.0], [2.5], [1.0], [4.0])), [[2.5], [1.0], [0.0]], 1e-9)
    # Backward in time.
    @test closeto(ode_emitted(k, ([0.0, 1.0], [2.5], [4.0], [1.0])), [[2.5], [1.0], [0.0]], 1e-9)

    # Dual numbers through the generated code, with the full specialization of the solver:
    # the derivative of the event time by the initial state and the parameter.
    D = integrator("drop", Vern9(), fall; TIGHT..., specialize = SciMLBase.FullSpecialize)
    drop = build_function(CasADiCodegen.Function("d", [x0, q], [D(x0, q, MX(0.5), MX(20.0))[3]]); nonzeros = true, expression = Val{false})
    event(v) = (res = [zeros(eltype(v), 1)]; drop(res, (v[1:2], v[3:3])); res[1][1])
    @test isapprox(ForwardDiff.gradient(event, at[1:3]), ForwardDiff.gradient(v -> landing([v; 0.5]), at[1:3]); atol = 1e-8)

    # Limits: a column of event functions, and free times only.
    @test_throws ArgumentError integrator("e", Vern9(), merge(swing, Dict("zero" => horzcat([w, w]))); TIGHT...)
    @test_throws ArgumentError integrator("e", Vern9(), swing, 0.0, 1.0; TIGHT...)
    @test_throws ArgumentError integrator("e", Vern9(), swing, 0.0, [1.0, 2.0]; TIGHT...)
    @test_throws ArgumentError integrator("e", Vern9(), swing; TIGHT..., direction = 2)
    @test_throws ArgumentError integrator("e", Vern9(), swing; TIGHT..., interp_points = -1)
end

# More than one event: `events` counts them, the entry "transition" gives the state after
# an event, and with `terminal = false` the integration goes on to the final time.
@testset "several events, with a jump of the state" begin
    # A ball that bounces: at the ground the speed changes its sign and loses a part, e.
    x, p = SX.sym("x", 2), SX.sym("p", 2)                                    # p = (g, e)
    ball = Dict("x" => x, "p" => p, "ode" => vertcat([x[2], -p[1]]), "zero" => x[1], "quad" => SX(1.0),
                "transition" => vertcat([x[1], -p[2] * x[2]]))
    F = integrator("ball", Vern9(), ball; events = 2, terminal = false, direction = -1, TIGHT...)
    @test [C.name_out(F, j) for j in 0:2] == ["xf", "qf", "te"] && C.size1(C.sparsity_out(F, 2)) == 2
    # From the height h with no speed: the ground at t1 = sqrt(2 h / g) and at t1 (1 + 2 e), then up with the speed e^2 g t1.
    times(v) = (t1 = sqrt(2 * v[1] / v[2]); [t1, t1 * (1 + 2 * v[3])])       # v = (h, g, e)
    final(v, T) = (t1 = sqrt(2 * v[1] / v[2]); d = T - t1 * (1 + 2 * v[3]); up = v[3]^2 * v[2] * t1;
                   [up * d - v[2] * d^2 / 2, up - v[2] * d])
    h0, q, tf = MX.sym("h0"), MX.sym("q", 2), MX.sym("tf")
    out = F(vertcat([h0, MX(0.0)]), q, MX(0.0), tf)
    every = vertcat([h0, q])
    f = CasADiCodegen.Function("bounces", [h0, q, tf], [out[1], out[2], out[3], jacobian(out[1], every), jacobian(out[3], every), jacobian(out[1], tf)])
    at, T = [5.0, 9.81, 0.8], 3.0
    expected = [final(at, T), [T], times(at), vec(ForwardDiff.jacobian(v -> final(v, T), at)), vec(ForwardDiff.jacobian(times, at)),
                ForwardDiff.derivative(s -> final(at, s), T)]
    @test closeto(ode_emitted(f, ([5.0], [9.81, 0.8], [T])), expected, 1e-8)
    @test closeto(casadi_eval(f, 5.0, [9.81, 0.8], T), expected, 1e-8)
    # The integration ends at the last event without a jump there: the state on the ground, before the second bounce.
    G = integrator("ball_end", Vern9(), ball; events = 2, direction = -1, TIGHT...)
    ended = G(vertcat([h0, MX(0.0)]), q, MX(0.0), tf)
    g = CasADiCodegen.Function("second", [h0, q, tf], [ended[1], ended[2], ended[3]])
    t1 = sqrt(2 * 5.0 / 9.81)
    @test closeto(ode_emitted(g, ([5.0], [9.81, 0.8], [T])), [[0.0, -0.8 * 9.81 * t1], [times(at)[2]], times(at)], 1e-9)
    # No "transition": the third time that an oscillator passes through zero, from (1, 0).
    y = SX.sym("y", 2)
    swing = Dict("x" => y, "ode" => vertcat([y[2], -y[1]]), "zero" => y[1])
    H = integrator("swing", Vern9(), swing; events = 3, TIGHT...)
    y0 = MX.sym("y0", 2)
    third = H(y0, MX(0.0), MX(50.0))
    h = CasADiCodegen.Function("third", [y0], [third[1], third[2], jacobian(third[2], y0)])
    @test closeto(ode_emitted(h, ([1.0, 0.0],)), [[0.0, -1.0], [π / 2, 3π / 2, 5π / 2], [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], 1e-9)
    # Fewer events than asked before the final time.
    err = try ode_emitted(g, ([5.0], [9.81, 0.8], [1.5])); nothing catch e; e end
    @test err isa ErrorException && occursin("no event", err.msg)
    @test_throws "events" integrator("none", Vern9(), swing; events = 0, TIGHT...)
    @test_throws "\"transition\", must have the size of the state" integrator("wide", Vern9(), merge(swing, Dict("transition" => vertcat([y, y]))); TIGHT...)
    @test_throws "\"transition\"" integrator("fixed", Vern9(), merge(delete!(copy(swing), "zero"), Dict("transition" => y)), 0.0, 1.0; TIGHT...)
    @test_throws "terminal" integrator("plain", Vern9(), delete!(copy(swing), "zero"); terminal = false, TIGHT...)
end

# Several event functions: the first element of "zero" that changes its sign selects the column of
# "transition", and the result "ie" names it. With `exact = false` fewer events than `events` can
# come before the final time; "ne" counts them, and a missing event has the final time as its time.
# The set takes about 20 s: the callback of several event functions compiles the solver once for
# Float64 and once for dual numbers, which take their own code path for the derivative of the time.
@testset "several event functions, and up to a number of events" begin
    # A body that moves at a constant speed between two walls, at 0 and at 1. At the wall 1 the
    # speed changes its sign and keeps the part e1, at the wall 0 the part e2. The third event function
    # t - 3 changes its sign from negative to positive, which `direction = -1` ignores.
    x, e, τ = SX.sym("x", 2), SX.sym("e", 2), SX.sym("t")
    walls = Dict("x" => x, "p" => e, "t" => τ, "ode" => vertcat([x[2], SX(0.0)]), "zero" => vertcat([x[1], 1 - x[1], τ - 3]),
                 "transition" => horzcat([vertcat([x[1], -e[2] * x[2]]), vertcat([x[1], -e[1] * x[2]]), -x]))
    F = integrator("walls", Vern9(), walls; events = 4, exact = false, terminal = false, direction = -1, TIGHT...)
    @test [C.name_out(F, j) for j in 0:3] == ["xf", "te", "ie", "ne"]
    # From the position a in (0, 1) with the speed w > 0, up to the final time T; v = (a, w, e1, e2, T).
    function flight(v)
        position, speed, t, times, hits = v[1], v[2], zero(eltype(v)), eltype(v)[], Float64[]
        while length(times) < 4
            # Loop invariant: the body is at `position` with `speed` at the time `t`, after the events in `times`.
            wall = speed > 0 ? 1.0 : 0.0
            arrival = t + (wall - position) / speed
            arrival < v[5] || break
            push!(times, arrival)
            push!(hits, wall == 1 ? 2.0 : 1.0)
            position, speed, t = wall, -(wall == 1 ? v[3] : v[4]) * speed, arrival
        end
        (; times, hits, state = [position + speed * (v[5] - t), speed])
    end
    s0, q, tf = MX.sym("s0", 2), MX.sym("q", 2), MX.sym("tf")
    out = F(s0, q, MX(0.0), tf)
    every = vertcat([s0, q, tf])
    f = CasADiCodegen.Function("f", [s0, q, tf], [out[1], out[2], out[3], out[4], jacobian(out[1], every), jacobian(out[2], every)])
    for T in (4.0, 9.0)                            # two and three events of four
        at = [0.5, 1.0, 0.5, 0.8, T]
        reference = flight(at)
        missing = 4 - length(reference.times)
        times(v) = [flight(v).times; fill(v[5], missing)]
        expected = [reference.state, times(at), [reference.hits; zeros(missing)], [4.0 - missing],
                    vec(ForwardDiff.jacobian(v -> flight(v).state, at)), vec(ForwardDiff.jacobian(times, at))]
        @test closeto(ode_emitted(f, (at[1:2], at[3:4], at[5:5])), expected, 1e-9)
        T == 9.0 && @test closeto(casadi_eval(f, at[1:2], at[3:4], T), expected, 1e-9)
    end

    # Dual numbers through the generated code: the event function that fired gives the derivative of its time.
    second = build_function(CasADiCodegen.Function("second", [s0, q], [F(s0, q, MX(0.0), MX(9.0))[2][2]]); nonzeros = true, expression = Val{false})
    timed(v) = (res = [zeros(eltype(v), 1)]; second(res, (v[1:2], v[3:4])); res[1][1])
    @test isapprox(ForwardDiff.gradient(timed, [0.5, 1.0, 0.5, 0.8]), ForwardDiff.gradient(v -> flight([v; 9.0]).times[2], [0.5, 1.0, 0.5, 0.8]); atol = 1e-8)

    # One event function: the count follows the times, and the integration ends at the last event or at the final time.
    # An elastic ball from the height 2 under the gravity 1 lands at 2 and at 6.
    fall = Dict("x" => x, "p" => e, "ode" => vertcat([x[2], -e[1]]), "zero" => x[1], "transition" => vertcat([x[1], -x[2]]))
    G = integrator("fall_upto", Vern9(), fall; events = 2, exact = false, direction = -1, TIGHT...)
    @test [C.name_out(G, j) for j in 0:2] == ["xf", "te", "ne"]
    @test closeto(casadi_eval(G, [2.0, 0.0], [1.0, 0.0], 0.0, 5.0), [[1.5, -1.0], [2.0, 5.0], [1.0]], 1e-9)

    @test_throws "one column per event function" integrator("wide", Vern9(), merge(walls, Dict("transition" => horzcat([x, x]))); TIGHT...)
    @test_throws "exact" integrator("plain", Vern9(), delete!(copy(fall), "zero"); exact = false, TIGHT...)
end

# With `adjoint`, a reverse derivative integrates the state and its multipliers
# backward in time, so that a gradient costs one integration for any number of
# parameters. The forward sensitivities are the oracle.
@testset "adjoint sensitivities" begin
    # A forced oscillator with the time in its right side, three parameters and a quadrature.
    x, p, t = SX.sym("x", 2), SX.sym("p", 3), SX.sym("t")
    dae = Dict("x" => x, "p" => p, "t" => t, "ode" => vertcat([x[2], -p[1] * x[1] + p[2] * sin(p[3] * t)]), "quad" => x[1]^2 * p[2])
    x0, q, t0, tf = MX.sym("x0", 2), MX.sym("q", 3), MX.sym("t0"), MX.sym("tf")
    inputs, point = [x0, q, t0, tf], ([0.4, -0.3], [1.3, 0.8, 2.1], [0.5], [3.0])
    function gradients(F, order)
        xf, qf = F(x0, q, t0, tf)
        cost = xf[1] * xf[2] + 3 * qf
        outputs = [cost; [gradient(cost, v) for v in inputs]]
        order == 2 && push!(outputs, jacobian(gradient(cost, x0), x0), jacobian(gradient(cost, q), x0))
        CasADiCodegen.Function("costs", inputs, outputs)
    end
    forward = integrator("AF", Vern9(), dae; TIGHT...)
    backward = integrator("AB", Vern9(), dae; adjoint = true, TIGHT...)
    reference = ode_emitted(gradients(forward, 2), point)
    f = gradients(backward, 1)
    @test closeto(ode_emitted(f, point), reference[1:5], 1e-9)
    @test closeto(casadi_eval(f, point[1], point[2], point[3][1], point[4][1]), reference[1:5], 1e-9)
    # The gradient made the integrator of the adjoint equations, and no integrator of sensitivity matrices.
    made = Set(call.name for (call, _) in values(CasADiCodegen.Codegen.JULIA_CALLS) if startswith(call.name, "AB"))
    @test made == Set(["AB", "AB_a"])
    # 2 states and the quadrature, their 3 multipliers, and 4 sums: for the parameters and the initial time.
    record = only(call for (call, _) in values(CasADiCodegen.Codegen.JULIA_CALLS) if call.name == "AB_a")
    @test record.nnz_in == [6, 5] && record.nnz_out == [10]
    # The second order differentiates the adjoint integrator.
    @test closeto(ode_emitted(gradients(backward, 2), point), reference, 1e-8)
    # Fixed times, and a model with no parameter and no time.
    fixed = integrator("AX", Vern9(), dae, 0.5, 3.0; adjoint = true, TIGHT...)
    xf, qf = fixed(x0, q)
    cost = xf[1] * xf[2] + 3 * qf
    g = CasADiCodegen.Function("fixed", [x0, q], [cost, gradient(cost, x0), gradient(cost, q)])
    @test closeto(ode_emitted(g, point[1:2]), reference[1:3], 1e-9)
    y = SX.sym("y", 2)
    plain = integrator("AP", Vern9(), Dict("x" => y, "ode" => vertcat([y[2], -y[1]])), 0.0, 1.0; adjoint = true, TIGHT...)
    turned = plain(x0)
    h = CasADiCodegen.Function("plain", [x0], [gradient(turned[1] + 2 * turned[2], x0)])
    @test closeto(ode_emitted(h, point[1:1]), [[cos(1.0) - 2 * sin(1.0), sin(1.0) + 2 * cos(1.0)]], 1e-10)
    # A grid of times: one backward integration, whose multipliers take the seed of each time on the way.
    times = [1.0, 3.0, 0.5, 2.0]
    function sampled(F)
        states, sums = F(x0, q)
        cost = states[1, 2] * states[2, 2] + 3 * sums[2] + C.sumsqr(states) + C.sumsqr(sums)
        CasADiCodegen.Function("grid", [x0, q], [gradient(cost, x0), gradient(cost, q), jacobian(gradient(cost, q), vertcat([x0, q]))])
    end
    grid = sampled(integrator("AG", Vern9(), dae, 0.5, times; adjoint = true, TIGHT...))
    expected = ode_emitted(sampled(integrator("FG", Vern9(), dae, 0.5, times; TIGHT...)), point[1:2])
    ours = ode_emitted(grid, point[1:2])
    @test closeto(ours[1:2], expected[1:2], 1e-9)
    @test closeto(ours[3:3], expected[3:3], 1e-8)
    # The states, the quadrature and the seeds at the 3 times after the initial time; the gradient by x0 and q.
    record = only(call for (call, _) in values(CasADiCodegen.Codegen.JULIA_CALLS) if call.name == "AG_a")
    @test record.nnz_in == [2, 3, 9, 9] && record.nnz_out == [5]
    # Only the last time: the cost of the reverse derivative of the arc with fixed times.
    last_cost = integrator("AL", Vern9(), dae, 0.5, [3.0]; adjoint = true, TIGHT...)(x0, q)
    k = CasADiCodegen.Function("last", [x0, q], [gradient(last_cost[1][1] * last_cost[1][2] + 3 * last_cost[2], q)])
    @test closeto(ode_emitted(k, point[1:2]), reference[3:3], 1e-9)
end

# The seeds of a least-squares cost are the residuals at the output times, so their error goes
# into the gradient with the weight of the sensitivities. At the minimum the terms cancel, and
# the gradient is as accurate as the states on the grid. Regression: interpolated states
# (`saveat`) of the integration without sensitivities gave a gradient 20 times less accurate
# than the forward sensitivities.
@testset "adjoint gradient of a least-squares fit at its minimum" begin
    x, p = SX.sym("x", 2), SX.sym("p", 4)
    lotka = Dict("x" => x, "p" => p, "ode" => vertcat([(p[1] - p[2] * x[2]) * x[1], (p[4] * x[1] - p[3]) * x[2]]))
    times, θ, u0 = collect(0.0:0.1:10.0), [1.5, 1.0, 3.0, 1.0], [1.0, 1.0]
    q = MX.sym("q", 4)
    reference = integrator("LR", Vern9(), lotka, 0.0, times; TIGHT...)
    data = reshape(casadi_eval(reference, u0, θ)[1], 2, :) .+ [0.01 * sin(7.0 * i + 3.0 * k) for i in 1:2, k in eachindex(times)]
    function fit(F)
        cost = C.sumsqr(MX(C.DM(data)) - F(MX(C.DM(u0)), q)) / length(data)
        CasADiCodegen.Function("fit", [q], [gradient(cost, q)])
    end
    exact = only(casadi_eval(fit(reference), θ))
    loose = (; abstol = 1e-10, reltol = 1e-10, specialize = SciMLBase.AutoSpecialize)
    by_adjoint = only(casadi_eval(fit(integrator("LA", Vern9(), lotka, 0.0, times; adjoint = true, loose...)), θ))
    @test maximum(abs, by_adjoint - exact) < 3e-10           # 1.9e-9 with interpolated states
end

# The reverse derivative through an event has the term of the event time: the multipliers jump
# where the integration ends. The exact solution of a body that falls is the oracle.
@testset "adjoint sensitivities through an event" begin
    x, g = SX.sym("x", 2), SX.sym("g", 1)
    F = integrator("AE", Vern9(), Dict("x" => x, "p" => g, "ode" => vertcat([x[2], -g[1]]), "zero" => x[1]); adjoint = true, TIGHT...)
    landing(v) = (v[2] + sqrt(v[2]^2 + 2 * v[3] * v[1])) / v[3]          # v = (h0, v0, g), from the time 0
    cost(v) = (v[2] - v[3] * landing(v)) + 2 * landing(v)                # the speed at the ground and twice the time
    x0, q = MX.sym("x0", 2), MX.sym("q", 1)
    xe, te = F(x0, q, MX(0.0), MX(20.0))
    f = CasADiCodegen.Function("f", [x0, q], [gradient(xe[2] + 2 * te, vertcat([x0, q]))])
    @test closeto(ode_emitted(f, ([10.0, 3.0], [9.81])), [ForwardDiff.gradient(cost, [10.0, 3.0, 9.81])], 1e-8)
end

@testset "later calls reuse the state of the solver" begin
    for (order, limit) in ((0, 1024), (1, 8192))
        f = with_derivatives(two_body_integrator(), order)
        g = build_function(f; nonzeros = true, expression = Val{false})
        res = [zeros(C.nnz_out(f, i - 1)) for i in 1:C.n_out(f)]
        arg = (ODE_X0, ODE_MU)
        g(res, arg); g(res, arg)
        @test (@allocated g(res, arg)) <= limit
        first_result = deepcopy(res)
        g(res, (2 * ODE_X0, ODE_MU))
        @test res != first_result
        g(res, arg)
        @test res == first_result
    end
end

@testset "tasks do not share the state of the solver" begin
    f = with_derivatives(two_body_integrator(), 1)
    g = build_function(f; nonzeros = true, expression = Val{false})
    run(scale) = (res = [zeros(4), zeros(16), zeros(4)]; g(res, (scale * ODE_X0, ODE_MU)); res)
    scales = [1.0 + 0.01k for k in 1:8]
    expected = map(run, scales)
    tasks = [Threads.@spawn run(s) for s in scales]
    @test map(fetch, tasks) == expected
end

@testset "generic number types" begin
    F = two_body_integrator()
    f0, f1 = with_derivatives(F, 0), with_derivatives(F, 1)
    g = build_function(f0; nonzeros = true, expression = Val{false})
    # One vector for the initial state and the parameter, so that the solver compiles for one dual number type.
    final(v) = (res = [zeros(eltype(v), 4)]; g(res, (v[1:4], v[5:5])); res[1])
    J = ForwardDiff.jacobian(final, [ODE_X0; ODE_MU])
    expected = ode_emitted(f1, (ODE_X0, ODE_MU))
    @test maximum(abs, vec(J[:, 1:4]) - expected[2]) <= 1e-9
    @test maximum(abs, J[:, 5] - expected[3]) <= 1e-9
end

@testset "a solver made by hand" begin
    # Classical Runge-Kutta with fixed steps: the package needs only a function
    # (initial state, parameters) -> final state.
    function runge_kutta(rhs!, n, m)
        function (x0, q)
            u, h = collect(float(eltype(x0)), x0), 1 / 200
            k = [similar(u) for _ in 1:4]
            for s in 0:199
                t = s * h
                rhs!(k[1], u, q, t)
                rhs!(k[2], u + h / 2 * k[1], q, t + h / 2)
                rhs!(k[3], u + h / 2 * k[2], q, t + h / 2)
                rhs!(k[4], u + h * k[3], q, t + h)
                u += h / 6 * (k[1] + 2k[2] + 2k[3] + k[4])
            end
            u
        end
    end
    x, p = SX.sym("x", 1), SX.sym("p", 1)
    F = CasADiCodegen.Codegen.ode_integrator("H", Dict("x" => x, "p" => p, "ode" => -p * x), 0.0, 1.0, runge_kutta)
    x0, q = MX.sym("x0", 1), MX.sym("q", 1)
    f = CasADiCodegen.Function("f", [x0, q], [F(x0, q), jacobian(F(x0, q), x0), jacobian(F(x0, q), q)])
    expected = [[3 * exp(-2)], [exp(-2)], [-3 * exp(-2)]]
    @test closeto(ode_emitted(f, ([3.0], [2.0])), expected, 1e-9)
    @test closeto(casadi_eval(f, [3.0], [2.0]), expected, 1e-9)
end

@testset "options and failure of the solver" begin
    x0, p = MX.sym("x0", 4), MX.sym("p", 1)
    F = two_body_integrator(; maxiters = 3)
    g = build_function(CasADiCodegen.Function("f", [x0, p], [F(x0, p)]); nonzeros = true, expression = Val{false})
    error_text = redirect_stderr(devnull) do          # the solver also writes a warning
        try
            g([zeros(4)], (ODE_X0, ODE_MU))
            ""
        catch err
            sprint(showerror, err)
        end
    end
    @test occursin("integrator \"F\"", error_text)
    @test occursin("MaxIters", error_text)
end

@testset "limits of the integrator" begin
    x, p = SX.sym("x", 2), SX.sym("p", 1)
    dae = Dict("x" => x, "p" => p, "ode" => -p * x)
    @test_throws "\"z\"" integrator("F", Vern9(), merge(dae, Dict("z" => SX.sym("z"))), 0.0, 1.0)
    @test_throws ArgumentError integrator("F", Vern9(), Dict("x" => x), 0.0, 1.0)
    @test_throws ArgumentError integrator("F", Vern9(), Dict("x" => x, "ode" => x[1]), 0.0, 1.0)
    @test integrator("F", Vern9(), Dict("x" => x, "ode" => vertcat([x[2], SX(1, 1)])), 0.0, 1.0) isa CasADiCodegen.Function
    @test_throws ArgumentError integrator("F", Vern9(), Dict("x" => SX.sym("x", 2, 2), "ode" => SX.sym("x", 2, 2)), 0.0, 1.0)
    @test_throws ArgumentError integrator("F", Vern9(), Dict("x" => x, "p" => SX.sym("p", 2, 2), "ode" => x), 0.0, 1.0)
    @test_throws ArgumentError integrator("F", Vern9(), merge(dae, Dict("t" => SX.sym("t", 2))), 0.0, 1.0)
    @test_throws ArgumentError integrator("not a name", Vern9(), dae, 0.0, 1.0)
    @test_throws ArgumentError integrator("F", Vern9(), dae, 1.0, 1.0)
    @test_throws ArgumentError integrator("F", Vern9(), dae, 0.0, Inf)
end
