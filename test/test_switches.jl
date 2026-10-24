# Right sides that switch on the state: the check of an integrator for operations whose
# derivatives miss the jump of the STM at the switching time, and the smooth helpers.
include("ode_helpers.jl")
using CasADiCodegen: Solvers, Solver, smooth_step, smooth_sign, smooth_abs, smooth_max, smooth_min
const K = CasADiCodegen.Codegen

"A model `x'' = u` whose `u` is the expression `law(x, b, t)`, with the parameter `b`."
function switched(law; quad = nothing)
    x, b, t = SX.sym("x", 2), SX.sym("b"), SX.sym("t")
    model = Dict{String,Any}("x" => x, "p" => b, "t" => t, "ode" => vertcat([x[2], law(x, b, t)]))
    quad === nothing || (model["quad"] = quad(x))
    model
end

@testset "an integrator warns about a switch on the state" begin
    on_state = switched((x, b, t) -> C.if_else(x[1] < 1, SX(1.0), b * x[1]))
    @test_logs (:warn, r"integrator \"F\".*OP_LT: 1 jump") integrator("F", Vern9(), on_state, 0.0, 2.0; TIGHT...)
    @test_throws ArgumentError integrator("F", Vern9(), on_state, 0.0, 2.0; nonsmooth = :error, TIGHT...)
    @test_throws ArgumentError integrator("F", Vern9(), on_state; nonsmooth = :sometimes, TIGHT...)
    # A switch at a fixed time, one on the parameters only, and a guard of an equality do not move with the state.
    @test_logs integrator("F", Vern9(), switched((x, b, t) -> C.if_else(t < 1, SX(1.0), b)), 0.0, 2.0; TIGHT...)
    @test_logs integrator("F", Vern9(), switched((x, b, t) -> C.if_else(b < 1, SX(1.0), b)), 0.0, 2.0; TIGHT...)
    @test_logs integrator("F", Vern9(), switched((x, b, t) -> C.if_else(x[1] == 0, SX(0.0), 1 / x[1])), 0.0, [1.0, 2.0]; TIGHT...)
    # A kink leaves the first derivatives right.
    kinked = switched((x, b, t) -> b; quad = x -> abs(x[1]))
    @test_logs (:warn, r"\"quad\".*OP_FABS: 1 kink.*order 2") integrator("F", Vern9(), kinked; TIGHT...)
end

@testset "an integrator of CasADi warns, natively and in generated code" begin
    x, b = MX.sym("x", 2), MX.sym("b")
    model = Dict("x" => x, "p" => b, "ode" => vertcat([x[2], C.if_else(x[1] < 1, MX(1.0), b)]))
    native = @test_logs integrator("bounce", "cvodes", model, 0.0, 2.0, Dict("nonsmooth" => "ignore"))
    f = CasADiCodegen.Function("f", [x, b], [native(x, MX(0, 1), b, MX(), MX(), MX(), MX())[1]])
    @test_logs (:warn, r"integrator \"bounce\"") match_mode = :any K.algorithm(f; expand = false, solvers = Solvers(ode = Solver(Vern9(); TIGHT...)))
    @test_logs K.algorithm(f; expand = false, solvers = Solvers(ode = Solver(Vern9(); TIGHT...), nonsmooth = :ignore))
    signed = Dict("x" => x, "ode" => vertcat([x[2], C.sign(x[1])]))
    @test_throws ArgumentError integrator("step", "rk", signed, 0.0, 1.0, Dict("nonsmooth" => "error"))
end

@testset "nonsmooth_operations finds the switches on chosen inputs" begin
    x, p = SX.sym("x", 2), SX.sym("p")
    inner = CasADiCodegen.Function("inner", [x, p], [C.sign(x[1]) + C.fmin(p, SX(2.0)) * x[2]], ["x", "p"], ["r"])
    X, P = MX.sym("x", 2), MX.sym("p")
    outer = CasADiCodegen.Function("outer", [X, P], [inner(X, P)], ["x", "p"], ["r"])
    @test CasADiCodegen.nonsmooth_operations(outer; inputs = ["x"]) == [:OP_SIGN => 1]
    @test CasADiCodegen.nonsmooth_operations(outer; inputs = [1], expand = false) == [:OP_SIGN => 1]
    @test sort(CasADiCodegen.nonsmooth_operations(outer)) == [:OP_FMIN => 1, :OP_SIGN => 1]
    # One input by itself, by name or by place.
    @test CasADiCodegen.nonsmooth_operations(outer; inputs = "x") == CasADiCodegen.nonsmooth_operations(outer; inputs = 1) == [:OP_SIGN => 1]
    @test_throws ArgumentError CasADiCodegen.nonsmooth_operations(outer; inputs = ["y"])
    # Regression: a short-circuit if_else counted twice in the matrix scan, at its comparison and at its Switch.
    a = MX.sym("a")
    ff = CasADiCodegen.Function("ff", [a], [C.if_else(a[1] > 0, 2a, 3a, true)])
    A = MX.sym("A", 1, 3)
    mapped = CasADiCodegen.Function("g", [A], [C.map(ff, 3)(A)])
    @test CasADiCodegen.nonsmooth_operations(mapped; expand = false) == [:OP_LT => 1]
end

@testset "smooth helpers, $kind" for kind in (:tanh, :algebraic)
    width = 1e-3
    @test smooth_step(0.0, width; kind) == 0.5 && smooth_abs(0.0, width; kind) == 0.0 && smooth_max(0.7, 0.7, width; kind) == 0.7
    s = range(-1, 1; length = 4001)
    @test maximum(abs.(smooth_abs.(s, width; kind) .- abs.(s))) <= 0.31 * width
    @test smooth_sign(1.0, width; kind) ≈ 1 atol = (kind === :tanh ? 1e-12 : 1e-6)
    @test smooth_min(-0.3, 0.4, width; kind) ≈ -0.3 atol = width
    # Symbolic expressions with a symbolic width, evaluated by CasADi, give the numbers.
    @testset "$T" for T in (SX, MX)
        a, b, w = T.sym("a"), T.sym("b"), T.sym("w")
        values = [smooth_step(a, w; kind), smooth_sign(a, w; kind), smooth_abs(a, w; kind), smooth_max(a, b, w; kind), smooth_min(a, 0.2, w; kind)]
        f = CasADiCodegen.Function("f", [a, b, w], values)
        expected = [smooth_step(0.3, 0.1; kind), smooth_sign(0.3, 0.1; kind), smooth_abs(0.3, 0.1; kind), smooth_max(0.3, -0.5, 0.1; kind), smooth_min(0.3, 0.2, 0.1; kind)]
        @test reduce(vcat, casadi_eval(f, 0.3, -0.5, 0.1)) ≈ expected rtol = 1e-14
    end
end

@testset "smooth helpers: derivative and contracts" begin
    @test ForwardDiff.derivative(s -> smooth_step(s, 0.01), 0.0) ≈ 1 / (2 * 0.01)
    @test_throws ArgumentError smooth_step(0.1, 0.0)
    @test_throws ArgumentError smooth_sign(0.1, 0.1; kind = :cubic)
end

# A switched right side with `modes`: the thrust m of x'' = m is +1 where S = 1 - x > 0 and -1
# beyond. The integrator finds each switch as an event, so the derivatives have the jump of the
# STM at the switching time. Oracle: the closed form with one switch, differentiated by ForwardDiff.
"The state at the time `T` from z = (x0, v0) with x0 < 1, with one switch where x reaches 1."
function bang(z, T)
    x0, v0 = z
    vs = sqrt(v0^2 + 2 * (1 - x0))
    τ = T - (vs - v0)
    [1 + vs * τ - τ^2 / 2, vs - τ]
end
"The double integrator with the mode `m` as its thrust, and its switching function."
function thrust_model(; written = m -> m)
    x, m = SX.sym("x", 2), SX.sym("m")
    Dict{String,Any}("x" => x, "ode" => vertcat([x[2], written(m)])), m => 1 - x[1]
end

@testset "modes: fixed times, the jump of the STM, the initial mode" begin
    model, modes = thrust_model()
    F = @test_logs integrator("F", Vern9(), model, 0.0, 2.0; modes, switches = 2, TIGHT...)
    @test [C.name_out(F, j) for j in 0:C.n_out(F)-1] == ["xf"]
    x0 = MX.sym("x0", 2)
    xf = F(x0)
    f = build_function(CasADiCodegen.Function("f", [x0], [xf, jacobian(xf, x0)]); nonzeros = true, expression = Val{false})
    emitted(v) = (res = [zeros(2), zeros(4)]; f(res, (v,)); res)
    z = [0.0, 0.0]
    stm = ForwardDiff.jacobian(v -> bang(v, 2.0), z)
    @test closeto(emitted(z), [bang(z, 2.0), vec(stm)], 1e-9)
    @test maximum(abs, stm - [1 2; 0 1]) > 0.1                  # the product of the flows without the jump
    # The second derivatives, from CasADi, which differentiates the same graph as the generated code.
    hessian = CasADiCodegen.Function("hessian", [x0], [jacobian(jacobian(xf[1], x0), x0)])
    @test closeto(casadi_eval(hessian, z), [vec(ForwardDiff.hessian(v -> bang(v, 2.0)[1], z))], 1e-9)
    # Beyond the surface the mode is -1; on the surface, the mode whose flow leaves it.
    @test closeto(emitted([1.5, 1.0])[1:1], [[1.5, -1.0]], 1e-9)
    @test closeto(emitted([1.0, 2.0])[1:1], [[3.0, 0.0]], 1e-9)
    # Dual numbers through the generated code take the derivatives of the event times too.
    if EXHAUSTIVE
        g = build_function(CasADiCodegen.Function("g", [x0], [xf]); nonzeros = true, expression = Val{false})
        final(v) = (res = [zeros(eltype(v), 2)]; g(res, (v,)); res[1])
        @test isapprox(ForwardDiff.jacobian(final, z), stm; atol = 1e-8)
    end
end

@testset "modes: free times, switch times, the bound and a vector of times" begin
    model, modes = thrust_model(written = m -> C.if_else(m > 0, SX(1.0), SX(-1.0)))   # a test of the mode does not warn
    G = @test_logs integrator("G", Vern9(), model; modes, switches = 1, switch_times = true, TIGHT...)
    @test [C.name_out(G, j) for j in 0:C.n_out(G)-1] == ["xf", "ts", "ns"]
    x0, tf = MX.sym("x0", 2), MX.sym("tf")
    out = G(x0, MX(0.0), tf)
    g = CasADiCodegen.Function("g", [x0, tf], [out[1], out[2], out[3], jacobian(out[2], x0)])
    switch_time(v) = sqrt(v[2]^2 + 2 * (1 - v[1])) - v[2]
    z = [0.0, 0.0]
    @test closeto(ode_emitted(g, (z, [2.0])), [bang(z, 2.0), [sqrt(2)], [1.0], ForwardDiff.gradient(switch_time, z)], 1e-9)
    # Over [0, 6] the state crosses x = 1 twice, more than the bound.
    @test_throws "raise switches" ode_emitted(g, (z, [6.0]))
    # A vector of times, on both sides of t0, carries the mode from one time to the next.
    H = integrator("H", Vern9(), model, 0.0, [2.0, 0.5, -0.5, 1.0]; modes, switches = 2, TIGHT...)
    h = CasADiCodegen.Function("h", [x0], [H(x0)])
    @test closeto(ode_emitted(h, (z,)), [[bang(z, 2.0); [0.125, 0.5]; [0.125, -0.5]; [0.5, 1.0]]], 1e-9)
end

@testset "modes: a sliding arc, the user's events, and the contracts" begin
    # x' = 1/2 - m with S = x: from x = 1 the flow reaches 0, and beyond it points back.
    y, n = SX.sym("y"), SX.sym("n")
    slide = integrator("slide", Vern9(), Dict("x" => y, "ode" => 0.5 - n), 0.0, 4.0; modes = n => y, switches = 3, TIGHT...)
    y0 = MX.sym("y0")
    @test_throws "slide" ode_emitted(CasADiCodegen.Function("s", [y0], [slide(y0)]), ([1.0],))
    # Two switching functions on one surface change their sign in one search: the second mode stays wrong.
    z, w = SX.sym("z", 2), SX.sym("w", 2)
    twice = integrator("twice", Vern9(), Dict("x" => z, "ode" => vertcat([z[2], (w[1] + w[2]) / 2])), 0.0, 2.0;
                       modes = w => vertcat([1 - z[1], 1 - z[1]]), switches = 2, TIGHT...)
    z0 = MX.sym("z0", 2)
    @test_throws "one surface" casadi_eval(CasADiCodegen.Function("t", [z0], [twice(z0)]), [0.0, 0.0])
    # An event function of the user beside the switch, at v = -1/2 after it. The integration ends
    # there, before the transition of the user, and the results are those without modes.
    model, modes = thrust_model()
    x = model["x"]
    model["zero"], model["transition"] = x[2] + 0.5, vertcat([x[1], -x[2]])
    E = integrator("E", Vern9(), model; modes, switches = 2, TIGHT...)
    @test [C.name_out(E, j) for j in 0:C.n_out(E)-1] == ["xf", "te"]
    x0 = MX.sym("x0", 2)
    out = E(x0, MX(0.0), MX(10.0))
    τ = sqrt(2) + 0.5
    @test closeto(ode_emitted(CasADiCodegen.Function("e", [x0], [out[1], out[2]]), ([0.0, 0.0],)),
                  [[1 + sqrt(2) * τ - τ^2 / 2, -0.5], [2 * sqrt(2) + 0.5]], 1e-9)
    # With `terminal = false` the transition turns v to +1/2, and the integration goes on to tf
    # through a second switch, at 3√2 + 1 with v = -√2. The next crossing of v = -1/2, after
    # the event that `events` counts, has no effect.
    W = integrator("W", Vern9(), model; modes, switches = 3, terminal = false, switch_times = true, TIGHT...)
    @test [C.name_out(W, j) for j in 0:C.n_out(W)-1] == ["xf", "te", "ts", "ns"]
    rest = 5 - 3 * sqrt(2)                                  # from the second switch to tf = 6
    @test closeto(ode_emitted(CasADiCodegen.Function("w", [x0], collect(W(x0, MX(0.0), MX(6.0)))), ([0.0, 0.0],)),
                  [[1 - sqrt(2) * rest + rest^2 / 2, rest - sqrt(2)], [2 * sqrt(2) + 0.5], [sqrt(2), 3 * sqrt(2) + 1, 6.0], [2.0]], 1e-9)
    # The mode is a symbol of its own, the surface does not depend on it, and the bound needs modes.
    m, n2 = x[1], SX.sym("n")
    plain = Dict("x" => x, "ode" => vertcat([x[2], SX(1.0)]))
    @test_throws ArgumentError integrator("bad", Vern9(), plain, 0.0, 1.0; modes = m => 1 - x[1], switches = 1, TIGHT...)
    @test_throws ArgumentError integrator("bad", Vern9(), plain, 0.0, 1.0; modes = n2 => 1 - x[1] + n2, switches = 1, TIGHT...)
    @test_throws ArgumentError integrator("bad", Vern9(), plain, 0.0, 1.0; switches = 1, TIGHT...)
    @test_throws ArgumentError integrator("bad", Vern9(), plain, 0.0, [1.0]; modes = n2 => 1 - x[1], switches = 1, switch_times = true, TIGHT...)
end

# Regression: after a switch the next search for an event started next to the root it had just
# passed, with the step of earlier calls, so a search could stop short of its final time and lose
# a switch, or give a wrong Jacobian, depending on the calls before it. Oracle: a central difference.
@testset "modes: a call does not depend on the calls before it" begin
    x, a, t, m = SX.sym("x", 2), SX.sym("a"), SX.sym("t"), SX.sym("m", 2)
    model = Dict{String,Any}("x" => x, "p" => a, "t" => t, "ode" => vertcat([x[2] + 0.1 * m[2], a * m[1]]),
                             "quad" => m[1] * x[1]^2 + m[2] * x[2] * a)
    F = integrator("B", Vern9(), model, 0.0, 3.0; modes = m => vertcat([0.1 * t - x[1], x[2] - 0.5]), switches = 8, TIGHT...)
    x0, p = MX.sym("x0", 2), MX.sym("a")
    out = F(x0, p)
    y = vertcat([out[1], out[2]])
    # CasADi evaluates the graph that the generated code runs, with the same solvers, at a smaller cost of compilation.
    g = CasADiCodegen.Function("g", [x0, p], [y, jacobian(y, vertcat([x0, p]))])
    call(v) = casadi_eval(g, v[1:2], v[3])
    h, base = 1e-5, [-0.1, 0.2, 1.0]
    for v in (base, base + [1e-4, 0, 0], base, base + [0, 1e-4, 0], base)
        jac = call(v)[2]
        central = reduce(hcat, [(call(v + h * e)[1] - call(v - h * e)[1]) / 2h for e in eachcol([1.0 0 0; 0 1 0; 0 0 1])])
        @test isapprox(reshape(jac, 3, 3), central; rtol = 1e-6, atol = 1e-6)
    end
    # The same model with the modes as states and the switches written as events of the user: 8 events in each call.
    X, b, τ = SX.sym("X", 4), SX.sym("a"), SX.sym("t")
    n, M = X[1:2], X[3:4]
    S = vertcat([0.1 * τ - n[1], n[2] - 0.5])
    hand = Dict{String,Any}("x" => X, "p" => b, "t" => τ, "ode" => vertcat([n[2] + 0.1 * M[2], b * M[1], SX(0.0), SX(0.0)]),
                            "zero" => C.times(M, S), "transition" => horzcat([vertcat([n, -M[1], M[2]]), vertcat([n, M[1], -M[2]])]))
    H = integrator("H", Vern9(), hand; direction = -1, events = 9, exact = false, TIGHT...)
    events(v) = only(last(casadi_eval(H, [v[1:2]; sign(-v[1]); sign(v[2] - 0.5)], v[3], 0.0, 3.0)))
    @test [events(v) for v in (base, base + [1e-4, 0, 0], base + [1e-4, 0, 0], base, base + [0, 1e-4, 0], base)] == fill(8.0, 6)
end

# The graph transform `smoothed`: each switch on the chosen inputs becomes a smooth form of
# width ε. Oracles: the smooth model written by hand, and the original model away from switches.
using CasADiCodegen: smoothed

"The values of the expressions `values` of the symbols `symbols` at the numbers `point`."
evaluated(symbols, values, point...) = reduce(vcat, casadi_eval(CasADiCodegen.Function("e", symbols, values), point...))

@testset "smoothed replaces switches by smooth forms" begin
    x, q, w = SX.sym("x"), SX.sym("q"), SX.sym("w")
    S = x^2 - 0.5
    signs = [smoothed(C.if_else(S > 0, SX(1.0), SX(-1.0)), w), smoothed(sign(S), w), smoothed(sign(S), w; kind = :algebraic)]
    s = 0.7^2 - 0.5
    @test evaluated([x, w], signs, 0.7, 0.03) ≈ [tanh(s / 0.03), tanh(s / 0.03), s / hypot(s, 0.03)] rtol = 1e-14
    # Nested switches converge to the original away from their surfaces.
    nested = C.if_else(C.and(q > 0, S > 0), C.fmin(C.fmax(2 * S, SX(0.0)), SX(1.0)), SX(0.0))
    @test abs(evaluated([x, q, w], [smoothed(nested, w)], 0.8, 0.3, 1e-6)[1] - 2 * (0.8^2 - 0.5)) < 1e-6
    # The soft condition is exactly 0 far from the surface, so a branch that is NaN there stays out.
    @test isfinite(evaluated([x, w], [smoothed(C.if_else(x > 0, sqrt(x), SX(0.0)), w)], -1.0, 1e-3)[1])
end

# Regression, issue #4: MX folds a product with an `if_else_zero` node into a new node, so
# the inner `if_else` of a nested one stayed a switch on a soft condition.
@testset "smoothed of a nested if_else, $T" for T in (SX, MX)
    l, w = T.sym("l"), T.sym("w")
    e = C.if_else(l > 1, T(-1.0), C.if_else(l < -1, T(1.0), T(0.0)))
    f = CasADiCodegen.Function("f", [l], [e])
    # The table of `smoothed`: if_else(c, a, b) becomes c a + (1 - c) b.
    function by_hand(lv, wv)
        c1, c2 = smooth_step(lv - 1, wv), smooth_step(-1 - lv, wv)
        -c1 + (1 - c1) * c2
    end
    points = ((0.0, 1.0), (-3.0, 1.0), (-1.0, 0.5), (2.0, 2.0))
    @test [only(evaluated([l, w], [smoothed(e, w)], p...)) for p in points] ≈ [by_hand(p...) for p in points] rtol = 1e-14
    @test [only(reduce(vcat, casadi_eval(smoothed(f), p...))) for p in points] ≈ [by_hand(p...) for p in points] rtol = 1e-14
end

@testset "smoothed keeps shared subexpressions" begin
    x, p = SX.sym("x", 2), SX.sym("p")
    e = sign(x[1] - p)
    f = CasADiCodegen.Function("f", [x, p], [e * (x[2] + k) for k in 1:50])
    @test C.n_instructions(smoothed(f)) < 2 * C.n_instructions(f)
end

@testset "smoothed keeps switches off the chosen inputs, and refuses a test of a soft value" begin
    x, y, p, w = SX.sym("x", 2), SX.sym("y"), SX.sym("p"), SX.sym("w")
    r = sqrt(x[1]^2 + x[2]^2)
    guarded = C.if_else(r == 0, SX(0.0), x[1] / r) + sign(p)       # an equality, and a switch on p only
    @test evaluated([x, p, w], [smoothed(guarded, w; inputs = x)], [0.0, 0.0], 0.5, 1.0) == [1.0]
    floored = CasADiCodegen.Function("g", [x], [floor(x[1]) + abs(x[2])])
    @test CasADiCodegen.nonsmooth_operations(smoothed(floored)) == [:OP_FLOOR => 1]
    @test_throws ArgumentError smoothed(C.if_else((x[1] > 0) + (y > 0), SX(1.0), SX(-1.0)), w)
    @test_throws ArgumentError smoothed((x[1] > 0) * (y > 0) == 1, w)
    @test_throws ArgumentError smoothed(sign(x[1]), MX.sym("w"))
    @test_throws ArgumentError smoothed(CasADiCodegen.Function("g", [x, w], [sign(x[1])], ["x", "width"], ["r"]))
end

@testset "smoothed of an MX function with a call" begin
    xs, ps = SX.sym("x", 2), SX.sym("p")
    inner = CasADiCodegen.Function("inner", [xs, ps], [sign(xs[1]) * ps + xs[2]])
    x, p = MX.sym("x", 2), MX.sym("p")
    f = CasADiCodegen.Function("f", [x, p], [C.if_else(x[2] < p, inner(x, p), MX(0.0)), abs(x[1])], ["x", "p"], ["a", "b"])
    g = smoothed(f)
    @test C.name(g) == "f_smoothed" && [C.name_in(g, i) for i in 0:2] == ["x", "p", "width"] && [C.name_out(g, j) for j in 0:1] == ["a", "b"]
    expanded = smoothed(C.expand(f))
    point = ([0.2, -0.1], 0.3, 0.1)
    @test reduce(vcat, casadi_eval(g, point...)) ≈ reduce(vcat, casadi_eval(expanded, point...)) rtol = 1e-14
    @test isempty(CasADiCodegen.nonsmooth_operations(g))
    # The expression form of MX.
    w = MX.sym("w")
    @test evaluated([x, w], [smoothed(abs(x[1]), w)], [0.2, 0.0], 0.1) ≈ [0.2 * tanh(2)] rtol = 1e-14
end

# Continuation in the width: each solve starts from the root of the one before.
using SimpleNonlinearSolve: SimpleNewtonRaphson

@testset "natural_continuation follows the root as the width falls" begin
    z, w = SX.sym("z"), SX.sym("width")
    R = CasADiCodegen.Function("r", [z, w], [smooth_abs(z - 1, w) - 0.5], ["z", "width"], ["r"])
    problem = SciMLBase.NonlinearProblem(SciMLBase.NonlinearFunction(R), [3.0], [1.0])
    path = CasADiCodegen.natural_continuation(problem, SimpleNewtonRaphson(), [1.0, 0.1, 0.01, 1e-3]; parameter = 1, abstol = 1e-12)
    @test first.(path) == [1.0, 0.1, 0.01, 1e-3] && abs(last(path).second.u[1] - 1.5) < 1e-6
    named = CasADiCodegen.natural_continuation(problem, SimpleNewtonRaphson(), [1.0, 0.1, 0.01, 1e-3]; parameter = :width, abstol = 1e-12)
    @test last(named).second.u == last(path).second.u
    @test last(CasADiCodegen.natural_continuation(problem, SimpleNewtonRaphson(), [1.0, 0.1]; parameter = "width", abstol = 1e-12)).second.u ==
          last(CasADiCodegen.natural_continuation(problem, SimpleNewtonRaphson(), [1.0, 0.1]; parameter = :width, abstol = 1e-12)).second.u
end

@testset "natural_continuation tries intermediate values" begin
    # Newton on tanh((z - 1)/ε) = 1/2 diverges from a start more than about ε from the root.
    z, w = SX.sym("z"), SX.sym("width")
    R = CasADiCodegen.Function("r", [z, w], [smooth_sign(z - 1, w) - 0.5], ["z", "width"], ["r"])
    problem = SciMLBase.NonlinearProblem(SciMLBase.NonlinearFunction(R), [1.5], [1.0])
    path = CasADiCodegen.natural_continuation(problem, SimpleNewtonRaphson(), [1.0, 0.1]; parameter = 1, abstol = 1e-12)
    @test length(path) > 2 && first(last(path)) == 0.1 && SciMLBase.successful_retcode(last(path).second)
    @test last(path).second.u[1] ≈ 1 + 0.1 * atanh(0.5) rtol = 1e-12
    stopped = CasADiCodegen.natural_continuation(problem, SimpleNewtonRaphson(), [1.0, 0.1]; parameter = 1, halvings = 0)
    @test first.(stopped) == [1.0, 0.1] && !SciMLBase.successful_retcode(last(stopped).second)
    @test_throws ArgumentError CasADiCodegen.natural_continuation(problem, SimpleNewtonRaphson(), Float64[]; parameter = 1)
    tupled = SciMLBase.NonlinearProblem((u, q) -> u .- q[1], [1.0], (1.0,))
    @test_throws ArgumentError CasADiCodegen.natural_continuation(tupled, SimpleNewtonRaphson(), [0.5]; parameter = 1)
end
