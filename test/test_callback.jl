# Julia functions called from CasADi graphs and from the generated code.
import ADTypes
import DifferentiationInterface
import ForwardDiff
import LinearAlgebra
using StaticArrays: SVector, SMatrix

module CallbackModels
drag(v) = [v[1]^2 * v[2], sin(v[1]) + v[3]]
drag_jacobian(v) = [2v[1]*v[2] v[1]^2 0.0; cos(v[1]) 0.0 1.0]
mix(A, v) = (A * v, [sum(A) * v[1]])
# A curve with its time derivatives, as an ephemeris gives them, and a function of two arguments.
using StaticArrays: SVector, SMatrix
curve(t) = SVector(sin(t[1]), t[1]^3, exp(2t[1]))
curve1(t) = SVector(cos(t[1]), 3t[1]^2, 2exp(2t[1]))
curve2(t) = SVector(-sin(t[1]), 6t[1], 4exp(2t[1]))
curve3(t) = SVector(-cos(t[1]), 6.0, 8exp(2t[1]))
# A function with a check of its own, which generated code must keep.
Base.@propagate_inbounds guarded(t) = (@boundscheck(t[1] <= 1.0 || throw(DomainError(t[1]))); SVector(2t[1]))
# A function of a matrix and a vector with two results, and its four Jacobian blocks.
blend(A, v) = (A * v, SVector(sum(A) * v[1]))
blend1(A, v) = (SMatrix{2,6}(v[1], 0, 0, v[1], v[2], 0, 0, v[2], v[3], 0, 0, v[3]), A,
                SMatrix{1,6}(ntuple(_ -> v[1], 6)), SMatrix{1,3}(sum(A), 0, 0))
pull(r, m) = [m[1] * r[1] * r[2], r[2]^2 + m[1]]
pull1(r, m) = ([m[1]*r[2] m[1]*r[1]; 0.0 2r[2]], [r[1] * r[2]; 1.0;;])               # blocks by r and by m
pull2(r, m) = ([0.0 m[1]; 0.0 0.0; m[1] 0.0; 0.0 2.0], [r[2]; 0.0; r[1]; 0.0;;],      # block 1 by r and by m
               [r[2] r[1]; 0.0 0.0], [0.0; 0.0;;])                                    # block 2 by r and by m
end

struct PlainCallback <: Callback end
casadi.eval_(::PlainCallback, arg) = [arg[1]]

const DRAG_X = [0.7, -1.3, 0.4]

emitted(f, res, arg; kwargs...) = build_function(f; nonzeros = true, expression = Val{false}, kwargs...)(res, arg)

@testset "a Julia function is a CasADi function" begin
    drag = callback(CallbackModels.drag, "drag", (3,), (2,))
    @test casadi.nonzeros(drag(casadi.DM(DRAG_X))) == CallbackModels.drag(DRAG_X)
    v = MX.sym("v", 3)
    f = CasADiCodegen.Function("f", [v], [2 * drag(v * 3)])
    @test casadi.nonzeros(f(casadi.DM(DRAG_X))) == 2 * CallbackModels.drag(3 * DRAG_X)
    @test emitted(f, [zeros(2)], (DRAG_X,))[1] == 2 * CallbackModels.drag(3 * DRAG_X)
    @test emitted(f, [zeros(2)], (DRAG_X,); expand = false, chunk_size = 1)[1] == 2 * CallbackModels.drag(3 * DRAG_X)
end

@testset "matrix arguments and many results" begin
    mix = callback(CallbackModels.mix, "mix", ((2, 3), 3), (2, 1))
    A, v = MX.sym("A", 2, 3), MX.sym("v", 3)
    both = mix(A, v)
    f = CasADiCodegen.Function("f", [A, v], [both[1] + both[2]])
    g = CasADiCodegen.Function("g", [A, v], [both[2] * 2])          # the first result is not read
    M = [1.0 2.0 3.0; -1.0 0.5 2.0]
    expected = CallbackModels.mix(M, DRAG_X)
    @test emitted(f, [zeros(2)], (vec(M), DRAG_X))[1] == expected[1] .+ expected[2]
    @test emitted(g, [zeros(1)], (vec(M), DRAG_X))[1] == 2 * expected[2]
    @test casadi_nonzeros(f(casadi.DM(M), casadi.DM(DRAG_X)))[1] == expected[1] .+ expected[2]
end

@testset "Jacobian from a Julia function" begin
    drag = callback(CallbackModels.drag, "drag", (3,), (2,); jacobian = CallbackModels.drag_jacobian)
    v = MX.sym("v", 3)
    f = CasADiCodegen.Function("f", [v], [jacobian(drag(2 * v), v)])
    expected = 2 * CallbackModels.drag_jacobian(2 * DRAG_X)
    @test reshape(casadi.nonzeros(f(casadi.DM(DRAG_X))), 2, 3) == expected
    @test reshape(emitted(f, [zeros(6)], (DRAG_X,))[1], 2, 3) == expected
end

@testset "Jacobian from an automatic differentiation backend" begin
    drag = callback(CallbackModels.drag, "drag", (3,), (2,); jacobian = ADTypes.AutoForwardDiff())
    v = MX.sym("v", 3)
    y = drag(v)
    f = CasADiCodegen.Function("f", [v], [jacobian(y, v)])
    @test reshape(emitted(f, [zeros(6)], (DRAG_X,))[1], 2, 3) ≈ CallbackModels.drag_jacobian(DRAG_X)
    @test reshape(casadi.nonzeros(f(casadi.DM(DRAG_X))), 2, 3) ≈ CallbackModels.drag_jacobian(DRAG_X)
    # Second derivatives: the backend differentiates its own Jacobian.
    h = CasADiCodegen.Function("h", [v], [jacobian(jacobian(y[1], v), v)])
    x = DRAG_X
    @test reshape(emitted(h, [zeros(9)], (x,))[1], 3, 3) ≈ [2x[2] 2x[1] 0; 2x[1] 0 0; 0 0 0]
    mix = callback(CallbackModels.mix, "mix", ((2, 3), 3), (2, 1); jacobian = ADTypes.AutoForwardDiff())
    A = MX.sym("A", 2, 3)
    m = CasADiCodegen.Function("m", [A, v], [jacobian(mix(A, v)[1], v), jacobian(mix(A, v)[2], A)])
    M = [1.0 2.0 3.0; -1.0 0.5 2.0]
    r = emitted(m, [zeros(6), zeros(6)], (vec(M), x))
    @test reshape(r[1], 2, 3) ≈ M && r[2] ≈ fill(x[1], 6)
    @test_throws "DifferentiationInterface" CasADiCodegen.Codegen.backend_jacobian(sin, :none, (1,), (1,))
end

@testset "derivatives of any order from a chain of Julia functions" begin
    M = CallbackModels
    curve = callback(M.curve, "curve", (1,), (3,); jacobian = (M.curve1, M.curve2, M.curve3))
    t = MX.sym("t")
    y = curve(2 * t)
    orders = [y, jacobian(y, t), jacobian(jacobian(y, t), t), jacobian(jacobian(jacobian(y, t), t), t)]
    f = CasADiCodegen.Function("f", [t], orders)
    tv = 0.3
    expected = [M.curve([2tv]), 2 * M.curve1([2tv]), 4 * M.curve2([2tv]), 8 * M.curve3([2tv])]
    @test all(casadi.nonzeros(r) ≈ e for (r, e) in zip(f(casadi.DM(tv)), expected))
    @test all(emitted(f, [zeros(3) for _ in 1:4], ([tv],)) .≈ expected)
    # The chain ends: one more derivative has no function.
    short = callback(M.curve, "short", (1,), (3,); jacobian = M.curve1)
    @test_throws casadi.SwigError jacobian(jacobian(short(t), t), t)
    # A written module calls each function of the chain by its name.
    text = sprint(io -> generate_julia(f, io; module_name = :Gen))
    @test occursin("checked(Main.CallbackModels.curve3,", text)

    # Two arguments: the second function gives one block per pair of a first block and an argument.
    pull = callback(M.pull, "pull", (2, 1), (2,); jacobian = (M.pull1, M.pull2))
    r, m = MX.sym("r", 2), MX.sym("m")
    z = pull(r, m)
    h = CasADiCodegen.Function("h", [r, m], [jacobian(jacobian(z, r), r), jacobian(jacobian(z, r), m), jacobian(jacobian(z, m), r)])
    rv, mv = [0.4, -1.1], [2.5]
    full(x) = M.pull(x[1:2], x[3:3])
    H = ForwardDiff.jacobian(x -> vec(ForwardDiff.jacobian(full, x)), [rv; mv])    # rows: vec of the 2 by 3 Jacobian
    got = emitted(h, [zeros(8), zeros(4), zeros(4)], (rv, mv))
    @test reshape(got[1], 4, 2) ≈ H[1:4, 1:2] && got[2] ≈ H[1:4, 3] && reshape(got[3], 2, 2) ≈ H[5:6, 1:2]
    # The backend gives the same second derivatives.
    auto = callback(M.pull, "auto", (2, 1), (2,); jacobian = ADTypes.AutoForwardDiff())
    za = auto(r, m)
    ha = CasADiCodegen.Function("ha", [r, m], [jacobian(jacobian(za, r), r), jacobian(jacobian(za, r), m), jacobian(jacobian(za, m), r)])
    @test all(emitted(ha, [zeros(8), zeros(4), zeros(4)], (rv, mv)) .≈ got)
end

@testset "scalar code around Julia functions" begin
    M = CallbackModels
    curve = callback(M.curve, "curve", (1,), (3,); jacobian = (M.curve1, M.curve2))
    mix = callback(M.blend, "blend", ((2, 3), 3), (2, 1); jacobian = M.blend1)
    x, t, A = MX.sym("x", 3), MX.sym("t"), MX.sym("A", 2, 3)
    c = curve(2 * t)
    y = LinearAlgebra.dot(c, x) * x + sin.(c) + mix(A, curve(t) + x)[2] * x
    f = CasADiCodegen.Function("f", [x, t, A], [y, jacobian(y, vcat(x, t)), mix(A, x)[1]])
    xv, tv, Av = DRAG_X, [0.3], [1.0 2.0 3.0; -1.0 0.5 2.0]
    expected = casadi_nonzeros(f(casadi.DM(xv), casadi.DM(tv), casadi.DM(Av)))
    blank() = [zeros(length(e)) for e in expected]
    # One function with local variables by default; matrix code with `expand = false` or when it is split.
    code(; kwargs...) = string(build_function(f; nonzeros = true, kwargs...))
    @test !occursin("workspace", code()) && occursin("checked", code())
    @test occursin("workspace", code(; expand = false)) && occursin("workspace", code(; chunk_size = 5))
    for kwargs in ((;), (; expand = false), (; chunk_size = 5))
        @test all(emitted(f, blank(), (xv, tv, vec(Av)); kwargs...) .≈ expected)
    end
    # No allocation in place and with static arrays.
    g, g! = build_function(f; expression = Val{false})
    out = (zeros(3), zeros(3, 4), zeros(2))
    static_arguments = (SVector{3}(xv), SVector{1}(tv), SMatrix{2,3}(Av))
    run(g, a) = g(a...)
    g!(out, (xv, tv, Av)); r = run(g, static_arguments)
    @test all(vec(a) ≈ e && vec(collect(b)) ≈ e for (a, b, e) in zip(out, r, expected))
    @test (@allocated g!(out, (xv, tv, Av))) == 0
    @test (@allocated run(g, static_arguments)) == 0
    # Other number types pass through the Julia functions.
    J = ForwardDiff.jacobian(v -> emitted(f, [similar(v, length(e)) for e in expected], (v, tv, vec(Av)))[1], xv)
    @test vec(J) ≈ expected[2][1:9]
    # A written module calls the functions by name.
    text = sprint(io -> generate_julia(f, io; module_name = :Gen))
    @test occursin("checked(Main.CallbackModels.curve1,", text) && occursin("const NWORK = 0", text)
    @test occursin("import StaticArrays\n", text)
    src = Module()
    Base.include_string(src, text)
    @test all(Base.invokelatest((r, a) -> src.Gen.f_nonzeros!(r, a), blank(), (xv, tv, vec(Av))) .≈ expected)
    # The Julia function keeps its own checks in every kind of code.
    guarded = callback(M.guarded, "guarded", (1,), (1,))
    h = CasADiCodegen.Function("h", [t], [guarded(t) + t])
    for kwargs in ((;), (; expand = false))
        @test emitted(h, [zeros(1)], ([0.5],); kwargs...)[1] == [1.5]
        @test_throws DomainError emitted(h, [zeros(1)], ([2.0],); kwargs...)
    end
end

@testset "every form of the code calls the Julia function" begin
    drag = callback(CallbackModels.drag, "drag", (3,), (2,); jacobian = CallbackModels.drag_jacobian)
    v = MX.sym("v", 3)
    f = CasADiCodegen.Function("f", [v], [drag(v), jacobian(drag(v), v)])
    expected = [CallbackModels.drag(DRAG_X), vec(CallbackModels.drag_jacobian(DRAG_X))]
    mod = Module()
    Core.eval(mod, build_function(f; nonzeros = true))
    @test Base.invokelatest((r, a) -> mod.f!(r, a), [zeros(2), zeros(6)], (DRAG_X,)) == expected
    text = sprint(io -> generate_julia(f, io; module_name = :Gen))
    @test occursin("checked(Main.CallbackModels.drag,", text) && occursin("checked(Main.CallbackModels.drag_jacobian,", text)
    unit = callback(LinearAlgebra.normalize, "unit", (3,), (3,))
    text = sprint(io -> generate_julia(CasADiCodegen.Function("u", [v], [unit(v)]), io))
    @test occursin("import LinearAlgebra", text) && occursin("checked(LinearAlgebra.normalize,", text)
end

# The work vector of matrix code and of split code belongs to the task. The Julia function gives
# way to the other tasks between a write to the work vector and the read of it, so that a vector
# that the tasks share would hold the numbers of another call.
@testset "tasks do not share the work vector" begin
    pause = callback(v -> (yield(); 2 .* v), "pause", (3,), (3,))
    v = MX.sym("v", 3)
    a = sin.(v)
    f = CasADiCodegen.Function("f", [v], [3 * a + casadi.times(pause(a), a)])
    inputs = [DRAG_X .+ k for k in 1:8]
    for kwargs in ((; expand = false), (; chunk_size = 2))
        @test occursin("workspace", string(build_function(f; nonzeros = true, kwargs...)))
        g = build_function(f; nonzeros = true, expression = Val{false}, kwargs...)
        tasks = [@async g([zeros(3)], (x,))[1] for x in inputs]
        @test all(map(fetch, tasks) .≈ [3 .* sin.(x) .+ 2 .* sin.(x) .^ 2 for x in inputs])
    end
end

@testset "limits of callbacks are named" begin
    v = MX.sym("v", 3)
    anonymous = callback(v -> 2 .* v, "anonymous", (3,), (3,))
    f = CasADiCodegen.Function("f", [v], [anonymous(v)])
    @test emitted(f, [zeros(3)], (DRAG_X,))[1] == 2 .* DRAG_X
    @test_throws "named function" generate_julia(f, IOBuffer())
    short = callback(v -> [1.0], "short", (3,), (2,))
    @test_throws DimensionMismatch emitted(CasADiCodegen.Function("s", [v], [short(v)]), [zeros(2)], (DRAG_X,))
    plain = CasADiCodegen.Function(callback(PlainCallback(), "plain"))
    s = MX.sym("s")
    # A callback made with the type of the front end has no record of its Julia function: generated code cannot call it.
    @test_throws "callback of a subtype of Callback" emitted(CasADiCodegen.Function("p", [s], [2 * plain(s)]), [zeros(1)], ([1.5],))
    @test_throws ArgumentError callback(sin, "bad name", (1,), (1,))
    @test_throws ArgumentError callback(sin, "shape", ((1, 2, 3),), (1,))
end

@testset "a Julia function in a branch runs only when the branch is selected" begin
    runs = Ref(0)
    counted = callback(v -> (runs[] += 1; [v[1]^2, v[2]]), "counted", (2,), (2,))
    a, c = MX.sym("a", 2), MX.sym("c")
    f = CasADiCodegen.Function("f", [c, a], [casadi.if_else(c > 0, counted(a), 2 * a, true)])
    g = build_function(f; nonzeros = true, expression = Val{false})
    @test g([zeros(2)], ([0.0], [3.0, 4.0])) == [[6.0, 8.0]]
    @test runs[] == 0
    @test g([zeros(2)], ([1.0], [3.0, 4.0])) == [[9.0, 4.0]]
    @test runs[] == 1
end
