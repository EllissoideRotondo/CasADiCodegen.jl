import ForwardDiff
import SpecialFunctions

include("interpreter.jl")
const SCALAR = casadi.Sparsity.dense(1, 1)
const COLUMN2 = casadi.Sparsity.dense(2, 1)
column(n) = casadi.Sparsity.dense(n, 1)

# f(x, p) = [x1 * x2 + p, sin(x1)], with slot reuse as CasADi produces it.
function demo_algorithm()
    Algorithm("f",
        [I(:OP_INPUT, 1, 1, 1), I(:OP_INPUT, 2, 1, 2), I(:OP_MUL, 2, 1, 2),
         I(:OP_INPUT, 3, 2, 1), I(:OP_ADD, 2, 2, 3), I(:OP_OUTPUT, 1, 2, 1),
         I(:OP_SIN, 1, 1), I(:OP_OUTPUT, 1, 1, 2)],
        3, [COLUMN2, SCALAR], [COLUMN2], ["x", "p"], ["y"])
end

demo_expected(x, p) = [x[1] * x[2] + p[1], sin(x[1])]

# Every way of running a alg, as (res, arg) -> res.
function evaluators(alg; chunk_size)
    mod = Module()
    Core.eval(mod, build_function(alg; nonzeros = true, name = :g!, chunk_size))
    src = Module()
    Base.include_string(src, sprint(io -> generate_julia(alg, io; module_name = :Gen, name = :g!, chunk_size)))
    g = compiled(alg; chunk_size)
    [
        "interpreter" => (res, arg) -> interpret!(res, alg, arg),
        "runtime function" => g,
        "expression" => (res, arg) -> Base.invokelatest((r, a) -> mod.g!(r, a), res, arg),
        "source" => (res, arg) -> Base.invokelatest((r, a) -> src.Gen.g_nonzeros!(r, a), res, arg),
    ]
end

@testset "emitter on a hand-written algorithm" begin
    alg = demo_algorithm()
    x, p = [0.3, -1.7], [2.5]
    for chunk_size in (1000, 3), (label, g) in evaluators(alg; chunk_size)
        @testset "$label, chunk_size = $chunk_size" begin
            res = g(results(alg), (x, p))
            @test res[1] == demo_expected(x, p)
        end
    end
end

@testset "constants reach outputs and operations" begin
    alg = Algorithm("k",
        [I(:OP_CONST, 1, 0, 0, 2.5), I(:OP_OUTPUT, 1, 1, 1), I(:OP_INPUT, 2, 1, 1),
         I(:OP_SUB, 1, 1, 2), I(:OP_OUTPUT, 1, 1, 2)],
        2, [SCALAR], [COLUMN2], ["x"], ["y"])
    for chunk_size in (1000, 2), (label, g) in evaluators(alg; chunk_size)
        @test g(results(alg), ([1.0],))[1] == [2.5, 1.5]
    end
end

@testset "generic number type" begin
    alg = demo_algorithm()
    for chunk_size in (1000, 2)
        g = compiled(alg; chunk_size)
        h(v) = g([zeros(eltype(v), 2)], (v[1:2], v[3:3]))[1]
        @test ForwardDiff.jacobian(h, [0.3, -1.7, 2.5]) == [-1.7 0.3 1.0; cos(0.3) 0.0 0.0]
        @test g([zeros(Float32, 2)], (Float32[0.3, -1.7], Float32[2.5]))[1] ≈ demo_expected([0.3, -1.7], [2.5]) rtol = 1e-6
    end
end

@testset "domain errors give NaN, as in CasADi" begin
    alg = Algorithm("d",
        [I(:OP_INPUT, 1, 1, 1), I(:OP_SQRT, 2, 1), I(:OP_OUTPUT, 1, 2, 1),
         I(:OP_LOG, 2, 1), I(:OP_OUTPUT, 1, 2, 2)],
        2, [SCALAR], [COLUMN2], ["x"], ["y"])
    for (label, g) in evaluators(alg; chunk_size = 1000)
        @test all(isnan, g(results(alg), ([-1.0],))[1])
    end
end

@testset "nanmath = false uses the throwing Base functions" begin
    alg = Algorithm("d",
        [I(:OP_INPUT, 1, 1, 1), I(:OP_SQRT, 2, 1), I(:OP_OUTPUT, 1, 2, 1),
         I(:OP_INPUT, 3, 1, 2), I(:OP_POW, 2, 1, 3), I(:OP_OUTPUT, 1, 2, 2),
         I(:OP_ERFINV, 2, 3), I(:OP_OUTPUT, 1, 2, 3)],
        3, [COLUMN2], [column(3)], ["x"], ["y"])
    mod = Module()
    Core.eval(mod, build_function(alg; nonzeros = true, name = :g!, nanmath = false))
    src = Module()
    source = sprint(io -> generate_julia(alg, io; module_name = :Gen, name = :g!, nanmath = false))
    Base.include_string(src, source)
    forms = [compiled(alg; nanmath = false),
             (res, arg) -> Base.invokelatest((r, a) -> mod.g!(r, a), res, arg),
             (res, arg) -> Base.invokelatest((r, a) -> src.Gen.g_nonzeros!(r, a), res, arg)]
    for g in forms
        @test g([zeros(3)], ([4.0, 0.5],))[1] == [2.0, 2.0, SpecialFunctions.erfinv(0.5)]
        @test_throws DomainError g([zeros(3)], ([-4.0, 0.5],))      # sqrt and pow leave their domain
        @test_throws DomainError g([zeros(3)], ([4.0, 1.5],))       # erfinv leaves its domain
    end
    @test !occursin("NaNMath", source) && occursin("import SpecialFunctions", source)
    @test !occursin("erfinv(x) =", source)
    # The default still gives NaN, and both settings agree inside the domain.
    @test all(isnan, compiled(alg)([zeros(3)], ([-4.0, 1.5],))[1])
    @test compiled(alg)([zeros(3)], ([4.0, 0.5],)) == forms[1]([zeros(3)], ([4.0, 0.5],))
    # Without NaNMath every real number type works.
    big_result = compiled(demo_algorithm(); nanmath = false)([zeros(BigFloat, 2)], (BigFloat[0.3, -1.7], BigFloat[2.5]))
    @test eltype(big_result[1]) == BigFloat
    @test big_result[1] == demo_expected(BigFloat[0.3, -1.7], BigFloat[2.5])
end

@testset "DEPENDENCIES lists the imports of the module" begin
    imported(source) = Tuple(sort!([Symbol(m[1]) for m in eachmatch(r"^import (\w+)$"m, source)]))
    x = SX.sym("x", 2)
    y = [sqrt(x[1]); x[2]^2]
    for (nanmath, expected) in ((true, (:NaNMath, :SparseArrays, :StaticArraysCore)), (false, (:SparseArrays, :StaticArraysCore)))
        source = sprint(io -> generate_julia(CasadiFunction("f", [x], [y, jacobian(y, x)]), io; module_name = :Gen, nanmath))
        mod = Module()
        Base.include_string(mod, source)
        @test mod.Gen.DEPENDENCIES == imported(source) == expected
    end
end

@testset "wrong argument shape" begin
    alg = demo_algorithm()
    # The checks are in the entry function, which the split does not change.
    for (label, g) in evaluators(alg; chunk_size = 3)
        label == "interpreter" && continue
        @test_throws ArgumentError g(results(alg), ([0.3, 1.0],))
        @test_throws ArgumentError g(results(alg), ([0.3], [1.0]))
        @test_throws ArgumentError g([zeros(1)], ([0.3, 1.0], [1.0]))
        @test_throws ArgumentError g((), ([0.3, 1.0], [1.0]))
    end
end

@testset "wrong keywords" begin
    @test_throws ArgumentError build_function(demo_algorithm(); nonzeros = true, chunk_size = 0)
    @test_throws ArgumentError build_function(demo_algorithm(); nonzeros = true, split_above = -1)
    @test_throws ArgumentError build_function(demo_algorithm(); nonzeros = true, name = Symbol("not valid"))
    @test_throws ArgumentError build_function(demo_algorithm(); nonzeros = true, expression = true)
    @test_throws ArgumentError generate_julia(demo_algorithm(), IOBuffer(); module_name = Symbol("not valid"))
end

@testset "source file runs without CasADi" begin
    alg = demo_algorithm()
    mktempdir() do dir
        path = joinpath(dir, "gen.jl")
        @test generate_julia(alg, path; module_name = :Gen, chunk_size = 3) == path
        script = """
            include($(repr(path)))
            res = Gen.f_nonzeros!([zeros(2)], ([0.3, -1.7], [2.5]))
            isdefined(Main, :CasADiCodegen) && error("CasADi was loaded")
            Gen.NNZ_IN == (2, 1) && Gen.NNZ_OUT == (2,) || error("bad metadata")
            Gen.SPARSITY_OUT[1] == (2, 1, [1, 3], [1, 2]) || error("bad sparsity")
            isdefined(Gen, :NaNMath) || error("the module must load NaNMath for sin")
            print(res[1][1], " ", res[1][2])
            """
        out = read(`$(Base.julia_cmd()) --startup-file=no --project=$(dirname(Base.active_project())) -e $script`, String)
        @test parse.(Float64, split(out)) == demo_expected([0.3, -1.7], [2.5])
    end
end

# ---------------------------------------------------------------------------
# Property: every emitted form agrees bit for bit with the interpreter, on
# random algorithms over the whole operation table, for any chunk size.
# ---------------------------------------------------------------------------
# OP_PRINTME is left out: it prints at each run, and test_nodes.jl checks it.
const OP_LIST = sort!(collect(setdiff(keys(CasADiCodegen.Codegen.OPERATIONS), [:OP_PRINTME])))
const N_ARG, N_SLOT = 3, 6

# A step is (operation choice, output slot, operand, operand); the builder maps
# it to a valid instruction, so every generated vector is a valid algorithm.
function random_algorithm(steps, constants)
    ins = [I(:OP_INPUT, k, 1, k) for k in 1:N_ARG]
    defined = collect(1:N_ARG)
    pick(i) = defined[mod1(i, length(defined))]
    for (n, (kind, out, a, b)) in enumerate(steps)
        slot = mod1(out, N_SLOT)
        if kind == 0
            push!(ins, I(:OP_CONST, slot, 0, 0, constants[mod1(n, length(constants))]))
        else
            op = OP_LIST[mod1(kind, length(OP_LIST))]
            arity = CasADiCodegen.Codegen.OPERATIONS[op].arity
            push!(ins, I(op, slot, pick(a), arity == 2 ? pick(b) : 0))
        end
        slot in defined || push!(defined, slot)
    end
    nout = length(defined)
    for (k, slot) in enumerate(defined)
        push!(ins, I(:OP_OUTPUT, 1, slot, k))
    end
    Algorithm("r", ins, N_SLOT, [column(N_ARG)], [column(nout)], ["x"], ["y"])
end

const step_gen = @composed function step(kind = Data.Integers(0, 80), out = Data.Integers(1, N_SLOT),
                                         a = Data.Integers(1, 12), b = Data.Integers(1, 12))
    (kind, out, a, b)
end
const float_gen = Data.Floats{Float64}()

@testset "property: emitted forms equal the interpreter" begin
    @check max_examples = 80 function emitted_equals_interpreter(
            steps = Data.Vectors(step_gen; min_size = 1, max_size = 40),
            constants = Data.Vectors(float_gen; min_size = 1, max_size = 4),
            x = Data.Vectors(float_gen; min_size = N_ARG, max_size = N_ARG),
            chunk_size = Data.Integers(1, 12))
        alg = random_algorithm(steps, constants)
        ref = interpret!(results(alg), alg, (x,))
        whole = compiled(alg; chunk_size = 10_000)(results(alg), (x,))
        split = compiled(alg; chunk_size)(results(alg), (x,))
        isequal(ref, whole) && isequal(ref, split)
    end
end

@testset "default chunk size" begin
    limit, size = CasADiCodegen.Codegen.UNSPLIT_LIMIT, CasADiCodegen.Codegen.DEFAULT_CHUNK_SIZE
    # One function below the limit, sub-functions above it, with the same results.
    x = SX.sym("x", 4)
    y = x
    for _ in 1:100
        y = [sin(y[1]) * y[2] + y[4]; y[3]^2 / (1 + y[1]^2); cos(y[4] * y[2]); y[1] * y[4] - y[2]] / 2
    end
    f = CasADiCodegen.Function("f", [x], [y])
    alg = Algorithm(f)
    @test size < length(alg.instructions) <= limit
    @test !occursin("_chunk", string(build_function(alg; nonzeros = true)))
    @test occursin("_chunk", string(build_function(alg; nonzeros = true, chunk_size = size)))
    @test occursin("_chunk", string(build_function(alg; nonzeros = true, split_above = size)))
    @test !occursin("_chunk", string(build_function(alg; nonzeros = true, chunk_size = 5, split_above = limit)))
    @test occursin("_chunk2(w)", sprint(io -> generate_julia(alg, io; split_above = 0)))
    @test !occursin("_chunk", sprint(io -> generate_julia(alg, io)))
    xv = ([0.1, 0.2, 0.3, 0.4],)
    @test compiled(alg)([zeros(4)], xv) == compiled(alg; chunk_size = size)([zeros(4)], xv)
    @test compiled(alg)([zeros(4)], xv) == compiled(alg; chunk_size = 50, split_above = 100)([zeros(4)], xv)
end
