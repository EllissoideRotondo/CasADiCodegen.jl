# Properties that use CasADi itself as the oracle.
import LinearAlgebra
import SpecialFunctions
using RuntimeGeneratedFunctions: RuntimeGeneratedFunction

emitted_eval(alg, args...; kwargs...) =
    compiled(alg; kwargs...)([fill(NaN, n) for n in alg.nnz_out], args)

# IEEE 754 requires these to be correctly rounded, so CasADi (C) and Julia must
# agree bit for bit. The others may differ in the last places between libraries.
const EXACT_OPS = Set([:OP_ASSIGN, :OP_ADD, :OP_SUB, :OP_MUL, :OP_DIV, :OP_NEG, :OP_SQRT, :OP_SQ,
    :OP_TWICE, :OP_LT, :OP_LE, :OP_EQ, :OP_NE, :OP_NOT, :OP_AND, :OP_OR, :OP_FLOOR, :OP_CEIL,
    :OP_FMOD, :OP_FABS, :OP_SIGN, :OP_COPYSIGN, :OP_IF_ELSE_ZERO, :OP_FMIN, :OP_FMAX, :OP_INV,
    :OP_REMAINDER, :OP_INPUT, :OP_OUTPUT, :OP_CONST])

same_bits(a, b) = isequal(a, b)
function op_function(op)
    arity, build = OP_BUILDERS[op]
    x, y = SX.sym("x"), SX.sym("y")
    CasADiCodegen.Function("f", [x, y], [arity == 1 ? build(x) : build(x, y)])
end

const OP_CASES = Dict(op => (f = op_function(op); (f, Algorithm(f))) for op in keys(OP_BUILDERS))

function op_agrees(op, a, b)
    f, alg = OP_CASES[op]
    expected = casadi_eval(f, a, b)[1][1]
    actual = emitted_eval(alg, [a], [b])[1][1]
    # CasADi's own erfinv is a rational approximation with a relative error up to
    # 2e-6 on (-0.7, 0.7) and near 1e-4 below -0.7. The emitted code calls the
    # accurate SpecialFunctions.erfinv, so it is tested as the inverse of erf and
    # only loosely against CasADi.
    op === :OP_ERFINV && return erfinv_agrees(a, expected, actual)
    # NaNMath.pow gives NaN for every negative base with a non-integer exponent.
    # The C library does not when the base or the exponent is infinite, for
    # example pow(-Inf, 0.5) == Inf and pow(-2, Inf) == Inf.
    exponent = op === :OP_CONSTPOW ? 2.5 : b
    if op in (:OP_POW, :OP_CONSTPOW) && a < 0 && !isinteger(exponent) && (isinf(a) || isinf(exponent))
        return isnan(actual)
    end
    op in EXACT_OPS ? same_bits(expected, actual) : close_enough(expected, actual)
end

function erfinv_agrees(a, expected, actual)
    abs(a) >= 1 && return isequal(expected, actual)
    isnan(a) && return isnan(actual)
    isapprox(SpecialFunctions.erf(actual), a; rtol = 1e-14, atol = 1e-300) &&
        isapprox(expected, actual; rtol = 1e-3)
end

@testset "operation table" begin
    # OP_ASSIGN has no front end builder, and no function of these tests holds it, so no test checks its meaning.
    # OP_PRINTME prints, and has a test of its own in test_nodes.jl.
    @test setdiff(keys(CasADiCodegen.Codegen.OPERATIONS), keys(OP_BUILDERS)) == Set([:OP_ASSIGN, :OP_PRINTME])
    @testset "$op is what the builder produces" for op in sort!(collect(keys(OP_BUILDERS)))
        @test any(ins -> ins.op === op, OP_CASES[op][2].instructions)
    end
end

@testset "property: each operation agrees with CasADi" begin
    @testset "$op" for op in sort!(collect(keys(OP_BUILDERS)))
        @check max_examples = 50 op_agrees(Data.Just(op), test_float, test_float)
    end
end

# The random search reaches infinities and other special values only rarely, so
# every pair of them is also checked directly.
const SPECIAL_VALUES = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0, 2.5, -2.5, 1e-300, -1e-300,
    5e-324, 1e300, -1e300, floatmax(), -floatmax(), Inf, -Inf, NaN, pi, -pi, pi / 2, 1 - eps(), -1 + eps()]

@testset "each operation agrees with CasADi on special values" begin
    @testset "$op" for op in sort!(collect(keys(OP_BUILDERS)))
        disagreements = [(a, b) for a in SPECIAL_VALUES, b in SPECIAL_VALUES if !op_agrees(op, a, b)]
        @test isempty(disagreements)
    end
end

# ---------------------------------------------------------------------------
# Random expression trees over the correctly rounded operations: the emitted
# function and its Jacobian must equal CasADi bit for bit.
# ---------------------------------------------------------------------------
# copysign is left out: it exposes the sign bit of a NaN (for example from
# fmod(0, 0)), which IEEE 754 leaves open and which differs between C and Julia.
const TREE_UNARY = [x -> -x, x -> C.sqrt(x), x -> x^2, x -> 2 * x, x -> C.abs(x), x -> C.sign(x),
    x -> C.floor(x), x -> C.ceil(x), x -> 1 / x, x -> C.not(x)]
const TREE_BINARY = [(x, y) -> x + y, (x, y) -> x - y, (x, y) -> x * y, (x, y) -> x / y,
    (x, y) -> C.lt(x, y), (x, y) -> C.le(x, y), (x, y) -> C.eq(x, y), (x, y) -> C.ne(x, y),
    (x, y) -> C.and(x, y), (x, y) -> C.or(x, y), (x, y) -> C.fmod(x, y),
    (x, y) -> C.fmin(x, y), (x, y) -> C.fmax(x, y),
    (x, y) -> C.if_else(x, y, SX(0.0)), (x, y) -> C.remainder(x, y)]

function tree_function(steps, constant)
    x = SX.sym("x", 3)
    nodes = Any[x[1], x[2], x[3], SX(constant)]
    pick(i) = nodes[mod1(i, length(nodes))]
    for (kind, i, j) in steps
        n = length(TREE_UNARY)
        push!(nodes, kind <= n ? TREE_UNARY[kind](pick(i)) :
                     TREE_BINARY[mod1(kind - n, length(TREE_BINARY))](pick(i), pick(j)))
    end
    y = vertcat(nodes[end], nodes[max(end - 1, 1)])
    CasADiCodegen.Function("t", [x], [y, jacobian(y, x)])
end

const tree_step = @composed function tree_step_(kind = Data.Integers(1, length(TREE_UNARY) + length(TREE_BINARY)),
                                                i = Data.Integers(1, 30), j = Data.Integers(1, 30))
    (kind, i, j)
end

const TREES, TREES_SKIPPED = Ref(0), Ref(0)

@testset "property: exact trees and their Jacobians equal CasADi" begin
    @check max_examples = 150 function tree_equals_casadi(
            steps = Data.Vectors(tree_step; min_size = 1, max_size = 25),
            constant = small_float,
            x = Data.Vectors(test_float; min_size = 3, max_size = 3),
            chunk_size = Data.SampledFrom([3, 7, 1000]))
        f = tree_function(steps, constant)
        alg = Algorithm(f)
        x = x .+ 0.0     # the front end turns -0.0 into 0.0 when it builds a DM from a vector
        TREES[] += 1
        # CasADi can put an operation that is not rounded correctly into a Jacobian; such a tree is counted, not compared.
        all(ins -> ins.op in EXACT_OPS, alg.instructions) || (TREES_SKIPPED[] += 1; return true)
        isequal(casadi_eval(f, x), emitted_eval(alg, x; chunk_size))
    end
    @test TREES_SKIPPED[] <= TREES[] ÷ 10
end

@testset "sparse outputs and metadata" begin
    x = SX.sym("x", 3)
    rosenbrock = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2 + x[3]^2
    grad, hess = gradient(rosenbrock, x), hessian(rosenbrock, x)
    f = CasADiCodegen.Function("rosen", [x], [rosenbrock, grad, hess], ["x"], ["f", "g", "H"])
    alg = Algorithm(f)
    @test alg.names_in == ["x"] && alg.names_out == ["f", "g", "H"]
    @test alg.nnz_out == [1, 3, 5]                 # the Hessian has 5 structural nonzeros
    @test (C.colind(alg.sparsity_out[3]), C.row(alg.sparsity_out[3])) == ([0, 2, 4, 5], [0, 1, 0, 1, 2])
    xv = [-1.2, 1.0, 0.7]
    @test isequal(casadi_eval(f, xv), emitted_eval(alg, xv))
    source = sprint(io -> generate_julia(f, io; module_name = :Rosen))
    @test occursin("rosen!", source)
    @test occursin("const SPARSITY_OUT = ((1, 1, [1, 2], [1]), (3, 1, [1, 4], [1, 2, 3]), (3, 3, [1, 3, 5, 6], [1, 2, 1, 2, 3]))", source)
end

@testset "large function is split into chunks" begin
    n, chunk_size = 60, 100
    x = SX.sym("x", n)
    terms = [(x[i] - x[i+1])^2 / (1 + C.abs(x[i])) for i in 1:n-1]
    y = vertcat(terms...)
    f = CasADiCodegen.Function("big", [x], [y, jacobian(y, x)])
    alg = Algorithm(f)
    @test length(alg.instructions) > 3 * chunk_size
    xv = collect(range(-2.0, 3.0; length = n))
    expected = casadi_eval(f, xv)
    g = compiled(alg; chunk_size)
    @test isequal(expected, g([zeros(k) for k in alg.nnz_out], (xv,)))
    @test isequal(expected, emitted_eval(alg, xv))          # default chunk size
    mktempdir() do dir
        path = generate_julia(f, joinpath(dir, "big.jl"); chunk_size)
        mod = Module()
        Base.include(mod, path)
        res = Base.invokelatest((r, a) -> mod.Big.big_nonzeros!(r, a), [zeros(k) for k in alg.nnz_out], (xv,))
        @test isequal(expected, res)
        @test occursin("big_nonzeros!_chunk4", read(path, String))
    end
end

@testset "a written module imports only what it uses" begin
    x, y = SX.sym("x"), SX.sym("y")
    source(expr) = sprint(io -> generate_julia(CasADiCodegen.Function("s", [x, y], [expr]), io))
    @test occursin("import SpecialFunctions", source(C.erf(x)))
    @test occursin("erfinv(x) =", source(C.erfinv(x)))
    @test occursin("import NaNMath", source(sin(x)))
    @test occursin("fmin(x, y) =", source(C.fmin(x, y)))
    @test occursin("fmin(x, y) =", sprint(io -> generate_julia(CasADiCodegen.Function("s", [x, y], [C.fmin(x, y)]), io; nanmath = false)))
    plain = source(x * y + 2)
    @test count("import", plain) == 1 && occursin("import StaticArraysCore", plain) && !occursin("fmin", plain)
end

@testset "MX functions" begin
    X = MX.sym("X", 2, 2)
    v = MX.sym("v", 2)
    a = SX.sym("a", 2)
    h = CasADiCodegen.Function("h", [a], [vertcat(a[1] * a[2], a[1] - a[2])])
    f = CasADiCodegen.Function("m", [X, v], [C.mtimes(X, v) + h(v)])
    alg = CasADiCodegen.Codegen.algorithm(f)               # CasADi expands it to SX
    Xv, vv = [1.0, 2.0, 3.0, 4.0], [0.5, -1.5]
    @test isequal(casadi_eval(f, reshape(Xv, 2, 2), vv), emitted_eval(alg, Xv, vv))

    # CasADi cannot expand a linear solve, so the matrix emitter takes the function.
    g = CasADiCodegen.Function("lin", [X, v], [C.solve(X, v)])
    solved = build_function(g; nonzeros = true, expression = Val{false})([zeros(2)], (Xv, vv))
    @test solved[1] ≈ reshape(Xv, 2, 2) \ vv
end

@testset "free symbols are rejected" begin
    x, p = SX.sym("x"), SX.sym("p")
    f = CasADiCodegen.Function("free", [x], [x * p], Dict("allow_free" => true))
    err = try generate_julia(f, IOBuffer()); nothing catch e; e end
    @test err isa ArgumentError && occursin("free", err.msg)
end

# The norms of LinearAlgebra on symbols take the same Julia code on numbers as the oracle.
const NORM_PS = (2, 1, Inf, -Inf, 0, 3, 1//2, -0.5, -1.5)

@testset "norm and opnorm of symbols as on numbers, $T" for T in (SX, MX)
    M = T.sym("M", 2, 3)
    d = T.sym("d", 3)
    D = C.diag(d)                                           # structural zeros count as elements
    outputs = [[LinearAlgebra.norm(M, p) for p in NORM_PS]; [LinearAlgebra.norm(D, p) for p in NORM_PS];
               [LinearAlgebra.norm([M[1, 1], M[2, 2], M[1, 3]], p) for p in NORM_PS];
               LinearAlgebra.norm([M[:, 1], M[:, 2]], 1); LinearAlgebra.opnorm(M, 1); LinearAlgebra.opnorm(M, Inf);
               LinearAlgebra.opnorm(M[:, 1]); LinearAlgebra.opnorm(M[1, :])]
    f = CasADiCodegen.Function("norms", [M, d], outputs)
    # Values with a zero element; all zeros, where the scaled formula must not give NaN; an infinite element.
    for (Mv, dv) in (([1.0 -2.0 0.5; 3.0 0.0 -1.0], [2.0, -0.5, 4.0]), (zeros(2, 3), zeros(3)),
                     ([1.0 -Inf 0.5; 3.0 0.25 -1.0], [2.0, -0.5, 4.0]))
        Dv = LinearAlgebra.diagm(dv)
        expected = [[LinearAlgebra.norm(Mv, p) for p in NORM_PS]; [LinearAlgebra.norm(Dv, p) for p in NORM_PS];
                    [LinearAlgebra.norm([Mv[1, 1], Mv[2, 2], Mv[1, 3]], p) for p in NORM_PS];
                    LinearAlgebra.norm([Mv[:, 1], Mv[:, 2]], 1); LinearAlgebra.opnorm(Mv, 1); LinearAlgebra.opnorm(Mv, Inf);
                    LinearAlgebra.opnorm(Mv[:, 1:1]); LinearAlgebra.opnorm(Mv[1:1, :])]
        @test all(map((a, b) -> isequal(a, b) || a ≈ b, [only(Matrix(r)) for r in f(C.DM(Mv), C.DM(dv))], expected))
    end
    @test Float64(LinearAlgebra.norm(T(2, 0), 3)) == LinearAlgebra.norm(zeros(2, 0), 3)
    @test_throws ArgumentError LinearAlgebra.opnorm(M)
    @test_throws ArgumentError LinearAlgebra.opnorm(M, 3)
end

@testset "a concatenation of SX and MX throws ArgumentError" begin
    s, m = SX.sym("s", 2), MX.sym("m", 2)
    err = try [s; m]; nothing catch e; e end
    @test err isa ArgumentError && occursin("SX", err.msg) && occursin("MX", err.msg)
    @test_throws ArgumentError [m s]
    @test_throws ArgumentError vertcat([s, m])
    @test_throws ArgumentError [m; [s[1], s[2]]]
end
