# Matrix (MX) functions emitted without expansion, with CasADi as the oracle.
import ForwardDiff
import LinearAlgebra
const MXAlgorithm = CasADiCodegen.Codegen.MXAlgorithm

"Evaluate the emitted matrix code on one vector of nonzeros per input."
mx_emitted(f, nz; kwargs...) =
    build_function(f; nonzeros = true, expression = Val{false}, expand = false, kwargs...)(
        [fill(NaN, n) for n in nnz_out(f)], Tuple(nz))

mx_ops(f) = Set(ins.op for ins in MXAlgorithm(f).instructions)

mx_same(f; kwargs...) = isequal(mx_casadi(f, mx_inputs(f)), mx_emitted(f, mx_inputs(f); kwargs...))

function mx_close(f; kwargs...)
    expected, actual = mx_casadi(f, mx_inputs(f)), mx_emitted(f, mx_inputs(f); kwargs...)
    length(expected) == length(actual) &&
        all(length(e) == length(a) && all(close_enough.(e, a)) for (e, a) in zip(expected, actual))
end

mx_error(f; expand = false, kwargs...) = try
    build_function(f; nonzeros = true, expand, kwargs...)
    nothing
catch err
    err
end

# The operations themselves come from the table of the scalar code (test_casadi.jl); matrix code adds
# the loop over the elements and the scalar operand, which the next test set checks.
@testset "conditional and constant power on matrices" begin
    x = MX.sym("x", 3)
    f = CasADiCodegen.Function("c", [x], [C.if_else(x, x + 1, MX(0.0)), x^2.5])
    @test mx_close(f)
end

@testset "elementwise results are bit for bit those of CasADi" begin
    x, y, s = MX.sym("x", 2, 3), MX.sym("y", 2, 3), MX.sym("s")
    f = CasADiCodegen.Function("x", [x, y, s], [C.times(x, y) + s, s / x - y, C.fmin(x, y) * 2, C.abs(x - s)])
    @test mx_same(f)
end

const STRUCTURAL = [
    "slice" => (:OP_GETNONZEROS, (A, B, v) -> A[2:3, 1]),
    "slice with a step" => (:OP_GETNONZEROS, (A, B, v) -> C.reshape(A, 9, 1)[1:4:9] * 2),
    "scattered nonzeros" => (:OP_VERTCAT, (A, B, v) -> C.vertcat([A[1, 1], A[3, 2], A[2, 3]]) * 2),
    "vertical join" => (:OP_VERTCAT, (A, B, v) -> C.vertcat([v, 2 * v, A[1:3, 2]]) + 1),
    "horizontal join" => (:OP_HORZCAT, (A, B, v) -> C.horzcat([A, 2 * B, v]) + 1),
    "diagonal join" => (:OP_DIAGCAT, (A, B, v) -> 2 * C.diagcat([A, 2 * v])),
    "vertical split" => (:OP_VERTSPLIT, (A, B, v) -> C.vertsplit(2 * v, 1)[3] + C.vertsplit(2 * v, 1)[1]),
    "horizontal split" => (:OP_HORZSPLIT, (A, B, v) -> C.horzsplit(A + B, 1)[2] * 2),
    "transpose" => (:OP_TRANSPOSE, (A, B, v) -> 2 * C.transpose(A)),
    "reshape" => (:OP_RESHAPE, (A, B, v) -> 2 * C.reshape(A + B, 1, 9)),
    "projection to a sparse pattern" => (:OP_PROJECT, (A, B, v) -> 2 * C.project(A + B, C.Sparsity.lower(3))),
    "projection to a dense pattern" => (:OP_PROJECT, (A, B, v) -> C.cos(C.project(A, C.Sparsity.diag(3)))),
    "repetition" => (:OP_HORZREPMAT, (A, B, v) -> 2 * C.repmat(2 * v, 1, 3)),
]

@testset "structural operations" begin
    @testset "$label" for (label, (op, build)) in STRUCTURAL
        A, B, v = MX.sym("A", 3, 3), MX.sym("B", 3, 3), MX.sym("v", 3)
        f = CasADiCodegen.Function("s", [A, B, v], [build(A, B, v)])
        @test op in mx_ops(f)
        @test mx_same(f)
    end
end

@testset "assignment to nonzeros" begin
    A, v = MX.sym("A", 3, 3), MX.sym("v", 3)
    M = 2 * A
    M[1:3, 2] = v
    N = 2 * A
    N[2, 1:3] = N[2, 1:3] + C.transpose(v)
    f = CasADiCodegen.Function("set", [A, v], [M, N * 3])
    @test !isempty(intersect(mx_ops(f), (:OP_SETNONZEROS, :OP_ADDNONZEROS, :OP_SUBASSIGN)))
    @test mx_same(f)
end

@testset "one value used twice by one operation" begin
    u = MX.sym("u", 3)
    f = CasADiCodegen.Function("twice", [u], [C.vertcat([u, u]) * 2, C.horzcat([u, 2 * u, u]) + 1])
    @test mx_same(f)
end

@testset "sums into nonzeros, as reverse derivatives make them" begin
    x, seed = MX.sym("x", 4), MX.sym("seed", 6)
    f = CasADiCodegen.Function("adj", [x, seed], [C.jtimes(C.vertcat([x[2:3], C.sin(x)]), x, seed, true)])
    @test :OP_ADDNONZEROS in mx_ops(f)
    @test mx_close(f)
end

@testset "sparse inputs and outputs" begin
    L = MX.sym("L", C.Sparsity.lower(3))
    D = MX.sym("D", C.Sparsity.diag(3))
    f = CasADiCodegen.Function("sp", [L, D], [C.transpose(L) * 2, L + D, C.vertcat([L, D])])
    @test nnz_in(f) == [6, 3]
    @test mx_same(f)
end

@testset "constants" begin
    v = MX.sym("v", 4)
    table = C.DM([10.0, -20.5, 30.25, 1e-3])
    f = CasADiCodegen.Function("k", [v], [v + table, v * 3.5, MX(table), MX(C.DM([7.0, 7.0]))])
    @test mx_same(f)
    ex = build_function(f; nonzeros = true, expand = false)
    @test occursin("k!_constants", string(ex))       # a table for the values that differ
    @test !occursin("7.0, 7.0", string(ex))          # equal values need no table
end

function call_case()
    a = SX.sym("a", 3)
    h = CasADiCodegen.Function("h", [a], [vertcat(a[1] * a[2] + a[3], sin(a[3])), a[1] - a[2]])
    p, q = MX.sym("p", 3), MX.sym("q", 3)
    m = CasADiCodegen.Function("m", [p, q], [h(C.times(p, q))[1] + q[1:2], C.transpose(C.reshape(p, 1, 3)) - h(q)[2]])
    v = MX.sym("v", 3)
    first_h, second_h = h(v), h(2 * v)
    inner = m(v, C.vertcat([second_h[1], first_h[2]]))
    CasADiCodegen.Function("top", [v], [inner[1] * 2, inner[2] + first_h[2], second_h[2]])
end

@testset "calls keep the function hierarchy" begin
    f = call_case()
    @test :OP_CALL in mx_ops(f)
    @test mx_close(f)
    ex = build_function(f; nonzeros = true, expand = false)
    definitions = [string(d.args[1].args[1]) for d in ex.args if d isa Expr && d.head === :function]
    @test definitions == ["top_h!", "top_m!", "top!"]     # each callee once, the entry last
    @test mx_close(f; chunk_size = 2)                       # callees and entry split into chunks
    @test isequal(mx_emitted(f, mx_inputs(f)), mx_emitted(f, mx_inputs(f); chunk_size = 2))
end

@testset "two callees with one name get two definitions" begin
    a = SX.sym("a")
    h1 = CasADiCodegen.Function("h", [a], [a * 2])
    h2 = CasADiCodegen.Function("h", [a], [a + 5])
    v = MX.sym("v")
    f = CasADiCodegen.Function("two", [v], [h1(v) - h2(v)])
    @test mx_same(f)
    @test occursin("two_h!", string(build_function(f; nonzeros = true, expand = false)))
    @test occursin("two_h_2!", string(build_function(f; nonzeros = true, expand = false)))
end

@testset "expand keyword" begin
    f = call_case()
    @test !occursin("top_h!", string(build_function(f; nonzeros = true)))                     # expanded to scalars
    @test occursin("top_h!", string(build_function(f; nonzeros = true, expand = false)))
    @test isapprox(build_function(f; nonzeros = true, expression = Val{false})([zeros(2), zeros(3), zeros(1)], (mx_inputs(f)[1],)),
                   mx_emitted(f, mx_inputs(f)); rtol = 1e-14)
    x = SX.sym("x", 2)
    s = CasADiCodegen.Function("s", [x], [x * 2])
    @test build_function(s; nonzeros = true, expand = false) == build_function(s; nonzeros = true)            # nothing to expand
end

@testset "every form of the matrix code gives the same numbers" begin
    f = call_case()
    x = mx_inputs(f)
    expected = mx_emitted(f, x)
    @testset "chunk_size = $chunk_size" for chunk_size in (2,)
        mod = Module()
        Core.eval(mod, build_function(f; nonzeros = true, expand = false, name = :g!, chunk_size))
        @test isequal(expected, Base.invokelatest((r, a) -> mod.g!(r, a), [zeros(n) for n in nnz_out(f)], Tuple(x)))
        src = Module()
        Base.include_string(src, sprint(io -> generate_julia(f, io; module_name = :Gen, name = :g!,
                                                             expand = false, chunk_size)))
        @test isequal(expected, Base.invokelatest((r, a) -> src.Gen.g_nonzeros!(r, a), [zeros(n) for n in nnz_out(f)], Tuple(x)))
        @test Base.invokelatest(() -> src.Gen.NNZ_OUT) == Tuple(nnz_out(f))
    end
end

@testset "matrix source file runs without CasADi" begin
    v = MX.sym("v", 3)
    a = SX.sym("a", 3)
    h = CasADiCodegen.Function("h", [a], [vertcat(a[1] * a[2], sin(a[3]))])
    f = CasADiCodegen.Function("file", [v], [C.vertcat([h(v), v]) + C.DM([1.0, 2.0, 3.0, 4.0, 5.0])])
    x = mx_inputs(f)
    mktempdir() do dir
        path = joinpath(dir, "gen.jl")
        @test generate_julia(f, path; module_name = :Gen, expand = false) == path
        script = """
            include($(repr(path)))
            res = Gen.file_nonzeros!([zeros(5)], ($(repr(x[1])),))
            isdefined(Main, :CasADiCodegen) && error("CasADi was loaded")
            Gen.NNZ_IN == (3,) && Gen.NNZ_OUT == (5,) || error("bad metadata")
            print(join(res[1], " "))
            """
        out = read(`$(Base.julia_cmd()) --startup-file=no --project=$(dirname(Base.active_project())) -e $script`, String)
        @test parse.(Float64, split(out)) == mx_emitted(f, x)[1]
    end
end

@testset "wrong argument shape in matrix code" begin
    v = MX.sym("v", 3)
    g = build_function(CasADiCodegen.Function("shape", [v], [C.transpose(v) * 2]); nonzeros = true, expression = Val{false}, expand = false)
    @test_throws ArgumentError g([zeros(3)], ([1.0, 2.0],))
    @test_throws ArgumentError g([zeros(2)], ([1.0, 2.0, 3.0],))
end

@testset "what the matrix emitter does not support is named" begin
    X, v = MX.sym("X", 2, 2), MX.sym("v", 2)
    err = mx_error(CasADiCodegen.Function("watched", [X, v], [C.norm_2(X) * v]))
    @test err isa ArgumentError && occursin("watched", err.msg) && occursin("OP_NORM2", err.msg)
    p = MX.sym("p")
    err = mx_error(CasADiCodegen.Function("free", [v], [v * p], Dict("allow_free" => true)))
    @test err isa ArgumentError && occursin("free", err.msg)
    inner = CasADiCodegen.Function("inner", [X, v], [C.norm_2(X) * v])
    err = mx_error(CasADiCodegen.Function("outer", [X, v], [inner(X, 2 * v)]); expand = true)
    @test err isa ArgumentError && occursin("inner", err.msg) && occursin("OP_NORM2", err.msg)
end

# Values near the unit matrix keep the linear systems well conditioned.
const SHIFT = C.DM([3.0 0 0; 0 3 0; 0 0 3])
const LINEAR = [
    "dense product" => (:OP_MTIMES, (A, B, L, u, v, s) -> C.mtimes(A, B)),
    "product with a vector" => (:OP_MTIMES, (A, B, L, u, v, s) -> C.mtimes(A, v) + u),
    "sparse product" => (:OP_MTIMES, (A, B, L, u, v, s) -> C.mtimes(L, C.transpose(L))),
    "sparse times dense" => (:OP_MTIMES, (A, B, L, u, v, s) -> C.mtimes(C.mtimes(L, A), L)),
    "product into a sparse pattern" => (:OP_MTIMES, (A, B, L, u, v, s) -> C.project(C.mtimes(A, A), C.Sparsity.lower(3)) * 2),
    "sum of each column" => (:OP_MTIMES, (A, B, L, u, v, s) -> C.sum1(A) * 2),
    "inner product" => (:OP_DOT, (A, B, L, u, v, s) -> C.dot(u, v) * s),
    "sum of squares" => (:OP_DOT, (A, B, L, u, v, s) -> C.sumsqr(L)),
    "bilinear form" => (:OP_BILIN, (A, B, L, u, v, s) -> C.bilin(A, u, v)),
    "sparse bilinear form" => (:OP_BILIN, (A, B, L, u, v, s) -> C.bilin(L, u, v)),
    "rank one update" => (:OP_RANK1, (A, B, L, u, v, s) -> C.rank1(A, s, u, v)),
    "sparse rank one update" => (:OP_RANK1, (A, B, L, u, v, s) -> C.rank1(L, s, u, v) * 2),
    "norm of Frobenius" => (:OP_NORMF, (A, B, L, u, v, s) -> C.norm_fro(A) + C.norm_2(v)),
    "norm 1" => (:OP_NORM1, (A, B, L, u, v, s) -> C.norm_1(v)),
    "norm infinity" => (:OP_NORMINF, (A, B, L, u, v, s) -> C.norm_inf(v - 1)),
    "smallest element" => (:OP_MMIN, (A, B, L, u, v, s) -> C.mmin(v) + C.mmin(L + 5)),
    "largest element" => (:OP_MMAX, (A, B, L, u, v, s) -> C.mmax(A) + C.mmax(L - 5)),
    "Kronecker product" => (:OP_KRON, (A, B, L, u, v, s) -> C.kron(A, B)),
    "sparse Kronecker product" => (:OP_KRON, (A, B, L, u, v, s) -> C.kron(L, B) * 2),
    "Kronecker product with a sparse second factor" => (:OP_KRON, (A, B, L, u, v, s) -> C.kron(v, L) + 1),
    "logarithm of the sum of the exponentials" => (:OP_LOGSUMEXP, (A, B, L, u, v, s) -> C.logsumexp(v) * s),
    "logarithm of the sum of the exponentials, largest element first" => (:OP_LOGSUMEXP, (A, B, L, u, v, s) -> C.logsumexp(-v)),
    "determinant" => (:OP_DETERMINANT, (A, B, L, u, v, s) -> C.det(A + SHIFT)),
    "inverse through a linear solve" => (:OP_SOLVE, (A, B, L, u, v, s) -> C.inv(A + SHIFT, "qr")),
    "linear solve" => (:OP_SOLVE, (A, B, L, u, v, s) -> C.solve(A + SHIFT, v)),
    "linear solve with a sparse matrix" => (:OP_SOLVE, (A, B, L, u, v, s) -> C.solve(L + C.DM(C.Sparsity.diag(3), C.DM([3.0, 3.0, 3.0])), v) * 2),
    "linear solve with the transpose" => (:OP_SOLVE, (A, B, L, u, v, s) -> C.solve(C.transpose(A + SHIFT), v)),
    "linear solve with many right sides" => (:OP_SOLVE, (A, B, L, u, v, s) -> C.solve(A + SHIFT, B)),
    # CasADi makes the right side of a solve dense, also when the user gives a sparse one.
    "linear solve with a sparse right side" => (:OP_SOLVE, (A, B, L, u, v, s) -> C.solve(A + SHIFT, L)),
]

linear_function(build) = (A = MX.sym("A", 3, 3); B = MX.sym("B", 3, 2); L = MX.sym("L", C.Sparsity.lower(3));
    u = MX.sym("u", 3); v = MX.sym("v", 3); s = MX.sym("s");
    CasADiCodegen.Function("la", [A, B, L, u, v, s], [build(A, B, L, u, v, s)]))

function mx_near(f; rtol = 1e-12, kwargs...)
    expected, actual = mx_casadi(f, mx_inputs(f)), mx_emitted(f, mx_inputs(f); kwargs...)
    length(expected) == length(actual) && all(isapprox.(expected, actual; rtol))
end

@testset "linear algebra" begin
    @testset "$label" for (label, (op, build)) in LINEAR
        f = linear_function(build)
        @test getfield(C, op) in [C.instruction_id(f, k) for k in 0:C.n_instructions(f)-1]
        @test mx_near(f)
        # The split does not depend on the operation, so one operation shows it.
        label == "linear solve" && @test mx_near(f; chunk_size = 1)
    end
end

# CasADi has no numerical evaluation for the inverse node, so Julia is the oracle.
@testset "matrix inverse node and elementwise reciprocal" begin
    A = MX.sym("A", 2, 2)
    f = CasADiCodegen.Function("both", [A], [C.inv_node(A), 1 / A])
    @test :OP_INVERSE in mx_ops(f)
    x = [[4.0, 1.0, 2.0, 3.0]]
    @test mx_emitted(f, x)[1] ≈ vec(inv([4.0 2.0; 1.0 3.0]))
    @test mx_emitted(f, x)[2] == 1 ./ x[1]
end

@testset "linear algebra in a written module" begin
    f = linear_function((A, B, L, u, v, s) -> C.solve(A + SHIFT, C.mtimes(L, v)) * C.det(A + SHIFT))
    text = sprint(io -> generate_julia(f, io; module_name = :Gen, name = :g!, expand = false))
    @test occursin("import LinearAlgebra", text)
    src = Module()
    Base.include_string(src, text)
    x = mx_inputs(f)
    @test isequal(mx_emitted(f, x), Base.invokelatest((r, a) -> src.Gen.g_nonzeros!(r, a), [zeros(3)], Tuple(x)))
    @test mx_near(f)
    plain = sprint(io -> generate_julia(linear_function((A, B, L, u, v, s) -> C.mtimes(A, v)), io; expand = false))
    @test !occursin("import LinearAlgebra", plain)      # products are plain loops
    big = MX.sym("S", C.Sparsity.lower(8))
    wide = CasADiCodegen.Function("wide", [big], [C.mtimes(big, C.transpose(big))])
    @test occursin("wide_nonzeros!_indices", sprint(io -> generate_julia(wide, io; expand = false)))
    @test mx_near(wide) && mx_near(wide; chunk_size = 1)
    @test isequal(mx_emitted(wide, mx_inputs(wide)),
                  (m = Module(); Core.eval(m, build_function(wide; nonzeros = true, expand = false));
                   Base.invokelatest((r, a) -> m.wide!(r, a), [zeros(64)], Tuple(mx_inputs(wide)))))
end

@testset "linear algebra accepts any real number type" begin
    A, v = MX.sym("A", 3, 3), MX.sym("v", 3)
    g = build_function(CasADiCodegen.Function("ad", [A, v], [C.solve(A, v), C.mtimes(A, v), C.det(A), C.inv_node(A)]); nonzeros = true,
                       expression = Val{false}, expand = false)
    M = [4.0 1.0 0.5; 0.2 3.0 0.1; 0.3 0.7 5.0]
    x = [0.3, -1.2, 0.8]
    run(k) = x -> g([similar(x, 3), similar(x, 3), similar(x, 1), similar(x, 9)], (vec(M), x))[k]
    @test ForwardDiff.jacobian(run(1), x) ≈ inv(M)
    @test ForwardDiff.jacobian(run(2), x) ≈ M
    dM = ForwardDiff.jacobian(m -> g([similar(m, 3), similar(m, 3), similar(m, 1), similar(m, 9)], (m, x))[3], vec(M))
    @test vec(dM) ≈ vec(LinearAlgebra.det(M) * inv(M)')
end

# ---------------------------------------------------------------------------
# Property: random matrix graphs over exact operations, with calls to a scalar
# and to a matrix function, equal CasADi bit for bit for any chunk size.
# ---------------------------------------------------------------------------
const GRAPH_H = let a = SX.sym("a", 3)
    CasADiCodegen.Function("h", [a], [vertcat(a[2] * a[3], a[1] - a[2], a[3]), a[1] * a[1]])
end
const GRAPH_M = let p = MX.sym("p", 3), q = MX.sym("q", 3)
    CasADiCodegen.Function("m", [p, q], [C.times(C.vertcat([p[2:3], q[1]]), q) + GRAPH_H(p)[1]])
end
const GRAPH_STEPS = [
    (x, y) -> x + y, (x, y) -> x - y, (x, y) -> C.times(x, y), (x, y) -> C.fmin(x, y), (x, y) -> C.fmax(x, y),
    (x, y) -> -x, (x, y) -> 2 * x, (x, y) -> C.abs(x), (x, y) -> x[1] * y, (x, y) -> x - y[3],
    (x, y) -> C.vertcat([x[2:3], y[1]]), (x, y) -> C.vertcat([y[3], x[1:2]]),
    (x, y) -> C.transpose(C.reshape(C.vertcat([x, y]), 2, 3))[1:3, 2],
    (x, y) -> C.horzsplit(C.horzcat([x, y]), 1)[2] + C.DM([1.0, -2.0, 0.5]),
    (x, y) -> GRAPH_H(x)[1], (x, y) -> GRAPH_H(y)[2] * x, (x, y) -> GRAPH_M(x, y),
    (x, y) -> C.mtimes(C.mtimes(x, C.transpose(y)), x), (x, y) -> C.dot(x, y) * y,
    (x, y) -> C.norm_1(x) - y, (x, y) -> C.norm_inf(y) * x, (x, y) -> C.mmin(x) + C.mmax(y) * y,
    (x, y) -> C.rank1(C.mtimes(y, C.transpose(x)), x[2], x, y)[1:3, 2], (x, y) -> C.bilin(C.mtimes(x, C.transpose(x)), y, x) * x,
]

function graph_function(steps)
    u, v = MX.sym("u", 3), MX.sym("v", 3)
    nodes = Any[u, v]
    for (kind, i, j) in steps
        # Loop invariant: each node is a vector of 3 numbers.
        node = GRAPH_STEPS[kind](nodes[mod1(i, length(nodes))], nodes[mod1(j, length(nodes))])
        # CasADi can simplify a step to a scalar, as (v - u[3]) - v to -u[3].
        C.numel(node) == 1 && (node = C.repmat(node, 3, 1))
        @assert size(node) == (3, 1)
        push!(nodes, node)
    end
    CasADiCodegen.Function("g", [u, v], [C.vertcat([nodes[end], nodes[max(end - 1, 1)]]), nodes[max(end - 2, 1)]])
end

const graph_step = @composed function graph_step_(kind = Data.Integers(1, length(GRAPH_STEPS)),
                                                  i = Data.Integers(1, 30), j = Data.Integers(1, 30))
    (kind, i, j)
end

@testset "property: random matrix graphs equal CasADi" begin
    # A graph in which CasADi simplifies a step to a scalar.
    simplified = graph_function([(10, 2, 1), (2, 9, 29), (10, 1, 20)])
    @test isequal(mx_casadi(simplified, [[0.3, -1.2, 0.8], [0.5, 2.0, -0.7]]), mx_emitted(simplified, [[0.3, -1.2, 0.8], [0.5, 2.0, -0.7]]; chunk_size = 1))
    @check max_examples = 80 function graph_equals_casadi(
            steps = Data.Vectors(graph_step; min_size = 1, max_size = 15),
            u = Data.Vectors(test_float; min_size = 3, max_size = 3),
            v = Data.Vectors(test_float; min_size = 3, max_size = 3),
            chunk_size = Data.SampledFrom([1, 4, 1000]))
        f = graph_function(steps)
        x = [u .+ 0.0, v .+ 0.0]
        isequal(mx_casadi(f, x), mx_emitted(f, x; chunk_size))
    end
end

@testset "map without expansion" begin
    x, p = SX.sym("x", 2), SX.sym("p")
    step = CasADiCodegen.Function("step", [x, p], [vertcat(x[2] * p, sin(x[1]) + p), x[1] * x[2]])
    X, P = MX.sym("X", 2, 5), MX.sym("P", 1, 5)
    both = C.map(step, 5)(X, P)
    y, z = both[1], both[2]
    f = CasADiCodegen.Function("f", [X, P], [y, z, jacobian(y, X), jacobian(C.sum2(z), P)])
    @test mx_same(f)
    @test mx_same(f; chunk_size = 3)
    # The body is one Julia function, which a loop over the columns calls (see "a map is a loop" in test_nodes.jl).
    source = string(build_function(f; nonzeros = true, expand = false))
    @test count("function f_step!(w)", source) == 1
    # A map inside a map, and a map of a matrix function.
    row = MX.sym("row", 2, 3)
    inner = CasADiCodegen.Function("inner", [row], [C.sum2(C.map(CasADiCodegen.Function("sq", [x], [x[1]^2 + x[2]]), 3)(row))])
    cube = MX.sym("cube", 2, 12)
    @test mx_same(CasADiCodegen.Function("g", [cube], [C.map(inner, 4)(cube)]))
end

# A copy that moves blocks with a fixed stride, as a transpose does, is one loop over the
# blocks; the size of its code does not depend on the number of blocks.
@testset "a transpose is a loop over blocks" begin
    transposed(n) = let X = MX.sym("X", 2, n), v = MX.sym("v", n)
        CasADiCodegen.Function("t", [X, v], [C.transpose(X), C.vertcat([C.transpose(v[1:n-1]), C.transpose(v[2:n])]) + X[:, 1:n-1]])
    end
    code_length(f) = length(string(build_function(f; nonzeros = true, expression = Val{true}, expand = false)))
    @test mx_same(transposed(5))
    @test code_length(transposed(300)) <= 1.1 * code_length(transposed(5))     # longer numbers only
end

@testset "a split whose first part has no nonzeros" begin
    # The parts of a sparse column are operands of a join; the first part is structurally zero.
    a = MX.sym("a", C.Sparsity.triplet(4, 1, [2, 3], [0, 0]))
    parts = C.vertsplit(2 * a, [0, 2, 3, 4])
    f = CasADiCodegen.Function("f", [a], [vertcat([parts[3], parts[2], parts[1]])])
    @test :OP_VERTSPLIT in mx_ops(f) && mx_same(f)
    # A constant column with one nonzero, between two parts that have none.
    c = MX(C.DM(C.Sparsity.triplet(4, 1, [2], [0]), C.DM([1.5])))
    s = MX.sym("s")
    g = CasADiCodegen.Function("g", [s], [horzcat([MX(1, 1), C.vertsplit(c, [0, 2, 3, 4])[2]]) * s])
    @test :OP_VERTSPLIT in mx_ops(g) && mx_same(g)
end

@testset "branches that run only when they are selected" begin
    a, c = MX.sym("a", 2), MX.sym("c")
    y = C.if_else(c > 0, sin(a) / a[1], 2 * a, true)
    f = CasADiCodegen.Function("f", [c, a], [y, jacobian(y, a)])
    several = C.conditional(c, [sin(a), cos(a), a], 3 * a, true)
    g = CasADiCodegen.Function("g", [c, a], [several, jacobian(several, a)])
    av = [0.3, 1.7]
    for selector in (1.0, 0.0, -2.0, 0.5, 2.0, 2.9, 3.0, -0.5, -1.0, NaN)
        @test isequal(mx_emitted(f, [[selector], av]), mx_casadi(f, [[selector], av]))
        @test isequal(mx_emitted(g, [[selector], av]), mx_casadi(g, [[selector], av]))
    end
    @test isequal(mx_emitted(g, [[1.0], av]; chunk_size = 2), mx_casadi(g, [[1.0], av]))
    source = string(build_function(g; nonzeros = true, expand = false))
    @test occursin("if trunc(selector) == 0", source) && occursin("elseif trunc(selector) == 2", source)
    # The branch that is not selected does not run: it would divide by zero here and give NaN in a sum.
    @test mx_emitted(f, [[0.0], [0.0, 1.0]])[1] == [0.0, 2.0]
    # A branch with another pattern than the switch.
    sparse_branch = C.if_else(c > 0, vertcat(a[1], MX(1, 1)), a, true)
    @test mx_same(CasADiCodegen.Function("s", [c, a], [sparse_branch]))
end

# Indices that are values of the graph: the place of a number in a grid, and
# nonzeros that are read or written at such places. CasADi builds them for
# an interpolant with `inline`, and for the derivatives of one.
const INDEXED = [
    "place in a grid" => (:OP_LOW, (v, x, i, j) -> C.low(v, i) + C.low(v, j[1])),
    "nonzeros at indices" => (:OP_GETNONZEROS_PARAM, (v, x, i, j) -> C.get_nz(v, false, i)),
    "nonzeros at indices plus a slice" => (:OP_GETNONZEROS_PARAM, (v, x, i, j) -> C.get_nz(v, false, i, C.Slice(1, 5, 2))),
    "nonzeros at a slice plus indices" => (:OP_GETNONZEROS_PARAM, (v, x, i, j) -> C.get_nz(v, false, C.Slice(1, 5, 2), i)),
    "nonzeros at sums of indices" => (:OP_GETNONZEROS_PARAM, (v, x, i, j) -> C.get_nz(v, false, i, j)),
    "assignment at indices" => (:OP_SETNONZEROS_PARAM, (v, x, i, j) -> (y = 2 * v; C.set_nz(y, x[1:2], false, i); y)),
    "sum at indices" => (:OP_ADDNONZEROS_PARAM, (v, x, i, j) -> C.jtimes(C.get_nz(v, false, i), v, x[1:2], true)),
    "sum at indices plus a slice" => (:OP_ADDNONZEROS_PARAM, (v, x, i, j) -> C.jtimes(C.get_nz(v, false, i, C.Slice(1, 5, 2)), v, reshape(x, 2, 2), true)),
    "sum at sums of indices" => (:OP_ADDNONZEROS_PARAM, (v, x, i, j) -> C.jtimes(C.get_nz(v, false, i, j), v, reshape(x, 2, 2), true)),
]

indexed_function(build) = (v = MX.sym("v", 6); x = MX.sym("x", 4); i = MX.sym("i", 2); j = MX.sym("j", 2);
    CasADiCodegen.Function("ix", [v, x, i, j], [build(v, x, i, j)]))

# A rising grid, and indices in the range, with a fraction, equal, and out of the range.
const INDEXED_INPUTS = [[[-1.0, 0.0, 0.5, 2.0, 3.0, 6.0], [1.5, -2.5, 3.5, 0.25], i, j]
                        for (i, j) in (([0.0, 1.0], [0.0, 3.0]), ([4.0, 2.7], [1.0, 0.0]), ([1.0, 1.0], [2.0, 2.0]),
                                       ([-1.0, 7.0], [0.0, 1.0]), ([0.5, 9.0], [-3.0, 6.0]))]

@testset "indices that are values" begin
    @testset "$label" for (label, (op, build)) in INDEXED
        f = indexed_function(build)
        @test op in mx_ops(f)
        g = build_function(f; nonzeros = true, expression = Val{false}, expand = false)
        @test all(isequal(mx_casadi(f, x), g([fill(NaN, n) for n in nnz_out(f)], Tuple(x))) for x in INDEXED_INPUTS)
    end
end

# CasADi 3.8.1 evaluates a write at a fixed slice plus indices wrongly, with values from other
# memory, so the oracle is the same write with each index written out: slice[s] + i[k].
@testset "a sum at a slice plus indices" begin
    places(i) = vertcat([1 + i[1], 3 + i[1], 1 + i[2], 3 + i[2]])
    f = indexed_function((v, x, i, j) -> C.jtimes(C.get_nz(v, false, C.Slice(1, 5, 2), i), v, reshape(x, 2, 2), true))
    reference = indexed_function((v, x, i, j) -> C.jtimes(C.get_nz(v, false, places(i)), v, x, true))
    @test :OP_ADDNONZEROS_PARAM in mx_ops(f)
    g = build_function(f; nonzeros = true, expression = Val{false}, expand = false)
    @test all(isequal(mx_casadi(reference, x), g([fill(NaN, n) for n in nnz_out(f)], Tuple(x))) for x in INDEXED_INPUTS)
    # The same node in a larger graph: the gradient of a sum of the nonzeros that are read.
    v, i = MX.sym("v", 6), MX.sym("i", 2)
    cost(read) = CasADiCodegen.Function("cost", [v, i], [gradient(C.dot(read, read), v)])
    @test :OP_ADDNONZEROS_PARAM in mx_ops(cost(C.get_nz(v, false, C.Slice(1, 5, 2), i)))
    h = build_function(cost(C.get_nz(v, false, C.Slice(1, 5, 2), i)); nonzeros = true, expression = Val{false}, expand = false)
    @test all(isequal(mx_casadi(cost(C.get_nz(v, false, places(i))), x[[1, 3]]), h([fill(NaN, 6)], Tuple(x[[1, 3]]))) for x in INDEXED_INPUTS)
end

@testset "linear interpolant with inline" begin
    grid, values = [0.0, 1.0, 2.5, 3.0, 4.0], [0.0, 1.0, 4.0, 9.0, 16.0]
    table = C.interpolant("table", "linear", [grid], values, Dict("inline" => true))
    x = MX.sym("x")
    f = CasADiCodegen.Function("line", [x], [table(x), C.jacobian(table(x), x)])
    out, inplace = build_function(f; expression = Val{false})       # CasADi cannot expand it, so the code is matrix code
    @test all(out([t])[k] ≈ dense_values(f(C.DM(t))[k]) for t in (-1.0, 0.0, 0.3, 1.0, 2.7, 3.99, 4.0, 7.0), k in 1:2)
    @test ForwardDiff.derivative(t -> out([t])[1][1], 2.7) ≈ 10.0
    # Two dimensions, with the values as an argument, and the derivative by the values.
    plane = C.interpolant("plane", "linear", [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], 2, Dict("inline" => true))
    p, c = MX.sym("p", 2), MX.sym("c", 18)
    g = CasADiCodegen.Function("plane_all", [p, c], [plane(p, c), C.jacobian(plane(p, c), c), C.jacobian(plane(p, c), p)])
    emitted = build_function(g; expression = Val{false})[1]
    coefficients = collect(1.0:18.0) .^ 1.5
    for point in ([0.5, 1.5], [1.0, 1.0], [2.0, 0.0], [-0.5, 2.5])
        expected, actual = g(C.DM(point), C.DM(coefficients)), emitted(point, coefficients)
        @test all(dense_values(expected[k]) ≈ vec(actual[k]) for k in 1:3)
    end
    text = sprint(io -> generate_julia(f, io; module_name = :Table, name = :line!))
    src = Module()
    Base.include_string(src, text)
    @test Base.invokelatest(t -> src.Table.line([t])[1], 2.7) ≈ out([2.7])[1]
end

@testset "monitor prints and passes its value" begin
    v = MX.sym("v", 3)
    f = CasADiCodegen.Function("watched", [v], [2 * C.monitor(v, "half of the result")])
    @test :OP_MONITOR in mx_ops(f)
    g = build_function(f; nonzeros = true, expression = Val{false}, expand = false)
    result, printed = zeros(3), ""
    mktemp() do path, io
        redirect_stdout(() -> g([result], ([1.0, 2.0, 3.0],)), io)
        close(io)
        printed = read(path, String)
    end
    @test result == [2.0, 4.0, 6.0]
    @test printed == "half of the result:\n[1.0, 2.0, 3.0]\n"
end

# Generated code never calls CasADi: a solver object of CasADi with no solver of
# the SciML ecosystem, and a class for which the package has no Julia code, are refused.
@testset "functions of CasADi without Julia code are refused" begin
    x, p = MX.sym("x", 2), MX.sym("p")
    decay = C.integrator("decay", "cvodes", Dict("x" => x, "p" => p, "ode" => -p * x), 0.0, 1.0)
    f = CasADiCodegen.Function("f", [x, p], [decay(x, MX(0, 1), p, MX(0, 1), MX(), MX(), MX())[1]])
    # Generated code never calls CasADi: a solver object of CasADi needs a solver of the SciML ecosystem.
    @test_throws "integrator \"cvodes\" of CasADi" build_function(f; nonzeros = true, expression = Val{false})
    @test_throws "give one with the option solvers = CasADiCodegen.Solvers(ode = ...)" generate_julia(f, IOBuffer())
    # A class for which the package has no Julia code.
    z = MX.sym("z")
    root = C.rootfinder("root", "nlpsol", Dict("x" => z, "p" => p, "g" => z^2 - p), Dict("nlpsol" => "ipopt"))
    f = CasADiCodegen.Function("f", [p], [root(p, p)])
    @test_throws "has the class ImplicitToNlp of CasADi, for which this package has no Julia code" build_function(f; expression = Val{false})
end

# A large sparse system is solved with the sparse QR factorization of CasADi,
# which keeps the work proportional to the nonzeros of the factors.
@testset "linear solve with a sparse factorization" begin
    n = 16
    # Three diagonals and the two far corners.
    entries = [(i, j) for i in 0:n-1 for j in 0:n-1 if abs(i - j) <= 1 || abs(i - j) == n - 1]
    band = C.Sparsity.triplet(n, n, first.(entries), last.(entries))
    A, b, B = MX.sym("A", band), MX.sym("b", n), MX.sym("B", n, 2)
    rows, columns = CasADiCodegen.Codegen.positions(band)
    values = [[rows[k] == columns[k] ? 4.0 + 0.1 * k : 0.3 + 0.05 * k for k in 1:C.nnz(band)], [0.5 * k - 2 for k in 1:n], [0.25 * k - 1 for k in 1:2n]]
    f = CasADiCodegen.Function("sparse_solve", [A, b, B], [C.solve(A, b), C.solve(C.transpose(A), b), C.solve(A, B)])
    expected = mx_casadi(f, values)
    g = build_function(f; nonzeros = true, expression = Val{false}, expand = false)
    results = [zeros(n), zeros(n), zeros(2n)]
    @test all(isapprox(a, e; rtol = 1e-12) for (a, e) in zip(g(results, Tuple(values)), expected))
    arguments = Tuple(values)
    @test @allocated(g(results, arguments)) == 0
    # Dual numbers: the derivative of the solution by the right side is the inverse.
    dense = zeros(n, n)
    foreach(k -> dense[rows[k], columns[k]] = values[1][k], 1:C.nnz(band))
    J = ForwardDiff.jacobian(r -> g([similar(r, n), similar(r, n), similar(r, 2n)], (values[1], r, values[3]))[1], values[2])
    @test J ≈ inv(dense)
    text = sprint(io -> generate_julia(f, io; module_name = :Banded, name = :solve!, expand = false))
    @test occursin("qr_factor!", text) && !occursin("import LinearAlgebra", text)
    src = Module()
    Base.include_string(src, text)
    @test Base.invokelatest((r, a) -> src.Banded.solve_nonzeros!(r, a), [zeros(n), zeros(n), zeros(2n)], Tuple(values)) == g(results, Tuple(values))
end
