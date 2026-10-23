# The modelling interface that `using CasADiCodegen` gives, with no module alias.
import LinearAlgebra
import SpecialFunctions
using SparseArrays: SparseMatrixCSC, nnz
using StaticArrays: SVector, SMatrix

module ApiUser
using CasADiCodegen
model() = (x = SX.sym("x", 2); (x, [sin(x[1]) * x[2]; x[1]^2]))
function build()
    x, y = model()
    CasadiFunction("f", [x], [y, jacobian(y, x)])
end
end

"Evaluate a scalar expression in the symbols `args` at the numbers `values`."
function evaluate(expr, args, values)
    g = build_function(CasADiCodegen.Function("t", collect(args), [expr]); nonzeros = true, expression = Val{false})
    g([zeros(1)], Tuple([v] for v in values))[1][1]
end

@testset "using CasADiCodegen is enough to write a model" begin
    f = ApiUser.build()
    g = build_function(f; nonzeros = true, expression = Val{false})
    res = g([zeros(2), zeros(3)], ([0.3, 1.7],))
    @test res[1] == [sin(0.3) * 1.7, 0.3^2]
    @test res[2] == [cos(0.3) * 1.7, 2 * 0.3, sin(0.3)]
end

@testset "Base functions work on symbols" begin
    x, y = SX.sym("x"), SX.sym("y")
    unary = [asinh, acosh, atanh, log1p, expm1, log10, log2, exp2, abs2, inv,
             SpecialFunctions.erf, SpecialFunctions.erfinv]
    @testset "$f" for f in unary
        a = f in (acosh,) ? 1.7 : 0.4
        @test evaluate(f(x), (x,), (a,)) ≈ f(a) rtol = 1e-7
    end
    # `inv` of a matrix is the matrix inverse, as in Julia, and not the reciprocal of each element.
    M = [4.0 1.0; 0.5 3.0]
    @testset "inverse of a matrix, $T" for T in (SX, MX)
        A = T.sym("A", 2, 2)
        for inverse in (T === MX ? (inv(A), inv(A, "qr")) : (inv(A),))
            g = build_function(CasADiCodegen.Function("t", [A], [inverse]); expression = Val{false})[1]
            @test g(M) ≈ inv(M) rtol = 1e-14
        end
        @test_throws DimensionMismatch inv(T.sym("B", 2, 3))
        T === MX && @test_throws DimensionMismatch inv(T.sym("B", 2, 3), "qr")
    end
    binary = [min, max, atan, hypot, rem, copysign]
    @testset "$f" for f in binary, (a, b) in ((0.4, -1.7),)
        @test evaluate(f(x, y), (x, y), (a, b)) ≈ f(a, b) rtol = 1e-14
        @test evaluate(f(x, b), (x,), (a,)) ≈ f(a, b) rtol = 1e-14
        @test evaluate(f(a, y), (y,), (b,)) ≈ f(a, b) rtol = 1e-14
    end

    # Logical operations give 1 or 0, as everywhere in CasADi.
    truth(e, a, b) = evaluate(e, (x, y), (a, b)) == 1
    for (a, b) in ((1.0, 2.0), (2.0, 1.0), (1.0, 1.0))
        @test truth(x != y, a, b) == (a != b)
        @test truth((x < y) & (x != y), a, b) == (a < b)
        @test truth((x < y) | (x > y), a, b) == (a != b)
        @test truth(!(x < y), a, b) == !(a < b)
        @test evaluate(ifelse(x < y, x, y), (x, y), (a, b)) == min(a, b)
        @test evaluate(ifelse(x < y, 10.0, y), (x, y), (a, b)) == (a < b ? 10.0 : b)
    end
end

@testset "matrix (MX) symbols get the same functions" begin
    m = MX.sym("m")
    f = CasADiCodegen.Function("t", [m], [hypot(min(m, 2.0), log1p(m)) + ifelse(m != 1.0, asinh(m), m)])
    a = 0.4
    @test build_function(f; nonzeros = true, expression = Val{false})([zeros(1)], ([a],))[1][1] ≈ hypot(min(a, 2.0), log1p(a)) + asinh(a)
end

@testset "LinearAlgebra functions" begin
    u, v = SX.sym("u", 3), SX.sym("v", 3)
    a, b = [1.0, -2.0, 0.5], [0.25, 3.0, -1.5]
    f = CasADiCodegen.Function("t", [u, v], [LinearAlgebra.dot(u, v), LinearAlgebra.cross(u, v)])
    res = build_function(f; nonzeros = true, expression = Val{false})([zeros(1), zeros(3)], (a, b))
    @test res[1][1] ≈ LinearAlgebra.dot(a, b)
    @test res[2] ≈ LinearAlgebra.cross(a, b)
    # `norm` is the Frobenius norm, for a vector and for a matrix, as in LinearAlgebra.
    for T in (SX, MX)
        w, A = T.sym("w", 3), T.sym("A", 2, 2)
        n = CasADiCodegen.Function("n", [w, A], [LinearAlgebra.norm(w), LinearAlgebra.norm(A)])
        values = C.nonzeros.(n(C.DM(a), C.DM([1.0 2.0; 3.0 4.0])))
        @test values[1][1] ≈ LinearAlgebra.norm(a) && values[2][1] ≈ sqrt(30)
    end
end

"The value of the expression `e`, a CasADi matrix or a Julia vector of symbols, at the numbers `values` of `symbols`, as a matrix."
function numeric(e, symbols, values)
    f = CasADiCodegen.Function("t", collect(symbols), [e isa AbstractVector ? vertcat(e) : e])
    Matrix(f(map(C.DM, values)...))
end

# The oracle is the same Julia code on numbers.
@testset "Julia array code on symbols, $T" for T in (SX, MX)
    x, M = T.sym("x", 3), T.sym("M", 2, 3)
    xv, Mv = [0.5, -1.25, 2.0], [1.0 -2.0 0.5; 3.0 0.25 -1.0]
    value(e) = numeric(e, (x, M), (xv, Mv))
    v = [x[1], x[2], x[3]]
    # Reductions of a CasADi matrix: over all its elements, along one dimension, or by iteration over the elements.
    @test only(value(sum(x))) ≈ sum(xv) && value(sum(M; dims = 1)) ≈ sum(Mv; dims = 1) && value(sum(M; dims = 2)) ≈ sum(Mv; dims = 2)
    @test only(value(sum(abs2, M))) ≈ sum(abs2, Mv) && only(value(maximum(x))) == maximum(xv) && only(value(minimum(M))) == minimum(Mv)
    @test only(value(prod(x))) ≈ prod(xv)
    # Broadcasting mixes CasADi matrices, Julia arrays of symbols and numbers, and gives a CasADi matrix.
    @test vec(value(x .+ v .* 2 .- [1.0, 2.0, 3.0])) ≈ xv .+ xv .* 2 .- [1.0, 2.0, 3.0]
    # A column and a row broadcast to a matrix, which CasADi does not do on whole matrices.
    @test value(M .+ transpose(x)) ≈ Mv .+ transpose(xv) && value(x .* [1.0 2.0]) ≈ xv .* [1.0 2.0]
    # A function that is not elementwise on a whole matrix applies to each element.
    @test value(inv.(M)) ≈ inv.(Mv) && value((u -> u * u).(M)) ≈ Mv .* Mv
    # In-place broadcasting into a Julia array of symbols and into a CasADi matrix.
    a, an = zeros(T, 5), zeros(5)
    a[2:4] .+= 2 .* x
    an[2:4] .+= 2 .* xv
    @views a[1:3] .-= x ./ 4
    @views an[1:3] .-= xv ./ 4
    @test vec(value(a)) ≈ an
    y = T(2, 3)
    y .= M .* 2 .+ 1
    @test value(y) ≈ Mv .* 2 .+ 1
    # Arithmetic with Julia arrays gives a CasADi matrix.
    @test vec(value(M * v)) ≈ Mv * xv && value([1.0 2.0; 0.0 1.0] * M) ≈ [1.0 2.0; 0.0 1.0] * Mv
    @test vec(value(x + 2v)) ≈ 3xv && only(value(LinearAlgebra.dot(v, x))) ≈ xv' * xv
    # Numbers promote to symbols, and a vector of indices selects elements.
    @test [x[1], 1.0] isa Vector{T}
    @test vec(value(x[[3, 1]])) == xv[[3, 1]] && value(M[[2, 1], 2:3]) == Mv[[2, 1], 2:3]
    @test vec(value(x[[true, false, true]])) == xv[[true, false, true]]
end

@testset "concatenation, zero, one, norm and init on symbols, $T" for T in (SX, MX)
    x, M = T.sym("x", 3), T.sym("M", 2, 3)
    xv, Mv = [0.5, -1.25, 2.0], [1.0 -2.0 0.5; 3.0 0.25 -1.0]
    value(e) = numeric(e, (x, M), (xv, Mv))
    v, vv = [x[1], x[2]], xv[1:2]
    # A Julia vector of symbols, numbers or columns joins a CasADi matrix as a column.
    @test [x; v] isa T && vec(value([x; v; 2.0])) ≈ [xv; vv; 2.0]
    @test vec(value([x; [x, 1.0]])) ≈ [xv; xv; 1.0]
    @test value([x[1:2] v]) ≈ [vv vv] && value([x[1:2] v; v x[1:2]]) ≈ [vv vv; vv vv]
    # Commas make a list of matrices, as the arguments of a function take them.
    @test [x, M] isa Vector{T} && length([x, M]) == 2
    @test_throws ArgumentError [x; [M]]
    # zero and one of a matrix have its shape; of the type, they are scalars.
    @test value(zero(M)) == zero(Mv) && value(zero(M) .+ M) == Mv && size(zero(T)) == (1, 1)
    @test value(one(M[:, 1:2])) == one(Mv[:, 1:2]) && only(value(one(x[1]))) == 1.0
    # Regression: zero of a scalar stays a scalar that converts to a number.
    @test Float64(zero(T(3.0))) == 0.0
    @test_throws DimensionMismatch one(M)
    @test only(value(LinearAlgebra.norm(v))) ≈ LinearAlgebra.norm(vv)
    @test vec(value(exp2(x))) ≈ exp2.(xv) && vec(value(exp2.(x))) ≈ exp2.(xv)
    @test only(value(sum(x; init = 1.5))) ≈ sum(xv; init = 1.5) && value(sum(M; dims = 2, init = 1.0)) ≈ sum(Mv; dims = 2, init = 1.0)
    @test only(value(sum(abs2, M; init = 2.0))) ≈ sum(abs2, Mv; init = 2.0)
end

@testset "a DM is a matrix of numbers to Julia code" begin
    Mv = [1.0 -2.0 0.5; 3.0 0.25 -1.0]
    d = C.DM(Mv)
    @test [a for a in d] == Mv && sum(d) == sum(Mv) && sum(abs2, d) == sum(abs2, Mv)
    @test sum(d; dims = 1) == sum(Mv; dims = 1) && maximum(d) == maximum(Mv)
    @test [C.DM([1.0, 2.0]); [7.0, 8.0]] isa C.DM && vec(Matrix([C.DM([1.0, 2.0]); [7.0, 8.0]])) == [1.0, 2.0, 7.0, 8.0]
end

@testset "a function takes Julia arrays of numbers" begin
    # Regression: a call on a Vector threw a MethodError, although a call by name took one.
    x, M = SX.sym("x", 2), SX.sym("M", 2, 2)
    f = CasadiFunction("f", [x, M, SX.sym("s")], [M * x, x[1]], ["x", "M", "s"], ["y", "first"])
    y, first = f([1.0, 2.0], [1.0 2.0; 3.0 4.0], 0.5)
    @test y isa C.DM && vec(Matrix(y)) == [5.0, 11.0] && only(first) == 1.0
    @test vec(Matrix(f([1, 2], C.DM([1.0 2.0; 3.0 4.0]), 1)[1])) == [5.0, 11.0]
end

@testset "tools: operations of a function and the first NaN" begin
    x, p = SX.sym("x", 2), SX.sym("p")
    f = CasADiCodegen.Function("f", [x, p], [vertcat(sqrt(x[1] - p) * x[2], log(x[2]) + sin(x[1]))])
    histogram = CasADiCodegen.instruction_histogram(f)
    @test histogram isa Vector{Pair{Symbol,Int}}
    @test Dict(histogram)[:OP_INPUT] == 3 && Dict(histogram)[:OP_SQRT] == 1 && Dict(histogram)[:OP_OUTPUT] == 2
    @test issorted(last.(histogram); rev = true)
    @test sum(last, histogram) == C.n_instructions(f)
    # No NaN, then the square root of a negative number, then a NaN in an argument.
    @test CasADiCodegen.nan_trace(f, [2.0, 3.0], 1.0) === nothing
    found = CasADiCodegen.nan_trace(f, [2.0, 3.0], 5.0)
    @test found.operation === :OP_SQRT && found.operands == (-3.0,) && isnan(found.result)
    @test C.instruction_id(f, found.instruction - 1) == C.OP_SQRT
    @test occursin("sqrt", found.text)
    given = CasADiCodegen.nan_trace(f, [NaN, 3.0], 1.0)
    @test given.operation === :OP_INPUT && given.text == "nonzero 1 of argument 1 (i0)"
    # A matrix function is expanded, and an argument with a sparse pattern is given whole or by its nonzeros.
    A = MX.sym("A", C.Sparsity.diag(2))
    g = CasADiCodegen.Function("g", [A], [log(A[1, 1]) + A[2, 2]])
    @test CasADiCodegen.nan_trace(g, [-1.0 0.0; 0.0 2.0]).operation === :OP_LOG
    @test CasADiCodegen.nan_trace(g, [1.0, 2.0]) === nothing
    # A constant of the function is an operand like any other.
    c = SX.sym("c")
    shifted = CasADiCodegen.Function("shifted", [c], [sqrt(c - 2.5)])
    @test CasADiCodegen.nan_trace(shifted, 1.0).operands == (-1.5,) && CasADiCodegen.nan_trace(shifted, 3.0) === nothing
    @test_throws ArgumentError CasADiCodegen.nan_trace(g, [1.0, 2.0, 3.0])
    @test_throws ArgumentError CasADiCodegen.nan_trace(f, [2.0, 3.0])
    # Functions with Julia functions in them: the histogram counts the calls, the trace needs scalar code.
    twice = callback(v -> 2 .* v, "twice", (2,), (2,))
    v = MX.sym("v", 2)
    h = CasADiCodegen.Function("h", [v], [twice(sin(v))])
    @test Dict(CasADiCodegen.instruction_histogram(h))[:OP_CALL] == 1
    @test Dict(CasADiCodegen.instruction_histogram(h; expand = false))[:OP_CALL] == 1
    @test_throws ArgumentError CasADiCodegen.nan_trace(h, [1.0, 2.0])
end

# The front end of CasADi gives an integer, or a whole Float64, in a Dict of options to CasADi as `true`.
# CasADiCodegen.jl gives each number to CasADi with its own type.
@testset "options of CasADi keep their numbers" begin
    x = MX.sym("x")
    # Integer: seven steps of the classic Runge-Kutta method on x' = -x, against its exact formula.
    rk = C.integrator("rk", "rk", Dict("x" => x, "ode" => -x), 0.0, 1.0, Dict("number_of_finite_elements" => 7))
    h = 1 / 7
    @test Float64(rk(Dict("x0" => C.DM(1.0)))["xf"]) ≈ (1 - h + h^2 / 2 - h^3 / 6 + h^4 / 24)^7 rtol = 1e-14
    # A whole Float64: the eigenvalue -1 is clipped to the margin 2.
    H = MX.sym("H")
    clip = CasADiCodegen.Function("clip", [H], [C.convexify(H, Dict("strategy" => "eigen-clip", "margin" => 2.0))])
    @test Float64(clip(C.DM(-1.0))) == 2.0
    # A Dict in a Dict: the Newton method of collocation needs more than one iteration.
    stiff = Dict("rootfinder" => "newton", "rootfinder_options" => Dict("max_iter" => 50, "error_on_fail" => true))
    cubic = C.integrator("cubic", "collocation", Dict("x" => x, "ode" => -x^3), 0.0, 1.0, stiff)
    @test Float64(cubic(Dict("x0" => C.DM(2.0)))["xf"]) ≈ 2 / sqrt(1 + 8) rtol = 1e-4
    # Vectors of integers and of whole Float64: the constraint z <= 0 selects the root -2 of z^2 = 4.
    z = MX.sym("z", 2)
    negative = C.rootfinder("negative", "kinsol", CasADiCodegen.Function("g", [z], [z .^ 2 .- 4]),
                            Dict("constraints" => [-1, -1], "u_scale" => [1.0, 2.0]))
    @test C.nonzeros(negative(C.DM([-1.0, -1.0]))) ≈ [-2.0, -2.0] rtol = 1e-6
end

# Regression: the front end called the const overload of `DaeBuilder::der`, which cannot
# make the derivative of a variable, so `eq(der(x), ...)` declared no state.
@testset "der of a DaeBuilder declares a differential state" begin
    dae = C.DaeBuilder("decay")
    x, a = C.add(dae, "x", "local", "continuous"), C.add(dae, "a", "parameter", "tunable")
    C.eq(dae, C.der(dae, x), -a * x)
    @test C.x(dae) == ["x"]
    f = C.create(dae, "f", ["x", "p"], ["ode"])
    @test Float64(f(C.DM(2.0), C.DM(3.0))) == -6.0
end

# Regression: with no LP64 BLAS behind libblastrampoline, as when Ipopt.jl is not loaded,
# the Ipopt of CasADi printed "no BLAS/LAPACK library loaded for dcopy_()" and returned garbage.
@testset "the Ipopt of CasADi solves with no other package loaded" begin
    x = SX.sym("x", 2)
    solver = C.nlpsol("nearest", "ipopt", Dict("x" => x, "f" => (x[1] - 1)^2 + (x[2] - 2)^2, "g" => x[1] + x[2]),
                      Dict("ipopt.print_level" => 0, "print_time" => false))
    @test C.nonzeros(solver(; x0 = [0.0, 0.0], lbg = 0.0, ubg = 1.0)["x"]) ≈ [0.0, 1.0] atol = 1e-7
end

@testset "block-triangular form of a sparsity pattern" begin
    # Rows are equations, columns unknowns: unknowns 3 and 1 form a loop, 4 needs 3, and 2 needs 4.
    rows, cols = [3, 3, 1, 1, 0, 0, 2, 2], [2, 0, 2, 0, 2, 3, 3, 1]
    pattern = C.triplet(C.Sparsity, 4, 4, rows, cols)
    form = CasADiCodegen.block_triangular_form(pattern)
    @test form.blocks == C.btf(pattern) == 3
    @test form.rowblock == form.colblock
    # The permuted matrix is block lower triangular, so the loop {1, 3} comes first, then 4, then 2.
    rowof, colof = invperm(form.rowperm), invperm(form.colperm)
    block(offsets, k) = searchsortedlast(offsets, k)
    @test [block(form.colblock, colof[j]) for j in 1:4] == [1, 3, 1, 2]
    @test all(block(form.rowblock, rowof[i+1]) >= block(form.colblock, colof[j+1]) for (i, j) in zip(rows, cols))
    @test CasADiCodegen.block_triangular_form(C.Sparsity(0, 0)).blocks == 0
end

@testset "linearization of a state-space model" begin
    x, u, p = SX.sym("x", 2), SX.sym("u"), SX.sym("p", 2)
    f = CasadiFunction("f", [x, u, p], [vertcat(x[2], -p[1] * sin(x[1]) + p[2] * u)])
    h = CasadiFunction("h", [x, u, p], [vertcat(x[1] * x[2], u^2)])
    lin, lin! = CasADiCodegen.linearization(f, h)
    x0, u0, p0 = [0.3, -0.7], [2.0], [9.81, 0.5]
    A, B, Cm, D = lin(x0, u0, p0)
    @test A ≈ [0 1; -9.81*cos(0.3) 0] && B ≈ [0; 0.5;;]
    @test Cm ≈ [-0.7 0.3; 0 0] && D ≈ [0; 4.0;;]
    # A matrix of one column, such as B for one input, is a matrix, also of static arrays.
    @test B isa Matrix{Float64} && D isa Matrix{Float64}
    @test lin(SVector(0.3, -0.7), SVector(2.0), SVector(9.81, 0.5))[2] isa SMatrix{2,1,Float64}
    # In place, and with `sparse` into the stored values of sparse matrices.
    out = (zeros(2, 2), zeros(2, 1), zeros(2, 2), zeros(2, 1))
    lin!(out, (x0, u0, p0))
    @test out[1] ≈ A && out[4] ≈ D
    _, sparse! = CasADiCodegen.linearization(f, h; sparse = true)
    S = map(i -> SparseMatrixCSC(C.sparsity_out(CasADiCodegen.linearization_function(f, h; sparse = true), i)), (0, 1, 2, 3))
    sparse!(S, (x0, u0, p0))
    @test nnz(S[1]) == 2 && Matrix(S[1]) ≈ A && Matrix(S[4]) ≈ D
    g = CasadiFunction("g", [x, p], [x])
    @test_throws "must take the arguments (x, u, p)" CasADiCodegen.linearization(g, h)
    @test CasADiCodegen.linearization(f, h; expression = Val{true}) isa Expr
end

@testset "linearization of a model with algebraic states" begin
    # 0 = z^3 + z - x1 - u has no closed-form solution, but dz = k (dx1 + du) with k = 1 / (3z^2 + 1).
    model(T) = (x = T.sym("x", 2); z = T.sym("z"); u = T.sym("u"); p = T.sym("p");
                (CasadiFunction("f", [x, z, u, p], [vertcat(x[2], -p * x[1] + z)]),
                 CasadiFunction("g", [x, z, u, p], [z^3 + z - x[1] - u]),
                 CasadiFunction("h", [x, z, u, p], [z + x[2]])))
    z0, x0, p0 = 0.5, [0.2, -0.3], [4.0]
    point = (x0, [z0], [z0^3 + z0 - x0[1]], p0)                # a point where g = 0
    k = 1 / (3z0^2 + 1)
    expected = ([0 1; k-4 0], [0; k;;], [k 1], [k;;])
    lin, lin! = CasADiCodegen.linearization(model(SX)...)
    @test all(map(≈, lin(point...), expected))
    @test all(map(≈, first(CasADiCodegen.linearization(model(MX)...))(point...), expected))
    _, sparse! = CasADiCodegen.linearization(model(SX)...; sparse = true)
    lin_casadi = CasADiCodegen.linearization_function(model(SX)...; sparse = true)
    out = map(i -> SparseMatrixCSC(C.sparsity_out(lin_casadi, i)), (0, 1, 2, 3))
    sparse!(out, point)
    @test all(map((S, E) -> Matrix(S) ≈ E, out, expected))
    f, g, h = model(SX)
    x, z, u, p = SX.sym("x", 2), SX.sym("z"), SX.sym("u"), SX.sym("p")
    wrong = CasadiFunction("g", [x, z, u, p], [vertcat(z, z)])
    @test_throws "must have the size of z" CasADiCodegen.linearization(f, wrong, h)
end
