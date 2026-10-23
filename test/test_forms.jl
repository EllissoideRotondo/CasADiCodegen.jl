# In-place and out-of-place functions on arrays in matrix shape.
using StaticArrays
using RuntimeGeneratedFunctions: RuntimeGeneratedFunction
import ForwardDiff
using SparseArrays

"Matrix shape of one CasADi result, with zero at each structural zero."
function dense_values(sp, nz)
    nrow, ncol, colptr, rowval = CasADiCodegen.Codegen.sparsity_tuple(sp)
    M = zeros(nrow, ncol)
    for c in 1:ncol, k in colptr[c]:colptr[c+1]-1
        M[rowval[k], c] = nz[k]
    end
    ncol == 1 ? vec(M) : M
end

"Matrix shape of one input, with a marker at each structural zero, which the code must ignore."
function dense_input(sp, nz)
    nrow, ncol, colptr, rowval = CasADiCodegen.Codegen.sparsity_tuple(sp)
    M = fill(NaN, nrow, ncol)
    for c in 1:ncol, k in colptr[c]:colptr[c+1]-1
        M[rowval[k], c] = nz[k]
    end
    ncol == 1 ? vec(M) : M
end

forms_expected(f, nz) = [dense_values(C.sparsity_out(f, j - 1), r) for (j, r) in enumerate(mx_casadi(f, nz))]
forms_inputs(f, nz) = Tuple(dense_input(C.sparsity_in(f, i - 1), v) for (i, v) in enumerate(nz))
static(a::AbstractVector) = SVector{length(a)}(a)
static(a::AbstractMatrix) = SMatrix{size(a)...}(a)
mutable_static(a::AbstractVector) = MVector{length(a),Float64}(undef)
mutable_static(a::AbstractMatrix) = MMatrix{size(a)...,Float64}(undef)
as_tuple(r) = r isa Tuple ? r : (r,)

"The dense results of the four forms, each as a vector of arrays."
function forms_results(f, nz; kwargs...)
    expected, x = forms_expected(f, nz), forms_inputs(f, nz)
    g, g! = build_function(f; expression = Val{false}, kwargs...)
    s, _ = build_function(f; expression = Val{false}, force_SA = true, kwargs...)
    [
        "out of place, arrays" => collect(as_tuple(g(x...))),
        "in place, arrays" => collect(g!(Tuple(fill(NaN, size(e)) for e in expected), x)),
        "in place, mutable static arrays" => collect(g!(Tuple(mutable_static(e) for e in expected), map(static, x))),
        "out of place, static arrays" => collect(as_tuple(g(map(static, x)...))),
        "out of place, static arrays forced" => collect(as_tuple(s(x...))),
    ]
end

forms_agree(f, nz; kwargs...) = all(forms_results(f, nz; kwargs...)) do (label, actual)
    expected = forms_expected(f, nz)
    length(actual) == length(expected) &&
        all(size(a) == size(e) && all(close_enough.(a, e)) for (a, e) in zip(actual, expected))
end

# Scalar code in one piece, split scalar code, and matrix code; the other products of type and option give one of these.
const FORM_CASES = ((SX, (;)), (SX, (; chunk_size = 3)), (MX, (; expand = false)))

# Sparse input, dense input, and dense, sparse and matrix outputs.
function forms_function(T)
    L, v = T.sym("L", C.Sparsity.lower(3)), T.sym("v", 2)
    y = C.vertcat([L[1, 1] * v[1] + L[3, 2], sin(v[2])])
    CasADiCodegen.Function("f", [L, v], [y, 2 * L + L * v[1], C.transpose(L), C.mtimes(v, C.transpose(v))])
end

@testset "output forms: values of CasADi" begin
    for (T, kwargs) in FORM_CASES
        f = forms_function(T)
        @test forms_agree(f, mx_inputs(f); kwargs...)
    end
end

@testset "output forms: types and shapes" begin
    f = forms_function(SX)
    nz = mx_inputs(f)
    x = forms_inputs(f, nz)
    g, g! = build_function(f; expression = Val{false})
    r = g(x...)
    @test r isa Tuple{Vector{Float64},Matrix{Float64},Matrix{Float64},Matrix{Float64}}
    @test r[2][1, 2] === 0.0 && r[3][2, 1] === 0.0
    r = g(map(static, x)...)
    @test r isa Tuple{SVector{2,Float64},SMatrix{3,3,Float64,9},SMatrix{3,3,Float64,9},SMatrix{2,2,Float64,4}}
    out = Tuple(fill(NaN, size(e)) for e in forms_expected(f, nz))
    @test g!(out, x) === out
    s, _ = build_function(f; expression = Val{false}, force_SA = true)
    @test s(x...) isa Tuple{SVector{2,Float64},SMatrix{3,3,Float64,9},SMatrix{3,3,Float64,9},SMatrix{2,2,Float64,4}}

    # One output: the array itself. A number serves as an input with one element.
    a = SX.sym("a")
    one_output = CasADiCodegen.Function("h", [a], [C.vertcat([a^2, 3 * a])])
    h, _ = build_function(one_output; expression = Val{false}, force_SA = true)
    @test h(2.0) === SVector(4.0, 6.0)
    h, h! = build_function(one_output; expression = Val{false})
    @test h(SVector(2.0)) === SVector(4.0, 6.0)
    @test h(2.0) == [4.0, 6.0]
    @test h!((zeros(2),), ([2.0],))[1] == [4.0, 6.0]

    # An output without structural nonzeros is a zero array, in each form.
    empty_output = CasADiCodegen.Function("e", [a], [C.vertcat([a, a]), SX(2, 3)])
    for kwargs in ((;), (; chunk_size = 1)), (label, r) in forms_results(empty_output, [[1.5]]; kwargs...)
        @test r[1] == [1.5, 1.5] && r[2] == zeros(2, 3)
    end

    # A function without inputs has no argument to choose static arrays by.
    constant = CasADiCodegen.Function("k", SX[], [SX([1.0, 2.0])])
    k, _ = build_function(constant; expression = Val{false})
    @test k() == [1.0, 2.0]
    k, _ = build_function(constant; expression = Val{false}, force_SA = true)
    @test k() === SVector(1.0, 2.0)

    # With `nonzeros`: the single function on vectors of nonzeros.
    nz_function = build_function(f; expression = Val{false}, nonzeros = true)
    @test isequal(nz_function([zeros(n) for n in nnz_out(f)], Tuple(nz)), mx_casadi(f, nz))
end

@testset "output forms: no allocation with static arrays" begin
    f = forms_function(SX)
    x = map(static, forms_inputs(f, mx_inputs(f)))
    run(s, x) = s(x...)
    for kwargs in ((;), (; force_SA = true))         # split code: see "static arrays from split scalar code"
        s, _ = build_function(f; expression = Val{false}, kwargs...)
        run(s, x)
        @test (@allocated run(s, x)) == 0
    end
end

@testset "output forms: number types" begin
    a = SX.sym("a", 2)
    f = CasADiCodegen.Function("h", [a], [C.vertcat([a[1] * a[2], sin(a[1]), 2])])
    g, _ = build_function(f; expression = Val{false})
    expected = [-1.7 0.3; cos(0.3) 0.0; 0.0 0.0]
    @test ForwardDiff.jacobian(v -> g(SVector{2}(v)), [0.3, -1.7]) == expected
    @test ForwardDiff.jacobian(g, [0.3, -1.7]) == expected
end

@testset "output forms: expression and source" begin
    f = forms_function(SX)
    nz = mx_inputs(f)
    expected, x = forms_expected(f, nz), forms_inputs(f, nz)
    for (force_SA, convert_input, is_static) in ((false, identity, false), (false, static, true), (true, identity, true))
        mod = Module()
        Core.eval(mod, build_function(f; force_SA))
        src = Module()
        Base.include_string(src, sprint(io -> generate_julia(f, io; module_name = :Gen, force_SA)))
        for m in (mod, Base.invokelatest(getglobal, src, :Gen))
            r = Base.invokelatest((a...) -> m.f(a...), map(convert_input, x)...)
            @test r[1] isa (is_static ? SVector{2,Float64} : Vector{Float64})
            @test all(all(close_enough.(a, e)) for (a, e) in zip(r, expected))
            out = Tuple(fill(NaN, size(e)) for e in expected)
            Base.invokelatest((o, a) -> m.f!(o, a), out, x)
            @test all(all(close_enough.(a, e)) for (a, e) in zip(out, expected))
        end
    end
    source = sprint(io -> generate_julia(f, io; module_name = :Gen))
    @test occursin("import StaticArraysCore", source)
    @test all(occursin(source), ("function f_nonzeros!(res, arg)", "function f!(res, arg)", "function f(arg1, arg2)",
                                 "function f(arg1::StaticArraysCore.StaticArray, arg2)"))
end

@testset "output forms: wrong use" begin
    f = forms_function(SX)
    @test_throws ArgumentError build_function(f; name = :f)
    @test_throws ArgumentError generate_julia(f, IOBuffer(); name = :f)
    @test build_function(f; name = :f, nonzeros = true) isa Expr
    g, g! = build_function(f; expression = Val{false})
    @test_throws ArgumentError g!((zeros(2), zeros(2, 2), zeros(3, 3), zeros(2, 2)), forms_inputs(f, mx_inputs(f)))
end

# Outputs with more elements than STATIC_LIMIT (1024) have no method with static arrays.
# The static method of a 2000 by 2000 diagonal Jacobian was one literal of 4 million
# values, which Julia refused to parse ("expression too large"). A static argument then
# gets ordinary arrays and one warning per function, and not a silent change of type.
@testset "output forms: limit of static arrays" begin
    diagonal(n) = (x = SX.sym("x", n); CasADiCodegen.Function("diagonal", [x], [C.jacobian(x .^ 2, x)]))
    g = build_function(diagonal(32); expression = Val{false})[1]     # 1024 elements
    @test (@test_logs g(SVector{32}(1.0:32.0))) isa SMatrix{32,32,Float64}

    large = diagonal(33)                                             # 1089 elements
    s = SVector{33}(1.0:33.0)
    g = build_function(large; expression = Val{false})[1]
    J = @test_logs (:warn, r"1089") (g(s); g(s))
    @test J isa Matrix{Float64} && J[end, end] == 66.0 && J[1, 2] == 0.0
    @test (@test_logs g(collect(s))) == J

    src = Module()
    Base.include_string(src, sprint(io -> generate_julia(large, io; module_name = :Large)))
    J = @test_logs (:warn, r"1089") Base.invokelatest(() -> (getglobal(src, :Large).diagonal(s); getglobal(src, :Large).diagonal(s)))
    @test J isa Matrix{Float64} && J[end, end] == 66.0
    @test_throws ArgumentError build_function(large; force_SA = true)
end

# A random pattern for a 3-by-3 input, from a mask of 9 entries with at least one nonzero.
function masked_function(T, mask)
    rows = [mod(k - 1, 3) for k in 1:9 if mask[k]]
    cols = [div(k - 1, 3) for k in 1:9 if mask[k]]
    X, v = T.sym("X", C.Sparsity.triplet(3, 3, rows, cols)), T.sym("v", 3)
    # The second output is full, because X + 1 is not zero at a structural zero. Each operation is
    # rounded correctly, so that a sum that cancels cannot enlarge a difference between two libraries.
    CasADiCodegen.Function("f", [X, v], [C.mtimes(X, v), X * 3 + (X + 1) * v[2], C.transpose(X)])
end

"The arrays of `expected` with a sparse matrix at each sparse pattern, and the same for the inputs."
sparse_outputs(f, expected) = Tuple(is_sparse_pattern(C.sparsity_out(f, j - 1)) ? SparseMatrixCSC(C.sparsity_out(f, j - 1)) : fill(NaN, size(e))
                                    for (j, e) in enumerate(expected))
sparse_inputs(f, nz) = Tuple(is_sparse_pattern(C.sparsity_in(f, i - 1)) ?
                             (S = SparseMatrixCSC(C.sparsity_in(f, i - 1)); nonzeros(S) .= v; S) : dense_values(C.sparsity_in(f, i - 1), v)
                             for (i, v) in enumerate(nz))
is_sparse_pattern(sp) = 0 < C.nnz(sp) < C.size1(sp) * C.size2(sp)

function sparse_agrees(f, nz; kwargs...)
    expected = forms_expected(f, nz)
    _, g! = build_function(f; expression = Val{false}, kwargs...)
    out = sparse_outputs(f, expected)
    patterns = [o isa SparseMatrixCSC ? (copy(o.colptr), copy(rowvals(o))) : nothing for o in out]
    g!(out, sparse_inputs(f, nz)) === out &&
        all(length(a) == length(e) && all(close_enough.(vec(Array(a)), vec(e))) for (a, e) in zip(out, expected)) &&
        all(p === nothing || (o.colptr, rowvals(o)) == p for (o, p) in zip(out, patterns))
end

@testset "sparse matrices in place" begin
    for (T, kwargs) in FORM_CASES
        f = forms_function(T)
        @test sparse_agrees(f, mx_inputs(f); kwargs...)
    end
    f = forms_function(SX)
    nz = mx_inputs(f)
    S = SparseMatrixCSC(C.sparsity_out(f, 1))
    @test S isa SparseMatrixCSC{Float64,Int} && size(S) == (3, 3) && nnz(S) == 6 && iszero(S)

    # No allocation: the function works on the stored values.
    _, g! = build_function(f; expression = Val{false})
    out, x = sparse_outputs(f, forms_expected(f, nz)), sparse_inputs(f, nz)
    g!(out, x)
    @test (@allocated g!(out, x)) == 0

    # A sparse matrix with another pattern is rejected, as input and as output.
    wrong = sparse([1, 2, 3, 3, 3, 1], [1, 2, 3, 1, 2, 3], ones(6))
    @test_throws "pattern of a sparse matrix" g!((out[1], wrong, out[3], out[4]), x)
    @test_throws "pattern of a sparse matrix" g!(out, (wrong, x[2]))
    @test_throws ArgumentError g!(out, (sparse(ones(3, 3)), x[2]))
    # A matrix that gains a stored value after a call with it is rejected at the next call.
    grown = out[2]
    grown[findfirst(c -> !(c[1] in rowvals(grown)[nzrange(grown, c[2])]), CartesianIndices(grown))] = 1.0
    @test_throws "pattern of a sparse matrix" g!((out[1], grown, out[3], out[4]), x)

    # A written module holds the method and imports SparseArrays only when a pattern is sparse.
    source = sprint(io -> generate_julia(f, io; module_name = :Gen))
    @test occursin("import SparseArrays", source)
    src = Module()
    Base.include_string(src, source)
    out = sparse_outputs(f, forms_expected(f, nz))
    Base.invokelatest((o, a) -> src.Gen.f!(o, a), out, x)
    @test all(all(close_enough.(Array(a), e)) for (a, e) in zip(out, forms_expected(f, nz)))
    a = SX.sym("a", 2)
    dense = sprint(io -> generate_julia(CasADiCodegen.Function("d", [a], [a * 2, C.mtimes(sin(a), C.transpose(a))]), io))
    @test !occursin("SparseArrays", dense) && !occursin("stored", dense)
end

@testset "a sparse matrix costs as much as its stored values after the first call" begin
    # Regression: the pattern of a SparseMatrixCSC was compared with that of the function at every call,
    # which cost five times the evaluation of this bidiagonal Jacobian.
    n = 1000
    x = SX.sym("x", n)
    f = CasADiCodegen.Function("j", [x], [CasADiCodegen.jacobian(vertcat([x[k]^2 + (k > 1 ? x[k-1] : 0.0) for k in 1:n]...), x)])
    _, g! = build_function(f; expression = Val{false})
    h! = build_function(f; expression = Val{false}, nonzeros = true)
    J, v = SparseMatrixCSC(C.sparsity_out(f, 0)), rand(n)
    g!((J,), (v,))
    @test nonzeros(J) == h!((zeros(nnz(J)),), (v,))[1]
    fastest(call) = minimum(@elapsed(for _ in 1:20; call(); end) for _ in 1:50)
    @test fastest(() -> g!((J,), (v,))) < 2 * fastest(() -> h!((nonzeros(J),), (v,)))
end

@testset "property: each form equals CasADi on random patterns" begin
    @check max_examples = 20 function forms_equal_casadi(
            mask = Data.Vectors(Data.Booleans(); min_size = 9, max_size = 9),
            values = Data.Vectors(test_float; min_size = 12, max_size = 12),
            matrix = Data.Booleans(), chunk_size = Data.SampledFrom([2, 1000]))
        assume!(any(mask))
        f = masked_function(matrix ? MX : SX, mask)
        nz = [values[1:count(mask)] .+ 0.0, values[10:12] .+ 0.0]
        forms_agree(f, nz; chunk_size, expand = !matrix) && sparse_agrees(f, nz; chunk_size, expand = !matrix)
    end
end

@testset "scalar code in place writes the arrays directly" begin
    f = forms_function(SX)                      # sparse patterns in an input and in two outputs
    nz = mx_inputs(f)
    expected, x = forms_expected(f, nz), forms_inputs(f, nz)
    _, g! = build_function(f; expression = Val{false})
    out = Tuple(fill(NaN, size(e)) for e in expected)
    g!(out, x)
    @test (@allocated g!(out, x)) == 0          # no view of the nonzeros
end

@testset "static arrays from split scalar code" begin
    f = forms_function(SX)
    nz = mx_inputs(f)
    expected = forms_expected(f, nz)
    x = map(static, forms_inputs(f, nz))
    run(s, x) = s(x...)
    split_code = string(build_function(f; chunk_size = 3))
    @test occursin("workspace", split_code) && !occursin("workspace", string(build_function(f)))
    @test count("function f_nonzeros!_chunk1(w)", split_code) == 1 && !occursin("f!_chunk", split_code)   # one set of sub-functions
    # No form of the split code allocates memory: the work vector belongs to the task.
    nonzero_form = build_function(f; expression = Val{false}, chunk_size = 3, nonzeros = true)
    _, inplace = build_function(f; expression = Val{false}, chunk_size = 3)
    res_nz, res = Tuple(zeros(C.nnz(sp)) for sp in C.sparsity_out.(Ref(f), 0:3)), Tuple(zeros(size(e)) for e in expected)
    arg_nz, arg = Tuple(nz), Tuple(forms_inputs(f, nz))
    nonzero_form(res_nz, arg_nz); inplace(res, arg)
    @test (@allocated nonzero_form(res_nz, arg_nz)) == 0
    @test (@allocated inplace(res, arg)) == 0
    @test all(all(close_enough.(a, e)) for (a, e) in zip(res, expected))
    for kwargs in ((; chunk_size = 3), (; chunk_size = 3, force_SA = true))
        s, _ = build_function(f; expression = Val{false}, kwargs...)
        r = run(s, x)
        @test r isa Tuple{SVector{2,Float64},SMatrix{3,3,Float64,9},SMatrix{3,3,Float64,9},SMatrix{2,2,Float64,4}}
        @test all(all(close_enough.(a, e)) for (a, e) in zip(r, expected))
        @test (@allocated run(s, x)) == 0       # the static arrays come from the work vector
    end

    # Other number types: dual numbers, and a type that is not stored inline.
    a = SX.sym("a", 2)
    h = CasADiCodegen.Function("h", [a], [C.vertcat([a[1] * a[2], sin(a[1]), 2])])
    s, _ = build_function(h; expression = Val{false}, chunk_size = 2)
    @test ForwardDiff.jacobian(v -> s(SVector{2}(v)), [0.3, -1.7]) == [-1.7 0.3; cos(0.3) 0.0; 0.0 0.0]
    b, _ = build_function(h; expression = Val{false}, chunk_size = 2, nanmath = false)
    big_result = b(SVector(big"0.3", big"-1.7"))
    @test big_result isa SVector{3,BigFloat} && big_result ≈ [big"0.3" * big"-1.7", sin(big"0.3"), 2]

    # A written module holds the split form and runs on its own.
    src = Module()
    Base.include_string(src, sprint(io -> generate_julia(f, io; module_name = :Gen, chunk_size = 3)))
    r = Base.invokelatest((a...) -> src.Gen.f(a...), x...)
    @test all(all(close_enough.(a, e)) for (a, e) in zip(r, expected))
end

@testset "matrix code allocates no memory in place and with static arrays" begin
    # A matrix function that calls a scalar function, with patterns that have structural zeros.
    a = SX.sym("a", 2)
    inner = CasADiCodegen.Function("inner", [a], [C.vertcat([a[1] * a[2], sin(a[1])])])
    L, v = MX.sym("L", C.Sparsity.lower(3)), MX.sym("v", 2)
    f = CasADiCodegen.Function("f", [L, v], [inner(v) + L[1, 1], 2 * L + L * v[1], C.mtimes(v, C.transpose(inner(v)))])
    nz = mx_inputs(f)
    expected = forms_expected(f, nz)
    x = forms_inputs(f, nz)
    xs = map(static, x)
    run(s, x) = s(x...)
    for kwargs in ((; expand = false), (; expand = false, chunk_size = 1))
        @test occursin("inner!(w)", string(build_function(f; kwargs...)))        # a called function takes the work vector only
        g, g! = build_function(f; expression = Val{false}, kwargs...)
        out = Tuple(fill(NaN, size(e)) for e in expected)
        g!(out, x); r = run(g, xs)
        @test all(all(close_enough.(a, e)) for (a, e) in zip(out, expected))
        @test all(all(close_enough.(a, e)) for (a, e) in zip(r, expected))
        @test (@allocated g!(out, x)) == 0
        @test (@allocated run(g, xs)) == 0
    end
end

module_optlevel(m::Module) = ccall(:jl_get_module_optlevel, Cint, (Any,), m)

@testset "optlevel sets the optimization level of the compiled functions" begin
    x = SX.sym("x", 2)
    f = CasADiCodegen.Function("f", [x], [[x[1]^2 + sin(x[2]); x[1] * x[2]]])
    expected = build_function(f; expression = Val{false})[1]([0.3, 1.7])
    for level in (0, 1, 3)
        g, _ = build_function(f; expression = Val{false}, optlevel = level)
        @test g([0.3, 1.7]) == expected
        @test module_optlevel(parentmodule(typeof(g).parameters[3])) == level
    end
    source = Module()
    Base.include_string(source, sprint(io -> generate_julia(f, io; module_name = :Gen, optlevel = 1)))
    @test module_optlevel(source.Gen) == 1
    @test_throws ArgumentError build_function(f; expression = Val{false}, optlevel = 4)
    @test_throws ArgumentError build_function(f; optlevel = 1)
end
