# Tools on the structure of a model: the block-triangular form of a sparsity pattern, and
# the matrices of the linearization of a state-space model.

# `Sparsity::btf` of CasADi returns its permutations and blocks through six arguments of
# type `std::vector<casadi_int>&`, which the SWIG front end drops: its `btf` returns the
# number of blocks only. The method is called here in libcasadi itself. An empty
# `std::vector` of libstdc++ is three null pointers (begin, end, end of the storage), so
# Julia passes six such triples, and frees the storage that CasADi gives them.
const BTF_SYMBOL = :_ZNK6casadi8Sparsity3btfERSt6vectorIxSaIxEES4_S4_S4_S4_S4_
const BTF_METHOD = Base.OncePerProcess{Ptr{Cvoid}}(() -> Front.Libdl.dlsym(Front.Libdl.dlopen(Front.CasADi_jll.libcasadi), BTF_SYMBOL))
const OPERATOR_DELETE = Base.OncePerProcess{Ptr{Cvoid}}(() -> Front.Libdl.dlsym(Front.Libdl.dlopen(Front.CasADi_jll.libcasadi), :_ZdlPv))

"""
    block_triangular_form(sparsity) -> NamedTuple

Return the block-triangular form of the sparsity pattern `sparsity`, which CasADi computes
with `Sparsity::btf` (the Dulmage-Mendelsohn decomposition).

With `A` a matrix of this pattern, `A[rowperm, colperm]` is block lower triangular. Its
diagonal block `k` has the rows `rowblock[k]:rowblock[k+1]-1` and the columns
`colblock[k]:colblock[k+1]-1`. For the Jacobian of square equations by their unknowns,
the blocks are the sets of equations to solve together, in order: the equations of block
`k` need only the unknowns of the blocks up to `k`.

The result has the fields `blocks`, `rowperm`, `colperm`, `rowblock`, `colblock`,
`coarse_rowblock` and `coarse_colblock`, all with indices from 1. The coarse blocks are
those of CasADi, plus one.

# Examples
```jldoctest
julia> using CasADiCodegen

julia> C = CasADiCodegen.C;

julia> pattern = C.triplet(C.Sparsity, 3, 3, [0, 1, 1, 2], [0, 0, 1, 2]);

julia> CasADiCodegen.block_triangular_form(pattern).blocks
3
```
"""
function block_triangular_form(sparsity::C.Sparsity)
    vectors = zeros(UInt, 3, 6)
    blocks = GC.@preserve sparsity vectors ccall(BTF_METHOD(), Clonglong,
                                                (Ptr{Cvoid}, Ptr{UInt}, Ptr{UInt}, Ptr{UInt}, Ptr{UInt}, Ptr{UInt}, Ptr{UInt}),
                                                sparsity.ptr, pointer(vectors, 1), pointer(vectors, 4), pointer(vectors, 7),
                                                pointer(vectors, 10), pointer(vectors, 13), pointer(vectors, 16))
    parts = map(1:6) do k
        start, stop = vectors[1, k], vectors[2, k]
        @ensures stop >= start && (stop - start) % sizeof(Int64) == 0 "block_triangular_form: CasADi returned a vector that is not one of libstdc++"
        values = start == 0 ? Int[] : Int.(copy(unsafe_wrap(Array, Ptr{Int64}(start), Int((stop - start) ÷ sizeof(Int64))))) .+ 1
        start == 0 || ccall(OPERATOR_DELETE(), Cvoid, (Ptr{Cvoid},), Ptr{Cvoid}(start))
        values
    end
    rowperm, colperm, rowblock, colblock, coarse_rowblock, coarse_colblock = parts
    @ensures blocks == C.btf(sparsity) && isperm(rowperm) && length(rowperm) == C.size1(sparsity) &&
             isperm(colperm) && length(colperm) == C.size2(sparsity) "block_triangular_form: the result of CasADi is not a permutation"
    rows, cols = C.get_triplet(sparsity)
    rowof, colof = invperm(rowperm), invperm(colperm)
    @ensures all(searchsortedlast(rowblock, rowof[i+1]) >= searchsortedlast(colblock, colof[j+1]) for (i, j) in zip(rows, cols)) "block_triangular_form: the permuted pattern is not block lower triangular"
    (; blocks = Int(blocks), rowperm, colperm, rowblock, colblock, coarse_rowblock, coarse_colblock)
end

"""
    linearization_function(f, h; sparse = false) -> CasADiCodegen.Function
    linearization_function(f, g, h; sparse = false) -> CasADiCodegen.Function

Return the CasADi function `(x, u, p) -> (A, B, C, D)` of the linearization of the
state-space model `x' = f(x, u, p)`, `y = h(x, u, p)`, with `A` of size `nx × nx`, `B` of
`nx × nu`, `C` of `ny × nx` and `D` of `ny × nu` for `x`, `u` and `y` of the sizes `nx`, `nu` and `ny`:

```math
A = ∂f/∂x, \\quad B = ∂f/∂u, \\quad C = ∂h/∂x, \\quad D = ∂h/∂u.
```

`f` and `h` are CasADi functions of the column vectors `x`, `u` and `p`, with one column
vector as their result; `f` has the size of `x`. Give `p` of no element for a model with
no parameter. The matrices are dense; with `sparse = true` they keep the sparsity patterns
of the Jacobians. [`linearization`](@ref linearization) gives the generated
code of this function.

With three functions of `(x, z, u, p)`, the model is the differential-algebraic system
`x' = f(x, z, u, p)`, `0 = g(x, z, u, p)`, `y = h(x, z, u, p)` with the algebraic
states `z`, and the result is the function `(x, z, u, p) -> (A, B, C, D)`. The
algebraic states are eliminated by the implicit function theorem: with `M = (∂g/∂z) \\ [∂g/∂x ∂g/∂u]`,

```math
[A \\; B] = [∂f/∂x \\; ∂f/∂u] - (∂f/∂z) M, \\quad [C \\; D] = [∂h/∂x \\; ∂h/∂u] - (∂h/∂z) M.
```

The result of `g` has the size of `z`. The matrices are those of the model at a point where
`g = 0` and `∂g/∂z` is invertible, which need not be a steady state. The caller ensures both
conditions: they are not checked, and at a point where `g ≠ 0` the matrices do not describe
the model.
"""
function linearization_function(f::C.Function, h::C.Function; sparse::Bool = false)
    check_linearization_arguments((f, h), ("x", "u", "p"))
    @requires C.size1_out(f, 0) == C.size1_in(f, 0) "linearization: the result of \"$(C.name(f))\" must have the size of x"
    x, u, p = linearization_symbols((f, h), 3)
    dx, y = f(x, u, p), h(x, u, p)
    matrix(J) = sparse ? J : Front.densify(J)
    jacobians = [matrix(Front.jacobian(e, v)) for (e, v) in ((dx, x), (dx, u), (y, x), (y, u))]
    lin = Front.Function("linearization", [x, u, p], jacobians, ["x", "u", "p"], ["A", "B", "C", "D"])
    @ensures C.size_out(lin, 0) == (C.size1_in(f, 0), C.size1_in(f, 0)) "linearization: A must be square"
    lin
end

function linearization_function(f::C.Function, g::C.Function, h::C.Function; sparse::Bool = false)
    check_linearization_arguments((f, g, h), ("x", "z", "u", "p"))
    @requires C.size1_out(f, 0) == C.size1_in(f, 0) "linearization: the result of \"$(C.name(f))\" must have the size of x"
    @requires C.size1_out(g, 0) == C.size1_in(g, 1) "linearization: the result of \"$(C.name(g))\" must have the size of z"
    x, z, u, p = linearization_symbols((f, g, h), 4)
    dx, residual, y = f(x, z, u, p), g(x, z, u, p), h(x, z, u, p)
    J = Front.jacobian
    # The sensitivities of z: dz = -M [dx; du], from 0 = ∂g/∂z dz + ∂g/∂x dx + ∂g/∂u du.
    M = C.solve(Front.densify(J(residual, z)), Front.densify(Front.horzcat([J(residual, x), J(residual, u)])))
    nx, nu = C.size1_in(f, 0), C.size1_in(f, 2)
    reduced(e) = Front.horzcat([J(e, x), J(e, u)]) - C.mtimes(J(e, z), M)
    AB, CD = reduced(dx), reduced(y)
    matrix(J) = sparse ? J : Front.densify(J)
    blocks = [matrix(AB[:, 1:nx]), matrix(AB[:, nx+1:nx+nu]), matrix(CD[:, 1:nx]), matrix(CD[:, nx+1:nx+nu])]
    lin = Front.Function("linearization", [x, z, u, p], blocks, ["x", "z", "u", "p"], ["A", "B", "C", "D"])
    @ensures C.size_out(lin, 0) == (nx, nx) && C.size_out(lin, 1) == (nx, nu) "linearization: A must be square and B have one column per input"
    lin
end

"Check that the functions `fs` take the column vectors `names` and return one column vector each, with the same argument sizes."
function check_linearization_arguments(fs, names)
    n = length(names)
    for g in fs
        @requires C.n_in(g) == n && C.n_out(g) == 1 && all(C.size2_in(g, i) == 1 for i in 0:n-1) && C.size2_out(g, 0) == 1 "linearization: \"$(C.name(g))\" must take the arguments ($(join(names, ", "))) as column vectors and return one column vector"
    end
    @requires all(C.size1_in(g, i) == C.size1_in(first(fs), i) for g in fs for i in 0:n-1) "linearization: \"$(join(map(C.name, fs), "\", \""))\" must take arguments of the same sizes"
    nothing
end

"New symbols of the `n` arguments of `fs[1]`: SX when every function of `fs` is an SX function, MX otherwise."
function linearization_symbols(fs, n)
    T = all(g -> C.is_a(g, "SXFunction"), fs) ? SX : MX
    Tuple(T.sym(C.name_in(fs[1], i), C.size1_in(fs[1], i)) for i in 0:n-1)
end

"""
    linearization(f, h; sparse = false, expression = Val{false}, options...) -> (lin, lin!)
    linearization(f, g, h; sparse = false, expression = Val{false}, options...) -> (lin, lin!)

Generate the code of the linearization of the state-space model `x' = f(x, u, p)`,
`y = h(x, u, p)` about a point. `lin(x, u, p)` returns the matrices `(A, B, C, D)` at the
point `(x, u, p)`, so that the deviations `δ` from it satisfy `δx' ≈ A δx + B δu` and
`δy ≈ C δx + D δu`. Each is a `Matrix`, also with one column, such as `B` for one input,
or an `SMatrix` when `x` is a static array. `lin!(out, (x, u, p))` writes them into the four
arrays of `out`.

With three functions, the model has algebraic states `z`: `x' = f(x, z, u, p)`,
`0 = g(x, z, u, p)`, `y = h(x, z, u, p)`. Then `lin(x, z, u, p)` returns the matrices of
the same form in `δx` and `δu` alone, with `δz` eliminated, at a point `(x, z, u, p)` where
`g = 0`, which the caller computes; `lin!(out, (x, z, u, p))` writes them.

The functions are those of [`build_function`](@ref) with `vector_columns = false` for
[`linearization_function`](@ref linearization_function)`(f, h; sparse)` or `(f, g, h; sparse)`,
which also states what `f`, `g` and `h` must be. With `sparse = true`, `lin!` writes into
`SparseMatrixCSC` arrays that have the patterns of the outputs of that function. The
functions are compiled by default; `expression = Val{true}` returns their code as one
`Expr`, in which `lin` and `lin!` are named `linearization` and `linearization!`. The
other `options` go to `build_function`.

# Examples
```jldoctest
julia> using CasADiCodegen

julia> x, u, p = SX.sym("x", 2), SX.sym("u"), SX.sym("p");

julia> f = CasadiFunction("f", [x, u, p], [vertcat(x[2], -p * x[1] + u)]);

julia> h = CasadiFunction("h", [x, u, p], [x[1:1]]);

julia> lin, lin! = CasADiCodegen.linearization(f, h);

julia> A, B, C, D = lin([0.0, 0.0], [0.0], [4.0]);

julia> A
2×2 Matrix{Float64}:
  0.0  1.0
 -4.0  0.0
```
"""
function linearization(f::C.Function, h::C.Function; sparse::Bool = false, expression = Val{false}, options...)
    @requires !haskey(options, :vector_columns) "linearization returns matrices, so it sets vector_columns = false itself"
    build_function(linearization_function(f, h; sparse); expression, vector_columns = false, options...)
end

function linearization(f::C.Function, g::C.Function, h::C.Function; sparse::Bool = false, expression = Val{false}, options...)
    @requires !haskey(options, :vector_columns) "linearization returns matrices, so it sets vector_columns = false itself"
    build_function(linearization_function(f, g, h; sparse); expression, vector_columns = false, options...)
end
