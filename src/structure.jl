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

Return the CasADi function `(x, u, p) -> (A, B, C, D)` of the linearization of the
state-space model `x' = f(x, u, p)`, `y = h(x, u, p)`:

```math
A = ∂f/∂x, \\quad B = ∂f/∂u, \\quad C = ∂h/∂x, \\quad D = ∂h/∂u.
```

`f` and `h` are CasADi functions of the column vectors `x`, `u` and `p`, with one column
vector as their result; `f` has the size of `x`. Give `p` of no element for a model with
no parameter. The matrices are dense; with `sparse = true` they keep the sparsity patterns
of the Jacobians. [`linearization`](@ref linearization) gives the generated
code of this function.
"""
function linearization_function(f::C.Function, h::C.Function; sparse::Bool = false)
    for g in (f, h)
        @requires C.n_in(g) == 3 && C.n_out(g) == 1 && all(C.size2_in(g, i) == 1 for i in 0:2) && C.size2_out(g, 0) == 1 "linearization: \"$(C.name(g))\" must take the arguments (x, u, p) as column vectors and return one column vector"
    end
    @requires all(C.size1_in(f, i) == C.size1_in(h, i) for i in 0:2) "linearization: \"$(C.name(f))\" and \"$(C.name(h))\" must take arguments of the same sizes"
    @requires C.size1_out(f, 0) == C.size1_in(f, 0) "linearization: the result of \"$(C.name(f))\" must have the size of x"
    T = C.is_a(f, "SXFunction") && C.is_a(h, "SXFunction") ? SX : MX
    x, u, p = (T.sym(C.name_in(f, i), C.size1_in(f, i)) for i in 0:2)
    dx, y = f(x, u, p), h(x, u, p)
    matrix(J) = sparse ? J : Front.densify(J)
    jacobians = [matrix(Front.jacobian(e, v)) for (e, v) in ((dx, x), (dx, u), (y, x), (y, u))]
    lin = Front.Function("linearization", [x, u, p], jacobians, ["x", "u", "p"], ["A", "B", "C", "D"])
    @ensures C.size_out(lin, 0) == (C.size1_in(f, 0), C.size1_in(f, 0)) "linearization: A must be square"
    lin
end

"""
    linearization(f, h; sparse = false, expression = Val{false}, options...) -> (lin, lin!)

Generate the code of the linearization of the state-space model `x' = f(x, u, p)`,
`y = h(x, u, p)` about a point. `lin(x, u, p)` returns the matrices `(A, B, C, D)` at the
point `(x, u, p)`, so that the deviations `δ` from it satisfy `δx' ≈ A δx + B δu` and
`δy ≈ C δx + D δu`; a matrix of one column, such as `B` for one input, is a `Vector`.
`lin!(out, (x, u, p))` writes them into the four arrays of `out`.

The functions are those of [`build_function`](@ref) for
[`linearization_function`](@ref linearization_function)`(f, h; sparse)`,
which also states what `f` and `h` must be. With `sparse = true`, `lin!` writes into
`SparseMatrixCSC` arrays that have the patterns of the outputs of that function. The
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
    build_function(linearization_function(f, h; sparse); expression, options...)
end
