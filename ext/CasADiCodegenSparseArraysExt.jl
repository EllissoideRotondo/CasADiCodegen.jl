module CasADiCodegenSparseArraysExt

import CasADiCodegen
import SparseArrays
import CasADiCodegen.Codegen.Helpers: stored, is_stored, PatternCheck
using CasADiCodegen.Codegen: @requires

# The methods of `stored` and `is_stored` for a sparse matrix, for generated code.
include_string(@__MODULE__, CasADiCodegen.Codegen.SPARSE_HELPER_SOURCE)

"""
    SparseMatrixCSC(sparsity::CasADiCodegen.Sparsity) -> SparseMatrixCSC{Float64,Int}

Make a sparse matrix of stored zeros with the sparsity pattern `sparsity`. Use it as an
output or an input of the in-place function of `build_function` or `generate_julia`, which
then reads and writes the stored values only.

# Examples
```julia
using CasADiCodegen, SparseArrays

x = SX.sym("x", 2)
f = CasadiFunction("f", [x], [jacobian([x[1]^2; x[1] * x[2]], x)])
J = SparseMatrixCSC(CasADiCodegen.sparsity_out(f, 0))      # 2×2 with 3 stored entries
_, g! = build_function(f; expression = Val{false})
g!((J,), ([2.0, 3.0],))                             # J is now [4.0 0.0; 3.0 2.0]
```
"""
function SparseArrays.SparseMatrixCSC(sparsity::CasADiCodegen.C.Sparsity)
    nrow, ncol, colptr, rowval = CasADiCodegen.Codegen.sparsity_tuple(sparsity)
    SparseArrays.SparseMatrixCSC(nrow, ncol, colptr, rowval, zeros(length(rowval)))
end

"""
    take_pattern!(A::SparseMatrixCSC, colptr, rowval) -> A

Give `A` the sparsity pattern of the column pointers `colptr` and the rows `rowval`, with
stored zeros, unless it has that pattern; `A` keeps its arrays, whose lengths change.
"""
function take_pattern!(A::SparseArrays.SparseMatrixCSC, colptr::Vector{Int}, rowval::Vector{Int})
    @requires length(colptr) == size(A, 2) + 1 && colptr[end] - 1 == length(rowval) "the pattern has $(length(colptr) - 1) columns, the matrix $(size(A, 2))"
    SparseArrays.getcolptr(A) == colptr && SparseArrays.rowvals(A) == rowval && return A
    copyto!(SparseArrays.getcolptr(A), colptr)
    copyto!(resize!(SparseArrays.rowvals(A), length(rowval)), rowval)
    fill!(resize!(SparseArrays.nonzeros(A), length(rowval)), zero(eltype(A)))
    A
end

end # module CasADiCodegenSparseArraysExt
