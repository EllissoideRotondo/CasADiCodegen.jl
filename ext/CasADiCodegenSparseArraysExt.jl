module CasADiCodegenSparseArraysExt

import CasADiCodegen
import SparseArrays
import CasADiCodegen.Codegen.Helpers: stored, is_stored, PatternCheck

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

end # module CasADiCodegenSparseArraysExt
