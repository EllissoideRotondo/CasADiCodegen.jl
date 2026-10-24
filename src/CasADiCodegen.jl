"""
    CasADiCodegen

The official CasADi front end for Julia, plus Julia code generation.

The module includes the SWIG-generated front end that ships with CasADi;
`CasADiCodegen.C` is the generated module. It has three changes: a number in a Dict of
options reaches CasADi with its type, where the generated code makes it a `bool`;
`der(dae, x)` of a `DaeBuilder` declares the differential state `x`, where the generated
code only reads an existing derivative; and `jacobian` of a graph with an integrator with
an event function gives the Jacobian of forward directions, where CasADi gives a wrong
one. The package adds
`build_function` and `generate_julia`, which turn a CasADi `Function` into
plain Julia code.
"""
module CasADiCodegen

import CasADi_jll
import CasADiSWIG_jll
import Libdl
import LinearAlgebra
import OpenBLAS32_jll
using PrecompileTools: @compile_workload

# The plugins of CasADi (Ipopt, Bonmin) call the LP64 BLAS through libblastrampoline,
# which Julia loads with an ILP64 BLAS only; with no LP64 one they compute garbage.
function __init__()
    if !Base.any(lib -> lib.interface == :lp64, LinearAlgebra.BLAS.lbt_get_config().loaded_libs)
        LinearAlgebra.BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
    end
    @assert Base.any(lib -> lib.interface == :lp64, LinearAlgebra.BLAS.lbt_get_config().loaded_libs)
end

# The official front end: the generated module `casadi`, its alias `C`, the
# convenience layer and its exports (`SX`, `MX`, `jacobian`, ...), with the changes
# of front_end.jl.
include("options.jl")
include("front_end.jl")
Base.include_dependency(CasADiSWIG_jll.casadi_native_body)
Base.include_dependency(CasADiSWIG_jll.casadi_jl)
include(typed_front_end, CasADiSWIG_jll.casadi_native_body)
SHAPED_ZEROS[] == 2 || throw(AssertionError("$(SHAPED_ZEROS[]) methods zero(::T) and one(::T) of the front end were found, not 2; the front end has changed"))

include("codegen.jl")

using .Codegen: build_function, generate_julia, instruction_histogram, nan_trace, differentiable, Solvers, Solver,
    Traced, traced, untraced, traced_function, block_triangular_form, linearization, linearization_function, reduce_index,
    initialization_problem, traced_problem

"""
    CasadiFunction(name, inputs, outputs[, options])

Make a function of CasADi from the symbolic `inputs` and `outputs`. `CasadiFunction`
is the front end's name for `CasADiCodegen.Function`, exported because a bare
`Function` is the abstract type of Julia.

```julia
x = SX.sym("x", 2)
f = CasadiFunction("f", [x], [sin(x[1]) * x[2]])
```
"""
CasadiFunction

"""
    casadi_function(name, inputs, outputs; input_names, output_names) -> CasADiCodegen.Function

Make a function of CasADi from Symbolics.jl expressions: `inputs` is a vector of symbols and
arrays of symbols, `outputs` a vector of expressions and arrays of expressions, and each
becomes one argument or one result of the function, with the shape of the array. Every symbol
of an output must be in `inputs`. `input_names` and `output_names` are vectors of strings;
without them CasADi names the arguments and results. The method is defined when Symbolics.jl
is loaded.

```julia
using Symbolics
@variables x y
f = casadi_function("f", [x, y], [sin(x) * y, Symbolics.gradient(sin(x) * y, [x, y])])
```
"""
function casadi_function end

export build_function, generate_julia, CasadiFunction, casadi_function

include("precompile.jl")

end # module CasADiCodegen
