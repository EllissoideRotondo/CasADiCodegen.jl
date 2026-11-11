"""
    CasADiCodegen

Julia code generation for CasADi functions.

The package works with CasADiNative.jl, the official Julia front end of CasADi, and
exports its names (`SX`, `MX`, `jacobian`, ...) and its module `casadi`, which holds every
function of CasADi. `build_function` and `generate_julia` turn a CasADi `Function` into
plain Julia code.
"""
module CasADiCodegen

import CasADiNative
using CasADiNative
import LinearAlgebra
using PrecompileTools: @compile_workload

"""
    casadi

The module of the CasADi front end, with every function and type of CasADi under its
Python name, such as `casadi.sumsqr`, `casadi.set_value` and `casadi.integrator`.
"""
const casadi = CasADiNative.casadi
const Function = CasADiNative.Function
const CasadiFunction = casadi.Function

for name in names(CasADiNative)
    name === :CasADiNative || @eval export $name
end
# The other names of the front end, for qualified access: `CasADiCodegen.densify`. The
# front end's alias `C` of `casadi` is left out.
for name in names(CasADiNative; all = true)
    text = string(name)
    (isempty(text) || text[1] in ('_', '#', '@') || name in (:eval, :include, :C) || isdefined(@__MODULE__, name)) && continue
    @eval const $name = CasADiNative.$name
end

include("codegen.jl")

using .Codegen: build_function, generate_julia, instruction_histogram, nan_trace, differentiable, Solvers, Solver,
    Traced, traced, untraced, traced_function, block_triangular_form, linearization, linearization_function, reduce_index,
    initialization_problem, traced_problem, nonsmooth_operations, smooth_step, smooth_sign, smooth_abs, smooth_max, smooth_min,
    smoothed, natural_continuation

include("models.jl")

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

export build_function, generate_julia, CasadiFunction, casadi_function, casadi

include("precompile.jl")

end # module CasADiCodegen
