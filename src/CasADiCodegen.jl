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

Make a CasADi function from the symbolic `inputs` and `outputs`, vectors of `SX` or `MX`.
`CasadiFunction` is the front end's name for `CasADiCodegen.Function`, exported because a
bare `Function` is Julia's abstract function type. `options` is a `Dict` of CasADi
options, such as `"allow_free" => true`.

CasADi throws a `SwigError` when an output has a symbol that is not an input, unless
`"allow_free"` is set, and [`build_function`](@ref) refuses such a function.

# Examples
```jldoctest
julia> using CasADiCodegen

julia> x = SX.sym("x", 2);

julia> f = CasadiFunction("f", [x], [sin(x[1]) * x[2]]);

julia> f(casadi.DM([0.0, 2.0]))
DM(0)
```
"""
CasadiFunction

"""
    casadi_function(name, inputs, outputs; input_names, output_names) -> CasADiCodegen.Function

Make a CasADi function from Symbolics.jl expressions: `inputs` is a vector of symbols and
arrays of symbols, `outputs` a vector of expressions and arrays of expressions, and each
becomes one argument or one result of the function, with the shape of the array. Every
symbol of an output must be in `inputs`. `input_names` and `output_names` are vectors of
strings, one per input and per output; without them CasADi names the arguments and
results. The method exists once Symbolics.jl is loaded.

Throws an `ArgumentError` if an output has a symbol that is not in `inputs`, or a complex
value. A wrong number of names gives CasADi's `SwigError`.

# Examples
```jldoctest
julia> using CasADiCodegen

julia> import Symbolics

julia> Symbolics.@variables x y;

julia> expr = sin(x) * y;

julia> grad = Symbolics.gradient(expr, [x, y]);

julia> f = casadi_function("f", [x, y], [expr, grad]);

julia> g, _ = build_function(f; expression = Val{false});

julia> g(0.0, 2.0)
([0.0], [2.0, 0.0])
```

Symbolics.jl exports a `build_function` of its own, so `import Symbolics`, not `using`,
keeps the `build_function` of this package unambiguous.
"""
function casadi_function end

export build_function, generate_julia, CasadiFunction, casadi_function, casadi
# The names that the manual calls with the prefix `CasADiCodegen.`; `docs/make.jl` checks
# that the reference lists each of them.
public Models, Traced, traced, untraced, traced_function, traced_problem, Solvers, Solver,
    linearization, linearization_function, block_triangular_form, reduce_index, nonsmooth_operations,
    smooth_step, smooth_sign, smooth_abs, smooth_max, smooth_min, smoothed, differentiable,
    instruction_histogram, nan_trace, initialization_problem, natural_continuation

"The version of CasADi whose serial form, C code and node data the readers of this package parse."
const CASADI_VERSION = "3.8.2"

# The readers parse formats that CasADi does not keep stable between versions, so another
# version is announced when the package loads; most readers also compare their code with
# CasADi at build time.
function __init__()
    found = casadi.CasadiMeta.version()
    found == CASADI_VERSION || @warn "CasADiCodegen.jl reads the internals of CasADi $CASADI_VERSION, and CasADi $found is loaded; generated code may be wrong or refused. Pin CasADiNative.jl to a version with CasADi $CASADI_VERSION."
    nothing
end

include("precompile.jl")

end # module CasADiCodegen
