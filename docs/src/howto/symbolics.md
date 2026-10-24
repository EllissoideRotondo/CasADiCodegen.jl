# Use a Symbolics.jl model

A Symbolics.jl expression becomes a CasADi function with
[`casadi_function`](@ref CasADiCodegen.casadi_function), a `CasADiCodegen.Function`
like the result of [`CasadiFunction`](@ref). [`build_function`](@ref) then generates Julia
code for it, `jacobian` gives its derivatives, and [the integrator page](@ref
"Integrate a differential equation") takes it as a right side. The method exists once Symbolics.jl is
loaded; without it the call raises a `MethodError`. The translation covers expressions
only: a ModelingToolkit system is not translated.

## Translate expressions

`casadi_function` takes the name of the function, the symbols that become its arguments,
and the expressions that become its results. A symbol, a vector or a matrix of symbols
makes one argument; an expression, a vector or a matrix of expressions makes one result,
with the same shape, matrices in column-major order. Pass an array of symbols whole, such as
`p` for `p[1:2]`. A constant becomes a constant of the function. The keywords `input_names`
and `output_names` are vectors of strings; without them, CasADi names the arguments and
results. Every symbol of a result must be in the arguments.

```@example symbolics
using CasADiCodegen
import Symbolics
using Symbolics: @variables

@variables x y p[1:2]
expr = sin(x) * y^2 + p[1] * exp(-x) / (1 + p[2]^2)

f = casadi_function("f", [x, y, p], [expr, Symbolics.gradient(expr, [x, y])];
                    input_names = ["x", "y", "p"], output_names = ["value", "gradient"])
g, g! = build_function(f; expression = Val{false})
g(0.3, -1.2, [2.0, 0.5])
```

Both packages export `build_function`. After `using Symbolics` and `using CasADiCodegen`
the bare name is ambiguous, so import Symbolics.jl or qualify the name.

## Supported operations

The translation accepts `+ - * / ^`, `sin cos tan asin acos atan sinh cosh tanh asinh acosh
atanh`, `exp exp2 expm1 log log2 log10 log1p sqrt inv`, `abs sign floor ceil min max hypot`,
`ifelse`, the comparisons `< > <= >= == !=` and the logical operators `& | !`. It stops with an `ArgumentError` that names an unsupported
operation, a function registered with `@register_symbolic`, an unevaluated `Differential` or a complex constant,
and a symbol that is not an argument. A `sum` or `prod` over a symbolic array raises the same error; apply `Symbolics.scalarize` to
the expression first.
