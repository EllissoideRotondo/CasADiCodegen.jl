# Use a Symbolics.jl model

[`casadi_function`](@ref CasADiCodegen.casadi_function) translates Symbolics.jl
expressions into a CasADi function, which [`build_function`](@ref) and
[the integrator page](@ref "Integrate a differential equation") take as any other. The
method exists once Symbolics.jl is loaded, and it translates expressions only, not a
ModelingToolkit system.

## Translate expressions

Each symbol or array of symbols makes one argument, and each expression or array of
expressions one result. Pass an array of symbols whole, such as `p` for `p[1:2]`.

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

Both packages export `build_function`, so `import Symbolics` rather than `using` it,
or qualify the name.

## Supported operations

The translation accepts `+ - * / ^`, `sin cos tan asin acos atan sinh cosh tanh asinh acosh
atanh`, `exp exp2 expm1 log log2 log10 log1p sqrt inv`, `abs sign floor ceil min max hypot`,
`ifelse`, the comparisons `< > <= >= == !=` and the logical operators `& | !`. It throws
an `ArgumentError` for any other operation, a function registered with
`@register_symbolic`, an unevaluated `Differential`, a complex constant, or a symbol
that is not an argument. Apply `Symbolics.scalarize` before a `sum` or `prod` over a
symbolic array.
