# Use the generated derivatives from an AD package

[`CasADiCodegen.differentiable`](@ref) makes a CasADi function a Julia function with
ChainRulesCore.jl rules, so that an AD package such as Zygote.jl takes its derivatives
from generated code instead of tracing it.

## Make a differentiable function

The result is called as the out-of-place function of [`build_function`](@ref), whose
keywords it takes:

```@example rules
using CasADiCodegen

x, p = SX.sym("x", 2), SX.sym("p")
f = CasadiFunction("f", [x, p], [p * x[1] * sin(x[2])])

d = CasADiCodegen.differentiable(f)
d([2.0, pi / 2], [3.0])
```

## Give the rules to an AD package

The rules exist once ChainRulesCore.jl is loaded. The pullback of the reverse rule
returns the gradients by `x` and by `p`:

```@example rules
import ChainRulesCore

y, pullback = ChainRulesCore.rrule(d, [2.0, pi / 2], [3.0])
pullback([1.0])
```

An AD package calls the rules in the same way:

```julia
import Zygote

dx, dp = Zygote.gradient((x, p) -> sum(abs2, d(x, p)), [2.0, pi / 2], [3.0])
```

The code of each rule is generated at its first use, so that call is slower.
