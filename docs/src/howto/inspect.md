# Inspect a model

When a generated function is slow or returns a NaN, two unexported functions look
into the model.

## Count the operations

[`CasADiCodegen.instruction_histogram`](@ref) counts the instructions of each
operation in the code that `build_function` emits:

```@example inspect
using CasADiCodegen

x = SX.sym("x", 2)
f = CasadiFunction("f", [x], [sqrt(x[1] - x[2]) * sin(x[1]) + x[2]^2])
CasADiCodegen.instruction_histogram(f)
```

## Find the source of a NaN

[`CasADiCodegen.nan_trace`](@ref) returns the first instruction that makes a NaN from
operands without one, the first NaN among the arguments, or `nothing`. With
`x = [2, 5]`, the square root takes a negative number:

```@example inspect
CasADiCodegen.nan_trace(f, [2.0, 5.0])
```
