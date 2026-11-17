# Getting started

We write a small model with CasADi, generate Julia functions for it and its Jacobian, call
them with arrays, static arrays and a sparse matrix, differentiate them, solve an ODE with
a model of the same kind, and write the functions to a file.

## Write the model

`using CasADiCodegen` loads [CasADiNative.jl](https://github.com/casadi/CasADiNative.jl),
the Julia front end of CasADi 3.8.2, whose calls follow the
[CasADi documentation](https://web.casadi.org/docs/) for Python, with the differences
listed in [The CasADi front end](@ref). `CasadiFunction` makes a CasADi function, since
`Function` is a Julia type:

```@example start
using CasADiCodegen

x = SX.sym("x", 3)
y = [sin(x[1]) * x[2]; x[3]^2 / (1 + x[1]^2)]
f = CasadiFunction("f", [x], [y, jacobian(y, x)])
nothing # hide
```

`f` has one input, `x`, and two outputs: `y` and its 2 by 3 Jacobian.

## Generate functions for this session

[`build_function`](@ref) returns an out-of-place and an in-place function:

```@example start
g, g! = build_function(f; expression = Val{false})
value, J = g([0.3, 1.7, -0.4])
J
```

An output with one column comes back as a `Vector`, any other as a `Matrix`
(`vector_columns = false` keeps the matrix shape). The in-place function takes a tuple of
outputs and a tuple of inputs:

```@example start
out = (zeros(2), zeros(2, 3))
g!(out, ([0.3, 1.7, -0.4],))
out[1]
```

## Avoid allocation with static arrays

With a static array as input, the out-of-place function returns static arrays and does
not allocate:

```@example start
using StaticArrays

value, J = g(SVector(0.3, 1.7, -0.4))
typeof(J)
```

`force_SA = true` returns static arrays for any input. Static arrays hold at most 1024
elements over all outputs of one function together; above that, a static input gives
ordinary arrays with a warning, and `force_SA = true` throws an `ArgumentError`.

## Fill a sparse Jacobian

The in-place function also fills a `SparseMatrixCSC` with the sparsity pattern of the
output, and rejects one with another pattern:

```@example start
using SparseArrays

Js = SparseMatrixCSC(casadi.sparsity_out(f, 1))   # pattern of the second output: CasADi counts from 0
g!((zeros(2), Js), ([0.3, 1.7, -0.4],))
Js
```

## Differentiate the generated code

The generated code takes the dual numbers of ForwardDiff.jl; here they reproduce the
Jacobian that CasADi built:

```@example start
import ForwardDiff

ForwardDiff.jacobian(x -> g(x)[1], [0.3, 1.7, -0.4]) ≈ J
```

## Solve an ODE

A CasADi function of the state, the parameters and the time is the right side of a SciML
problem, with generated code for its Jacobians. The states and parameters take the names
of the scalar symbols, and each result after the first is an observed variable named after
the result:

```@example start
using OrdinaryDiffEqTsit5: Tsit5
import SciMLBase

u, τ, t = SX.sym("u"), SX.sym("τ"), SX.sym("t")
lag = CasadiFunction("lag", [u, τ, t], [(sin(t) - u) / τ, sin(t) - u], ["u", "τ", "t"], ["du", "error"])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(lag), [0.0], (0.0, 10.0), [0.5])
solution = SciMLBase.solve(problem, Tsit5())
solution[:error][end], SciMLBase.remake(problem; p = [:τ => 2.0]).p
```

A new parameter value needs no new code. [Use the generated code with SciML packages](@ref)
covers the other problems.

## Write the code to a file

[`generate_julia`](@ref) writes the functions to a module that runs without
CasADiCodegen.jl; each package of `F.DEPENDENCIES` must be in the environment that loads
the file:

```@example start
path = joinpath(mktempdir(), "f.jl")
generate_julia(f, path)                       # the module F, after the function f
include(path)
F.f([0.3, 1.7, -0.4])[1], F.DEPENDENCIES
```

## Choose a form

| Call | Result | Needs CasADiCodegen.jl to run |
|---|---|---|
| `build_function(f)` | An `Expr`, for `eval` or a macro | Yes |
| `build_function(f; expression = Val{false})` | Compiled functions | Yes |
| `build_function(f; expression = Val{false}, cache = true)` | The same, cached for later sessions | Yes |
| `generate_julia(f, path)` | A module in a file | No, unless the model holds a SciML solver |

The first call of a function compiles it, which takes seconds for a graph of tens of
thousands of instructions; [Large functions and the cache on disk](@ref) shortens that.

## Next steps

- A model written as Julia code: [From a Julia problem to generated code](@ref).
- A model saved by CasADi from Python, MATLAB or C++: [Load a model from another language](@ref).
- An integrator inside the graph, with its derivatives in a fit:
  [Estimate parameters through an integrator](@ref).
- Equations with algebraic states: [Model physical systems with differential-algebraic equations](@ref).
- What the package supports, and where it differs from CasADi: [Scope and differences from CasADi](@ref).
