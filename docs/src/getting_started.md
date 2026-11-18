# Getting started

We write a small model with CasADi and generate Julia functions for it and its Jacobian.
We call them with arrays, static arrays and a sparse matrix, differentiate them, solve an
ODE with a model of the same kind, and save the functions to a file.

## Define a function and its Jacobian with CasADi

`using CasADiCodegen` loads [CasADiNative.jl](https://github.com/casadi/CasADiNative.jl),
the Julia front end of CasADi 3.8.2. Its calls follow the
[CasADi documentation](https://web.casadi.org/docs/) for Python, with the differences
listed in [Differences from CasADi's Python API](@ref). `CasadiFunction` makes a CasADi
function, since
`Function` is a Julia type:

```@example start
using CasADiCodegen

x = SX.sym("x", 3)
y = [sin(x[1]) * x[2]; x[3]^2 / (1 + x[1]^2)]
f = CasadiFunction("f", [x], [y, jacobian(y, x)])
nothing # hide
```

`f` has one input, `x`, and two outputs: `y` and its 2 by 3 Jacobian.

## Generate Julia functions with `build_function`

[`build_function`](@ref) returns an out-of-place and an in-place function:

```@example start
g, g! = build_function(f; expression = Val{false})

x0 = [0.3, 1.7, -0.4]
value, J = g(x0)
J
```

An output with one column comes back as a `Vector`, any other as a `Matrix`
(`vector_columns = false` keeps the matrix shape). The in-place function takes a tuple of
outputs and a tuple of inputs:

```@example start
out = (zeros(2), zeros(2, 3))
g!(out, (x0,))
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

`force_SA = true` returns static arrays for any input.

!!! details "Size limit of static arrays: 1024 elements"
    Static arrays hold at most 1024 elements over all outputs of one function together.
    Above that, a static input gives ordinary arrays with a warning, and
    `force_SA = true` throws an `ArgumentError`.

## Fill a sparse Jacobian in place

The in-place function also fills a `SparseMatrixCSC` that has the sparsity pattern of
the output, and rejects one with another pattern:

```@example start
using SparseArrays

pattern = casadi.sparsity_out(f, 1)   # second output: CasADi counts from 0
Js = SparseMatrixCSC(pattern)
g!((zeros(2), Js), (x0,))
Js
```

## Differentiate the generated code with ForwardDiff

The generated code accepts ForwardDiff.jl dual numbers. Here they reproduce the Jacobian
that CasADi built:

```@example start
import ForwardDiff

ForwardDiff.jacobian(x -> g(x)[1], x0) ≈ J
```

## Use a CasADi function as the right side of an ODE

A CasADi function of the state, the parameters and the time can be the right side of a
SciML problem, with generated code for its Jacobians. The states and parameters take the
names of the scalar symbols, and each result after the first is an observed variable named
after that result:

```@example start
using OrdinaryDiffEqTsit5: Tsit5
import SciMLBase

u = SX.sym("u")
τ = SX.sym("τ")
t = SX.sym("t")
error_u = sin(t) - u                 # the input sin(t) minus the state
du = error_u / τ
lag = CasadiFunction("lag", [u, τ, t], [du, error_u], ["u", "τ", "t"], ["du", "error"])

u0 = [0.0]
tspan = (0.0, 10.0)
p0 = [0.5]
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(lag), u0, tspan, p0)
solution = SciMLBase.solve(problem, Tsit5())

slower = SciMLBase.remake(problem; p = [:τ => 2.0])
solution[:error][end], slower.p
```

A new parameter value needs no new code.
[Use the generated code with SciML packages](@ref) covers the other problems.

## Save the code to a file with `generate_julia`

[`generate_julia`](@ref) writes the functions to a module that runs without
CasADiCodegen.jl. Each package in `F.DEPENDENCIES` must be in the environment that loads
the file:

```@example start
path = joinpath(mktempdir(), "f.jl")
generate_julia(f, path)                       # the module F, after the function f
include(path)
F.f(x0)[1], F.DEPENDENCIES
```

## Compare the ways to generate code

| Call | Result | Needs CasADiCodegen.jl to run |
|---|---|---|
| `build_function(f)` | An `Expr`, for `eval` or a macro | Yes |
| `build_function(f; expression = Val{false})` | Compiled functions | Yes |
| `build_function(f; expression = Val{false}, cache = true)` | The same, cached for later sessions | Yes |
| `generate_julia(f, path)` | A module in a file | No, unless the model holds a SciML solver |

The first call of a function compiles it, which takes seconds for a graph of tens of
thousands of operations; [Speed up compilation of large models](@ref) shortens that.

## Read next, by kind of model

- A model written as Julia code: [Generate code from an existing SciML problem](@ref).
- A model saved by CasADi from Python, MATLAB or C++:
  [Load a model from another language](@ref).
- An integrator inside the graph, with its derivatives in a fit:
  [Fit ODE parameters to data in three ways](@ref).
- Equations with algebraic states:
  [Model a circuit and a pendulum as DAEs](@ref).
- What the package supports, and where it differs from CasADi:
  [Scope and differences from CasADi](@ref).
