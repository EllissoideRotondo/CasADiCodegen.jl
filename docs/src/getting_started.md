# Getting started

We write a small model with CasADi, generate Julia functions for it and its Jacobian,
and call them with arrays, static arrays and a sparse matrix; we then write the same
functions to a file.

## Write the model

`using CasADiCodegen` loads [CasADiNative.jl](https://github.com/casadi/CasADiNative.jl),
the Julia front end of CasADi 3.8.2, whose calls follow the
[CasADi documentation](https://web.casadi.org/docs/) for Python, with these differences:

- `SX`, `MX`, `DM`, `jacobian`, `gradient`, `hessian`, `substitute`, `nlpsol` and the
  `Opti` functions `Opti`, `variable`, `parameter`, `minimize`, `subject_to`, `bounded`,
  `set_initial`, `solve` and `value` are exported. Every other function is in the
  exported module `casadi`, as in Python: `casadi.set_value`, `casadi.sumsqr`.
- `CasadiFunction` makes a CasADi function, since `Function` is a Julia type.
- Operations have their Julia meaning: `*` is the matrix product, `sin.(x)` and `x .^ 2`
  act on each element, and `[a; b]` concatenates.
- Indices start at 1, except in CasADi methods that take the index of an input or output.

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
elements over all outputs of one function, such as the Jacobian of 32 states. Above that
limit, the function returns ordinary arrays with a warning, and an explicit request for
static arrays throws an `ArgumentError`: `force_SA = true` here, or `inplace = false` in a
[SciML constructor](@ref "Give a CasADi function to a SciML problem"), which applies the
limit to each generated array.

## Fill a sparse Jacobian

The in-place function also fills a `SparseMatrixCSC` with the sparsity pattern of the
output, and rejects one with another pattern:

```@example start
using SparseArrays

Js = SparseMatrixCSC(casadi.sparsity_out(f, 1))   # pattern of the second output: CasADi counts from 0
g!((zeros(2), Js), ([0.3, 1.7, -0.4],))
Js
```

## Write the code to a file

[`generate_julia`](@ref) writes the functions to a module that runs without
CasADiCodegen.jl:

```@example start
path = joinpath(mktempdir(), "f.jl")
generate_julia(f, path; module_name = :F)
include(path)
F.f([0.3, 1.7, -0.4])[1]
```

```@example start
F.f!(out, ([0.3, 1.7, -0.4],))
out[2]
```

The module also holds `F.f_nonzeros!` (see [Choose a form](@ref)) and these constants:

| Constant | Content |
|---|---|
| `NAME`, `NAMES_IN`, `NAMES_OUT` | Names of the function, its inputs and outputs |
| `NNZ_IN`, `NNZ_OUT` | Numbers of structural nonzeros of inputs and outputs |
| `NWORK` | Length of the work vector |
| `SPARSITY_IN`, `SPARSITY_OUT` | Patterns `(nrow, ncol, colptr, rowval)`, indices from 1 |
| `DEPENDENCIES` | Packages that the module imports |

Each package of `DEPENDENCIES` must be in the environment that loads the file:

```@example start
F.DEPENDENCIES
```

A model with an integrator, a rootfinder or a solver also needs CasADiCodegen.jl,
SciMLBase.jl and the package of the algorithm.

## Choose a form

| Call | Result | Needs CasADiCodegen.jl to run |
|---|---|---|
| `build_function(f)` | An `Expr`, for `eval` or a macro | Yes |
| `build_function(f; expression = Val{false})` | Compiled functions | Yes |
| `build_function(f; expression = Val{false}, cache = true)` | The same, cached for later sessions | Yes |
| `generate_julia(f, path)` | A module in a file | No, unless the model holds a SciML solver |

`nonzeros = true` gives one function `(res, arg) -> res` on the structural nonzeros of
each input and output, in column-major order, as CasADi evaluates it; `generate_julia`
writes it as `<name>_nonzeros!`.
