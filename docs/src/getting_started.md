# Getting started

In this tutorial, we write a small model with CasADi, generate Julia functions for
the model and its Jacobian, and call them with ordinary arrays, static arrays and a
sparse matrix. We then write the same functions to a file and compare the forms of
the generated code and what each one costs. The examples need CasADiCodegen.jl,
StaticArrays.jl and the standard library SparseArrays. The file that we write in
[Write the code to a file](@ref) also needs StaticArraysCore.jl and NaNMath.jl in the
environment.

## Write the model

CasADi builds a model from symbols. An `SX` expression is a graph of scalar
operations, an `MX` expression is a graph of matrix operations and of calls to other
functions, and a `DM` is a matrix of numbers. A `Function` of CasADi maps a list of
inputs to a list of outputs, and CasADi differentiates it from its graph.

`using CasADiCodegen` loads the official CasADi front end, the SWIG interface that ships
with CasADi 3.8.1. Its calls are those of Python, and the
[CasADi documentation](https://web.casadi.org/docs/) describes them. Four rules
apply in Julia:

- The front end exports the common names: `SX`, `MX`, `DM`, `jacobian`, `gradient`,
  `hessian`, `vertcat`, `horzcat`, `substitute` and others. A number or a Julia array of
  numbers is accepted as the value of a symbol in `substitute` and as the point of the
  Taylor series `CasADiCodegen.C.taylor` and `CasADiCodegen.C.mtaylor`.
- Each other function of CasADi is in the module `CasADiCodegen.C`, for example
  `CasADiCodegen.C.sumsqr` and `CasADiCodegen.C.nlpsol`.
- `CasadiFunction` makes a function of CasADi, because a bare `Function` is the abstract
  type of Julia. `CasADiCodegen.Function` is another name for the same type.
- Elements have the indices of Julia: `x[1]` is the first one. Methods of CasADi that
  take an index of an input or an output keep the indices of CasADi, which start at 0.

```@example start
using CasADiCodegen

x = SX.sym("x", 3)
y = [sin(x[1]) * x[2]; x[3]^2 / (1 + x[1]^2)]
f = CasadiFunction("f", [x], [y, jacobian(y, x)])
nothing # hide
```

We have a function `f` with one input, the vector `x` of three elements, and two
outputs: the value `y` and its 2 by 3 Jacobian.

## Generate functions for this session

[`build_function`](@ref) returns a pair: an out-of-place and an in-place function.
We ask for compiled functions with `expression = Val{false}` and call the first one:

```@example start
g, g! = build_function(f; expression = Val{false})
value, J = g([0.3, 1.7, -0.4])
J
```

The out-of-place function takes one array per input and returns one array per
output. CasADi has no vector type, so every output is a matrix, which the function
returns as a `Vector` when it has one column and as a `Matrix` otherwise; a scalar is a
vector of one element. With `vector_columns = false` it keeps the matrix shape for every
output, also for a column. With one output the function returns the array itself.
[Choose a form](@ref) lists the other forms.

The in-place function takes a tuple of outputs and a tuple of inputs. We give it
arrays of the right sizes:

```@example start
out = (zeros(2), zeros(2, 3))
g!(out, ([0.3, 1.7, -0.4],))
out[1]
```

## Avoid allocation with static arrays

When the first argument is a static array, the out-of-place function returns static
arrays and allocates no memory. The example loads StaticArrays.jl itself, because
CasADiCodegen.jl does not export its names.

```@example start
using StaticArrays

value, J = g(SVector(0.3, 1.7, -0.4))
typeof(J)
```

`force_SA = true` makes the out-of-place function return static arrays for any
argument. Static arrays suit small outputs only: when the outputs have more than 1024
elements in all, such as the Jacobian of a model of more than 32 states, the method with
static arrays compiles more slowly, and for a dense Jacobian it also runs more slowly
than the method with ordinary arrays. The out-of-place function then
returns ordinary arrays, with a warning at the first call with a static array, and
`force_SA = true` throws an `ArgumentError`. The in-place
function also accepts `MVector` and `MMatrix`.

## Fill a sparse Jacobian

The in-place function also fills a `SparseMatrixCSC` that has the pattern of the
output. We make one from the sparsity pattern of the Jacobian with
`SparseMatrixCSC(sparsity)`; the function then writes the stored values only, and
it rejects a matrix with another pattern.

```@example start
using SparseArrays

Js = SparseMatrixCSC(CasADiCodegen.C.sparsity_out(f, 1))   # pattern of the second output: CasADi counts from 0
g!((zeros(2), Js), ([0.3, 1.7, -0.4],))
Js
```

A structural zero is an element that the sparsity pattern leaves out. A structural
zero of an output is zero, and that of an input is not read.

## Write the code to a file

[`generate_julia`](@ref) writes the functions to one module. For a model such as this
one, the module does not need CasADiCodegen.jl to run. We write it to a temporary
file and include it:

```@example start
path = joinpath(mktempdir(), "f.jl")
generate_julia(f, path; module_name = :F)
include(path)
F.f([0.3, 1.7, -0.4])[1]
```

`F.f!` is the in-place function, with the arguments of `g!` above:

```@example start
F.f!(out, ([0.3, 1.7, -0.4],))
out[2]
```

The keyword `module_name` names the module; by default the name is that of `f` with
its first letter in upper case. The module holds `F.f`, `F.f!` and `F.f_nonzeros!`
(see [Choose a form](@ref)) and constants that describe the function:

| Constant | Content |
|---|---|
| `NAME`, `NAMES_IN`, `NAMES_OUT` | Name of the function, and tuples of the names of its inputs and outputs |
| `NNZ_IN`, `NNZ_OUT` | Tuples of the numbers of structural nonzeros of the inputs and outputs |
| `NWORK` | Length of the work vector |
| `SPARSITY_IN`, `SPARSITY_OUT` | Tuples of sparsity patterns, each `(nrow, ncol, colptr, rowval)` with indices from 1 |
| `DEPENDENCIES` | Sorted tuple of the names of the packages that the module imports |

The module imports StaticArraysCore.jl for the method with static arrays, NaNMath.jl
unless `nanmath = false`, and SparseArrays whenever an input or an output has
structural zeros, as the Jacobian of this model has, even when every caller passes
dense arrays; SpecialFunctions.jl, LinearAlgebra and the packages of Julia functions
in the model are imported when the code calls them. Each package must be in the
environment that loads the file, and a package that includes the file must list the
packages of `DEPENDENCIES` as its own dependencies:

```@example start
F.DEPENDENCIES
```

A model with an integrator, a rootfinder or an optimization solver gives a module
that also loads CasADiCodegen.jl, SciMLBase.jl and the package of the algorithm.

## Choose a form

| Call | Result | Needs CasADiCodegen.jl to run |
|---|---|---|
| `build_function(f)` | An `Expr` with the definitions, for `eval` or a macro | Yes |
| `build_function(f; expression = Val{false})` | Compiled functions for the current session | Yes |
| `build_function(f; expression = Val{false}, cache = true)` | The same, compiled once and loaded in later sessions | Yes |
| `generate_julia(f, path)` | A file with one module, for `include` or for a package | No, unless the model holds a SciML solver |

With `nonzeros = true`, `build_function` returns the single function
`(res, arg) -> res`, which evaluates the function as CasADi does. `arg[i]` is a vector
of the structural nonzeros of input `i`, and the function writes those of output `j`
into the vector `res[j]`; the nonzeros are in column-major order. `generate_julia`
writes this form as the function `<name>_nonzeros!` of the module.
