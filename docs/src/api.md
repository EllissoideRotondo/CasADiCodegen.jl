# API reference

This page and [SciML problems and functions](@ref) list the functions and types of
CasADiCodegen.jl, grouped by task. The package exports `build_function`,
`generate_julia`, `CasadiFunction`, `casadi_function` and `casadi`; the other names below
are public and need the prefix `CasADiCodegen.`, except `callback`, `integrator` and
`rootfinder`, which are functions of the front end to which the package adds methods.

## The CasADi front end

`using CasADiCodegen` loads CasADiNative.jl, the Julia front end of CasADi 3.8.2, whose
calls follow the [CasADi documentation](https://web.casadi.org/docs/) for Python, with
these differences:

- `SX`, `MX`, `DM`, `jacobian`, `gradient`, `hessian`, `substitute`, `nlpsol`,
  `integrator`, `rootfinder`, `callback` and the `Opti` functions `Opti`, `variable`,
  `parameter`, `minimize`, `subject_to`, `bounded`, `set_initial`, `solve` and `value` are
  exported. Every other function is in the exported module `casadi`, as in Python:
  `casadi.set_value`, `casadi.sumsqr`.
- `CasadiFunction` makes a CasADi function, since `Function` is a Julia type; the type of
  the result is `CasADiCodegen.Function`.
- Operations have their Julia meaning: `*` is the matrix product, `sin.(x)` and `x .^ 2`
  act on each element, and `[a; b]` concatenates.
- Indices start at 1, except in CasADi methods that take the index of an input or output,
  such as `casadi.sparsity_out(f, 0)`.
- A chained comparison such as `-2 <= w[1] <= 2` is not a constraint, since Julia
  evaluates it with `&&`; write `bounded(-2, w[1], 2)`.

## Code generation

```@docs
CasadiFunction
CasADiCodegen.casadi_function
build_function
generate_julia
```

## Models written as equations

The submodule `CasADiCodegen.Models` exports these names, which `using CasADiCodegen.Models`
gives; `SciMLBase.ODEProblem` of a `System` is on [SciML problems and functions](@ref).

```@docs
CasADiCodegen.Models
CasADiCodegen.Models.Differential
CasADiCodegen.Models.Equation
CasADiCodegen.Models.System
CasADiCodegen.Models.unknowns
CasADiCodegen.Models.algebraic
CasADiCodegen.Models.parameters
CasADiCodegen.Models.observed
CasADiCodegen.Models.equations
CasADiCodegen.Models.invariants
CasADiCodegen.Models.differentiation_index
Base.Dict(::CasADiCodegen.Models.System)
```

## Julia models traced into CasADi

```@docs
CasADiCodegen.Traced
CasADiCodegen.traced
CasADiCodegen.untraced
CasADiCodegen.traced_function(::AbstractString, ::Any, ::AbstractArray{<:Real})
CasADiCodegen.traced_function(::AbstractString, ::Union{SciMLBase.ODEProblem,SciMLBase.NonlinearProblem})
CasADiCodegen.traced_problem(::AbstractString, ::SciMLBase.ODEProblem)
Base.ifelse(::CasADiCodegen.Traced, ::Real, ::Real)
```

## Julia functions inside a model

```@docs
CasADiCodegen.callback(::Any, ::AbstractString, ::Tuple, ::Tuple)
```

## SciML solvers inside a model

`integrator` and `rootfinder` are the functions of the CasADi front end. With a
solver name such as `"cvodes"` or `"kinsol"`, they make a solver object of CasADi
itself, as in Python. The methods below take a SciML algorithm instead; they exist
once SciMLBase.jl is loaded, which each SciML solver package does.

```@docs
CasADiCodegen.integrator(::AbstractString, ::SciMLBase.AbstractODEAlgorithm, ::Any, ::Real, ::Real)
CasADiCodegen.rootfinder(::AbstractString, ::SciMLBase.AbstractNonlinearAlgorithm, ::CasADiCodegen.Function)
```

## Solvers of CasADi in generated code

```@docs
CasADiCodegen.Solvers
CasADiCodegen.Solver
```

## Linearization, the order of equations and index reduction

```@docs
CasADiCodegen.linearization
CasADiCodegen.linearization_function
CasADiCodegen.block_triangular_form
CasADiCodegen.reduce_index
```

## Discontinuous models

```@docs
CasADiCodegen.nonsmooth_operations
CasADiCodegen.smooth_step
CasADiCodegen.smooth_sign
CasADiCodegen.smooth_abs
CasADiCodegen.smooth_max
CasADiCodegen.smooth_min
CasADiCodegen.smoothed
```

## Automatic differentiation rules and sparse matrices

```@docs
CasADiCodegen.differentiable
SparseArrays.SparseMatrixCSC(::CasADiCodegen.Sparsity)
```

## Inspection

```@docs
CasADiCodegen.instruction_histogram
CasADiCodegen.nan_trace
```
