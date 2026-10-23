# API reference

This page lists the functions and types of CasADiCodegen.jl, grouped by task. The
package exports `build_function`, `generate_julia` and `CasadiFunction`, and `using CasADiCodegen`
also gives the names that the front end of CasADi exports, such as `SX`, `MX`, `DM`,
`Opti`, `jacobian`, `gradient` and `vertcat`. The whole front end, with the functions of
the Python API of CasADi, is the module `CasADiCodegen.C`, and a CasADi function is a
`CasADiCodegen.Function`, which `CasadiFunction` makes. The other names of this page need
the prefix `CasADiCodegen.`, except `callback`, `integrator` and `rootfinder`, which are
exported functions of the front end to which the package adds methods; the entries below
write the prefix for all of them.

## Code generation

```@docs
CasadiFunction
build_function
generate_julia
```

## Julia models traced into CasADi

```@docs
CasADiCodegen.Traced
CasADiCodegen.traced
CasADiCodegen.untraced
CasADiCodegen.traced_function(::AbstractString, ::Any, ::AbstractArray{<:Real})
CasADiCodegen.traced_function(::AbstractString, ::Union{SciMLBase.ODEProblem,SciMLBase.NonlinearProblem})
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

## SciML problems from solvers and models of CasADi

These constructors make a SciML problem from a solver object of CasADi, from a
`DaeBuilder` or from an `Opti`, with generated code for the model and its derivatives.

```@docs
SciMLBase.ODEProblem(::CasADiCodegen.Function, ::AbstractVector{<:Real}, ::AbstractVector{<:Real})
SciMLBase.ODEProblem(::CasADiCodegen.C.DaeBuilder, ::Union{Nothing,Tuple{Real,Real}})
SciMLBase.NonlinearProblem(::CasADiCodegen.Function, ::AbstractVector{<:Real}, ::AbstractVector{<:Real}...)
SciMLBase.OptimizationProblem(::CasADiCodegen.Function, ::AbstractVector{<:Real}, ::AbstractVector{<:Real})
SciMLBase.OptimizationProblem(::CasADiCodegen.C.Opti)
```

## Functions of SciML problems

```@docs
SciMLBase.ODEFunction(::CasADiCodegen.Function)
SciMLBase.NonlinearFunction(::CasADiCodegen.Function)
SciMLBase.OptimizationFunction(::CasADiCodegen.Function, ::Union{Nothing,CasADiCodegen.Function})
```

## Constraints of an Opti object

```@docs
CasADiCodegen.bounded
```

## Linearization and the order of equations

```@docs
CasADiCodegen.linearization
CasADiCodegen.linearization_function
CasADiCodegen.block_triangular_form
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
