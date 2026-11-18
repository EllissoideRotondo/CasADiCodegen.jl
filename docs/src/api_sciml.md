# SciML problems and functions

The methods on this page exist once SciMLBase.jl is loaded, which every SciML solver
package does. The bifurcation problem also needs BifurcationKit.jl. Names follow the
conventions of the [API reference](@ref).

## Make a SciML problem from a CasADi model or solver

These constructors make a SciML problem, with generated code for the model and its
derivatives, from a CasADi function or solver object, a `DaeBuilder`, a
`CasADiCodegen.Models.System` or an `Opti`.

```@docs
SciMLBase.ODEProblem(::CasADiCodegen.Function, ::AbstractVector{<:Real}, ::AbstractVector{<:Real})
SciMLBase.ODEProblem(::CasADiCodegen.casadi.DaeBuilder, ::Union{Nothing,Tuple{Real,Real}})
SciMLBase.ODEProblem(::CasADiCodegen.Models.System, ::Any, ::Tuple{Real,Real})
SciMLBase.NonlinearProblem(::CasADiCodegen.Function, ::AbstractVector{<:Real}, ::AbstractVector{<:Real}...)
SciMLBase.SDEProblem(::CasADiCodegen.Function, ::CasADiCodegen.Function, ::Any, ::Any)
SciMLBase.DiscreteProblem(::CasADiCodegen.Function, ::Any, ::Any)
SciMLBase.ImplicitDiscreteProblem(::CasADiCodegen.Function, ::Any, ::Tuple)
SciMLBase.OptimizationProblem(::CasADiCodegen.Function, ::AbstractVector{<:Real}, ::AbstractVector{<:Real})
SciMLBase.OptimizationProblem(::CasADiCodegen.casadi.Opti)
```

## Make the function of a SciML problem

```@docs
SciMLBase.ODEFunction(::CasADiCodegen.Function)
SciMLBase.SDEFunction(::CasADiCodegen.Function, ::CasADiCodegen.Function)
SciMLBase.DiscreteFunction(::CasADiCodegen.Function)
SciMLBase.ImplicitDiscreteFunction(::CasADiCodegen.Function)
SciMLBase.NonlinearFunction(::CasADiCodegen.Function)
SciMLBase.OptimizationFunction(::CasADiCodegen.Function, ::Union{Nothing,CasADiCodegen.Function})
CasADiCodegen.initialization_problem(::SciMLBase.ODEProblem)
CasADiCodegen.natural_continuation(::SciMLBase.AbstractNonlinearProblem, ::Any, ::Any)
```

## Make a BifurcationKit problem

```@docs
BifurcationKit.BifurcationProblem(::CasADiCodegen.Function, ::Any, ::Any, ::Union{Symbol,Integer})
```
