# SciML problems and functions

The methods of this page exist once SciMLBase.jl is loaded, which each SciML solver
package does, and the bifurcation problem once BifurcationKit.jl is loaded too. Their
names follow the conventions of the [API reference](@ref).

## SciML problems from solvers and models of CasADi

These constructors make a SciML problem from a solver object of CasADi, from a
`DaeBuilder` or from an `Opti`, with generated code for the model and its derivatives.

```@docs
SciMLBase.ODEProblem(::CasADiCodegen.Function, ::AbstractVector{<:Real}, ::AbstractVector{<:Real})
SciMLBase.ODEProblem(::CasADiCodegen.C.DaeBuilder, ::Union{Nothing,Tuple{Real,Real}})
SciMLBase.NonlinearProblem(::CasADiCodegen.Function, ::AbstractVector{<:Real}, ::AbstractVector{<:Real}...)
SciMLBase.SDEProblem(::CasADiCodegen.Function, ::CasADiCodegen.Function, ::Any, ::Any)
SciMLBase.DiscreteProblem(::CasADiCodegen.Function, ::Any, ::Any)
SciMLBase.ImplicitDiscreteProblem(::CasADiCodegen.Function, ::Any, ::Tuple)
SciMLBase.OptimizationProblem(::CasADiCodegen.Function, ::AbstractVector{<:Real}, ::AbstractVector{<:Real})
SciMLBase.OptimizationProblem(::CasADiCodegen.C.Opti)
```

## Functions of SciML problems

```@docs
SciMLBase.ODEFunction(::CasADiCodegen.Function)
SciMLBase.SDEFunction(::CasADiCodegen.Function, ::CasADiCodegen.Function)
SciMLBase.DiscreteFunction(::CasADiCodegen.Function)
SciMLBase.ImplicitDiscreteFunction(::CasADiCodegen.Function)
SciMLBase.NonlinearFunction(::CasADiCodegen.Function)
SciMLBase.OptimizationFunction(::CasADiCodegen.Function, ::Union{Nothing,CasADiCodegen.Function})
CasADiCodegen.initialization_problem(::SciMLBase.ODEProblem)
```

## Bifurcation problems

```@docs
BifurcationKit.BifurcationProblem(::CasADiCodegen.Function, ::Any, ::Any, ::Union{Symbol,Integer})
```

## Constraints of an Opti object

```@docs
CasADiCodegen.bounded
```

