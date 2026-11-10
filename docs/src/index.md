# CasADiCodegen.jl: Julia code generation for CasADi functions

CasADiCodegen.jl generates plain Julia code from the functions of
[CasADi](https://web.casadi.org/). A model and its derivatives, built with CasADi through
its Julia front end [CasADiNative.jl](https://github.com/casadi/CasADiNative.jl), become
Julia functions that run without CasADi and accept arrays, static arrays, sparse matrices
and ForwardDiff.jl dual numbers.

Features:

- Functions for the current session with [`build_function`](@ref), or a module on disk
  with [`generate_julia`](@ref), each in an out-of-place and an in-place form.
- Models saved by CasADi from Python, MATLAB or C++, generic Julia code traced into
  CasADi, and equations with time derivatives compiled by `CasADiCodegen.Models`.
- Calls to Julia functions inside a model, with user or automatic derivatives.
- Integrators, rootfinders and optimization solvers inside a model, solved by SciML
  packages in the generated code.
- CasADi functions as `ODEFunction`, `NonlinearFunction` and `OptimizationFunction`.

To evaluate a model with CasADi itself, in C or with its solvers, use the CasADi front
end and its C code generation instead.

## Installation

The package supports Julia 1.12 and 1.13:

```julia-repl
pkg> add https://github.com/Orbitology/CasADiCodegen.jl
```

CasADiNative.jl, a dependency, brings the binaries of CasADi 3.8.2.

## Contents

### Getting started

```@contents
Pages = ["getting_started.md"]
Depth = 2
```

### Tutorials

The tutorials solve the problems of the ModelingToolkit.jl documentation with CasADi
models and SciML solvers.

```@contents
Pages = [
    "tutorials/ode_modeling.md",
    "tutorials/nonlinear.md",
    "tutorials/optimization.md",
    "tutorials/sparse_jacobians.md",
    "tutorials/dae_models.md",
    "tutorials/initialization.md",
    "tutorials/events.md",
    "tutorials/bang_bang.md",
    "tutorials/stochastic.md",
    "tutorials/discrete.md",
    "tutorials/callable_parameters.md",
    "tutorials/bifurcation.md",
    "tutorials/traced_problems.md",
    "tutorials/linear_analysis.md",
    "tutorials/independent_variable.md",
    "tutorials/perturbation.md",
    "tutorials/attractors.md",
    "tutorials/parameter_estimation.md",
    "tutorials/dynamic_optimization.md",
    "tutorials/large_models.md",
]
Depth = 1
```

### Models

```@contents
Pages = [
    "howto/equations.md",
    "howto/load_model.md",
    "howto/symbolics.md",
    "howto/traced_models.md",
    "howto/callbacks.md",
    "howto/integrators.md",
    "howto/events.md",
    "howto/discontinuities.md",
    "howto/rootfinders.md",
    "howto/linearize.md",
]
Depth = 1
```

### Solvers and SciML

```@contents
Pages = [
    "howto/casadi_solvers.md",
    "howto/sciml_functions.md",
    "howto/autodiff_rules.md",
]
Depth = 1
```

### Large models and debugging

```@contents
Pages = [
    "howto/large_functions.md",
    "howto/inspect.md",
]
Depth = 1
```

### Concepts

```@contents
Pages = [
    "explanation/matrix_code.md",
    "explanation/integrator_derivatives.md",
    "explanation/performance.md",
    "explanation/scope.md",
]
Depth = 1
```

### Reference

```@contents
Pages = ["api.md", "api_sciml.md"]
Depth = 1
```

## Contributing

- Report bugs on the [issue tracker](https://github.com/Orbitology/CasADiCodegen.jl/issues).
- Run the tests with `julia --project=. -e 'using Pkg; Pkg.test()'`; one group with
  `Pkg.test(; test_args = ["mx"])`. On Julia 1.12, also pass
  `julia_args = ["--check-bounds=auto"]` (see [Keep bounds checks off in tests](@ref)).
- Build this manual with `julia --project=docs docs/make.jl` after
  `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`.

## Reproducibility

The examples run when the manual is built, with the packages below. Times in tables were
measured once, on Linux with an AMD Ryzen 9 5950X, unless a page says otherwise.

```@raw html
<details><summary>Julia and the packages of this build</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

## Acknowledgements

CasADi and CasADiNative.jl are the work of the CasADi developers: J. A. E. Andersson,
J. Gillis, G. Horn, J. B. Rawlings and M. Diehl, "CasADi: a software framework for
nonlinear optimization and optimal control", *Mathematical Programming Computation*
11 (2019), 1–36. The forms of the generated functions follow `build_function` of
Symbolics.jl.
