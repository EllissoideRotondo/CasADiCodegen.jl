# CasADiCodegen.jl

[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://AstroGuild.github.io/CasADiCodegen.jl/dev/)
[![Coverage](https://codecov.io/gh/AstroGuild/CasADiCodegen.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/AstroGuild/CasADiCodegen.jl)

CasADiCodegen.jl is a package for generating plain Julia code from the functions of
[CasADi](https://web.casadi.org/), the framework for symbolic modelling and
algorithmic differentiation. A model is built and differentiated with CasADi, through
the Julia front end of CasADi that the package includes, and CasADiCodegen.jl then
emits the model and its derivatives as Julia functions, which run without CasADi and
accept Julia arrays, static arrays, sparse matrices and the dual numbers of
ForwardDiff.jl.

Features include:

- Julia functions for the current session with `build_function`, or a Julia module
  on disk with `generate_julia`, each with an out-of-place and an in-place form.
- Models that were built in Python, MATLAB or C++ and saved by CasADi, and models
  written as generic Julia code, which a number type traces into CasADi.
- Calls to Julia functions inside a model, with derivatives from Julia functions or
  from an automatic differentiation backend.
- Integrators, rootfinders and optimization solvers inside a model, which the
  generated code solves with the SciML packages.
- CasADi functions as the functions of SciML problems: `ODEFunction`,
  `NonlinearFunction` and `OptimizationFunction`.

CasADiCodegen.jl is meant for a model that is defined with CasADi and must run as
Julia code, in a simulation, in an optimization with a Julia solver, or inside a SciML
problem. A model that is written as generic Julia code enters CasADi by tracing: the
number type `CasADiCodegen.Traced` records the operations of the Julia function as a
CasADi graph, which CasADi differentiates and the package emits. The interface of
`build_function` follows that of [Symbolics.jl](https://docs.sciml.ai/Symbolics/stable/).

## Installation

CasADiCodegen.jl supports Julia 1.12 and 1.13. Install it from GitHub:

```julia-repl
pkg> add https://github.com/AstroGuild/CasADiCodegen.jl
```

The binaries of CasADi 3.8.1 come with the package.

## Example

We define a function of two variables and its gradient with CasADi, and then build
Julia functions that evaluate both:

```julia
using CasADiCodegen

x = SX.sym("x", 2)
y = x[1]^2 + sin(x[2])
f = CasadiFunction("f", [x], [y, gradient(y, x)])

f_julia, f_julia! = build_function(f; expression = Val{false})
value, derivative = f_julia([2.0, 0.0])

value       # [4.0]
derivative  # [4.0, 1.0]
```

- `using CasADiCodegen` loads the front end of CasADi: `SX`, `MX`, `gradient`,
  `jacobian` and the other common names.
- `CasadiFunction` makes a function of CasADi. A bare `Function` is a type of Julia.
- `f_julia` allocates its outputs. `f_julia!` writes into arrays that the caller gives.
- `generate_julia(f, path)` writes the same functions to a Julia module, named `F`
  after the function `f` by default.

## Tutorials and Documentation

For information on using the package, see the
[documentation](https://AstroGuild.github.io/CasADiCodegen.jl/dev/). It starts
with a [tutorial](https://AstroGuild.github.io/CasADiCodegen.jl/dev/getting_started/)
that builds a model, generates Julia functions for it and compares the forms of the
generated code. Tutorials then solve the problems of the ModelingToolkit.jl
documentation with CasADi models, from ODE and DAE models to nonlinear systems, static
and dynamic optimization, events and parameter estimation. Its how-to guides cover models from other languages, Julia
models traced into CasADi, calls to Julia code, integrators, events and rootfinders,
the replacement of the solvers of CasADi by SciML solvers, CasADi functions in SciML
problems, and large models; the concept pages explain the matrix code, the
derivatives of integrators, the cost of the generated code and the differences from
CasADi, and the API reference ends the manual.

## Limits

- A model must have a fixed number of inputs and outputs of fixed sizes, as every
  CasADi function has.
- Some CasADi nodes have no generated code, such as cone constraints and Functional
  Mock-up Units.
- The package is tested on Linux; macOS is not tested. It does not load on Windows,
  because the Windows build of the CasADi front end lacks functions that it needs.

The page [Scope and differences from CasADi](https://AstroGuild.github.io/CasADiCodegen.jl/dev/explanation/scope/)
of the manual lists all limits.

## Acknowledgements

CasADi is the work of the CasADi developers; see J. A. E. Andersson, J. Gillis,
G. Horn, J. B. Rawlings and M. Diehl, "CasADi: a software framework for nonlinear
optimization and optimal control", *Mathematical Programming Computation* 11
(2019), 1–36. Anthropic Claude Code assisted with code, tests and documentation.

## License

CasADiCodegen.jl is released under the MIT license; see [LICENSE](LICENSE).
