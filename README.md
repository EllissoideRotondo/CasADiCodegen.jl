# CasADiCodegen.jl

[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://orbitology.github.io/CasADiCodegen.jl/dev/)
[![Coverage](https://codecov.io/gh/Orbitology/CasADiCodegen.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Orbitology/CasADiCodegen.jl)

CasADiCodegen.jl is a package for generating plain Julia code from the functions of
[CasADi](https://web.casadi.org/). A model is built and differentiated with CasADi, through
[CasADiNative.jl](https://github.com/casadi/CasADiNative.jl), the official Julia front
end of CasADi, and CasADiCodegen.jl then emits the model and its derivatives as Julia
functions, which run without CasADi and accept Julia arrays, static arrays, sparse
matrices and the dual numbers of ForwardDiff.jl.

A model enters as a CasADi function built in Julia, as a file saved by CasADi from Python,
MATLAB or C++, as generic Julia code traced into CasADi, or as equations with time
derivatives. The generated code can call Julia functions and SciML solvers, and it serves
as the function of a SciML problem. The interface of `build_function` follows that of
[Symbolics.jl](https://docs.sciml.ai/Symbolics/stable/). To evaluate a model with CasADi
itself, in C or with its solvers, use the CasADi front end and its C code generation
instead.

## Installation

CasADiCodegen.jl supports Julia 1.12 and 1.13. Install it from GitHub:

```julia-repl
pkg> add https://github.com/Orbitology/CasADiCodegen.jl
```

The package depends on CasADiNative.jl, which brings the binaries of CasADi 3.8.2.

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

## Documentation

The [manual](https://orbitology.github.io/CasADiCodegen.jl/dev/) starts with
[Getting started](https://orbitology.github.io/CasADiCodegen.jl/dev/getting_started/).
A model must have a fixed number of inputs and outputs of fixed sizes, as every CasADi
function has; [Scope and differences from CasADi](https://orbitology.github.io/CasADiCodegen.jl/dev/explanation/scope/)
lists what is supported. The package is tested on Linux, macOS and Windows.

## Acknowledgements

CasADi is the work of the CasADi developers; see J. A. E. Andersson, J. Gillis,
G. Horn, J. B. Rawlings and M. Diehl, "CasADi: a software framework for nonlinear
optimization and optimal control", *Mathematical Programming Computation* 11
(2019), 1–36. Anthropic Claude Code assisted with code, tests and documentation.

## License

CasADiCodegen.jl is released under the MIT license; see [LICENSE](LICENSE).
