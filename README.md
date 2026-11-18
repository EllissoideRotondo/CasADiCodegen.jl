# CasADiCodegen.jl

[![Docs: dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://orbitology.github.io/CasADiCodegen.jl/dev/)
[![Coverage](https://codecov.io/gh/Orbitology/CasADiCodegen.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Orbitology/CasADiCodegen.jl)

CasADiCodegen.jl turns [CasADi](https://web.casadi.org/) functions into plain Julia
code. You build a model and its derivatives with CasADi, through
[CasADiNative.jl](https://github.com/casadi/CasADiNative.jl), the official Julia front
end of CasADi, and get Julia functions that run without CasADi. They accept Julia
arrays, static arrays, sparse matrices and ForwardDiff.jl dual numbers.

A model can be a CasADi function built in Julia, a file saved by CasADi from Python,
MATLAB or C++, generic Julia code traced into CasADi, or equations with time
derivatives. The generated code can call Julia functions and SciML solvers, and it can
serve as the function of a SciML problem. `build_function` works like its namesake in
[Symbolics.jl](https://docs.sciml.ai/Symbolics/stable/). To run a model with CasADi
itself, in C or with CasADi's solvers, use the CasADi front end and its C code
generation instead.

## Install from GitHub

CasADiCodegen.jl supports Julia 1.12 and 1.13:

```julia-repl
pkg> add https://github.com/Orbitology/CasADiCodegen.jl
```

The dependency CasADiNative.jl brings the CasADi 3.8.2 binaries.

## Generate a function and its gradient

We define a function of two variables and its gradient with CasADi, then build Julia
functions that evaluate both:

```julia
using CasADiCodegen

x = SX.sym("x", 2)
y = x[1]^2 + sin(x[2])
f = CasadiFunction("f", [x], [y, gradient(y, x)])

f_julia, f_julia! = build_function(f; expression = Val{false})

x0 = [2.0, 0.0]
value, derivative = f_julia(x0)

value       # [4.0]
derivative  # [4.0, 1.0]
```

- `using CasADiCodegen` loads the CasADi front end: `SX`, `MX`, `gradient`, `jacobian`
  and the other common names.
- `CasadiFunction` makes a CasADi function, since `Function` is a Julia type.
- `f_julia` allocates its outputs. `f_julia!` writes into arrays that you pass.

<details>
<summary>Save the same functions to a file</summary>

`generate_julia` writes a module that runs without CasADiCodegen.jl. It is named `F`
after the function `f` by default:

```julia
path = "f.jl"
generate_julia(f, path)

include(path)
F.f(x0)
```

</details>

## Where to read next

The [manual](https://orbitology.github.io/CasADiCodegen.jl/dev/) starts with
[Getting started](https://orbitology.github.io/CasADiCodegen.jl/dev/getting_started/).
A model must have a fixed number of inputs and outputs of fixed sizes, as every CasADi
function has.
[Scope and differences from CasADi](https://orbitology.github.io/CasADiCodegen.jl/dev/explanation/scope/)
lists what is supported. The package is tested on Linux, macOS and Windows.

## Credits

CasADi is the work of the CasADi developers; see J. A. E. Andersson, J. Gillis,
G. Horn, J. B. Rawlings and M. Diehl, "CasADi: a software framework for nonlinear
optimization and optimal control", *Mathematical Programming Computation* 11
(2019), 1–36. Anthropic Claude Code assisted with code, tests and documentation.

## License

CasADiCodegen.jl is released under the MIT license; see [LICENSE](LICENSE).
