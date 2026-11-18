# CasADiCodegen.jl: Julia code generation for CasADi functions

CasADiCodegen.jl turns [CasADi](https://web.casadi.org/) functions into plain Julia
code. You build a model and its derivatives with CasADi, through its Julia front end
[CasADiNative.jl](https://github.com/casadi/CasADiNative.jl), and get Julia functions that
run without CasADi. They accept arrays, static arrays, sparse matrices and ForwardDiff.jl
dual numbers.

A model can be a CasADi function built in Julia, a file saved by CasADi from Python,
MATLAB or C++, generic Julia code traced into CasADi, or equations compiled by
`CasADiCodegen.Models`. The generated code can call Julia functions and SciML solvers, and
it can serve as the function of a SciML problem.

To run a model with CasADi itself, in C or with CasADi's solvers, use the CasADi front end
and its C code generation instead.

## Install from GitHub

The package supports Julia 1.12 and 1.13:

```julia-repl
pkg> add https://github.com/Orbitology/CasADiCodegen.jl
```

The dependency CasADiNative.jl brings the CasADi 3.8.2 binaries.

Start with [Getting started](@ref); [Scope and differences from CasADi](@ref) lists what
the package supports.

## Report bugs, run the tests and build this manual

- Report bugs on the [issue tracker](https://github.com/Orbitology/CasADiCodegen.jl/issues).
- Run the tests with `julia --project=. -e 'using Pkg; Pkg.test()'`, or one group with
  `Pkg.test(; test_args = ["mx"])`. On Julia 1.12, also pass
  `julia_args = ["--check-bounds=auto"]` (see [Keep bounds checks off in tests](@ref)).
- Build this manual with `julia --project=docs docs/make.jl` after
  `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`.

## [Package versions used to build this manual](@id reproducibility)

The examples run when the manual is built, with the packages below. Times in tables were
measured once, on Linux with an AMD Ryzen 9 5950X, unless a page says otherwise.

!!! details "Julia version and package list of this build"
    ```@example
    using InteractiveUtils # hide
    versioninfo() # hide
    ```

    ```@example
    using Pkg # hide
    Pkg.status() # hide
    ```

## Credits

CasADi and CasADiNative.jl are the work of the CasADi developers: J. A. E. Andersson,
J. Gillis, G. Horn, J. B. Rawlings and M. Diehl, "CasADi: a software framework for
nonlinear optimization and optimal control", *Mathematical Programming Computation*
11 (2019), 1–36. The forms of the generated functions follow `build_function` of
Symbolics.jl.
