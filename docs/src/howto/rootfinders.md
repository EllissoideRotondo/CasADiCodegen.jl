# Solve an equation

[`CasADiCodegen.rootfinder`](@ref) makes a CasADi function that solves
`f(z, p1, p2, ...) = 0` for `z` with a SciML nonlinear solver; the method that takes a
SciML algorithm exists once a solver package, such as SimpleNonlinearSolve.jl, is
loaded. Solve Kepler's equation for the eccentric anomaly and generate the root with
its derivative by the mean anomaly:

```@example root
using CasADiCodegen
using SimpleNonlinearSolve: SimpleNewtonRaphson

E, M, e = SX.sym("E"), SX.sym("M"), SX.sym("e")
kepler = CasadiFunction("kepler", [E, M, e], [E - e * sin(E) - M])
anomaly = rootfinder("anomaly", SimpleNewtonRaphson(), kepler; abstol = 1e-14)

m, ec = MX.sym("m"), MX.sym("ec")
root = anomaly(m, m, ec)                      # the mean anomaly is the first guess
f = CasadiFunction("f", [m, ec], [root, jacobian(root, m)])
g, g! = build_function(f; expression = Val{false})
g([1.0], [0.3])
```

The function takes a first guess of `z` and then the other arguments of `f`; all
keywords, such as `abstol` and `maxiters`, go to the solver.

## Derivatives

CasADi differentiates the root to any order by the implicit function theorem; `g!` of
the example took 0.13 µs without allocation on Julia 1.13. `generate_julia` writes a rootfinder to a module under the rule for
integrators (see [Write the model to a file](@ref)).

A rootfinder of CasADi itself also works in generated code: the package generates Julia
code for `newton`, while `kinsol` and `fast_newton` need a SciML solver (see
[Replace the solvers of CasADi](@ref)). The code of `newton` uses a sparse LU for a
Jacobian of 16 rows or more with at most a quarter of its elements stored; on the 2D
Bratu problem of 1024 unknowns, a solve took 8.8 ms, against 29 s dense and 6.2 ms in
CasADi.
