# Solve a system of nonlinear equations

A steady state of a differential equation is a root of its right side. In this tutorial,
we solve for steady states with generated code for the residual and its Jacobian, change
parameters and guesses by name, and solve a discretized boundary value problem with a
sparse Jacobian.

## The steady state of the Lorenz system

The steady states of the Lorenz system solve

```math
\begin{aligned}
0 &= \sigma (y - x), \\
0 &= x (\rho - z) - y, \\
0 &= x y - \beta z.
\end{aligned}
```

We give each state and parameter its own scalar symbol, so that the problem has names
for them. The CasADi function takes the unknowns and the parameters, and returns the
residual:

```@example nonlinear
using CasADiCodegen
import SciMLBase

x, y, z = SX.sym("x"), SX.sym("y"), SX.sym("z")
σ, ρ, β = SX.sym("σ"), SX.sym("ρ"), SX.sym("β")
equations = vcat(σ * (y - x), x * (ρ - z) - y, x * y - β * z)
lorenz = CasadiFunction("lorenz", [vcat(x, y, z), vcat(σ, ρ, β)], [equations])
nothing # hide
```

`SciMLBase.NonlinearFunction` generates code for the residual and its Jacobian (see
[Give a CasADi function to a SciML problem](@ref)):

```@example nonlinear
using SimpleNonlinearSolve: SimpleNewtonRaphson

f = SciMLBase.NonlinearFunction(lorenz)
problem = SciMLBase.NonlinearProblem(f, [1.0, 0.0, 0.0], [10.0, 26.0, 8 / 3])
solution = SciMLBase.solve(problem, SimpleNewtonRaphson())
solution.retcode, solution.u
```

From the guess ``(1, 0, 0)`` the method reaches the origin. The generated Jacobian
`jac(J, u, p)` writes into a matrix:

```@example nonlinear
J = zeros(3, 3)
f.jac(J, solution.u, problem.p)
J
```

With `jac = false` the function holds the residual only, and the solver differentiates
it itself.

## Change the parameters and the guess by name

A solution is indexed by the name of a state, and `SciMLBase.remake` takes values as
name-value pairs. For ``\rho > 1`` there are two more steady states,
``x = y = \pm\sqrt{\beta (\rho - 1)}``, ``z = \rho - 1``:

```@example nonlinear
convection = SciMLBase.remake(problem; u0 = [:x => 10.0, :y => 10.0, :z => 20.0])
solution = SciMLBase.solve(convection, SimpleNewtonRaphson())
solution[:x], solution[:z]
```

The parameters are an argument of the generated code, so a new ``\rho`` needs no new
build:

```@example nonlinear
solution = SciMLBase.solve(SciMLBase.remake(convection; p = [:ρ => 28.0]), SimpleNewtonRaphson())
solution[[:x, :y, :z]]
```

## An array equation with a sparse Jacobian

We discretize the Bratu equation ``u'' + \lambda e^{u} = 0`` on ``(0, 1)`` with
``u(0) = u(1) = 0`` by central differences on ``n`` interior points, ``h = 1/(n + 1)``.
The unknown is one symbol of ``n`` elements, padded with the boundary zeros:

```@example nonlinear
n = 8
h = 1 / (n + 1)
u, λ = SX.sym("u", n), SX.sym("λ")
padded = vcat(0, u, 0)
residual = (padded[1:n] - 2 * padded[2:(n + 1)] + padded[3:(n + 2)]) / h^2 + λ * exp.(u)
bratu = CasadiFunction("bratu", [u, λ], [residual])
nothing # hide
```

The Jacobian is tridiagonal. With `sparse = true`, which needs SparseArrays, the
function holds a `jac_prototype` with the sparsity pattern, and the generated Jacobian
writes its stored values:

```@example nonlinear
using SparseArrays

f = SciMLBase.NonlinearFunction(bratu; sparse = true)
f.jac_prototype
```

The elements of `u` are named `u_0`, `u_1`, ..., as in CasADi. From ``u = 0`` the method
finds the lower of the two solutions for ``\lambda = 1``:

```@example nonlinear
problem = SciMLBase.NonlinearProblem(f, zeros(n), [1.0])
solution = SciMLBase.solve(problem, SimpleNewtonRaphson())
solution.retcode, solution.u
```

On a large grid, a solver of NonlinearSolve.jl such as `NewtonRaphson` factorizes the
sparse `jac_prototype` at each step.

## Small systems on static arrays

With `inplace = false`, the residual returns an `SVector` and the Jacobian an `SMatrix`,
which keeps a small solve off the heap:

```@example nonlinear
using StaticArrays: SVector

f = SciMLBase.NonlinearFunction(lorenz; inplace = false)
problem = SciMLBase.NonlinearProblem(f, SVector(10.0, 10.0, 20.0), SVector(10.0, 26.0, 8 / 3))
solution = SciMLBase.solve(problem, SimpleNewtonRaphson())
solution.u
```

The Jacobian, as a static array of at most 1024 elements, limits this form to 32 unknowns;
a larger system throws an `ArgumentError` (see [Avoid allocation with static arrays](@ref)).
