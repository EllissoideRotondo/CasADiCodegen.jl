# Solve a system of nonlinear equations

A steady state of a system of differential equations is a point at which its right side
vanishes, so that finding one is the solution of a system of nonlinear equations rather
than the integration of an initial value problem. In this tutorial, we write such a
system with CasADi, give it to a Newton method with generated code for the residual and
its Jacobian, and change its parameters and its initial guess by name; we then write a
discretized boundary value problem as an array equation with a sparse Jacobian, and solve
a small system on static arrays. The examples need CasADiCodegen.jl, SciMLBase.jl,
SimpleNonlinearSolve.jl, StaticArrays.jl and the standard library SparseArrays.

## The steady state of the Lorenz system

The Lorenz system has the states ``x``, ``y`` and ``z`` and the parameters ``\sigma``,
``\rho`` and ``\beta``, and its steady states are the solutions of

```math
\begin{aligned}
0 &= \sigma (y - x), \\
0 &= x (\rho - z) - y, \\
0 &= x y - \beta z.
\end{aligned}
```

We make one scalar symbol for each state and each parameter, so that the problem has a
name for each, and collect them into the two arguments of a CasADi function, the unknowns
and the parameters in this order. Its first result is the residual, a column vector of
the size of the unknowns:

```@example nonlinear
using CasADiCodegen
import SciMLBase

x, y, z = SX.sym("x"), SX.sym("y"), SX.sym("z")
σ, ρ, β = SX.sym("σ"), SX.sym("ρ"), SX.sym("β")
equations = vertcat(σ * (y - x), x * (ρ - z) - y, x * y - β * z)
lorenz = CasadiFunction("lorenz", [vertcat(x, y, z), vertcat(σ, ρ, β)], [equations])
nothing # hide
```

`SciMLBase.NonlinearFunction` turns the CasADi function into the function of a
`NonlinearProblem`, with generated code for the residual and for its Jacobian by the
unknowns, which CasADi computes from the graph of the equations (see
[Give a CasADi function to a SciML problem](@ref)). The problem takes the initial guess
and the values of the parameters as vectors in the order of the symbols, and the Newton
method of SimpleNonlinearSolve.jl calls the generated Jacobian at each step:

```@example nonlinear
using SimpleNonlinearSolve: SimpleNewtonRaphson

f = SciMLBase.NonlinearFunction(lorenz)
problem = SciMLBase.NonlinearProblem(f, [1.0, 0.0, 0.0], [10.0, 26.0, 8 / 3])
solution = SciMLBase.solve(problem, SimpleNewtonRaphson())
solution.retcode, solution.u
```

From the guess ``(1, 0, 0)`` the method reaches the origin, which is a steady state for
all values of the parameters. The generated Jacobian is a function `jac(J, u, p)` that
writes into a matrix, so that we can also call it ourselves, here at the solution:

```@example nonlinear
J = zeros(3, 3)
f.jac(J, solution.u, problem.p)
J
```

With `jac = false` the function holds the residual only, and the solver differentiates
the generated residual itself, by automatic differentiation in the case of
`SimpleNewtonRaphson`.

## Change the parameters and the guess by name

`NonlinearFunction` gives the problem the names of the scalar symbols of the CasADi
function, so that a solution is indexed by the name of a state and `SciMLBase.remake`
takes new values of the parameters or of the guess as pairs of a name and a value. For
``\rho > 1`` the Lorenz system has two more steady states,
``x = y = \pm\sqrt{\beta (\rho - 1)}`` and ``z = \rho - 1``, and a guess near the first
one leads the method there:

```@example nonlinear
convection = SciMLBase.remake(problem; u0 = [:x => 10.0, :y => 10.0, :z => 20.0])
solution = SciMLBase.solve(convection, SimpleNewtonRaphson())
solution[:x], solution[:z]
```

A new value of ``\rho`` moves this steady state, and the generated code serves the new
problem without being built again, since the parameters are an argument of the function:

```@example nonlinear
solution = SciMLBase.solve(SciMLBase.remake(convection; p = [:ρ => 28.0]), SimpleNewtonRaphson())
solution[[:x, :y, :z]]
```

## An array equation with a sparse Jacobian

A boundary value problem discretized on a grid gives a large system in which each
equation involves a few neighboring unknowns. We take the Bratu equation
``u'' + \lambda e^{u} = 0`` on the interval ``(0, 1)`` with ``u(0) = u(1) = 0``, and
replace the second derivative by central differences on ``n`` interior points with the
step ``h = 1/(n + 1)``. The unknowns are one symbol `u` of ``n`` elements, and the
boundary values enter as the zeros that pad it. The second difference is then one
expression of three slices of the padded vector, with one residual element per interior
point:

```@example nonlinear
n = 8
h = 1 / (n + 1)
u, λ = SX.sym("u", n), SX.sym("λ")
padded = vertcat(0, u, 0)
residual = (padded[1:n] - 2 * padded[2:(n + 1)] + padded[3:(n + 2)]) / h^2 + λ * exp(u)
bratu = CasadiFunction("bratu", [u, λ], [residual])
nothing # hide
```

Each element of the residual depends on the unknown at its own point and on its two
neighbors, so that the Jacobian is tridiagonal and most of its elements are zero for a
fine grid. With `sparse = true` the function also holds a `jac_prototype`, a
`SparseMatrixCSC` with the sparsity pattern of the Jacobian, and the generated Jacobian
writes its stored values; the standard library SparseArrays must be loaded:

```@example nonlinear
using SparseArrays

f = SciMLBase.NonlinearFunction(bratu; sparse = true)
f.jac_prototype
```

The elements of a symbol of several elements are named `u_0`, `u_1`, ..., as CasADi
names them, so that `solution[:u_0]` is `solution.u[1]`, and the parameter keeps the
name `λ`. From the guess ``u = 0`` the method
finds the lower of the two solutions that the equation has for ``\lambda = 1``:

```@example nonlinear
problem = SciMLBase.NonlinearProblem(f, zeros(n), [1.0])
solution = SciMLBase.solve(problem, SimpleNewtonRaphson())
solution.retcode, solution.u
```

On a grid of thousands of points, a solver of NonlinearSolve.jl such as `NewtonRaphson`
factorizes the sparse matrix of the `jac_prototype` at each step instead of a dense
matrix.

## Small systems on static arrays

For a system of a few unknowns, a solve on static arrays keeps the vectors and the
matrices of the solver off the heap. With `inplace = false`, `NonlinearFunction` holds
out-of-place functions that return an `SVector` for the residual and an `SMatrix` for
the Jacobian, which we use with a guess and parameters of type `SVector`:

```@example nonlinear
using StaticArrays: SVector

f = SciMLBase.NonlinearFunction(lorenz; inplace = false)
problem = SciMLBase.NonlinearProblem(f, SVector(10.0, 10.0, 20.0), SVector(10.0, 26.0, 8 / 3))
solution = SciMLBase.solve(problem, SimpleNewtonRaphson())
solution.u
```

The generated static arrays have at most 4096 elements, so that this form with the
Jacobian allows at most 64 unknowns; for a larger system, `inplace = false` throws an
`ArgumentError`, and the in-place form serves with a dense or a sparse Jacobian.
