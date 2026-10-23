# Add noise to a differential equation

A deterministic model gives one trajectory for one initial state, whereas many physical
systems are driven by perturbations that are too fast or too many to model one by one,
and which are better described as random forces. A stochastic differential equation (SDE)
adds such a force to the right side, and its solution is then a random process, of which
a solver computes one realization per noise path. In this tutorial, we write the drift
and the noise of an SDE as two CasADi functions and solve the problem with
StochasticDiffEq.jl. We start with one source of noise that drives all the equations,
give each equation a source of its own, mix the sources through a matrix, and finally
estimate the mean and the variance of the state from an ensemble of paths.

The examples need CasADiCodegen.jl, SciMLBase.jl, the solvers `SRIW1` of
StochasticDiffEqHighOrder.jl and `LambaEM` of StochasticDiffEqLowOrder.jl,
DiffEqNoiseProcess.jl for a scalar Wiener process, and the standard libraries
SparseArrays and Statistics.

```@example sde
using CasADiCodegen, SparseArrays, Statistics
import SciMLBase
using StochasticDiffEqHighOrder: SRIW1
using StochasticDiffEqLowOrder: LambaEM
using DiffEqNoiseProcess: WienerProcess
```

## Drift, noise and Brownian motion

An Itô SDE for the state ``u`` reads

```math
du = f(u, p, t)\, dt + g(u, p, t)\, dW,
```

where ``f`` is the drift, ``W`` is a vector of ``m`` independent Wiener processes
(standard Brownian motions), and the noise ``g`` is a matrix with one row per state and
one column per Wiener process. In the notation of the Langevin equation, common in
physics, the same model is ``u' = f + g \xi``, where each element of the white noise
``\xi = dW/dt`` is a Brownian variable; column ``j`` of ``g`` then holds the coefficients
of the Brownian variable ``\xi_j`` in each equation. Two special cases need no matrix.
With diagonal noise, each state has a Wiener process of its own, which enters its
equation alone, so that ``g`` is a vector of the size of the state; with scalar noise, a
single Wiener process drives all the equations, and ``g`` is again a vector, its only
column.

[`SDEProblem`](@ref SciMLBase.SDEProblem(::CasADiCodegen.Function, ::CasADiCodegen.Function, ::Any, ::Any))
takes the drift and the noise as two CasADi functions with the same arguments, the
state, the parameters and the time, and generates code for both and for the Jacobian of
the drift. A noise that returns a column vector is diagonal noise unless the problem
receives a scalar Wiener process; a noise that returns a matrix gives the problem the
`noise_rate_prototype` that StochasticDiffEq.jl needs for it.

## One source of noise for all equations

We take the Lorenz equations, with parameter values for which the deterministic system
settles at a stable equilibrium rather than on its chaotic attractor, and add to each
equation a noise whose magnitude is 0.3 times the magnitude of its state, so that the
noise is multiplicative:

```math
\begin{aligned}
x' &= \sigma (y - x) + 0.3 x \xi, \\
y' &= x (\rho - z) - y + 0.3 y \xi, \\
z' &= x y - \beta z + 0.3 z \xi.
\end{aligned}
```

The drift is the right side of the deterministic model, and the noise is the column of
the coefficients of ``\xi``. Both functions take the state, the parameters and the time,
in this order; as for an ODE, a model without parameters or time takes fewer arguments:

```@example sde
x, y, z = SX.sym("x"), SX.sym("y"), SX.sym("z")
σ, ρ, β, t = SX.sym("σ"), SX.sym("ρ"), SX.sym("β"), SX.sym("t")
u, p = vertcat(x, y, z), vertcat(σ, ρ, β)
drift = CasadiFunction("lorenz", [u, p, t], [vertcat(σ * (y - x), x * (ρ - z) - y, x * y - β * z)])
noise = CasadiFunction("noise", [u, p, t], [0.3 * u])
u0, parameters = [5.0, 5.0, 1.0], [10.0, 2.33, 26.0]
nothing # hide
```

Since ``\xi`` is the same in the three equations, the problem gets one Wiener process,
`WienerProcess(0.0, 0.0, 0.0)`, which starts at the time 0 with the value 0; the third
value starts the auxiliary process that methods of high order, such as `SRIW1`, need for
their iterated integrals. `SRIW1` is a
method of strong order 1.5 with adaptive steps for diagonal and scalar noise, and the
option `seed` fixes the random numbers, so that the same call gives the same path again:

```@example sde
problem = SciMLBase.SDEProblem(drift, noise, u0, (0.0, 100.0), parameters; noise = WienerProcess(0.0, 0.0, 0.0))
solution = SciMLBase.solve(problem, SRIW1(); seed = 1)
solution.retcode, solution.u[end]
```

The solution knows the names of the symbols, as for an ODE, so that `solution[:x]` is the
path of ``x`` and `problem.ps[:ρ]` the value of ``\rho``. Since one Wiener process drives
all the states, their perturbations are correlated: a positive increment of ``W`` pushes
``x``, ``y`` and ``z`` up together, each in proportion to its magnitude, so that the noise
has a stronger effect on ``x`` and ``y``, which take larger values, than on ``z``.

## A source of noise for each equation

For uncorrelated noise, each equation gets a Brownian variable of its own,
``\xi_x``, ``\xi_y`` and ``\xi_z``. The noise matrix is then diagonal, with the
coefficients `0.3 * u` on its diagonal, and diagonal noise is written as the vector of
that diagonal. The function `noise` is thus the same as before, and only the Wiener
process of the problem changes: without the option `noise`, the problem has one Wiener
process per state.

```@example sde
problem = SciMLBase.SDEProblem(drift, noise, u0, (0.0, 100.0), parameters)
solution = SciMLBase.solve(problem, SRIW1(); seed = 1)
solution.u[end]
```

The seed is the same as in the previous section, but the path differs, because three
Wiener processes now draw three random numbers at each step, of which each state
receives its own.

## Mixed sources of noise

When a Brownian variable enters several equations and an equation receives several
Brownian variables, the noise is a full matrix. We let a first source ``\xi_1`` act on
``x`` and ``y``, and a second one ``\xi_2``, with a constant intensity, act on ``y`` and
``z``:

```math
\begin{aligned}
x' &= \sigma (y - x) + 0.3 x \xi_1, \\
y' &= x (\rho - z) - y + 0.3 y \xi_1 + 0.1 \xi_2, \\
z' &= x y - \beta z + 0.3 z \xi_2.
\end{aligned}
```

The noise function returns the ``3 \times 2`` matrix of the coefficients. An element
that is zero for all states is a structural zero of CasADi, `SX(1, 1)`, so that the
sparsity pattern of the matrix holds the four nonzeros only:

```@example sde
o = SX(1, 1)
mixed = CasadiFunction("mixed", [u, p, t], [[0.3x o; 0.3y 0.1; o 0.3z]])
problem = SciMLBase.SDEProblem(drift, mixed, u0, (0.0, 100.0), parameters)
size(problem.noise_rate_prototype)
```

The problem has a `noise_rate_prototype` of zeros of the size of the matrix, from which
the solver learns that there are two Wiener processes. `SRIW1` accepts diagonal and
scalar noise only; for a noise matrix we take `LambaEM`, the Euler-Maruyama method with
adaptive steps:

```@example sde
solution = SciMLBase.solve(problem, LambaEM(); seed = 1)
solution.retcode, solution.u[end]
```

For a large model whose noise matrix has few nonzeros, `sparse = true` makes the
prototype a `SparseMatrixCSC` with the sparsity pattern of the noise, and the Jacobian
of the drift sparse too. The generated noise then writes the nonzeros only, and the
solver multiplies the sparse matrix by the increments of the Wiener processes:

```@example sde
sparse_problem = SciMLBase.SDEProblem(drift, mixed, u0, (0.0, 100.0), parameters; sparse = true)
sparse_problem.noise_rate_prototype
```

For a matrix as small as this one, the dense form is faster.

## Statistics of an ensemble

One path is one realization of a random process, so that a statement about the model,
such as its mean state at a time, needs many paths. An `EnsembleProblem` solves the
problem once per path; its option `seed` gives each path a seed of its own, derived from
this one, so that the paths are independent and the ensemble can be reproduced. We
solve 1000 paths of the model with uncorrelated noise up to ``t = 10`` and save the
states at the times 1, 5 and 10:

```@example sde
problem = SciMLBase.SDEProblem(drift, noise, u0, (0.0, 10.0), parameters)
ensemble = SciMLBase.EnsembleProblem(problem)
paths = SciMLBase.solve(ensemble, SRIW1(), SciMLBase.EnsembleSerial(); trajectories = 1000,
                        seed = 1, saveat = [1.0, 5.0, 10.0])
final_x = [path[:x][end] for path in paths.u]
mean(final_x), std(final_x), std(final_x) / sqrt(length(final_x))
```

The last number is the standard error of the mean, the standard deviation divided by the
square root of the number of paths, and a mean from another set of paths agrees with
this one to a few standard errors.
