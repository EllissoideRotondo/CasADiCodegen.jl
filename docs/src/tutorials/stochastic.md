# Add noise to a differential equation

In this tutorial, we write the drift and the noise of a stochastic differential equation
(SDE) as two CasADi functions and solve the problem with StochasticDiffEq.jl, with scalar,
diagonal and matrix noise, and then estimate the statistics of an ensemble of paths.

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

where ``f`` is the drift, ``W`` is a vector of ``m`` independent Wiener processes, and the
noise ``g`` has one row per state and one column per Wiener process. In the Langevin
notation ``u' = f + g \xi``, column ``j`` of ``g`` holds the coefficients of the white
noise ``\xi_j``. With diagonal noise, each state has its own Wiener process, and with
scalar noise one Wiener process drives all equations; in both cases ``g`` is a vector.

[`SDEProblem`](@ref SciMLBase.SDEProblem(::CasADiCodegen.Function, ::CasADiCodegen.Function, ::Any, ::Any))
takes the drift and the noise as two CasADi functions of the state, the parameters and
the time, and generates code for both and for the Jacobian of the drift. A noise that
returns a column vector is diagonal unless the problem receives a scalar Wiener process; a
noise that returns a matrix sets the `noise_rate_prototype` of the problem.

## One source of noise for all equations

We take the Lorenz equations at parameters with a stable equilibrium and add a
multiplicative noise of 0.3 times each state:

```math
\begin{aligned}
x' &= \sigma (y - x) + 0.3 x \xi, \\
y' &= x (\rho - z) - y + 0.3 y \xi, \\
z' &= x y - \beta z + 0.3 z \xi.
\end{aligned}
```

The noise is the column of the coefficients of ``\xi``; as for an ODE, a model without
parameters or time takes fewer arguments:

```@example sde
x, y, z = SX.sym("x"), SX.sym("y"), SX.sym("z")
σ, ρ, β, t = SX.sym("σ"), SX.sym("ρ"), SX.sym("β"), SX.sym("t")
u, p = vcat(x, y, z), vcat(σ, ρ, β)
drift = CasadiFunction("lorenz", [u, p, t], [vcat(σ * (y - x), x * (ρ - z) - y, x * y - β * z)])
noise = CasadiFunction("noise", [u, p, t], [0.3 * u])
u0, parameters = [5.0, 5.0, 1.0], [10.0, 2.33, 26.0]
nothing # hide
```

One ``\xi`` drives the three equations, so the problem gets one Wiener process, whose
third initial value starts the auxiliary process that high-order methods such as `SRIW1`
need. `SRIW1` has strong order 1.5 and adaptive steps, and `seed` makes the path
reproducible:

```@example sde
problem = SciMLBase.SDEProblem(drift, noise, u0, (0.0, 100.0), parameters; noise = WienerProcess(0.0, 0.0, 0.0))
solution = SciMLBase.solve(problem, SRIW1(); seed = 1)
solution.retcode, solution.u[end]
```

The solution is indexed by the names of the symbols, as for an ODE. One Wiener process
drives all states, so their perturbations are correlated.

## A source of noise for each equation

For uncorrelated noise, each equation gets its own ``\xi``, and the noise matrix is the
diagonal `0.3 * u`. The function `noise` is the same; without the option `noise`, the
problem has one Wiener process per state:

```@example sde
problem = SciMLBase.SDEProblem(drift, noise, u0, (0.0, 100.0), parameters)
solution = SciMLBase.solve(problem, SRIW1(); seed = 1)
solution.u[end]
```

With the same seed, the path differs, because each state receives its own random
numbers.

## Mixed sources of noise

When a source of noise enters several equations and an equation receives several
sources, the noise is a full matrix:

```math
\begin{aligned}
x' &= \sigma (y - x) + 0.3 x \xi_1, \\
y' &= x (\rho - z) - y + 0.3 y \xi_1 + 0.1 \xi_2, \\
z' &= x y - \beta z + 0.3 z \xi_2.
\end{aligned}
```

The noise function returns the ``3 \times 2`` matrix, with the structural zero
`SX(1, 1)` where an element is zero for all states:

```@example sde
o = SX(1, 1)
mixed = CasadiFunction("mixed", [u, p, t], [[0.3x o; 0.3y 0.1; o 0.3z]])
problem = SciMLBase.SDEProblem(drift, mixed, u0, (0.0, 100.0), parameters)
size(problem.noise_rate_prototype)
```

The size of the prototype tells the solver that there are two Wiener processes. `SRIW1`
accepts only diagonal and scalar noise, so we take `LambaEM`, the adaptive
Euler-Maruyama method:

```@example sde
solution = SciMLBase.solve(problem, LambaEM(); seed = 1)
solution.retcode, solution.u[end]
```

For a large noise matrix with few nonzeros, `sparse = true` makes the prototype and the
Jacobian of the drift sparse:

```@example sde
sparse_problem = SciMLBase.SDEProblem(drift, mixed, u0, (0.0, 100.0), parameters; sparse = true)
sparse_problem.noise_rate_prototype
```

For a matrix as small as this one, the dense form is faster.

## Statistics of an ensemble

An `EnsembleProblem` solves the problem once per path, and its `seed` gives each path a
seed of its own, so that the ensemble is reproducible. We solve 1000 paths with
uncorrelated noise:

```@example sde
problem = SciMLBase.SDEProblem(drift, noise, u0, (0.0, 10.0), parameters)
ensemble = SciMLBase.EnsembleProblem(problem)
paths = SciMLBase.solve(ensemble, SRIW1(), SciMLBase.EnsembleSerial(); trajectories = 1000,
                        seed = 1, saveat = [1.0, 5.0, 10.0])
final_x = [path[:x][end] for path in paths.u]
mean(final_x), std(final_x), std(final_x) / sqrt(length(final_x))
```

The last number is the standard error of the mean.
