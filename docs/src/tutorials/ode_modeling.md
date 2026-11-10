# Build and fit an ODE model

In this tutorial, we write a small differential equation with CasADi, solve it with
OrdinaryDiffEq.jl, extend it with observed variables, inputs, components and a
second-order equation, and fit the parameters of a Lotka-Volterra model to data.

## A first-order lag

The first model is a first-order lag, whose state ``x`` follows an input ``f(t)`` with
the time constant ``\tau``:

```math
\dot{x} = \frac{f(t) - x}{\tau}.
```

We take ``f = 1``. The CasADi function takes the state, the parameters and the time, in
this order, and `SciMLBase.ODEFunction` generates code for the right side and its
Jacobian (see [Give a CasADi function to a SciML problem](@ref)):

```@example ode
using CasADiCodegen
using OrdinaryDiffEqTsit5: Tsit5
import SciMLBase

x, τ, t = SX.sym("x"), SX.sym("τ"), SX.sym("t")
lag = CasadiFunction("lag", [x, τ, t], [(1 - x) / τ])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(lag), [0.0], (0.0, 10.0), [3.0])
solution = SciMLBase.solve(problem, Tsit5())
solution(0:2:10)
```

The initial state and the parameters are vectors in the order of the arguments of
`lag`.

## Change the parameters and the initial state

The problem carries the names of the scalar symbols, so `remake` and indexing work by
name:

```@example ode
problem2 = SciMLBase.remake(problem; u0 = [:x => 0.5], p = [:τ => 1 / 3])
solution2 = SciMLBase.solve(problem2, Tsit5())
solution2[:x][end]
```

A map of names keeps the parameters that it does not name; inside a loss, a plain
vector avoids the lookup. A `remake` without `p` shares the parameter vector of the
original problem, so give the copy its own vector, `remake(problem; p = copy(problem.p))`,
before you change it in place.

## Compute an intermediate variable

Each result of the CasADi function after the first is an observed variable, which the
solution computes from the state under the name of the result. The two lists of names
after the outputs name the arguments and the results:

```@example ode
rhs = (1 - x) / τ
lag = CasadiFunction("lag", [x, τ, t], [rhs, rhs], ["x", "τ", "t"], ["dx", "RHS"])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(lag), [0.0], (0.0, 10.0), [3.0])
solution = SciMLBase.solve(problem, Tsit5())
solution[:RHS][end]
```

A second index selects one saved time:

```@example ode
solution[:x, 2]
```

## An input that depends on time

An input that depends on time, such as ``f(t) = \sin t``, is an expression of `t`. We
also return it as an observed variable:

```@example ode
input = sin(t)
lag_sin = CasadiFunction("lag_sin", [x, τ, t], [(input - x) / τ, input], ["x", "τ", "t"], ["dx", "f"])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(lag_sin), [0.0], (0.0, 10.0), [3.0])
solution = SciMLBase.solve(problem, Tsit5())
solution[[:x, :f]][end]
```

## An input given by data

A table of measured values has no expression that CasADi can differentiate.
[`callback`](@ref CasADiCodegen.callback) wraps a Julia function as a function that an
`MX` graph can call; an `SX` graph cannot call it. The tuples `(1,), (1,)` give the
sizes of the arguments and the results, and a static array as result keeps the call free
of allocation. The derivative of the table is zero between its jumps, which we pass as
`jacobian`:

```@example ode
using StaticArrays: SVector

const TABLE = [0.3, -1.2, 0.8, 0.1, -0.5, 1.4, -0.9, 0.6, 0.0, -0.3]
hold(t) = SVector(t[1] >= 10 ? TABLE[end] : TABLE[floor(Int, t[1]) + 1])
table = callback(hold, "table", (1,), (1,); jacobian = t -> SVector(0.0))

x, τ, t = MX.sym("x"), MX.sym("τ"), MX.sym("t")
input = table(t)
lag_table = CasadiFunction("lag_table", [x, τ, t], [(input - x) / τ, input], ["x", "τ", "t"], ["dx", "f"])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(lag_table), [0.0], (0.0, 10.0), [0.75])
solution = SciMLBase.solve(problem, Tsit5(); tstops = 1:9)
solution[[:x, :f]][end]
```

`tstops` ends a step at each jump of the input.
[Call a Julia function from the model](@ref) describes the other forms of the derivatives.

## A model made of components

A component is a Julia function that makes the symbols of one lag, prefixed with its
name, and returns them with the right side for a given input. We connect two lags in
series, with the state of the first as the input of the second:

```@example ode
function first_order_lag(name, input)
    x, τ = SX.sym(name * "_x"), SX.sym(name * "_τ")
    return (; x, τ, dx = (input - x) / τ)
end

t = SX.sym("t")
lag_1 = first_order_lag("lag_1", 1.5)
lag_2 = first_order_lag("lag_2", lag_1.x)
connected = CasadiFunction("connected", [vcat(lag_1.x, lag_2.x), vcat(lag_1.τ, lag_2.τ), t],
                           [vcat(lag_1.dx, lag_2.dx)])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(connected), [-0.5, 1.0], (0.0, 10.0), [2.0, 4.0])
solution = SciMLBase.solve(problem, Tsit5())
solution[:lag_2_x][end]
```

The connection is a substitution of expressions, so the model holds only the two
state equations.

## Derivatives and their sparsity

An implicit solver such as `Rodas5P` calls the generated Jacobian. We evaluate it at the
initial state:

```@example ode
J = zeros(2, 2)
problem.f.jac(J, problem.u0, problem.p, 0.0)
J
```

The first state does not depend on the second, so one element is zero. With
`sparse = true`, which needs SparseArrays, the function holds a `jac_prototype` with this
pattern, and the Jacobian code writes the stored values only:

```@example ode
using SparseArrays
using OrdinaryDiffEqRosenbrock: Rodas5P

f_sparse = SciMLBase.ODEFunction(connected; sparse = true)
problem = SciMLBase.ODEProblem(f_sparse, [-0.5, 1.0], (0.0, 10.0), [2.0, 4.0])
solution = SciMLBase.solve(problem, Rodas5P())
f_sparse.jac_prototype
```

The sparse form pays off only for a large model in which each state depends on a few
others.

## An equation of higher order

An equation of order ``n`` becomes ``n`` first-order equations. We take a Lorenz
system whose first equation is of the second order:

```math
\ddot{x} = \sigma (y - x), \qquad \dot{y} = x (\rho - z) - y, \qquad \dot{z} = x y - \beta z.
```

The velocity becomes the state `vx`. The values ``\sigma = 28``, ``\rho = 10`` are
those of the ModelingToolkit.jl example:

```@example ode
x, vx, y, z, t = SX.sym("x"), SX.sym("vx"), SX.sym("y"), SX.sym("z"), SX.sym("t")
σ, ρ, β = SX.sym("σ"), SX.sym("ρ"), SX.sym("β")
lorenz = CasadiFunction("lorenz", [vcat(x, vx, y, z), vcat(σ, ρ, β), t],
                        [vcat(vx, σ * (y - x), x * (ρ - z) - y, x * y - β * z)])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(lorenz), [1.0, 2.0, 0.0, 0.0], (0.0, 100.0),
                               [28.0, 10.0, 8 / 3])
solution = SciMLBase.solve(problem, Tsit5())
solution[[:x, :vx]][end]
```

The submodule `CasADiCodegen.Models` makes the derivative symbols itself (see
[Write a model as equations](@ref)): `Differential(t)` is the derivative, `~` separates the two
sides, and [`System`](@ref CasADiCodegen.Models.System) reduces the order:

```@example ode
using CasADiCodegen.Models

D = Differential(t)
lorenz2 = System([D(D(x)) ~ σ * (y - x), D(y) ~ x * (ρ - z) - y, D(z) ~ x * y - β * z], t;
                 parameters = [σ, ρ, β], name = "lorenz2")
equations(lorenz2)
```

The velocity is the symbol `dx` of `D(x)`, and
`SciMLBase.ODEProblem(lorenz2, [x => 1.0, D(x) => 2.0, y => 0.0, z => 0.0], (0.0, 100.0), [σ => 28.0, ρ => 10.0, β => 8 / 3])`
gives the problem above with its initial values by name.

## Fit the parameters to data

The Lotka-Volterra model of prey ``x`` and predators ``y`` is

```math
\dot{x} = (\alpha - \beta y) x, \qquad \dot{y} = (\delta x - \gamma) y.
```

We make noisy data from known parameters:

```@example ode
using OrdinaryDiffEqTsit5: AutoTsit5
using OrdinaryDiffEqRosenbrock: Rosenbrock23
using Random: Xoshiro

x, y, t = SX.sym("x"), SX.sym("y"), SX.sym("t")
α, β, γ, δ = SX.sym("α"), SX.sym("β"), SX.sym("γ"), SX.sym("δ")
lotka_volterra = CasadiFunction("lotka_volterra", [vcat(x, y), vcat(α, β, γ, δ), t],
                                [vcat((α - β * y) * x, (δ * x - γ) * y)])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(lotka_volterra), [1.0, 1.0], (0.0, 10.0),
                               [1.5, 1.0, 3.0, 1.0])
times = 0.0:0.1:10.0
data = Array(SciMLBase.solve(problem, Tsit5(); saveat = times))
data .+= 0.01 .* randn(Xoshiro(1), size(data))
nothing # hide
```

The loss remakes the problem with the plain parameter vector `q` and solves it with
`AutoTsit5`, which switches to `Rosenbrock23` if the proposed parameters make the model
stiff. Under ForwardDiff.jl, `q` holds dual numbers, which the generated code accepts:

```@example ode
function loss(q, (problem, times, data))
    new_problem = SciMLBase.remake(problem; p = q)
    solution = SciMLBase.solve(new_problem, AutoTsit5(Rosenbrock23()); saveat = times)
    return sum(abs2, Array(solution) .- data) / length(data)
end

loss([1.5, 1.0, 3.0, 1.0], (problem, times, data))
```

We minimize the loss with BFGS within bounds:

```@example ode
using Optimization, OptimizationOptimJL
import ForwardDiff

f_loss = SciMLBase.OptimizationFunction(loss, Optimization.AutoForwardDiff())
fit = SciMLBase.OptimizationProblem(f_loss, [1.2, 0.8, 2.5, 0.8], (problem, times, data);
                                    lb = fill(0.1, 4), ub = fill(4.0, 4))
result = SciMLBase.solve(fit, Optim.BFGS())
result.u
```

The fit recovers the true parameters ``(1.5, 1, 3, 1)``:

```@example ode
result.u .- [1.5, 1.0, 3.0, 1.0]
```
