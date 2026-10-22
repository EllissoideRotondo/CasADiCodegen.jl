# Build and fit an ODE model

In this tutorial, we write a small differential equation with CasADi and solve it with
OrdinaryDiffEq.jl. We then extend it step by step: new values of the parameters, an
intermediate variable that the solution computes on demand, an input that depends on
time or on tabulated data, a model made of two components, and its sparse Jacobian.
At the end we fit the parameters of a Lotka-Volterra model to data, which needs a new
problem for each set of parameters and derivatives of the loss through the solver.
The examples need CasADiCodegen.jl, SciMLBase.jl, OrdinaryDiffEqTsit5.jl,
OrdinaryDiffEqRosenbrock.jl, StaticArrays.jl and the standard library SparseArrays, and
the fit at the end also needs Optimization.jl, OptimizationOptimJL.jl (which exports the
module `Optim`), ForwardDiff.jl and the standard library Random.

## A first-order lag

The first model is a first-order lag, whose state ``x`` follows an input ``f(t)`` with
the time constant ``\tau``:

```math
\dot{x} = \frac{f(t) - x}{\tau}.
```

We take the constant input ``f = 1`` first. A CasADi function of the state, the
parameters and the time, in this order, gives the right side, and
`SciMLBase.ODEFunction` turns it into the function of an ODE problem with generated
code for the right side and its Jacobian (see
[Give a CasADi function to a SciML problem](@ref)):

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
`lag`, here `[0.0]` and `[3.0]`, and the solution interpolates between its steps, so
that `solution(0:2:10)` gives the state at six times.

## Change the parameters and the initial state

`ODEFunction` gives the problem the names of the scalar symbols of the arguments of
`lag`, so that a problem can be remade, and a solution indexed, by name. `remake` copies the problem
with the new values:

```@example ode
problem2 = SciMLBase.remake(problem; u0 = [:x => 0.5], p = [:τ => 1 / 3])
solution2 = SciMLBase.solve(problem2, Tsit5())
solution2[:x][end]
```

With the shorter time constant, the state is close to the input at the end of the
interval. A map of names keeps the parameters that it does not name; inside a loss, a
plain vector avoids the lookup of the names. A `remake` without `p` shares the parameter
vector of the original problem, so that a change made in place to the copy also changes
the original; to change the values in place, give the copy its own vector first, as in
`remake(problem; p = copy(problem.p))`.

## Compute an intermediate variable

A model often has variables that are not states but that we want to see, such as the
right side itself. Each result of the CasADi function after the first one is such an
observed variable: the solution computes it from the state with generated code, under
the name of the result. The two lists of names that `CasadiFunction` accepts after
the outputs name the arguments and the results; the states and the parameters keep the
names of their scalar symbols, and the result names name the observed variables:

```@example ode
rhs = (1 - x) / τ
lag = CasadiFunction("lag", [x, τ, t], [rhs, rhs], ["x", "τ", "t"], ["dx", "RHS"])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(lag), [0.0], (0.0, 10.0), [3.0])
solution = SciMLBase.solve(problem, Tsit5())
solution[:RHS][end]
```

A name indexes the time series of a state or of an observed variable, and a second
index selects one saved time, here the second one:

```@example ode
solution[:x, 2]
```

## An input that depends on time

The time is the third argument of the function, so an input that is an expression of
time, such as ``f(t) = \sin t``, enters the model like any other expression. We also
return the input as an observed variable:

```@example ode
input = sin(t)
lag_sin = CasadiFunction("lag_sin", [x, τ, t], [(input - x) / τ, input], ["x", "τ", "t"], ["dx", "f"])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(lag_sin), [0.0], (0.0, 10.0), [3.0])
solution = SciMLBase.solve(problem, Tsit5())
solution[[:x, :f]][end]
```

## An input given by data

An input that is known only through measured values, such as a table that holds its
value on each second, has no expression that CasADi can differentiate.
[`callback`](@ref CasADiCodegen.callback) makes a Julia function of this kind a function
that an `MX` graph can call; an `SX` graph holds scalar operations only and cannot call
it. The generated code then calls the Julia function directly. The tuples `(1,), (1,)`
give the size of each argument and of each result. The Julia function takes one array
for each argument and returns one array, or a tuple with one array for each result; a
static array as result keeps the call free of allocation. The derivative of the table
by time is zero between its jumps, which we give as the `jacobian`, a function of the
same arguments that returns the 1×1 Jacobian as an array of one element:

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

The keyword `tstops` makes the solver end a step at each jump of the input, so that no
step crosses a jump, which would cost the solver rejected steps and accuracy.
[Call a Julia function from the model](@ref) describes the other forms of the
derivatives and how the generated code calls the function.

## A model made of components

A larger model is easier to write from components. A component here is a Julia
function that makes the symbols of one lag, with names that start with the name of the
component, and returns them together with the right side for a given input. We connect
two lags in series: the first one has the constant input 1.5, and the state of the
first one is the input of the second one.

```@example ode
function first_order_lag(name, input)
    x, τ = SX.sym(name * "_x"), SX.sym(name * "_τ")
    return (; x, τ, dx = (input - x) / τ)
end

t = SX.sym("t")
lag_1 = first_order_lag("lag_1", 1.5)
lag_2 = first_order_lag("lag_2", lag_1.x)
connected = CasadiFunction("connected", [vertcat(lag_1.x, lag_2.x), vertcat(lag_1.τ, lag_2.τ), t],
                           [vertcat(lag_1.dx, lag_2.dx)])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(connected), [-0.5, 1.0], (0.0, 10.0), [2.0, 4.0])
solution = SciMLBase.solve(problem, Tsit5())
solution[:lag_2_x][end]
```

The connection is a substitution of expressions: the input of `lag_2` is the
expression `lag_1.x`, so that the model holds only the two equations of the states, as
if we had eliminated the inputs by hand.

## Derivatives and their sparsity

The function of the problem holds generated code for the Jacobian of the right side by
the state, which an implicit solver such as `Rodas5P` calls in place of finite
differences or automatic differentiation. We evaluate it at the initial state:

```@example ode
J = zeros(2, 2)
problem.f.jac(J, problem.u0, problem.p, 0.0)
J
```

The Jacobian of the two lags in series has three nonzero elements of four, because the
first state does not depend on the second one. With `sparse = true`, the function
holds a `jac_prototype` with this sparsity pattern, and the Jacobian code writes the
stored values only. The standard library SparseArrays must be loaded:

```@example ode
using SparseArrays
using OrdinaryDiffEqRosenbrock: Rodas5P

f_sparse = SciMLBase.ODEFunction(connected; sparse = true)
problem = SciMLBase.ODEProblem(f_sparse, [-0.5, 1.0], (0.0, 10.0), [2.0, 4.0])
solution = SciMLBase.solve(problem, Rodas5P())
f_sparse.jac_prototype
```

For a model of two states, the sparse form saves nothing and adds the work of the
sparse linear algebra; it pays off for a large model whose states each depend on a few
others, as in a model made of many components.

## Fit the parameters to data

The Lotka-Volterra model describes a prey population ``x`` and a predator population
``y``:

```math
\dot{x} = (\alpha - \beta y) x, \qquad \dot{y} = (\delta x - \gamma) y.
```

We make data by solving the model with known parameters and adding noise, and then
recover the parameters by minimizing the mean squared difference between a solution
and the data. The state and the parameters are vectors of named scalar symbols:

```@example ode
using OrdinaryDiffEqTsit5: AutoTsit5
using OrdinaryDiffEqRosenbrock: Rosenbrock23
using Random: Xoshiro

x, y, t = SX.sym("x"), SX.sym("y"), SX.sym("t")
α, β, γ, δ = SX.sym("α"), SX.sym("β"), SX.sym("γ"), SX.sym("δ")
lotka_volterra = CasadiFunction("lotka_volterra", [vertcat(x, y), vertcat(α, β, γ, δ), t],
                                [vertcat((α - β * y) * x, (δ * x - γ) * y)])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(lotka_volterra), [1.0, 1.0], (0.0, 10.0),
                               [1.5, 1.0, 3.0, 1.0])
times = 0.0:0.1:10.0
data = Array(SciMLBase.solve(problem, Tsit5(); saveat = times))
data .+= 0.01 .* randn(Xoshiro(1), size(data))
nothing # hide
```

The loss needs a problem with the parameters that the optimizer proposes. Because the
proposed parameters can make the model stiff, the loss solves it with `AutoTsit5`, which
switches to the stiff solver `Rosenbrock23` when needed. Because the
parameters of the problem are a plain vector, `remake(problem; p = q)` with the new
vector `q` gives that problem, without names to look up. The optimizer differentiates
the loss with ForwardDiff.jl, so `q` is then a vector of dual numbers, which the
generated code accepts as it accepts any number type. The loss takes the problem, the
times and the data as its second argument, so that it uses no global variable:

```@example ode
function loss(q, (problem, times, data))
    new_problem = SciMLBase.remake(problem; p = q)
    solution = SciMLBase.solve(new_problem, AutoTsit5(Rosenbrock23()); saveat = times)
    return sum(abs2, Array(solution) .- data) / length(data)
end

loss([1.5, 1.0, 3.0, 1.0], (problem, times, data))
```

We minimize the loss with BFGS within bounds on the parameters, starting from values
near the true ones:

```@example ode
using Optimization, OptimizationOptimJL
import ForwardDiff

f_loss = SciMLBase.OptimizationFunction(loss, Optimization.AutoForwardDiff())
fit = SciMLBase.OptimizationProblem(f_loss, [1.2, 0.8, 2.5, 0.8], (problem, times, data);
                                    lb = fill(0.1, 4), ub = fill(4.0, 4))
result = SciMLBase.solve(fit, Optim.BFGS())
result.u
```

The difference from the true parameters ``\alpha = 1.5``, ``\beta = 1``,
``\gamma = 3`` and ``\delta = 1`` is small:

```@example ode
result.u .- [1.5, 1.0, 3.0, 1.0]
```
