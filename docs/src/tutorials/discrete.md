# Discrete-time systems

Many models change their state at separate instants rather than continuously: an
epidemic counted once a day, a population censused once a year, a digital controller
that updates its output at each tick of its clock, or a differential equation after a
numerical method has replaced it by its steps. Such a discrete-time system is a map,
which gives the state at the next step from the state at the current one, and its
solution is the sequence of states that the map produces from an initial state. In this
tutorial, we write the map of a discrete-time system as a CasADi function and step it
with the `FunctionMap` solver of OrdinaryDiffEqFunctionMap.jl. We start with an epidemic
model, then write a recursion over two past values as a map of a larger state, and
finally solve an implicit map, in which the next state is the solution of a system of
equations, with `IDSolve` of ImplicitDiscreteSolve.jl.

The examples need CasADiCodegen.jl, SciMLBase.jl, OrdinaryDiffEqFunctionMap.jl,
ImplicitDiscreteSolve.jl and StaticArrays.jl.

```@example discrete
using CasADiCodegen, StaticArrays
import SciMLBase
using OrdinaryDiffEqFunctionMap: FunctionMap
using ImplicitDiscreteSolve: IDSolve
```

## A map from one step to the next

A discrete-time system with the state ``u`` and the parameters ``p`` is the map

```math
u_{k+1} = f(u_k, p, t_{k+1}),
```

where ``t_k = t_0 + k \, \Delta t`` is the time of step ``k``. The time that the map
receives is that of the new state, which matters only when ``f`` depends on the time.
[`DiscreteProblem`](@ref SciMLBase.DiscreteProblem(::CasADiCodegen.Function, ::Any, ::Any))
takes the map as a CasADi function `f(u)`, `f(u, p)` or `f(u, p, t)` of the state, the
parameters and the time, in this order, whose first result is the next state, a column
vector of the size of the state.
The solver `FunctionMap` applies the map once per step, with ``\Delta t = 1`` unless
`solve` receives another `dt`, so that a problem over the times ``(0, n)`` has ``n``
steps. The step ``\Delta t`` sets the number of steps and the times of the solution
only, and the map does not receive it: a map that depends on a length of step takes it
as a parameter, and the times of the solution are those of the model only when that
parameter equals `dt`.

## An epidemic model

The SIR model divides a population into the susceptible ``S``, the infected ``I`` and the
recovered ``R``. During one step of length ``\delta t``, a susceptible person meets ``c``
others per unit of time and is infected by an infected one with the probability
``\beta``, while an infected person recovers at the rate ``\gamma``. A rate ``r`` that
acts during ``\delta t`` turns into the proportion ``1 - e^{-r \delta t}`` of the group
that changes, so that the number of new infections and of recoveries in one step are

```math
\begin{aligned}
\text{infection} &= \left(1 - e^{-\beta c \delta t I / (S + I + R)}\right) S, \\
\text{recovery} &= \left(1 - e^{-\gamma \delta t}\right) I,
\end{aligned}
```

and the map moves these numbers from one group to the next:

```math
S_{k+1} = S_k - \text{infection}, \quad
I_{k+1} = I_k + \text{infection} - \text{recovery}, \quad
R_{k+1} = R_k + \text{recovery}.
```

The length of a step ``\delta t`` is a parameter of the model. We keep the default
`dt = 1` of the solver, so that the times of the solution count the steps, and the 100
steps below are 10 units of the time of the model. The CasADi function
takes the state and the parameters; a second result, the size ``N = S + I + R`` of the
population, becomes an observed variable of the solution, that is, a quantity computed
from the state at each step, with the name that the function gives to that result
(the names of the arguments and of the first result are not used):

```@example discrete
S, I, R = SX.sym("S"), SX.sym("I"), SX.sym("R")
β, c, γ, δt = SX.sym("β"), SX.sym("c"), SX.sym("γ"), SX.sym("δt")
proportion(rate) = 1 - exp(-rate * δt)
infection = proportion(β * c * I / (S + I + R)) * S
recovery = proportion(γ) * I
sir = CasadiFunction("sir", [vertcat(S, I, R), vertcat(β, c, γ, δt)],
                     [vertcat(S - infection, I + infection - recovery, R + recovery), S + I + R],
                     ["u", "p"], ["next", "N"])
nothing # hide
```

We start with 990 susceptible and 10 infected people and take 100 steps:

```@example discrete
problem = SciMLBase.DiscreteProblem(sir, [990.0, 10.0, 0.0], (0.0, 100.0), [0.05, 10.0, 0.25, 0.1])
solution = SciMLBase.solve(problem, FunctionMap())
solution.u[end]
```

The solution has a state per step, the initial one included, and it knows the names of
the scalar symbols, as for an ODE, so that `solution[:I]` is the sequence of the
infected, `problem.ps[:β]` the value of ``\beta``, and `solution[:N]` the sequence of the
observed size of the population, which the map conserves. We look at the infected at the
steps 0, 50 and 100, and at the smallest and the largest size of the population:

```@example discrete
solution[:I][[1, 51, 101]], extrema(solution[:N])
```

`solve` without a solver also chooses `FunctionMap` for a `DiscreteProblem` when
OrdinaryDiffEqFunctionMap.jl is loaded.

For a small state, the out-of-place form with static arrays avoids the arrays on the
heap: with `inplace = false`, the generated map returns a new `SVector` instead of
writing into an array, the state must be an `SVector`, and parameters as an `SVector`
keep the step free of allocations too:

```@example discrete
static = SciMLBase.DiscreteProblem(sir, SVector(990.0, 10.0, 0.0), (0.0, 100.0), SVector(0.05, 10.0, 0.25, 0.1);
                                   inplace = false)
SciMLBase.solve(static, FunctionMap()).u[end]
```

## Several past values

A recursion that reaches back more than one step, such as the Fibonacci series
``x_{k+1} = x_k + x_{k-1}``, is a map of a larger state, which holds the values that the
recursion still needs. Here the state is the pair ``(x_{k-1}, x_k)``, and the map moves
the newer value into the place of the older one:

```@example discrete
older, newer = SX.sym("older"), SX.sym("newer")
fibonacci = CasadiFunction("fibonacci", [vertcat(older, newer)], [vertcat(newer, older + newer)])
series = SciMLBase.solve(SciMLBase.DiscreteProblem(fibonacci, [1.0, 1.0], (0.0, 10.0)), FunctionMap())
series[:newer]
```

The initial state holds the two values before the first step, and each step adds one
term of the series. A delay of ``d`` steps in a larger model works in the same way, with
``d`` copies of the delayed variable in the state.

## Implicit maps

In an implicit map, the next state appears on both sides of the equations, which then
define it as the solution of the system

```math
0 = F(u_{k+1}, u_k, p, t_{k+1}).
```

Implicit maps arise from implicit methods of integration, and from models with algebraic
equations, each of which relates variables of the same step and contains no value of the
previous step. As an
example, we take the stiff equation ``y' = -\lambda (y - \cos t)`` with ``\lambda = 50``,
whose solution approaches ``\cos t`` within a time of order ``1/\lambda``. With the step
``h = 0.1``, the explicit Euler method multiplies the error by ``1 - \lambda h = -4`` at
each step and diverges, while the backward Euler method

```math
y_{k+1} = y_k - h \lambda \left(y_{k+1} - \cos t_{k+1}\right)
```

is stable for every step. We add the error ``e = y - \cos t`` as an algebraic variable:
the residual function of `ImplicitDiscreteProblem` has one result only, and thus no
observed variables, so that a quantity derived from the state becomes an element of the
state with an equation of its own.

[`ImplicitDiscreteProblem`](@ref SciMLBase.ImplicitDiscreteProblem(::CasADiCodegen.Function, ::Any, ::Tuple))
takes the residual ``F`` as a CasADi function `F(u⁺, u)`, `F(u⁺, u, p)` or
`F(u⁺, u, p, t)` of the next state, the current state, the parameters and the time. Its
one result has an element per element of the state; a longer residual makes each step a
least-squares problem. The states have the names of the symbols of the next state:

```@example discrete
y, e, y_prev, e_prev = SX.sym("y"), SX.sym("e"), SX.sym("y_prev"), SX.sym("e_prev")
λ, h, t = SX.sym("λ"), SX.sym("h"), SX.sym("t")
euler = CasadiFunction("backward_euler", [vertcat(y, e), vertcat(y_prev, e_prev), vertcat(λ, h), t],
                       [vertcat(y - y_prev + h * λ * (y - cos(t)), e - (y - cos(t)))])
implicit = SciMLBase.ImplicitDiscreteProblem(euler, [0.0, -1.0], (0.0, 2.0), [50.0, 0.1])
nothing # hide
```

At each step, `IDSolve` solves the residual for the next state by Newton's method, which
differentiates the generated code with dual numbers, since a SciML implicit discrete
function has no place for a Jacobian. `IDSolve` writes the state in place, so that it
needs the in-place form, not static arrays. Two of its defaults do not suit this model.
It adapts the length of its steps to the convergence of Newton's method, whereas the
residual assumes the step ``h``, so we fix the step with `dt = 0.1` and
`adaptive = false`. And it takes the initial state as the state before the first time:
before the first step, it solves ``F(u, u_0, p, t_0) = 0`` and saves the solution
``u``, not ``u_0``, as the first state. The option `initializealg = SciMLBase.NoInit()`
skips this solve, so that the solution starts from ``u_0``, which must then satisfy the
algebraic equation, as ``e = 0 - \cos 0 = -1`` does. For a residual that discretizes a
differential equation, such as this one, the default would take a step of the method
before ``t_0`` and save its result at ``t_0``, so that `NoInit` is the usual choice:

```@example discrete
path = SciMLBase.solve(implicit, IDSolve(); dt = 0.1, adaptive = false, initializealg = SciMLBase.NoInit())
path[:y][1:4], path[:e][end]
```

The state reaches ``\cos t`` within a few steps, after which the error stays close to
``\sin t / \lambda``, the lag with which the exact solution, too, follows its moving
target.
