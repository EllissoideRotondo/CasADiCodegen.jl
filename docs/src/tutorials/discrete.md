# Discrete-time systems

A discrete-time system is a map that gives the state at the next step from the state at
the current one. In this tutorial, we write such maps as CasADi functions, step an
explicit map with `FunctionMap` of OrdinaryDiffEqFunctionMap.jl and solve an implicit
map with `IDSolve` of ImplicitDiscreteSolve.jl.

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

where ``t_k = t_0 + k \, \Delta t`` is the time of step ``k``; the map receives the time
of the new state.
[`DiscreteProblem`](@ref SciMLBase.DiscreteProblem(::CasADiCodegen.Function, ::Any, ::Any))
takes the map as a CasADi function `f(u)`, `f(u, p)` or `f(u, p, t)`, whose first result
is the next state as a column vector. `FunctionMap` applies the map once per step, with
``\Delta t = 1`` unless `solve` receives another `dt`. The map does not receive
``\Delta t``, so a map that depends on a step length takes it as a parameter.

## An epidemic model

The SIR model divides a population into the susceptible ``S``, the infected ``I`` and the
recovered ``R``. A rate ``r`` that acts during a step ``\delta t`` changes the proportion
``1 - e^{-r \delta t}`` of a group, so that one step has

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

The step ``\delta t`` is a parameter, and the solver keeps `dt = 1`, so the times of the
solution count the steps. A second result of the function, here the population
``N = S + I + R``, becomes an observed variable of the solution under the name of that
result:

```@example discrete
S, I, R = SX.sym("S"), SX.sym("I"), SX.sym("R")
β, c, γ, δt = SX.sym("β"), SX.sym("c"), SX.sym("γ"), SX.sym("δt")
proportion(rate) = 1 - exp(-rate * δt)
infection = proportion(β * c * I / (S + I + R)) * S
recovery = proportion(γ) * I
sir = CasadiFunction("sir", [vcat(S, I, R), vcat(β, c, γ, δt)],
                     [vcat(S - infection, I + infection - recovery, R + recovery), S + I + R],
                     ["u", "p"], ["next", "N"])
nothing # hide
```

We start with 990 susceptible and 10 infected people and take 100 steps:

```@example discrete
problem = SciMLBase.DiscreteProblem(sir, [990.0, 10.0, 0.0], (0.0, 100.0), [0.05, 10.0, 0.25, 0.1])
solution = SciMLBase.solve(problem, FunctionMap())
solution.u[end]
```

The solution is indexed by the names of the scalar symbols, as for an ODE. The map
conserves the observed population:

```@example discrete
solution[:I][[1, 51, 101]], extrema(solution[:N])
```

With OrdinaryDiffEqFunctionMap.jl loaded, `solve` without a solver also chooses
`FunctionMap`. For a small state, `inplace = false` generates a map that returns an
`SVector`; the state must then be an `SVector`, and `SVector` parameters keep the step
free of allocations:

```@example discrete
static = SciMLBase.DiscreteProblem(sir, SVector(990.0, 10.0, 0.0), (0.0, 100.0), SVector(0.05, 10.0, 0.25, 0.1);
                                   inplace = false)
SciMLBase.solve(static, FunctionMap()).u[end]
```

## Several past values

A recursion over more than one past value, such as the Fibonacci series
``x_{k+1} = x_k + x_{k-1}``, is a map of a larger state that holds the values the
recursion needs:

```@example discrete
older, newer = SX.sym("older"), SX.sym("newer")
fibonacci = CasadiFunction("fibonacci", [vcat(older, newer)], [vcat(newer, older + newer)])
series = SciMLBase.solve(SciMLBase.DiscreteProblem(fibonacci, [1.0, 1.0], (0.0, 10.0)), FunctionMap())
series[:newer]
```

A delay of ``d`` steps works in the same way, with ``d`` copies of the delayed variable in
the state.

## Implicit maps

In an implicit map, the next state is the solution of the system

```math
0 = F(u_{k+1}, u_k, p, t_{k+1}).
```

As an example, we take the stiff equation ``y' = -\lambda (y - \cos t)`` with
``\lambda = 50`` and the step ``h = 0.1``, for which explicit Euler diverges, and solve it
with the backward Euler method, which is stable for every step:

```math
y_{k+1} = y_k - h \lambda \left(y_{k+1} - \cos t_{k+1}\right)
```

[`ImplicitDiscreteProblem`](@ref SciMLBase.ImplicitDiscreteProblem(::CasADiCodegen.Function, ::Any, ::Tuple))
takes the residual ``F`` as a CasADi function `F(u⁺, u)`, `F(u⁺, u, p)` or
`F(u⁺, u, p, t)`, whose one result has an element per state; a longer residual makes each
step a least-squares problem. The residual has no observed variables, so we add the error
``e = y - \cos t`` as an algebraic state:

```@example discrete
y, e, y_prev, e_prev = SX.sym("y"), SX.sym("e"), SX.sym("y_prev"), SX.sym("e_prev")
λ, h, t = SX.sym("λ"), SX.sym("h"), SX.sym("t")
euler = CasadiFunction("backward_euler", [vcat(y, e), vcat(y_prev, e_prev), vcat(λ, h), t],
                       [vcat(y - y_prev + h * λ * (y - cos(t)), e - (y - cos(t)))])
implicit = SciMLBase.ImplicitDiscreteProblem(euler, [0.0, -1.0], (0.0, 2.0), [50.0, 0.1])
nothing # hide
```

`IDSolve` applies Newton's method with dual numbers through the generated code and needs
the in-place form. Two of its defaults do not suit a discretized ODE: it adapts its step,
whereas the residual assumes ``h``, so we pass `dt = 0.1` and `adaptive = false`; and it
solves ``F(u, u_0, p, t_0) = 0`` before the first step and saves ``u`` in place of
``u_0``, which `initializealg = SciMLBase.NoInit()` skips. With `NoInit`, ``u_0`` must
satisfy the algebraic equation, as ``e = -1`` does:

```@example discrete
path = SciMLBase.solve(implicit, IDSolve(); dt = 0.1, adaptive = false, initializealg = SciMLBase.NoInit())
path[:y][1:4], path[:e][end]
```

The state reaches ``\cos t`` within a few steps, after which the error stays close to the
lag ``\sin t / \lambda`` of the exact solution.
