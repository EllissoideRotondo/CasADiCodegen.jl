# Estimate parameters through an integrator

In this tutorial, we fit the four parameters of a Lotka-Volterra model to samples of
its solution by three routes: a loss whose gradient and Hessian CasADi generates through
the integrator, the adjoint method of SciMLSensitivity.jl with generated Jacobians, and a
Julia loss around `remake` that ForwardDiff.jl differentiates.

## Write the model

The model of prey ``x_1`` and predators ``x_2`` is

```math
\dot{x}_1 = (\alpha - \beta x_2)\, x_1, \qquad \dot{x}_2 = (\delta x_1 - \gamma)\, x_2.
```

An integrator takes the model as a dictionary of `MX` expressions (see
[Integrate a differential equation](@ref)), with the four parameters in one vector:

```@example estimation
using CasADiCodegen
using OrdinaryDiffEqVerner: Vern9

x, p = MX.sym("x", 2), MX.sym("p", 4)
α, β, γ, δ = p[1], p[2], p[3], p[4]
model = Dict("x" => x, "p" => p, "ode" => vcat((α - β * x[2]) * x[1], (δ * x[1] - γ) * x[2]))
nothing # hide
```

## Make the data

Given a vector of times, `integrator` returns `F(x0, p)`, whose result holds the state at
each time in one column; the solver runs once and stops at each time:

```@example estimation
times = collect(0.0:0.1:10.0)
F = integrator("lotka_volterra", Vern9(), model, 0.0, times; abstol = 1e-10, reltol = 1e-10)
simulate = build_function(F; expression = Val{false})[1]    # the out-of-place function
X = simulate([1.0, 1.0], [1.5, 1.0, 3.0, 1.0])
size(X)
```

The data are this solution for ``(\alpha, \beta, \gamma, \delta) = (1.5, 1, 3, 1)`` with noise
of standard deviation 0.01:

```@example estimation
using Random: Xoshiro

data = X .+ 0.01 .* randn(Xoshiro(1), size(X))
nothing # hide
```

## Write the loss

The loss is the mean squared error of the solution for ``\theta``. The data are a second
argument `d`, so a fit to other data reuses the generated code. Calling `F` on symbols
places the integration in the graph of the loss:

```@example estimation
θ = MX.sym("θ", 4)
d = MX.sym("d", 2, length(times))
loss = sum(abs2, F(MX(DM([1.0, 1.0])), θ) - d) / length(data)
objective = CasadiFunction("loss", [θ, d], [loss])
nothing # hide
```

`gradient(loss, θ)` differentiates through the integrator, by default with forward
sensitivities integrated with the state. We evaluate the loss and its gradient away from
the optimum:

```@example estimation
loss_and_gradient = build_function(CasadiFunction("loss_and_gradient", [θ, d], [loss, gradient(loss, θ)]);
                                   expression = Val{false})[1]
value, ∇loss = loss_and_gradient([1.3, 0.8, 2.7, 1.1], data)
∇loss
```

With many more parameters than states, `adjoint = true` in `integrator` integrates the
adjoint equations backward instead (see [How an integrator is differentiated](@ref)).

## Fit with a quasi-Newton method

`SciMLBase.OptimizationFunction` takes the unknown first and the data as the problem
parameter (see [Give a CasADi function to a SciML problem](@ref)). It generates the
gradient and, by default, the Hessian; a quasi-Newton method needs no Hessian, so
`hess = false` skips its compilation:

```@example estimation
import SciMLBase

f = SciMLBase.OptimizationFunction(objective; hess = false)
problem = SciMLBase.OptimizationProblem(f, [1.2, 0.8, 2.5, 0.8], data; lb = fill(0.1, 4), ub = fill(3.0, 4))
nothing # hide
```

We solve with the L-BFGS approximation of Ipopt and silence its output:

```@example estimation
using OptimizationIpopt: IpoptOptimizer

quiet = Dict{String,Any}("print_level" => 0, "sb" => "yes")
lbfgs = IpoptOptimizer(; hessian_approximation = "limited-memory", additional_options = quiet)
solution = SciMLBase.solve(problem, lbfgs)
solution.u
```

The fit is close to the true parameters ``(1.5, 1, 3, 1)``, and the statistics give the
number of iterations:

```@example estimation
solution.stats.iterations
```

## Fit with the exact Hessian

By default, the function also holds the Hessian, which CasADi computes from
second-order sensitivities; its code takes longer to compile, but Newton steps need fewer
iterations:

```@example estimation
f_exact = SciMLBase.OptimizationFunction(objective)
problem_exact = SciMLBase.OptimizationProblem(f_exact, [1.2, 0.8, 2.5, 0.8], data; lb = fill(0.1, 4), ub = fill(3.0, 4))
solution_exact = SciMLBase.solve(problem_exact, IpoptOptimizer(; additional_options = quiet))
solution_exact.u, solution_exact.stats.iterations
```

In this run, the fit agreed with L-BFGS to about five digits in about half of the
iterations.

## Compute the gradient with SciMLSensitivity

The second route integrates in Julia and generates only the right side, with its
Jacobians by the state (`jac = true`) and by the parameters (`paramjac = true`), which
SciMLSensitivity.jl uses with `autojacvec = false`. The parameters are named scalar
symbols, which the later sections use:

```@example estimation
u, t = SX.sym("u", 2), SX.sym("t")
q = [SX.sym(name) for name in ("α", "β", "γ", "δ")]
right_side = CasadiFunction("right_side", [u, vcat(q...), t], [vcat((q[1] - q[2] * u[2]) * u[1], (q[4] * u[1] - q[3]) * u[2])])
ode = SciMLBase.ODEProblem(SciMLBase.ODEFunction(right_side; jac = true, paramjac = true), [1.0, 1.0], (0.0, 10.0),
                           [1.3, 0.8, 2.7, 1.1])
nothing # hide
```

The loss is a sum over the samples, so its derivative by the state at sample `i` is a
jump of the adjoint, which `dgdu!` writes:

```@example estimation
using SciMLSensitivity: adjoint_sensitivities, InterpolatingAdjoint

function dgdu!(out, u, p, t, i)
    out .= 2 .* (u .- view(data, :, i)) ./ length(data)
    return nothing
end

forward = SciMLBase.solve(ode, Vern9(); abstol = 1e-10, reltol = 1e-10)
_, ∇sciml = adjoint_sensitivities(forward, Vern9(); t = times, dgdu_discrete = dgdu!,
                                  sensealg = InterpolatingAdjoint(autojacvec = false), abstol = 1e-10, reltol = 1e-10)
maximum(abs, vec(∇sciml) - ∇loss) / maximum(abs, ∇loss)
```

The two gradients agree to the tolerance of the solvers.

## Differentiate through `remake` and the solver

The third route remakes `ode` with each trial ``\theta`` and solves it. `remake` shares
the `ODEFunction` with the original, so nothing is compiled again, and the generated right
side computes with the dual numbers of ForwardDiff.jl:

```@example estimation
import ForwardDiff

function sciml_loss(θ, (problem, data))
    solution = SciMLBase.solve(SciMLBase.remake(problem; p = θ), Vern9(); saveat = times, abstol = 1e-10, reltol = 1e-10)
    sum(abs2, Array(solution) .- data) / length(data)
end

∇forward = ForwardDiff.gradient(θ -> sciml_loss(θ, (ode, data)), [1.3, 0.8, 2.7, 1.1])
maximum(abs, ∇forward - ∇loss) / maximum(abs, ∇loss)
```

The explicit method `Vern9` needs neither `jac` nor `paramjac`, which `jac = false` and
`paramjac = false` leave out; an implicit method also calls `jac`, which accepts dual
numbers.

## Set the parameters by name

The problem knows the names of the scalar symbols, and the elements `u_0`, `u_1` of a
vector symbol, counted from 0 (see
[`ODEFunction`](@ref SciMLBase.ODEFunction(::CasADiCodegen.Function))).
SymbolicIndexingInterface.jl reads and writes by these names: `setp` writes in place into
the `Float64` vector, which cannot hold dual numbers, and `setsym_oop` returns new vectors
of the element type of the values, which suits a loss under ForwardDiff.jl:

```@example estimation
using SymbolicIndexingInterface: getp, setp, setsym_oop

trial = SciMLBase.remake(ode; p = copy(ode.p))
setp(trial, [:α, :δ])(trial, [1.4, 0.9])
trial.p, getp(trial, :δ)(trial)
```

A loss that fits α and δ alone, with the others kept, gives the corresponding elements of
the gradient above:

```@example estimation
function partial_loss(θ, (problem, set, data))
    u0, p = set(problem, θ)
    solution = SciMLBase.solve(SciMLBase.remake(problem; u0, p), Vern9(); saveat = times, abstol = 1e-10, reltol = 1e-10)
    sum(abs2, Array(solution) .- data) / length(data)
end

set_αδ = setsym_oop(ode, [:α, :δ])
ForwardDiff.gradient(θ -> partial_loss(θ, (ode, set_αδ, data)), [1.3, 1.1]) - ∇forward[[1, 4]]
```

`setsym_oop` looks the names up once, whereas `remake(ode; p = [:α => 1.4])` looks them up
at each call (see the table at the end).

For a plain parameter vector, `canonicalize` of SciMLStructures.jl with `Tunable()`
returns the vector itself, and `replace` returns a copy with the element type of the new
values:

```@example estimation
import SciMLStructures
using SciMLStructures: Tunable

tunables, _, _ = SciMLStructures.canonicalize(Tunable(), ode.p)
tunables === ode.p, eltype(SciMLStructures.replace(Tunable(), ode.p, ForwardDiff.Dual.([1.5, 1.0, 3.0, 1.0], 1.0)))
```

SciMLStructures.jl does not define `replace!` for a vector; use `copyto!(trial.p, θ)` or
`setp`.

Zygote.jl differentiates `remake(problem; p = θ)` and the solve through the adjoint method
of the keyword `sensealg`, such as `InterpolatingAdjoint(autojacvec = false)`. It throws
"Mutating arrays is not supported" for `setsym_oop`, `setp` and `remake` with names, so a
Zygote loss that fits α and δ builds the vector without mutation, as
`p = [θ[1]; problem.p[2:3]; θ[2]]`.

## Fit with the ForwardDiff gradient

With `Optimization.AutoForwardDiff()`, the Julia loss becomes an optimization problem
with the ODE problem and the data as its parameter. We use the start point, bounds and
method of the first fit:

```@example estimation
import Optimization

f_forward = SciMLBase.OptimizationFunction(sciml_loss, Optimization.AutoForwardDiff())
problem_forward = SciMLBase.OptimizationProblem(f_forward, [1.2, 0.8, 2.5, 0.8], (ode, data); lb = fill(0.1, 4), ub = fill(3.0, 4))
solution_forward = SciMLBase.solve(problem_forward, lbfgs)
maximum(abs, solution_forward.u - solution.u), solution_forward.stats.iterations
```

The fit agrees with the first one, in a similar number of iterations; `partial_loss`
fits α and δ alone in the same way.

The first route gives the loss, gradient and Hessian as generated code with no automatic
differentiation in Julia; the second fits a loss written for SciMLSensitivity.jl; the
third keeps a Julia loss that can change any field by `remake`, at a cost that grows with
the number of parameters. The cost of one call, measured once with the settings of this
page, was:

| Call | Time |
|:-----|-----:|
| `remake(ode; p = θ)` | 30 ns |
| `setsym_oop` of α and δ, then `remake` | 0.2 µs |
| `remake(ode; p = [:α => 1.4, :δ => 0.9])` | 8.7 µs |
| `sciml_loss` (one solve) | 60 µs |
| Loss and gradient of the generated code, first route | 89 µs |
| Gradient by `adjoint_sensitivities`, second route | 937 µs |
| Gradient by ForwardDiff.jl through `sciml_loss`, third route | 102 µs |

For four parameters, the first and third routes cost about the same, and the adjoint of
the second costs ten times as much.
