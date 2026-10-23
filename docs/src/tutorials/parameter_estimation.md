# Estimate parameters through an integrator

In this tutorial, we fit the four parameters of a Lotka-Volterra model to samples of
its solution by three routes. In the first, the integrator is a node of the CasADi graph,
so that CasADi differentiates the loss through the solver and the integration, the
gradient and the Hessian of the loss are all generated code, with no automatic
differentiation in Julia. We fit with this loss first by a quasi-Newton method and then
with the exact Hessian. In the second route, SciMLSensitivity.jl computes the same
gradient from the generated Jacobians of the right side. In the third, the loss is a Julia
function that makes a SciML problem with the trial parameters by `remake` and solves it,
ForwardDiff.jl differentiates it through the solver, and the parameters to fit can be
chosen by name. The examples need CasADiCodegen.jl, SciMLBase.jl,
OrdinaryDiffEqVerner.jl, OptimizationIpopt.jl, SciMLSensitivity.jl, ForwardDiff.jl,
Optimization.jl, SymbolicIndexingInterface.jl, SciMLStructures.jl and the standard
library Random.

## Write the model

The Lotka-Volterra model describes a prey population ``x_1`` and a predator population
``x_2``:

```math
\dot{x}_1 = (\alpha - \beta x_2)\, x_1, \qquad \dot{x}_2 = (\delta x_1 - \gamma)\, x_2.
```

An integrator of CasADiCodegen.jl takes the model as a dictionary of `MX` expressions:
the state under `"x"`, the parameters under `"p"` and the right side under `"ode"`
(see [Integrate a differential equation](@ref)). We join the four parameters in one
vector, since the fit will treat them as one unknown:

```@example estimation
using CasADiCodegen
using OrdinaryDiffEqVerner: Vern9

x, p = MX.sym("x", 2), MX.sym("p", 4)
α, β, γ, δ = p[1], p[2], p[3], p[4]
model = Dict("x" => x, "p" => p, "ode" => vertcat((α - β * x[2]) * x[1], (δ * x[1] - γ) * x[2]))
nothing # hide
```

## Make the data

The samples are taken at the 101 times 0, 0.1, ..., 10. When `integrator` receives a
vector of times instead of a final time, the function that it returns, `F(x0, p)`,
gives a matrix with the state at each time in one column; the solver runs once from
the first time and stops at each of the others, as `solve` with `saveat` and `tstops`
does in SciML:

```@example estimation
times = collect(0.0:0.1:10.0)
F = integrator("lotka_volterra", Vern9(), model, 0.0, times; abstol = 1e-10, reltol = 1e-10)
simulate = build_function(F; expression = Val{false})[1]    # the out-of-place function
X = simulate([1.0, 1.0], [1.5, 1.0, 3.0, 1.0])
size(X)
```

The data are this solution, for the parameters ``\alpha = 1.5``, ``\beta = 1``,
``\gamma = 3`` and ``\delta = 1`` and the initial state ``(1, 1)``, with noise of
standard deviation 0.01 added to each sample:

```@example estimation
using Random: Xoshiro

data = X .+ 0.01 .* randn(Xoshiro(1), size(X))
nothing # hide
```

## Write the loss

The loss is the mean squared difference between the solution for the parameters
``\theta`` and the data. We make it a CasADi function of two inputs, ``\theta`` and a
2 by 101 symbol `d` for the data, so that the data enter the generated code as an
argument; a fit to other data then reuses the same code, whereas data written into the
graph as a `DM` would need a new function. Calling `F` on symbols places the
integration in the graph of the loss, and its initial state is then a constant `MX`:

```@example estimation
θ = MX.sym("θ", 4)
d = MX.sym("d", 2, length(times))
loss = sum(abs2, F(MX(DM([1.0, 1.0])), θ) - d) / length(data)
objective = CasadiFunction("loss", [θ, d], [loss])
nothing # hide
```

`gradient(loss, θ)` differentiates the loss through the integrator. By default, CasADi
does so with forward sensitivities: the generated code integrates the state together
with its derivatives by the four parameters in one run of the solver, over the same
times. We generate the loss with its gradient and evaluate both at a point away from
the solution:

```@example estimation
loss_and_gradient = build_function(CasadiFunction("loss_and_gradient", [θ, d], [loss, gradient(loss, θ)]);
                                   expression = Val{false})[1]
value, ∇loss = loss_and_gradient([1.3, 0.8, 2.7, 1.1], data)
∇loss
```

With many more parameters than states, `adjoint = true` in the call of `integrator`
makes the reverse derivatives integrate the adjoint equations backward instead, once
over all the times; for four parameters and two states, the forward sensitivities are
the cheaper choice (see [How an integrator is differentiated](@ref)).

## Fit with a quasi-Newton method

`SciMLBase.OptimizationFunction` turns the loss into the function of a SciML
optimization problem, whose first input is the unknown and whose second input is the
parameter of the problem, here the data (see
[Give a CasADi function to a SciML problem](@ref)). It generates the gradient and, by
default, the Hessian. A quasi-Newton method builds its own approximation of the
Hessian from the gradients, so we leave the Hessian out with `hess = false`, which
saves the compilation of its code. The problem holds the start point and the bounds
0.1 and 3 on each parameter:

```@example estimation
import SciMLBase

f = SciMLBase.OptimizationFunction(objective; hess = false)
problem = SciMLBase.OptimizationProblem(f, [1.2, 0.8, 2.5, 0.8], data; lb = fill(0.1, 4), ub = fill(3.0, 4))
nothing # hide
```

We solve it with Ipopt and its limited-memory approximation of the Hessian (L-BFGS), and
silence the output of Ipopt with the options `print_level` and `sb`:

```@example estimation
using OptimizationIpopt: IpoptOptimizer

quiet = Dict{String,Any}("print_level" => 0, "sb" => "yes")
lbfgs = IpoptOptimizer(; hessian_approximation = "limited-memory", additional_options = quiet)
solution = SciMLBase.solve(problem, lbfgs)
solution.u
```

The fit is close to the true parameters ``(1.5, 1, 3, 1)``, and the statistics of the solution give the number of iterations of Ipopt:

```@example estimation
solution.stats.iterations
```

## Fit with the exact Hessian

Without `hess = false`, the function also holds the Hessian of the loss, which CasADi
computes from second-order sensitivities of the integration; its code takes longer to
compile than that of the gradient, but Newton steps with the exact Hessian converge in
fewer iterations. We solve the same problem with this function and with the default
options of Ipopt, which use the Hessian:

```@example estimation
f_exact = SciMLBase.OptimizationFunction(objective)
problem_exact = SciMLBase.OptimizationProblem(f_exact, [1.2, 0.8, 2.5, 0.8], data; lb = fill(0.1, 4), ub = fill(3.0, 4))
solution_exact = SciMLBase.solve(problem_exact, IpoptOptimizer(; additional_options = quiet))
solution_exact.u, solution_exact.stats.iterations
```

In this run, the parameters agreed with those of the L-BFGS fit to about five digits,
the accuracy that the default tolerance of Ipopt gives at this optimum, and Ipopt took
about half of the iterations of L-BFGS.

## Compute the gradient with SciMLSensitivity

The second route keeps the integration in Julia and generates only the right side and
its derivatives. A CasADi function of the state, the parameters and, optionally, the
time becomes an `ODEFunction` with generated code for the Jacobian by the state
(`jac = true`) and by the parameters (`paramjac = true`). These two Jacobians are what
the adjoint methods of SciMLSensitivity.jl need when they are told not to compute
vector-Jacobian products themselves (`autojacvec = false`). The parameters are four scalar
symbols joined by `vertcat`, so that the problem knows them by the names α, β, γ and δ,
which the last sections use; `SX` and `MX` symbols serve alike. The problem holds the
point ``(1.3, 0.8, 2.7, 1.1)`` of the gradient above as its parameters:

```@example estimation
u, t = SX.sym("u", 2), SX.sym("t")
q = [SX.sym(name) for name in ("α", "β", "γ", "δ")]
right_side = CasadiFunction("right_side", [u, vertcat(q...), t], [vertcat((q[1] - q[2] * u[2]) * u[1], (q[4] * u[1] - q[3]) * u[2])])
ode = SciMLBase.ODEProblem(SciMLBase.ODEFunction(right_side; jac = true, paramjac = true), [1.0, 1.0], (0.0, 10.0),
                           [1.3, 0.8, 2.7, 1.1])
nothing # hide
```

`adjoint_sensitivities` integrates the adjoint equations backward along a solution.
The loss is a sum over the samples, so its derivative by the state at the sample `i`,
the index of the time in `times`, enters as a jump of the adjoint at that time, which the
function `dgdu!` writes:

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

The third route keeps the loss in Julia: for each trial ``\theta``, it makes a problem
with these parameters from `ode` and solves it at the sample times. `SciMLBase.remake`
copies a problem with the fields that it is given replaced and shares the others with
the original, among them the `ODEFunction` with its generated code, so that the loss
compiles nothing after its first call with each element type of ``\theta``. The
parameters of `ode` are the `Vector{Float64}` given to `ODEProblem`, which
`remake(ode; p = θ)` replaces by ``\theta`` of any element type. When ``\theta`` holds the
dual numbers of ForwardDiff.jl, the solver promotes the initial state to the same type,
and the generated right side, which is generic Julia code, computes with them, so that
forward-mode differentiation runs through every step of the solver. The problem and the
data enter the loss as its second argument, a tuple, so that one loss serves other
problems and other data at the same sample times:

```@example estimation
import ForwardDiff

function sciml_loss(θ, (problem, data))
    solution = SciMLBase.solve(SciMLBase.remake(problem; p = θ), Vern9(); saveat = times, abstol = 1e-10, reltol = 1e-10)
    sum(abs2, Array(solution) .- data) / length(data)
end

∇forward = ForwardDiff.gradient(θ -> sciml_loss(θ, (ode, data)), [1.3, 0.8, 2.7, 1.1])
maximum(abs, ∇forward - ∇loss) / maximum(abs, ∇loss)
```

The gradient agrees with that of the generated code to the tolerance of the solvers. The
explicit method Vern9 calls the right side alone, so that this route needs neither `jac`
nor `paramjac`, which `ODEFunction` generates by default and `jac = false` and
`paramjac = false` leave out; an implicit method also calls `jac`, which takes dual
numbers as well.

## Set the parameters by name

The problem has the names of the symbols of `right_side` as its symbolic system, the
object through which SciMLBase.jl looks up a name: the names of scalar symbols, such as
α, and for a symbol of several elements, such as `u`, the names `u_0` and `u_1` of its
elements, counted from 0, so that `u_0` is ``x_1``. A parameter symbol of several
elements also has its own name for all of them (see
[`ODEFunction`](@ref SciMLBase.ODEFunction(::CasADiCodegen.Function))). The functions of
SymbolicIndexingInterface.jl read and write a state or a parameter by these names.
`getp` returns a function that reads parameters from a problem, and `setp` one that writes
them in place into its vector of `Float64`, which cannot hold dual numbers. `setsym_oop`
returns a function that gives a new initial state and a new parameter vector, of which
the one that receives the values takes their element type, and is therefore the setter
for a loss that ForwardDiff.jl differentiates; `remake` then promotes the other. We set two parameters of a copy of `ode`, with a
parameter vector of its own, and read one back:

```@example estimation
using SymbolicIndexingInterface: getp, setp, setsym_oop

trial = SciMLBase.remake(ode; p = copy(ode.p))
setp(trial, [:α, :δ])(trial, [1.4, 0.9])
trial.p, getp(trial, :δ)(trial)
```

`setsym_oop` suits a fit of some of the parameters, with the others kept at their values
in the problem. The loss below fits α and δ alone, and its gradient at the values of
`ode` agrees with the first and the last element of the gradient above to the tolerance
of the solver:

```@example estimation
function partial_loss(θ, (problem, set, data))
    u0, p = set(problem, θ)
    solution = SciMLBase.solve(SciMLBase.remake(problem; u0, p), Vern9(); saveat = times, abstol = 1e-10, reltol = 1e-10)
    sum(abs2, Array(solution) .- data) / length(data)
end

set_αδ = setsym_oop(ode, [:α, :δ])
ForwardDiff.gradient(θ -> partial_loss(θ, (ode, set_αδ, data)), [1.3, 1.1]) - ∇forward[[1, 4]]
```

`setsym_oop` looks the names up once, when it is called on them, whereas
`remake(ode; p = [:α => 1.4])` looks them up at each call, which costs more (see the
table at the end).

Code written for SciML problems whose parameter object has several portions reaches the
parameters to tune through the interface of SciMLStructures.jl. For a plain vector,
`canonicalize` with the portion `Tunable()` returns the vector itself, and `replace`
returns a copy with other values, of their element type, so that such code runs on
these problems unchanged:

```@example estimation
import SciMLStructures
using SciMLStructures: Tunable

tunables, _, _ = SciMLStructures.canonicalize(Tunable(), ode.p)
tunables === ode.p, eltype(SciMLStructures.replace(Tunable(), ode.p, ForwardDiff.Dual.([1.5, 1.0, 3.0, 1.0], 1.0)))
```

SciMLStructures.jl does not define `replace!` for a vector; `copyto!(trial.p, θ)` or
a setter of `setp` writes in place instead.

Reverse-mode packages such as Zygote.jl differentiate `remake(problem; p = θ)` and the
solve through the adjoint method that the keyword `sensealg` of `solve` names, such as
`InterpolatingAdjoint(autojacvec = false)`, which uses the generated Jacobians.
`setsym_oop` and `remake` with names write into a copy of the parameter vector, and
`setp` into the vector itself; for each of them Zygote.jl throws "Mutating arrays is not
supported". A loss for Zygote.jl that fits
α and δ alone therefore builds the whole vector without mutation, as in
`p = [θ[1]; problem.p[2:3]; θ[2]]`, and passes it to `remake`.

## Fit with the ForwardDiff gradient

`SciMLBase.OptimizationFunction` with `Optimization.AutoForwardDiff()` turns the Julia
loss into the function of an optimization problem whose gradient ForwardDiff.jl computes.
The second argument of the loss is the parameter of the optimization problem, here the
tuple of the ODE problem and the data. We solve it from the same start point and with the
same bounds and the same L-BFGS method of Ipopt as the first fit:

```@example estimation
import Optimization

f_forward = SciMLBase.OptimizationFunction(sciml_loss, Optimization.AutoForwardDiff())
problem_forward = SciMLBase.OptimizationProblem(f_forward, [1.2, 0.8, 2.5, 0.8], (ode, data); lb = fill(0.1, 4), ub = fill(3.0, 4))
solution_forward = SciMLBase.solve(problem_forward, lbfgs)
maximum(abs, solution_forward.u - solution.u), solution_forward.stats.iterations
```

The fit agrees with the first one to the accuracy of the default tolerance of Ipopt, in
a similar number of iterations. A fit of α and δ alone takes `partial_loss` in the same
way, with the tuple `(ode, set_αδ, data)` as the parameter and a start point and bounds
of two elements.

The three routes give one gradient and one fit to the tolerance of the solvers, and they
differ in what they suit. The first gives the loss, its gradient and its Hessian as one
generated function each, ready for an optimizer, with no automatic differentiation in
Julia. The second suits a problem whose loss and solver options are already written for
SciMLSensitivity.jl and needs the derivatives of the right side only. The third keeps the
loss as Julia code around a SciML problem, which may change other fields of the problem
by `remake` or fit some parameters by name, at the cost of differentiating the solver with
dual numbers, whose work grows with the number of parameters. For many more parameters
than states, the adjoint equations, of the first route with `adjoint = true` or of the
second route, grow with the number of states instead. The costs of one call, measured
once with the settings of this page, were as follows:

| Call | Time |
|:-----|-----:|
| `remake(ode; p = θ)` | 30 ns |
| `setsym_oop` of α and δ, then `remake` | 0.2 µs |
| `remake(ode; p = [:α => 1.4, :δ => 0.9])` | 8.7 µs |
| `sciml_loss` (one solve) | 60 µs |
| Loss and gradient of the generated code, first route | 89 µs |
| Gradient by `adjoint_sensitivities`, second route | 937 µs |
| Gradient by ForwardDiff.jl through `sciml_loss`, third route | 102 µs |

For four parameters, the generated loss and gradient of the first route, which integrate
the state and its sensitivities in one run of the solver, cost about as much as the
ForwardDiff.jl gradient of the third, and the adjoint of the second route costs ten times
as much.
