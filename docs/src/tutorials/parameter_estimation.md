# Estimate parameters through an integrator

In this tutorial, we fit the four parameters of a Lotka-Volterra model to samples of
its solution by two routes. In the first, the integrator is a node of the CasADi graph,
so that CasADi differentiates the loss through the solver and the integration, the
gradient and the Hessian of the loss are all generated code, with no automatic
differentiation in Julia. We fit with this loss first by a quasi-Newton method and then
with the exact Hessian. In the second route, SciMLSensitivity.jl computes the same
gradient from the generated Jacobians of the right side. The examples need CasADiCodegen.jl, SciMLBase.jl,
OrdinaryDiffEqVerner.jl, OptimizationIpopt.jl, SciMLSensitivity.jl and the standard
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
vector-Jacobian products themselves (`autojacvec = false`):

```@example estimation
u, q, t = SX.sym("u", 2), SX.sym("q", 4), SX.sym("t")
right_side = CasadiFunction("right_side", [u, q, t], [vertcat((q[1] - q[2] * u[2]) * u[1], (q[4] * u[1] - q[3]) * u[2])])
ode = SciMLBase.ODEProblem(SciMLBase.ODEFunction(right_side; jac = true, paramjac = true), [1.0, 1.0], (0.0, 10.0),
                           [1.3, 0.8, 2.7, 1.1])
nothing # hide
```

`adjoint_sensitivities` integrates the adjoint equations backward along a solution.
The loss is a sum over the samples, so its derivative by the state at the sample `i`
enters as a jump of the adjoint at that time, which the function `dgdu!` writes:

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

The two gradients agree to the tolerance of the solvers. The first route gives the
loss, its gradient and its Hessian as one generated function each, ready for an
optimizer; the second one suits a problem whose loss and solver options are already
written for SciMLSensitivity.jl and needs the derivatives of the right side only.
