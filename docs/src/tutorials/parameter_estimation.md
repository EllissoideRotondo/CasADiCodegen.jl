# Fit ODE parameters to data in three ways

In this tutorial, we fit the four parameters of a Lotka-Volterra model to samples of
its solution in three ways: a loss whose gradient and Hessian CasADi generates through
the integrator, the adjoint method of SciMLSensitivity.jl with generated Jacobians, and a
Julia loss around `remake` that ForwardDiff.jl differentiates.

## Write the predator-prey model

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
dynamics = vcat((α - β * x[2]) * x[1], (δ * x[1] - γ) * x[2])
model = Dict("x" => x, "p" => p, "ode" => dynamics)
nothing # hide
```

## Generate noisy measurements

Given a vector of times, `integrator` returns `F(x0, p)`, whose result holds the state at
each time in one column; the solver runs once and stops at each time:

```@example estimation
times = collect(0.0:0.1:10.0)
F = integrator("lotka_volterra", Vern9(), model, 0.0, times;
               abstol = 1e-10, reltol = 1e-10)
simulate = build_function(F; expression = Val{false})[1]    # the out-of-place function
x_start = [1.0, 1.0]
p_true = [1.5, 1.0, 3.0, 1.0]
X = simulate(x_start, p_true)
size(X)
```

The data are this solution for ``(\alpha, \beta, \gamma, \delta) = (1.5, 1, 3, 1)`` with
noise of standard deviation 0.01:

```@example estimation
using Random: Xoshiro

data = X .+ 0.01 .* randn(Xoshiro(1), size(X))
nothing # hide
```

## Write the fitting error as a CasADi function

The loss is the mean squared error of the solution for ``\theta``. The data are a second
argument `d`, so a fit to other data reuses the generated code. Calling `F` on symbols
places the integration in the graph of the loss:

```@example estimation
θ = MX.sym("θ", 4)
d = MX.sym("d", 2, length(times))
trajectory = F(MX(DM([1.0, 1.0])), θ)
loss = sum(abs2, trajectory - d) / length(data)
objective = CasadiFunction("loss", [θ, d], [loss])
nothing # hide
```

`gradient(loss, θ)` differentiates through the integrator, by default with forward
sensitivities integrated with the state. We evaluate the loss and its gradient away from
the optimum:

```@example estimation
dloss = gradient(loss, θ)
with_gradient = CasadiFunction("loss_and_gradient", [θ, d], [loss, dloss])
loss_and_gradient = build_function(with_gradient; expression = Val{false})[1]
trial = [1.3, 0.8, 2.7, 1.1]
value, ∇loss = loss_and_gradient(trial, data)
∇loss
```

With many more parameters than states, `adjoint = true` in `integrator` integrates the
adjoint equations backward instead (see [How an integrator is differentiated](@ref)).

## Fit using only the generated gradient

`SciMLBase.OptimizationFunction` takes the unknown first and the data as the problem
parameter (see [Use the generated code with SciML packages](@ref)). It generates the
gradient and, by default, the Hessian; a quasi-Newton method needs no Hessian, so
`hess = false` skips its compilation:

```@example estimation
import SciMLBase

f = SciMLBase.OptimizationFunction(objective; hess = false)
guess = [1.2, 0.8, 2.5, 0.8]
lower, upper = fill(0.1, 4), fill(3.0, 4)
problem = SciMLBase.OptimizationProblem(f, guess, data; lb = lower, ub = upper)
nothing # hide
```

We solve with the L-BFGS approximation of Ipopt and silence its output:

```@example estimation
using OptimizationIpopt: IpoptOptimizer

quiet = Dict{String,Any}("print_level" => 0, "sb" => "yes")
lbfgs = IpoptOptimizer(; hessian_approximation = "limited-memory",
                       additional_options = quiet)
solution = SciMLBase.solve(problem, lbfgs)
solution.u
```

The fit is close to the true parameters ``(1.5, 1, 3, 1)``, and the statistics give the
number of iterations:

```@example estimation
solution.stats.iterations
```

## Fit in fewer iterations with the generated Hessian

By default, the function also holds the Hessian, which CasADi computes from
second-order sensitivities; its code takes longer to compile, but Newton steps need fewer
iterations:

```@example estimation
f_exact = SciMLBase.OptimizationFunction(objective)
problem_exact = SciMLBase.OptimizationProblem(f_exact, guess, data; lb = lower, ub = upper)
newton = IpoptOptimizer(; additional_options = quiet)
solution_exact = SciMLBase.solve(problem_exact, newton)
solution_exact.u, solution_exact.stats.iterations
```

## Get the gradient from SciMLSensitivity instead

The second way integrates in Julia and generates only the right side and its
Jacobians, which SciMLSensitivity.jl uses with `autojacvec = false`:

```@example estimation
u, t = SX.sym("u", 2), SX.sym("t")
q = [SX.sym(name) for name in ("α", "β", "γ", "δ")]
du = vcat((q[1] - q[2] * u[2]) * u[1], (q[4] * u[1] - q[3]) * u[2])
right_side = CasadiFunction("right_side", [u, vcat(q...), t], [du])
rhs = SciMLBase.ODEFunction(right_side; jac = true, paramjac = true)
ode = SciMLBase.ODEProblem(rhs, [1.0, 1.0], (0.0, 10.0), [1.3, 0.8, 2.7, 1.1])
nothing # hide
```

`dgdu!` gives the derivative of the loss by the state at sample `i`, and the gradient
agrees with the first way to the tolerance of the solvers:

```@example estimation
using SciMLSensitivity: adjoint_sensitivities, InterpolatingAdjoint

function dgdu!(out, u, p, t, i)
    out .= 2 .* (u .- view(data, :, i)) ./ length(data)
    return nothing
end

forward = SciMLBase.solve(ode, Vern9(); abstol = 1e-10, reltol = 1e-10)
adjoint_method = InterpolatingAdjoint(autojacvec = false)
_, ∇sciml = adjoint_sensitivities(forward, Vern9(); t = times, dgdu_discrete = dgdu!,
                                  sensealg = adjoint_method, abstol = 1e-10, reltol = 1e-10)
maximum(abs, vec(∇sciml) - ∇loss) / maximum(abs, ∇loss)
```

## Differentiate the Julia solver with ForwardDiff

The third way remakes `ode` with each trial ``\theta``, which compiles nothing again,
and ForwardDiff.jl differentiates through the solver and the generated right side:

```@example estimation
import ForwardDiff

function sciml_loss(θ, (problem, data))
    trial_problem = SciMLBase.remake(problem; p = θ)
    solution = SciMLBase.solve(trial_problem, Vern9(); saveat = times,
                               abstol = 1e-10, reltol = 1e-10)
    sum(abs2, Array(solution) .- data) / length(data)
end

∇forward = ForwardDiff.gradient(θ -> sciml_loss(θ, (ode, data)), trial)
maximum(abs, ∇forward - ∇loss) / maximum(abs, ∇loss)
```

An implicit method also calls the generated `jac`, which accepts dual numbers.
