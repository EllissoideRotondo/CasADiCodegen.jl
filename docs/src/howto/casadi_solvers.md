# Run CasADi solver objects with SciML solvers

Generated code never calls CasADi, so a CasADi solver object in a model, such as a
`cvodes` integrator, needs a SciML solver in its place;
[Functions the package rejects, and what to do instead](@ref) lists the objects. The
examples use a `cvodes` integrator for the planar two-body problem:

```@example solvers
using CasADiCodegen
using OrdinaryDiffEqVerner: Vern9
import SciMLBase

x, mu = MX.sym("x", 4), MX.sym("mu")
r3 = (x[1]^2 + x[2]^2)^1.5
ode = vcat(x[3], x[4], -mu * x[1] / r3, -mu * x[2] / r3)
model = Dict("x" => x, "p" => mu, "ode" => ode)
times = [1.0, 2.0, 3.0]
cv = casadi.integrator("cv", "cvodes", model, 0.0, times)
nothing # hide
```

## Replace a solver in generated code

Pass the keyword `solvers`, a [`CasADiCodegen.Solvers`](@ref), to `build_function` or
`generate_julia`. Each field holds an algorithm, or a [`CasADiCodegen.Solver`](@ref) for an
algorithm with options:

```@example solvers
using CasADiCodegen: Solvers, Solver

x0, p = MX.sym("x0", 4), MX.sym("p")
# 4 by 3: one column per output time
final = cv(Dict("x0" => x0, "p" => p))["xf"]
f = CasadiFunction("f", [x0, p], [final, jacobian(final[:, 3], x0)])

solvers = Solvers(ode = Solver(Vern9(); abstol = 1e-13, reltol = 1e-13))
g, g! = build_function(f; expression = Val{false}, solvers)
states, stm = g([1.0, 0.0, 0.0, 1.0], [1.0])
states
```

Without `solvers`, `build_function` throws an `ArgumentError` that names the field to
set.

| Field of `Solvers` | Replaces | With |
|---|---|---|
| `ode` | Integrator `cvodes` | An ODE algorithm, such as `Vern9()` |
| `dae` | Integrator `idas` with algebraic states | An algorithm that accepts a mass matrix, such as `Rodas5P()` |
| `nonlinear` | Rootfinders `kinsol` and `fast_newton` | A nonlinear algorithm, as for `rootfinder` |
| `optimization` | Solvers of `nlpsol` and of `conic` | An algorithm of Optimization.jl, such as `Ipopt.Optimizer()` with OptimizationMOI.jl |
| `overrides` | One solver object each | A tuple of `Solver(object, alg; options...)`, which has priority over the other fields |

The integrators `rk` and `collocation` and the rootfinder `newton` need no field, since
the package generates code for them.

!!! details "Replace one solver object, or pass options to Ipopt"
    An override replaces one solver object by a SciML solver, also an `rk`, `collocation`
    or `newton` object that would otherwise get generated code. In
    `Solver(object, alg; options...)`, `object` is the CasADi solver object, found by its
    name and class:

    ```julia
    R = casadi.integrator("R", "rk", model, 0.0, 1.0)
    solvers = Solvers(overrides = (Solver(R, Vern9(); abstol = 1e-12),))

    import Ipopt, OptimizationMOI
    ipopt = OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer,
                                                        "print_level" => 0, "tol" => 1e-12)
    solvers = Solvers(optimization = ipopt)
    ```

## What carries over from the CasADi solver

- The code uses the tolerances of the CasADi integrator unless `Solver` gives `abstol` or
  `reltol`.
- A replaced integrator or rootfinder is an `integrator` or `rootfinder` of this package;
  [Supported functions, models and solvers](@ref) lists the derivatives of each replaced
  solver.
- A replaced `nlpsol` solver treats a bound as active when
  ``|x - \mathrm{bound}| \le 10^{-6} \max(1, |\mathrm{bound}|)``. It uses dense matrices
  and ignores the initial guesses of the multipliers and the options of the CasADi
  solver, such as those of `ipopt`; give these with the algorithm.
- `generate_julia` writes the algorithm and its options to a source file, so they must be
  values that a file can hold (see [Write an integrator to a source file](@ref)).

## Replace an integrator that has events

A CasADi integrator with event functions in `"zero"` and the option `"transition"` is
replaced by the solver of `ode`, or `dae` for algebraic states, and with `rk` or
`collocation` by an override. Events follow the rules of CasADi, with an error after more
than `max_events` events (20 by default), and an event at an output time is applied there
(see [Where results differ from CasADi](@ref)).

For a ball that bounces with the restitution `e`:

```julia
h, v, e = MX.sym("h"), MX.sym("v"), MX.sym("e")             # height, speed, restitution
falling = Dict("x" => vcat(h, v), "p" => e, "ode" => vcat(v, -9.81), "zero" => h)
after = Dict("x" => vcat(h, v), "p" => e, "post_x" => vcat(h, -e * v))
bounce = CasadiFunction("transition", after, casadi.event_in(), casadi.event_out())
options = Dict("transition" => bounce)
ball = casadi.integrator("ball", "cvodes", falling, 0.0, [0.5, 1.0, 2.0], options)
```

Take `jacobian` of an expression and densify it, not `casadi.jacobian` of a function:

```julia
s0, q = MX.sym("s0", 2), MX.sym("q")
final = casadi.vec(ball(Dict("x0" => s0, "p" => q))["xf"])
J = jacobian(final, vcat(s0, q))
f = CasadiFunction("f", [s0, q], [final, casadi.densify(J)])
```

With algebraic states, the algebraic equations are solved again after each transition.

## Make a SciML problem from a CasADi solver object

To call a SciML solver directly, make a SciML problem from the CasADi solver object:

```@example solvers
u0 = [1.0, 0.0, 0.0, 1.0]
params = [1.0]
problem = SciMLBase.ODEProblem(cv, u0, params)    # the output times of cv are `saveat`
solution = SciMLBase.solve(problem, Vern9(); abstol = 1e-13, reltol = 1e-13)
solution.u[end]
```

| Constructor | CasADi object | Problem |
|---|---|---|
| `ODEProblem(I, x0, p; z0, u)` | Integrator `cvodes`, `idas`, `rk` or `collocation` | State of differential states, algebraic states from a consistent `z0`, and quadratures; a mass matrix for algebraic states; a callback for events; `abstol` and `reltol` from `cvodes` and `idas` |
| `ODEProblem(dae, tspan)` | Model of a `DaeBuilder` | Differential then algebraic states from their start values; parameters `p` then inputs `u`; a mass matrix for algebraic states; the time span of `dae` by default |
| `NonlinearProblem(R, z0, others...)` | Rootfinder `newton`, `fast_newton` or `kinsol` | The residual and its Jacobian; the other arguments of `R` are the parameters |
| `OptimizationProblem(opti; sparse = false)` | `Opti` stack | Objective, constraints, initial values, parameter values and bounds at the time of the call; a bound on a variable stays a constraint |
| `OptimizationProblem(S, x0, p; lbx, ubx, lbg, ubg, sparse = false)` | Solver of `nlpsol` or of `conic` | Objective and constraints with derivatives, sparse with `lag_h` for `sparse = true` (see [Use the generated code with SciML packages](@ref)); bounds with the names and defaults of CasADi. For `conic`, `p` holds `h`, `g` and `a` (see [`OptimizationProblem`](@ref SciMLBase.OptimizationProblem(::CasADiCodegen.Function, ::AbstractVector{<:Real}, ::AbstractVector{<:Real}))) |

For an integrator with controls and several output times, `u` has one block of controls
per output time. A model of a `DaeBuilder` whose variables are scalars gives a solution
indexed by name:

```@example solvers
dae = casadi.DaeBuilder("decay")
y = casadi.add(dae, "y", "local", "continuous")
k = casadi.add(dae, "k", "parameter", "tunable")
casadi.eq(dae, casadi.der(dae, y), -k * y)
casadi.set_start(dae, "y", 1.0)
casadi.set_start(dae, "k", 0.5)
decay = SciMLBase.ODEProblem(dae, (0.0, 2.0))
solution = SciMLBase.solve(decay, Vern9(); abstol = 1e-12, reltol = 1e-12)
solution[:y][end] ≈ exp(-1.0)
```
