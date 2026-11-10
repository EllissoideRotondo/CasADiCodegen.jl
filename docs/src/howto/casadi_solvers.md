# Replace the solvers of CasADi

Generated code never calls CasADi, so each solver object of CasADi in a model, such as
an integrator `cvodes` or a solver of `nlpsol`, needs a SciML solver in its place;
[What is refused](@ref) lists the objects without one. The examples use an integrator
`cvodes` for the planar two-body problem:

```@example solvers
using CasADiCodegen
using OrdinaryDiffEqVerner: Vern9
import SciMLBase

x, mu = MX.sym("x", 4), MX.sym("mu")
r3 = (x[1]^2 + x[2]^2)^1.5
ode = vcat(x[3], x[4], -mu * x[1] / r3, -mu * x[2] / r3)
cv = CasADiCodegen.C.integrator("cv", "cvodes", Dict("x" => x, "p" => mu, "ode" => ode), 0.0, [1.0, 2.0, 3.0])
nothing # hide
```

## Replace a solver in generated code

Pass the keyword `solvers`, a [`CasADiCodegen.Solvers`](@ref), to `build_function` or
`generate_julia`. Each field holds an algorithm, or a [`CasADiCodegen.Solver`](@ref) for an
algorithm with options:

```@example solvers
using CasADiCodegen: Solvers, Solver

x0, p = MX.sym("x0", 4), MX.sym("p")
final = cv(Dict("x0" => x0, "p" => p))["xf"]                    # 4 by 3: one column per output time
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
| `ode` | Integrator `cvodes` | An ODE algorithm; supports quadratures, controls `u`, several output times, forward and adjoint sensitivities of any order, and events with forward derivatives only |
| `dae` | Integrator `idas` with algebraic states | An algorithm that accepts a mass matrix, such as `Rodas5P()`; the same derivatives |
| `nonlinear` | Rootfinders `kinsol` and `fast_newton` | A nonlinear algorithm, as for `rootfinder` |
| `optimization` | Solvers of `nlpsol` and of `conic` (quadratic problems; no cone constraints) | An algorithm of Optimization.jl, such as `Ipopt.Optimizer()` with OptimizationMOI.jl |
| `overrides` | One solver object each | A tuple of `Solver(object, alg; options...)`, which has priority over the other fields |

The integrators `rk` and `collocation` and the rootfinder `newton` need no field, since
the package generates code for them; an override replaces them by a SciML solver. In
`Solver(object, alg; options...)`, `object` is the solver object of CasADi, found by its
name and class:

```julia
R = CasADiCodegen.C.integrator("R", "rk", Dict("x" => x, "p" => mu, "ode" => ode), 0.0, 1.0)
solvers = Solvers(overrides = (Solver(R, Vern9(); abstol = 1e-12),))

import Ipopt, OptimizationMOI
ipopt = OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer, "print_level" => 0, "tol" => 1e-12)
solvers = Solvers(optimization = ipopt)
```

## What a replaced solver does

- The code uses the tolerances of the integrator of CasADi unless `Solver` gives
  `abstol` or `reltol`.
- A replaced integrator or rootfinder is an `integrator` or `rootfinder` of this package,
  so it takes dual numbers and has derivatives of any order, in forward and reverse mode;
  an integrator with events is the exception (see [Integrators with events](@ref)).
- A replaced solver of `nlpsol` has derivatives of any order, and treats a bound as active
  when ``|x - \mathrm{bound}| \le 10^{-6} \max(1, |\mathrm{bound}|)``. It solves in
  `Float64` with dense matrices and ignores the initial guesses of the multipliers and the
  options of the solver of CasADi, such as those of `ipopt`; give these with the algorithm.
- Hessians are built only for an algorithm that needs them by the traits
  `SciMLBase.requireshessian`, `requiresconshess` or `requireslagh`.
- `generate_julia` writes the algorithm and its options to a source file, so they must be
  values that a file can hold (see [Write the model to a file](@ref)).

## Integrators with events

An integrator of CasADi with event functions in `"zero"` and the option `"transition"`
is replaced by the solver of `ode`, or `dae` for algebraic states, and with `rk` or
`collocation` by an override. Events follow the rules of CasADi, with an error after more
than `max_events` events (20 by default). An event
that falls on an output time is applied there, which CasADi 3.8.2 misses (see
[Differences from CasADi](@ref)). The replaced integrator takes `Float64` only, unlike the
`integrator` of this package with events (see [Stop at an event](@ref events)).

For a ball that bounces with the restitution `e`:

```julia
h, v, e = MX.sym("h"), MX.sym("v"), MX.sym("e")             # height, speed, restitution
model = Dict("x" => vcat(h, v), "p" => e, "ode" => vcat(v, -9.81), "zero" => h)
bounce = CasadiFunction("transition", Dict("x" => vcat(h, v), "p" => e, "post_x" => vcat(h, -e * v)),
                        CasADiCodegen.C.event_in(), CasADiCodegen.C.event_out())
ball = CasADiCodegen.C.integrator("ball", "cvodes", model, 0.0, [0.5, 1.0, 2.0], Dict("transition" => bounce))
```

CasADi differentiates such an integrator by forward directions only, so take `jacobian`
of an expression, which is dense, since `CasADiCodegen.C.jacobian(F)` of a function that
holds it throws:

```julia
s0, q = MX.sym("s0", 2), MX.sym("q")
final = CasADiCodegen.C.vec(ball(Dict("x0" => s0, "p" => q))["xf"])
J = jacobian(final, vcat(s0, q))
f = CasadiFunction("f", [s0, q], [final, CasADiCodegen.C.densify(J)])
```

With algebraic states, the algebraic equations are solved again after each transition.

## Solve the problem in Julia

To call a SciML solver directly, make a SciML problem from the solver object of CasADi:

```@example solvers
problem = SciMLBase.ODEProblem(cv, [1.0, 0.0, 0.0, 1.0], [1.0])    # the output times of cv are `saveat`
solution = SciMLBase.solve(problem, Vern9(); abstol = 1e-13, reltol = 1e-13)
solution.u[end]
```

| Constructor | Object of CasADi | Problem |
|---|---|---|
| `ODEProblem(I, x0, p; z0, u)` | Integrator `cvodes`, `idas`, `rk` or `collocation` | State of differential states, algebraic states from a consistent `z0`, and quadratures; a mass matrix for algebraic states; a callback for events; `abstol` and `reltol` from `cvodes` and `idas` |
| `ODEProblem(dae, tspan)` | Model of a `DaeBuilder` | Differential then algebraic states from their start values; parameters `p` then inputs `u`; a mass matrix for algebraic states; the time span of `dae` by default |
| `NonlinearProblem(R, z0, others...)` | Rootfinder `newton`, `fast_newton` or `kinsol` | The residual and its Jacobian; the other arguments of `R` are the parameters |
| `OptimizationProblem(opti; sparse = false)` | `Opti` stack | Objective, constraints, initial values, parameter values and bounds at the time of the call; a bound on a variable stays a constraint |
| `OptimizationProblem(S, x0, p; lbx, ubx, lbg, ubg, sparse = false)` | Solver of `nlpsol` or of `conic` | Objective and constraints with derivatives, sparse with `lag_h` for `sparse = true` (see [Give a CasADi function to a SciML problem](@ref)); bounds with the names and defaults of CasADi. For `conic`, `p` has the nonzeros of `h`, then `g`, then the nonzeros of `a`, for the problem: minimize ``\tfrac12 x^T h x + g^T x`` subject to ``\mathrm{lba} \le a x \le \mathrm{uba}`` |

For an integrator with controls and several output times, `u` has one block of controls
per output time. A model of a `DaeBuilder` whose variables are scalars gives a solution
indexed by name:

```@example solvers
dae = CasADiCodegen.C.DaeBuilder("decay")
y = CasADiCodegen.C.add(dae, "y", "local", "continuous")
k = CasADiCodegen.C.add(dae, "k", "parameter", "tunable")
CasADiCodegen.C.eq(dae, CasADiCodegen.C.der(dae, y), -k * y)
CasADiCodegen.C.set_start(dae, "y", 1.0)
CasADiCodegen.C.set_start(dae, "k", 0.5)
solution = SciMLBase.solve(SciMLBase.ODEProblem(dae, (0.0, 2.0)), Vern9(); abstol = 1e-12, reltol = 1e-12)
solution[:y][end] ≈ exp(-1.0)
```
