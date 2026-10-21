# Replace the solvers of CasADi

A model that exists already often holds solver objects of CasADi itself: an
integrator made with `CasADiCodegen.C.integrator(name, "cvodes", ...)`, a rootfinder,
or a solver of `nlpsol`. Generated code never calls CasADi, so we give each such
object a solver of the SciML ecosystem, which needs SciMLBase.jl and the package of
the algorithm.

In the examples of this page, we use an integrator `cvodes` for the planar two-body
problem:

```@example solvers
using CasADiCodegen
using OrdinaryDiffEqVerner: Vern9
import SciMLBase

x, mu = MX.sym("x", 4), MX.sym("mu")
r3 = (x[1]^2 + x[2]^2)^1.5
ode = vertcat(x[3], x[4], -mu * x[1] / r3, -mu * x[2] / r3)
I = CasADiCodegen.C.integrator("I", "cvodes", Dict("x" => x, "p" => mu, "ode" => ode), 0.0, [1.0, 2.0, 3.0])
nothing # hide
```

## Replace a solver in generated code

Give `build_function` or `generate_julia` the keyword `solvers` with a
[`CasADiCodegen.Solvers`](@ref). Each field holds an algorithm, or a
[`CasADiCodegen.Solver`](@ref) when the algorithm has options.

Here we ask for the states at the three output times and the Jacobian of the final
state by the initial state, with `Vern9` at tight tolerances:

```@example solvers
using CasADiCodegen: Solvers, Solver

x0, p = MX.sym("x0", 4), MX.sym("p")
final = I(Dict("x0" => x0, "p" => p))["xf"]                    # 4 by 3: one column per output time
f = CasadiFunction("f", [x0, p], [final, jacobian(final[:, 3], x0)])

solvers = Solvers(ode = Solver(Vern9(); abstol = 1e-13, reltol = 1e-13))
g, g! = build_function(f; expression = Val{false}, solvers)
states, stm = g([1.0, 0.0, 0.0, 1.0], [1.0])
states
```

Without `solvers`, `build_function` throws an `ArgumentError`. The message names the
function and the field of `Solvers` to set.

| Field of `Solvers` | Replaces | With |
|---|---|---|
| `ode` | Integrator `cvodes` | An ODE algorithm. Supported: quadratures, controls (the input `u` of a CasADi integrator, constant between two output times), several output times, forward and adjoint sensitivities, derivatives of any order, event functions with forward derivatives only (see [Integrators with events](@ref)) |
| `dae` | Integrator `idas` with algebraic states | An algorithm that accepts a mass matrix, such as `Rodas5P()`; the same derivatives |
| `nonlinear` | Rootfinders `kinsol` and `fast_newton` | A nonlinear algorithm, as for `rootfinder` |
| `optimization` | Solvers of `nlpsol` and of `conic` (quadratic problems; no cone constraints) | An algorithm of Optimization.jl, such as `Ipopt.Optimizer()` with OptimizationMOI.jl |
| `overrides` | One solver object each | A tuple of `Solver(object, alg; options...)`. An override has priority over the other fields |

The integrators `rk` and `collocation` and the rootfinder `newton` need no field:
the package generates Julia code for them. An override replaces them by a SciML
solver.

In `Solver(object, alg; options...)`, `object` is the solver object of CasADi itself.
The generated code finds the object, and the functions that CasADi derives from it,
by its name and its class. The two examples below replace one integrator `R` and a
solver of `nlpsol`:

```julia
R = CasADiCodegen.C.integrator("R", "rk", Dict("x" => x, "p" => mu, "ode" => ode), 0.0, 1.0)
solvers = Solvers(overrides = (Solver(R, Vern9(); abstol = 1e-12),))

import Ipopt, OptimizationMOI
ipopt = OptimizationMOI.MOI.OptimizerWithAttributes(Ipopt.Optimizer, "print_level" => 0, "tol" => 1e-12)
solvers = Solvers(optimization = ipopt)
```

## What a replaced solver does

The code uses the tolerances of the integrator of CasADi, unless the `Solver(...)`
gives `abstol` or `reltol`. An integrator or a rootfinder becomes an `integrator` or a
`rootfinder` of this package, so the code takes dual numbers, and CasADi gets its
derivatives as for those functions; an integrator with an event function is the
exception (see [Integrators with events](@ref)).

For a solver of `nlpsol`, the generated code solves the problem with the bounds of
the call and then computes the multipliers in the form that CasADi expects: the
multiplier of a bound that is not active is exactly zero, and the others solve the
stationarity condition by least squares. CasADi makes the derivatives of the
solution from these results, since it reads the active bounds from the signs of the
multipliers, so derivatives of any order need no other code. A bound counts as active
when the solution is within 1e-6 of it, relative to the bound when that is more than
1. The problem is solved with `Float64` only and with dense matrices, and the first
guesses of the multipliers are not used. The options of the solver of CasADi, such as
those of `ipopt`, are not read, so they must go with the algorithm.

`generate_julia` writes a replaced solver to the module under the rule for
integrators (see [Write the model to a file](@ref)): the algorithm and its options
must be values that a source file can hold.

Reverse mode works through an integrator that a solver replaces, because CasADi gets
the adjoint sensitivities as the reverse derivative of the chain of calls between
the output times. Each call integrates the adjoint equations backward, as
`integrator` does with `adjoint = true`, also for a model with algebraic states. An
integrator with events is the exception: it has forward derivatives only.

## Integrators with events

An integrator of CasADi whose model has the entry `"zero"` has event functions, and
its option `"transition"` gives the state after an event. The generated code
replaces it by one Julia function that integrates to all the output times with the
solver of the field `ode`, or of `dae` for algebraic states; an integrator `rk` or
`collocation` with events needs an override. The events follow the rules of CasADi:
an event is a change of sign of an element of `"zero"` from positive to negative,
the transition then sets the state, and a call throws after more than `max_events`
events, where `max_events` is an option of the integrator of CasADi (20 by default).
The numbers must be `Float64`: this replaced integrator does not take dual numbers,
unlike the `integrator` with an event function of this package
(see [Stop at an event](@ref events)).

As an example, we model a ball that bounces with the restitution `e`:

```julia
h, v, e = MX.sym("h"), MX.sym("v"), MX.sym("e")             # height, speed, restitution
model = Dict("x" => vertcat(h, v), "p" => e, "ode" => vertcat(v, -9.81), "zero" => h)
bounce = CasADiCodegen.C.Function("transition", Dict("x" => vertcat(h, v), "p" => e, "post_x" => vertcat(h, -e * v)),
                           CasADiCodegen.C.event_in(), CasADiCodegen.C.event_out())
ball = CasADiCodegen.C.integrator("ball", "cvodes", model, 0.0, [0.5, 1.0, 2.0], Dict("transition" => bounce))
```

CasADi differentiates such an integrator by forward directions only. `jacobian` of an
expression that calls it gives a dense Jacobian, and the method `jacobian()` of a
`Function` that holds it throws an error; [Differences from CasADi](@ref) gives the
reason.

We take the Jacobian of the final states by the initial state and the restitution:

```julia
s0, q = MX.sym("s0", 2), MX.sym("q")
final = CasADiCodegen.C.vec(ball(Dict("x0" => s0, "p" => q))["xf"])
J = jacobian(final, vertcat(s0, q))
f = CasadiFunction("f", [s0, q], [final, CasADiCodegen.C.densify(J)])
```

With algebraic states, the event function can depend on them, and the forward
derivatives of the event time and of the states after it are exact; after the transition,
the package solves the algebraic equations again for the algebraic states and for their
derivatives.

## Solve the problem in Julia

To call a SciML solver directly, we make a SciML problem from the solver object of
CasADi. The problem holds generated code for its functions and their derivatives:

```@example solvers
problem = SciMLBase.ODEProblem(I, [1.0, 0.0, 0.0, 1.0], [1.0])    # the output times of I are `saveat`
solution = SciMLBase.solve(problem, Vern9(); abstol = 1e-13, reltol = 1e-13)
solution.u[end]
```

| Constructor | Object of CasADi | Problem |
|---|---|---|
| `ODEProblem(I, x0, p; z0, u)` | Integrator `cvodes`, `idas`, `rk` or `collocation` | The state has the differential states, the algebraic states from the guess `z0` and the quadratures; algebraic states give a mass matrix; event functions give a callback with the transition; the tolerances of CasADi are `abstol` and `reltol` |
| `NonlinearProblem(R, z0, others...)` | Rootfinder `newton`, `fast_newton` or `kinsol` | The residual and its Jacobian; the other arguments of `R` are the parameters |
| `OptimizationProblem(S, x0, p; lbx, ubx, lbg, ubg)` | Solver of `nlpsol` or of `conic` | The objective and the constraints with their derivatives; the bounds have the names and the default values of CasADi. For `conic`, `p` has the nonzeros of `h`, then `g`, then the nonzeros of `a`, where CasADi writes the quadratic problem as minimizing ``\tfrac12 x^T h x + g^T x`` subject to ``\mathrm{lba} \le a x \le \mathrm{uba}`` |

For an integrator with controls and several output times, `u` has one block of
controls per output time, and the problem has a callback that changes the controls
at each output time.

## Limits

[What is refused](@ref) lists each object that has no replacement, with what to do.
