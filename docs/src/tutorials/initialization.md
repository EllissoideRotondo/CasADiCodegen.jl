# Initialize differential-algebraic equations

The initial values of a differential-algebraic equation (DAE) must satisfy its algebraic
equations. In this tutorial, we give a SciML problem the conditions that we know, such as
a position or a velocity, as a CasADi function, and the solver computes consistent initial
values before its first step.

```@example init
using CasADiCodegen, LinearAlgebra
import SciMLBase
using OrdinaryDiffEqRosenbrock: Rodas5P
using OrdinaryDiffEqTsit5: Tsit5
using NonlinearSolve: LevenbergMarquardt, TrustRegion
```

## Consistent initial values

A semi-explicit DAE has differential variables ``x`` and algebraic variables ``z``,

```math
x' = f(x, z, t), \qquad 0 = g(x, z, t),
```

and its initial values are consistent when ``g(x(t_0), z(t_0), t_0) = 0``. The modeler
must add the right number of conditions, on a variable or on a derivative, so that
together with ``g = 0`` they determine all initial values.

[`ODEFunction`](@ref SciMLBase.ODEFunction(::CasADiCodegen.Function)) takes these
conditions as the keyword `initialization`: a CasADi function with the arguments of the
right side whose result must vanish at the initial time. The initialization problem has
the whole state as unknowns, and the algebraic equations (the zero rows of the mass
matrix) and the conditions as equations; the initial state of the problem is only a
guess.

## Write the pendulum in index-1 form

A pendulum of unit mass on a rod of unit length has the position ``(x, y)`` and the
multiplier ``\lambda``:

```math
x'' = \lambda x, \qquad y'' = \lambda y - g, \qquad 0 = x^2 + y^2 - 1.
```

These equations have index 3 (see
[Model physical systems with differential-algebraic equations](@ref)). We write the
index-1 form of the method of dummy derivatives: ``y`` and ``v_y`` are the differential
states, and ``x``, ``v_x`` and ``\lambda`` are algebraic, determined by the constraint and
its first two time derivatives, which CasADi computes:

```@example init
y, vy, x, vx, λ = (SX.sym(name) for name in ("y", "vy", "x", "vx", "λ"))
g, x0, y0, vy0, t = (SX.sym(name) for name in ("g", "x0", "y0", "vy0", "t"))
variables, slopes = (x, vx, y, vy), (vx, λ * x, vy, λ * y - g)
ddt(e) = sum(jacobian(e, s) * ds for (s, ds) in zip(variables, slopes))

constraint = x^2 + y^2 - 1
u, p = vcat(y, vy, x, vx, λ), vcat(g, x0, y0, vy0)
rhs = CasadiFunction("pendulum", [u, p, t], [vcat(vy, λ * y - g, constraint, ddt(constraint), ddt(ddt(constraint)))])
mass = Diagonal([1.0, 1.0, 0.0, 0.0, 0.0])
nothing # hide
```

The parameters hold the gravity `g` and the targets `x0`, `y0` and `vy0` of the
conditions. Each problem uses the Levenberg-Marquardt method for the initialization,
since it uses the generated Jacobian, accepts any number of equations, and tolerates a
singular Jacobian:

```@example init
initialize = SciMLBase.OverrideInit(; nlsolve = LevenbergMarquardt())
conditions(rows...) = CasadiFunction("conditions", [u, p, t], [vcat(rows...)])

function pendulum(guess, rows...; tspan = (0.0, 1.5), parameters = [1.0, 1.0, 0.0, 0.0])
    f = SciMLBase.ODEFunction(rhs; mass_matrix = mass, initialization = conditions(rows...),
                              specialize = SciMLBase.AutoSpecialize)
    SciMLBase.ODEProblem(f, guess, tspan, parameters; initializealg = initialize, abstol = 1e-8, reltol = 1e-8)
end
nothing # hide
```

`AutoSpecialize` gives all these problems one function type, so the solver compiles
once. The guess is in the order (y, vy, x, vx, λ).

## Start from a position

We start the pendulum at `x = 1`, `y = 0`, with the guess `λ = 1`:

```@example init
problem = pendulum([0.0, 0.0, 1.0, 0.0, 1.0], x - x0, y - y0)
solution = SciMLBase.solve(problem, Rodas5P())
solution.retcode, solution.u[1]
```

The last three elements of the right side, the constraint and its derivatives, vanish
at the start:

```@example init
rhs(solution.u[1], problem.p, 0.0)
```

The multiplier leaves its guess for the value 0 of a pendulum at rest at the side. Any
variable can carry a condition, here `λ = 0` or `vy = 0` with `x = 1`:

```@example init
from_lambda = SciMLBase.solve(pendulum([1.0, 0.0, 1.0, 0.0, 0.0], x - x0, λ), Rodas5P())
from_speed = SciMLBase.solve(pendulum([1.0, 0.0, 1.0, 0.0, 0.0], x - x0, vy - vy0), Rodas5P())
from_lambda.u[1], from_speed.u[1]
```

At `x = 1` the constraint has a double root in `y`, so the solver converges slowly and
stops at a height of up to about ``10^{-4}``. A condition can be any equation, and it can
contain parameters, as `vy = g`:

```@example init
upward = SciMLBase.solve(pendulum([1.0, 0.0, 1.0, 0.0, 0.0], x - x0, vy - g; tspan = (0.0, 3.0)), Rodas5P())
upward.u[1]
```

A condition on the derivative of a variable that is not a state can call `rhs` on the
arguments of the conditions function.

## Count the conditions

Five variables and three algebraic equations need two conditions.
[`initialization_problem`](@ref CasADiCodegen.initialization_problem) returns the
initialization problem at the guess:

```@example init
under = pendulum([0.0, 0.0, 1.0, 0.0, 1.0], x - x0)
over = pendulum([0.0, 0.0, 1.0, 0.0, 1.0], x - x0, y - y0, vy - vy0)
typeof(CasADiCodegen.initialization_problem(problem)).name.name,
typeof(CasADiCodegen.initialization_problem(under)).name.name,
typeof(CasADiCodegen.initialization_problem(over)).name.name
```

A square system is a `NonlinearProblem`, and any other count gives a
`NonlinearLeastSquaresProblem`. An underdetermined system returns a solution near the
guess, and an overdetermined one succeeds when its conditions agree:

```@example init
SciMLBase.solve(under, Rodas5P()).retcode, SciMLBase.solve(over, Rodas5P()).retcode
```

A square system can be singular: with `x = 1` and `y = 0`, `vy` is free and keeps its
guess. Contradictory conditions fail; with `vy = 2` the constraint needs `λ = -4`, not 1:

```@example init
target = [1.0, 1.0, 0.0, 2.0]
failed = SciMLBase.solve(pendulum([0.0, 2.0, 1.0, 0.0, 1.0], x - x0, y - y0, vy - vy0, λ - 1; parameters = target), Rodas5P())
failed.retcode
```

SciMLBase reports `InitialFailure` rather than integrate from inconsistent values.

## Solve the initialization problem alone

Solving the initialization problem alone shows the residual of the algebraic equations
followed by the conditions:

```@example init
inconsistent = pendulum([0.0, 2.0, 1.0, 0.0, 1.0], x - x0, y - y0, vy - vy0, λ - 1; parameters = target)
alone = SciMLBase.solve(CasADiCodegen.initialization_problem(inconsistent), LevenbergMarquardt())
alone.retcode, round.(alone.resid; digits = 3)
```

No state satisfies all residuals. With `λ = 0` in place of `λ = 1`, the residual
vanishes:

```@example init
consistent = pendulum([0.0, 0.0, 1.0, 0.0, 1.0], x - x0, y - y0, vy - vy0, λ; parameters = [1.0, 1.0, 0.0, 0.0])
alone = SciMLBase.solve(CasADiCodegen.initialization_problem(consistent), LevenbergMarquardt())
alone.retcode, maximum(abs, alone.resid)
```

The solution is indexed by name, and any method of NonlinearSolve.jl applies, such as
the trust-region method on a square problem:

```@example init
square = SciMLBase.solve(CasADiCodegen.initialization_problem(problem), TrustRegion())
square.retcode, square[:λ]
```

## Change the conditions

A new target is a new parameter value and needs no new code:

```@example init
moving = pendulum([1.0, 0.0, 1.0, 0.0, 0.0], x - x0, vy - vy0)
moving.ps[:vy0] = -0.1
SciMLBase.solve(moving, Rodas5P())[:vy][1]
```

Other conditions need a new conditions function, here `y = 0` and `vx = 0`:

```@example init
sideways = SciMLBase.solve(pendulum([0.0, 0.0, 1.0, 0.0, 0.0], y - y0, vx), Rodas5P())
sideways.u[1]
```

Only the generated code of the new conditions is compiled.

## Choose the nonlinear solver

The guess selects one of several solutions. At `x = 0.6` the pendulum hangs at
`y = -0.8` or stands at `y = 0.8`, and Levenberg-Marquardt stays near the guess `y = -1`:

```@example init
low = SciMLBase.solve(pendulum([-1.0, 0.0, 0.6, 0.0, 0.0], x - 0.6, vy - vy0), Rodas5P())
low[:y][1]
```

Without `initializealg`, OrdinaryDiffEq.jl uses its default solver, which needs
OrdinaryDiffEqNonlinearSolve.jl and starts with Broyden's method; it ignores the generated
Jacobian and can jump to the other solution.

## Solve for a parameter

Parameters can be unknowns of the initialization. In a model where `x + y` is the
parameter `total`, we give two of the three and compute the third; `y` is an observed
variable:

```@example init
x, total, x0, y0 = SX.sym("x"), SX.sym("total"), SX.sym("x0"), SX.sym("y0")
q = vcat(total, x0, y0)
sum_model = CasadiFunction("sum_model", [x, q, t], [-x, total - x], ["u", "p", "t"], ["du", "y"])
nothing # hide
```

The keyword `solved_parameters` names the parameters to solve for; their values in the
problem are guesses. With `x = 1` and `y = 2`:

```@example init
both = CasadiFunction("both", [x, q, t], [vcat(x - x0, (total - x) - y0)])
f = SciMLBase.ODEFunction(sum_model; initialization = both, solved_parameters = [:total])
solution = SciMLBase.solve(SciMLBase.ODEProblem(f, [0.0], (0.0, 1.0), [0.0, 1.0, 2.0]), Tsit5();
                           initializealg = initialize, abstol = 1e-8, reltol = 1e-8)
solution.ps[:total]
```

To solve for `x` instead, the condition is `y = 1` alone:

```@example init
only_y = CasadiFunction("only_y", [x, q, t], [(total - x) - y0])
f = SciMLBase.ODEFunction(sum_model; initialization = only_y)
solution = SciMLBase.solve(SciMLBase.ODEProblem(f, [1.0], (0.0, 1.0), [4.0, 0.0, 1.0]), Tsit5();
                           initializealg = initialize, abstol = 1e-8, reltol = 1e-8)
solution.u[1], solution[:y][1]
```

## Start at a steady state

Conditions on the derivatives initialize an ODE too. The Lotka-Volterra system is at a
steady state when its right side vanishes:

```@example init
r, s = SX.sym("r"), SX.sym("s")
prey = vcat(r, s)
slope = vcat(1.5r - r * s, -3s + r * s)
lotka = CasadiFunction("lotka", [prey], [slope])
steady = SciMLBase.ODEFunction(lotka; initialization = CasadiFunction("steady", [prey], [slope]))
solution = SciMLBase.solve(SciMLBase.ODEProblem(steady, [2.0, 1.0], (0.0, 10.0)), Tsit5();
                           initializealg = initialize, abstol = 1e-8, reltol = 1e-8)
solution.u[1], solution.u[end]
```

The guess selects the steady state ``(3, 1.5)`` over the origin. A condition can also
be an expression of the state, here `r' = 0` and `r + s = 2`:

```@example init
observed = CasadiFunction("observed", [prey], [vcat(slope[1], r + s - 2)])
solution = SciMLBase.solve(SciMLBase.ODEProblem(SciMLBase.ODEFunction(lotka; initialization = observed), [1.0, 1.0], (0.0, 10.0)),
                           Tsit5(); initializealg = initialize, abstol = 1e-8, reltol = 1e-8)
solution.u[1]
```
