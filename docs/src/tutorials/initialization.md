# Initialize differential-algebraic equations

The initial values of a differential-algebraic equation (DAE) must satisfy its algebraic
equations. In this tutorial, we give the conditions that we know, such as a position or
a velocity, to [`ODEFunction`](@ref SciMLBase.ODEFunction(::CasADiCodegen.Function)) as
the keyword `initialization`: a CasADi function with the arguments of the right side whose
result must vanish at the initial time. Before its first step, the solver computes the
whole state from the algebraic equations (the zero rows of the mass matrix) and the
conditions; the initial state of the problem is only a guess.

```@example init
using CasADiCodegen, LinearAlgebra
import SciMLBase
using OrdinaryDiffEqRosenbrock: Rodas5P
using OrdinaryDiffEqTsit5: Tsit5
using NonlinearSolve: LevenbergMarquardt, TrustRegion
```

## Write the pendulum as an index-1 DAE

A pendulum of unit mass on a rod of unit length, with the multiplier ``\lambda``, has
index 3:

```math
x'' = \lambda x, \qquad y'' = \lambda y - g, \qquad 0 = x^2 + y^2 - 1.
```

The problem of a `System` (see
[Model a circuit and a pendulum as DAEs](@ref)) makes its
`ODEFunction` without the keyword `initialization`, so we write the right side by hand, in
the index-1 form of dummy derivatives: ``y`` and ``v_y`` are differential, and ``x``,
``v_x`` and ``\lambda`` are algebraic, fixed by the constraint and its first two time
derivatives, which CasADi computes:

```@example init
y, vy, x, vx, λ = (SX.sym(name) for name in ("y", "vy", "x", "vx", "λ"))
g, x0, y0, vy0, t = (SX.sym(name) for name in ("g", "x0", "y0", "vy0", "t"))
variables, slopes = (x, vx, y, vy), (vx, λ * x, vy, λ * y - g)
ddt(e) = sum(jacobian(e, s) * ds for (s, ds) in zip(variables, slopes))

constraint = x^2 + y^2 - 1
u, p = vcat(y, vy, x, vx, λ), vcat(g, x0, y0, vy0)
dynamics = vcat(vy, λ * y - g, constraint, ddt(constraint), ddt(ddt(constraint)))
rhs = CasadiFunction("pendulum", [u, p, t], [dynamics])
mass = Diagonal([1.0, 1.0, 0.0, 0.0, 0.0])
nothing # hide
```

The parameters are the gravity `g` and the targets `x0`, `y0` and `vy0`. The
Levenberg-Marquardt method solves the initialization: it uses the generated Jacobian,
accepts any number of equations and tolerates a singular Jacobian. The guess is in the
order (y, vy, x, vx, λ):

```@example init
initialize = SciMLBase.OverrideInit(; nlsolve = LevenbergMarquardt())
conditions(rows...) = CasadiFunction("conditions", [u, p, t], [vcat(rows...)])

function pendulum(guess, rows...; tspan = (0.0, 1.5), parameters = [1.0, 1.0, 0.0, 0.0])
    f = SciMLBase.ODEFunction(rhs; mass_matrix = mass, initialization = conditions(rows...),
                              specialize = SciMLBase.AutoSpecialize)
    SciMLBase.ODEProblem(f, guess, tspan, parameters; initializealg = initialize,
                         abstol = 1e-8, reltol = 1e-8)
end
nothing # hide
```

## Start the pendulum at a given position

We start the pendulum at `x = 1`, `y = 0`, with the guess `λ = 1`:

```@example init
problem = pendulum([0.0, 0.0, 1.0, 0.0, 1.0], x - x0, y - y0)
solution = SciMLBase.solve(problem, Rodas5P())
solution.retcode, solution.u[1]
```

The constraint and its derivatives, the last three elements of the right side, vanish:

```@example init
rhs(solution.u[1], problem.p, 0.0)
```

The multiplier leaves its guess for 0, the value at rest at the side. Any variable can
carry a condition, here `λ = 0` or `vy = 0` with `x = 1`:

```@example init
by_lambda = pendulum([1.0, 0.0, 1.0, 0.0, 0.0], x - x0, λ)
by_speed = pendulum([1.0, 0.0, 1.0, 0.0, 0.0], x - x0, vy - vy0)
from_lambda = SciMLBase.solve(by_lambda, Rodas5P())
from_speed = SciMLBase.solve(by_speed, Rodas5P())
from_lambda.u[1], from_speed.u[1]
```

The height is of order ``10^{-4}``, at a double root of the constraint.

## Count how many initial conditions are needed

Five variables and three algebraic equations need two conditions.
[`initialization_problem`](@ref CasADiCodegen.initialization_problem) returns the
problem at the guess, a `NonlinearProblem` when it is square and a
`NonlinearLeastSquaresProblem` otherwise:

```@example init
under = pendulum([0.0, 0.0, 1.0, 0.0, 1.0], x - x0)
over = pendulum([0.0, 0.0, 1.0, 0.0, 1.0], x - x0, y - y0, vy - vy0)
typeof(CasADiCodegen.initialization_problem(problem)).name.name,
typeof(CasADiCodegen.initialization_problem(under)).name.name,
typeof(CasADiCodegen.initialization_problem(over)).name.name
```

An underdetermined system returns a solution near the guess, and an overdetermined one
succeeds when its conditions agree:

```@example init
SciMLBase.solve(under, Rodas5P()).retcode, SciMLBase.solve(over, Rodas5P()).retcode
```

Contradictory conditions fail; with `vy = 2` the constraint needs `λ = -4`, not 1:

```@example init
target = [1.0, 1.0, 0.0, 2.0]                             # g, x0, y0, vy0
conflicting = pendulum([0.0, 2.0, 1.0, 0.0, 1.0], x - x0, y - y0, vy - vy0, λ - 1;
                       parameters = target)
failed = SciMLBase.solve(conflicting, Rodas5P())
failed.retcode
```

## Solve for the initial values without integrating

Solved alone, the initialization problem shows the residual of the algebraic equations,
then of the conditions; no state satisfies them all:

```@example init
inconsistent = pendulum([0.0, 2.0, 1.0, 0.0, 1.0], x - x0, y - y0, vy - vy0, λ - 1;
                        parameters = target)
initial_values = CasADiCodegen.initialization_problem(inconsistent)
alone = SciMLBase.solve(initial_values, LevenbergMarquardt())
alone.retcode, round.(alone.resid; digits = 3)
```

Its solution is indexed by name, and any method of NonlinearSolve.jl applies:

```@example init
square = SciMLBase.solve(CasADiCodegen.initialization_problem(problem), TrustRegion())
square.retcode, square[:λ]
```

## Change the starting values and the conditions

A new target is a new parameter value and needs no new code:

```@example init
moving = pendulum([1.0, 0.0, 1.0, 0.0, 0.0], x - x0, vy - vy0)
moving.ps[:vy0] = -0.1
SciMLBase.solve(moving, Rodas5P())[:vy][1]
```

Other conditions need a new conditions function, and only its code is compiled.

## Find a parameter during initialization

In a model where `x + y` is the parameter `total`, we give two of the three and compute
the third; `y` is an observed variable:

```@example init
x, total, x0, y0 = SX.sym("x"), SX.sym("total"), SX.sym("x0"), SX.sym("y0")
q = vcat(total, x0, y0)
sum_model = CasadiFunction("sum_model", [x, q, t], [-x, total - x],
                           ["u", "p", "t"], ["du", "y"])
nothing # hide
```

The keyword `solved_parameters` names the parameters to solve for, whose values in the
problem are guesses. With `x = 1` and `y = 2`:

```@example init
both = CasadiFunction("both", [x, q, t], [vcat(x - x0, (total - x) - y0)])
f = SciMLBase.ODEFunction(sum_model; initialization = both, solved_parameters = [:total])
values_sum = [0.0, 1.0, 2.0]                              # total (a guess), x0, y0
sum_problem = SciMLBase.ODEProblem(f, [0.0], (0.0, 1.0), values_sum)
solution = SciMLBase.solve(sum_problem, Tsit5(); initializealg = initialize,
                           abstol = 1e-8, reltol = 1e-8)
solution.ps[:total]
```

## Start the predator-prey model at equilibrium

Conditions on the derivatives initialize an ODE too, here the Lotka-Volterra system at a
steady state:

```@example init
r, s = SX.sym("r"), SX.sym("s")
prey = vcat(r, s)
slope = vcat(1.5r - r * s, -3s + r * s)
lotka = CasadiFunction("lotka", [prey], [slope])
at_rest = CasadiFunction("steady", [prey], [slope])
steady = SciMLBase.ODEFunction(lotka; initialization = at_rest)
equilibrium = SciMLBase.ODEProblem(steady, [2.0, 1.0], (0.0, 10.0))
solution = SciMLBase.solve(equilibrium, Tsit5(); initializealg = initialize,
                           abstol = 1e-8, reltol = 1e-8)
solution.u[1], solution.u[end]
```

The guess selects the steady state ``(3, 1.5)`` over the origin.
