# Initialize differential-algebraic equations

A differential-algebraic equation (DAE) cannot start from arbitrary values: its algebraic
equations, and the constraints hidden in them, must hold at the initial time, and the
values that make them hold are often unknown to the modeler, who knows instead a position,
a velocity or a steady state. In this tutorial, we give a SciML problem the conditions that
we know as a CasADi function, and the solver computes the initial values that satisfy them
and the algebraic equations together before its first step. We start a pendulum from a
position, from a velocity and from equations, see what happens when the conditions are
too few or too many, change them without new code, solve for a parameter, start a
Lotka-Volterra system at its steady state, and finally solve the initialization problem
alone to find out why a set of conditions cannot be met.

The examples need CasADiCodegen.jl, SciMLBase.jl, the standard library LinearAlgebra,
the solvers `Rodas5P` of OrdinaryDiffEqRosenbrock.jl and `Tsit5` of OrdinaryDiffEqTsit5.jl,
and NonlinearSolve.jl, whose Levenberg-Marquardt method solves the initialization
problems.

```@example init
using CasADiCodegen, LinearAlgebra
import SciMLBase
using OrdinaryDiffEqRosenbrock: Rodas5P
using OrdinaryDiffEqTsit5: Tsit5
using NonlinearSolve: LevenbergMarquardt, TrustRegion
```

## Consistent initial values

A DAE in semi-explicit form has differential variables ``x`` and algebraic variables
``z``,

```math
x' = f(x, z, t), \qquad 0 = g(x, z, t),
```

and its initial values are consistent when ``g(x(t_0), z(t_0), t_0) = 0``. An ODE has no
algebraic equation, so that every initial state is consistent. The difficulty of a DAE is
that the modeler must choose the right number of conditions, on the right variables, so
that together with ``g = 0`` they determine all initial values: fixing a differential
variable is a condition, fixing an algebraic one is another, and fixing a derivative is a
third kind, since ``x' = f`` ties it to the state.

[`ODEFunction`](@ref SciMLBase.ODEFunction(::CasADiCodegen.Function)) takes these
conditions as the keyword `initialization`: a CasADi function with the arguments of the
right side whose result is a vector of residuals that must vanish at the initial time.
The initialization problem then has the whole state as its unknowns, and the algebraic
equations of the right side, which are the rows of zeros of the mass matrix, together
with the conditions as its equations. The initial state of the SciML problem is only a
guess from which a nonlinear solver starts.

## Write the pendulum in index-1 form

A pendulum of unit mass on a rod of unit length has the position ``(x, y)`` and the
multiplier ``\lambda``, proportional to the tension of the rod:

```math
x'' = \lambda x, \qquad y'' = \lambda y - g, \qquad 0 = x^2 + y^2 - 1.
```

These equations have index 3, and the tutorial [Model physical systems with
differential-algebraic equations](@ref) shows how to reduce them to index 1 by
differentiating the constraint twice. Here we write the form that a modeling language
derives with the method of dummy derivatives: the height ``y`` and its velocity
``v_y`` are the differential states, and ``x``, its velocity ``v_x`` and ``\lambda`` are
algebraic, determined by the constraint and its first and second time derivatives. The
time derivative of an expression is the sum of its partial derivatives times the
derivatives of the variables, which CasADi computes for us:

```@example init
y, vy, x, vx, λ = (SX.sym(name) for name in ("y", "vy", "x", "vx", "λ"))
g, x0, y0, vy0, t = (SX.sym(name) for name in ("g", "x0", "y0", "vy0", "t"))
variables, slopes = (x, vx, y, vy), (vx, λ * x, vy, λ * y - g)
ddt(e) = sum(jacobian(e, s) * ds for (s, ds) in zip(variables, slopes))

constraint = x^2 + y^2 - 1
u, p = vertcat(y, vy, x, vx, λ), vertcat(g, x0, y0, vy0)
rhs = CasadiFunction("pendulum", [u, p, t], [vertcat(vy, λ * y - g, constraint, ddt(constraint), ddt(ddt(constraint)))])
mass = Diagonal([1.0, 1.0, 0.0, 0.0, 0.0])
nothing # hide
```

Besides the gravity `g`, the parameters hold the target values `x0`, `y0` and `vy0` of
the conditions below, so that a later section can change them without new code. Each
problem of this tutorial comes from the same right side and its own conditions. We give
the problems the Levenberg-Marquardt method as their initialization algorithm, since it
uses the Jacobian that CasADi generates, accepts more or fewer equations than unknowns,
and does not fail where that Jacobian is singular:

```@example init
initialize = SciMLBase.OverrideInit(; nlsolve = LevenbergMarquardt())
conditions(rows...) = CasadiFunction("conditions", [u, p, t], [vertcat(rows...)])

function pendulum(guess, rows...; tspan = (0.0, 1.5), parameters = [1.0, 1.0, 0.0, 0.0])
    f = SciMLBase.ODEFunction(rhs; mass_matrix = mass, initialization = conditions(rows...),
                              specialize = SciMLBase.AutoSpecialize)
    SciMLBase.ODEProblem(f, guess, tspan, parameters; initializealg = initialize, abstol = 1e-8, reltol = 1e-8)
end
nothing # hide
```

`specialize = SciMLBase.AutoSpecialize` gives the functions of all these problems one
type, so that the solvers compiled for the first problem serve the others. The guess is a
vector in the order of the state, (y, vy, x, vx, λ).

## Start from a position

We start the pendulum at the far right, `x = 1` and `y = 0`, with the guess `λ = 1` and
zeros for the other variables:

```@example init
problem = pendulum([0.0, 0.0, 1.0, 0.0, 1.0], x - x0, y - y0)
solution = SciMLBase.solve(problem, Rodas5P())
solution.retcode, solution.u[1]
```

The conditions hold at the first saved state, and so do the algebraic equations, as the
right side shows: its last three elements, the residuals of the constraint and of its two
derivatives, are zero to the tolerance:

```@example init
rhs(solution.u[1], problem.p, 0.0)
```

The conditions must be satisfied, while the guesses are only where the solver starts:
`λ` does not keep the guess 1, but takes the value 0 that the second derivative of the
constraint requires of a pendulum at rest at the side.

Any variable can carry a condition. With `x = 1` and `λ = 0` the solver finds the height,
starting from the guess `y = 1`; with `x = 1` and `vy = 0` it finds both the height and the
multiplier:

```@example init
from_lambda = SciMLBase.solve(pendulum([1.0, 0.0, 1.0, 0.0, 0.0], x - x0, λ), Rodas5P())
from_speed = SciMLBase.solve(pendulum([1.0, 0.0, 1.0, 0.0, 0.0], x - x0, vy - vy0), Rodas5P())
from_lambda.u[1], from_speed.u[1]
```

At `x = 1` the constraint has the double root `y = 0`, where its derivative by `y`
vanishes, so the nonlinear solver converges slowly and stops at a small height, up to
about ``10^{-4}``, the square root of its tolerance. A condition can be any equation, and the
condition `y = 0` is the residual `y` itself. A condition can also contain parameters, as
`vy = g`, which gives the pendulum a vertical speed equal to the gravity:

```@example init
upward = SciMLBase.solve(pendulum([1.0, 0.0, 1.0, 0.0, 0.0], x - x0, vy - g; tspan = (0.0, 3.0)), Rodas5P())
upward.u[1]
```

In the index-1 form a velocity is a state, so that a condition on the derivative of `y`
is a condition on `vy`. A condition on the derivative of another variable is a condition
on the right side, and the conditions function can call `rhs` on its own arguments to
write it.

## Count the conditions

The pendulum has five variables and three algebraic equations, so two conditions
complete a square system; with one condition there are fewer equations than unknowns, and
with three there are more. [`initialization_problem`](@ref CasADiCodegen.initialization_problem)
returns the initialization problem of a SciML problem, at its guess:

```@example init
under = pendulum([0.0, 0.0, 1.0, 0.0, 1.0], x - x0)
over = pendulum([0.0, 0.0, 1.0, 0.0, 1.0], x - x0, y - y0, vy - vy0)
typeof(CasADiCodegen.initialization_problem(problem)).name.name,
typeof(CasADiCodegen.initialization_problem(under)).name.name,
typeof(CasADiCodegen.initialization_problem(over)).name.name
```

A system with as many equations as unknowns is a `NonlinearProblem`, and any other count
gives a `NonlinearLeastSquaresProblem`, whose solution minimizes the sum of the squares
of the residuals. An underdetermined system has many solutions, and the solver returns
one of them near the guess; an overdetermined system has a solution only when its
conditions agree, as `x = 1`, `y = 0` and `vy = 0` do:

```@example init
SciMLBase.solve(under, Rodas5P()).retcode, SciMLBase.solve(over, Rodas5P()).retcode
```

A square system can be singular too. With `x = 1` and `y = 0` the constraint and its
first derivative leave `vy` free, since `x vx + y vy = 0` reduces to `vx = 0`; the
initialization above kept the guess `vy = 0`, and a different guess would give a
different start.

When the conditions contradict each other, the initialization fails and so does the
solve. With `vy = 2` and `λ = 1` at the side, the second derivative of the constraint
needs `λ = -4`:

```@example init
target = [1.0, 1.0, 0.0, 2.0]
failed = SciMLBase.solve(pendulum([0.0, 2.0, 1.0, 0.0, 1.0], x - x0, y - y0, vy - vy0, λ - 1; parameters = target), Rodas5P())
failed.retcode
```

The least-squares solver stops at the smallest sum of squares that it finds, which is
not zero, and SciMLBase then reports `InitialFailure` instead of integrating from values
that do not satisfy the model.

## Solve the initialization problem alone

When an initialization fails, solving its problem alone shows how far the conditions are
from agreement. The problem of `failed` above is a least-squares problem in the five
variables, with the three algebraic equations followed by the four conditions as its
residual:

```@example init
inconsistent = pendulum([0.0, 2.0, 1.0, 0.0, 1.0], x - x0, y - y0, vy - vy0, λ - 1; parameters = target)
alone = SciMLBase.solve(CasADiCodegen.initialization_problem(inconsistent), LevenbergMarquardt())
alone.retcode, round.(alone.resid; digits = 3)
```

The method stops after its largest number of iterations at a point where no residual is
zero: it can trade the conditions against the equations of the model, but no state
satisfies them all. With
`λ = 0` in place of `λ = 1`, the conditions agree with the model, and the residual
vanishes:

```@example init
consistent = pendulum([0.0, 0.0, 1.0, 0.0, 1.0], x - x0, y - y0, vy - vy0, λ; parameters = [1.0, 1.0, 0.0, 0.0])
alone = SciMLBase.solve(CasADiCodegen.initialization_problem(consistent), LevenbergMarquardt())
alone.retcode, maximum(abs, alone.resid)
```

The initialization problem has the names of the variables and of the solved parameters,
so that its solution answers to them, as `alone[:λ]`. It is an ordinary SciML problem,
and any method of NonlinearSolve.jl can solve it; a square problem, as that of
`problem` above, can also go to the trust-region method:

```@example init
square = SciMLBase.solve(CasADiCodegen.initialization_problem(problem), TrustRegion())
square.retcode, square[:λ]
```

## Change the conditions

The targets of the conditions are parameters, so that a new value of one needs no new
code. We start the pendulum at the side with the speed `vy = -0.1` by changing the
parameter `vy0` of a problem with the conditions `x = x0` and `vy = vy0`:

```@example init
moving = pendulum([1.0, 0.0, 1.0, 0.0, 0.0], x - x0, vy - vy0)
moving.ps[:vy0] = -0.1
SciMLBase.solve(moving, Rodas5P())[:vy][1]
```

Other conditions need a new conditions function and a new problem. To fix the height and
the horizontal velocity, `y = 0` and `vx = 0`, and let the initialization find `x`, we
build the problem from those conditions:

```@example init
sideways = SciMLBase.solve(pendulum([0.0, 0.0, 1.0, 0.0, 0.0], y - y0, vx), Rodas5P())
sideways.u[1]
```

With `AutoSpecialize` this problem has the type of the others, and the solve compiles
only the generated code of the new conditions.

## Choose the nonlinear solver

An initialization problem can have several solutions, and the guess selects one of them.
At `x = 0.6` the pendulum hangs at `y = -0.8` or stands at `y = 0.8`, and the
Levenberg-Marquardt method, which uses the Jacobian, stays near the guess `y = -1`:

```@example init
low = SciMLBase.solve(pendulum([-1.0, 0.0, 0.6, 0.0, 0.0], x - 0.6, vy - vy0), Rodas5P())
low[:y][1]
```

Without `initializealg`, OrdinaryDiffEq.jl solves the initialization problem with its
default nonlinear solver, which needs OrdinaryDiffEqNonlinearSolve.jl loaded. That solver
begins with Broyden's method, which does not use the generated Jacobian and can leave the
guess for the other solution, here the height 0.8; this is why the problems of this
tutorial name their method.

## Solve for a parameter

Parameters can be unknowns of the initialization as well. A model in which the sum of two
quantities is a parameter `total` lets the modeler give any two of `x`, `y` and `total`
and compute the third. We write the model as an ODE in `x` with `y = total - x` as an
observed variable, the second result of the right side:

```@example init
x, total, x0, y0 = SX.sym("x"), SX.sym("total"), SX.sym("x0"), SX.sym("y0")
q = vertcat(total, x0, y0)
sum_model = CasadiFunction("sum_model", [x, q, t], [-x, total - x], ["u", "p", "t"], ["du", "y"])
nothing # hide
```

The keyword `solved_parameters` names the parameters that the initialization solves for.
With the conditions `x = 1` and `y = 2`, the value of `total` in the problem is a guess,
and the solution holds the value that the initialization found:

```@example init
both = CasadiFunction("both", [x, q, t], [vertcat(x - x0, (total - x) - y0)])
f = SciMLBase.ODEFunction(sum_model; initialization = both, solved_parameters = [:total])
solution = SciMLBase.solve(SciMLBase.ODEProblem(f, [0.0], (0.0, 1.0), [0.0, 1.0, 2.0]), Tsit5();
                           initializealg = initialize, abstol = 1e-8, reltol = 1e-8)
solution.ps[:total]
```

To give `y` and `total` and solve for `x` instead, the conditions hold `y = 1` alone,
and `total` is an ordinary parameter:

```@example init
only_y = CasadiFunction("only_y", [x, q, t], [(total - x) - y0])
f = SciMLBase.ODEFunction(sum_model; initialization = only_y)
solution = SciMLBase.solve(SciMLBase.ODEProblem(f, [1.0], (0.0, 1.0), [4.0, 0.0, 1.0]), Tsit5();
                           initializealg = initialize, abstol = 1e-8, reltol = 1e-8)
solution.u[1], solution[:y][1]
```

## Start at a steady state

An ODE has no algebraic equations, but conditions on its derivatives still make an
initialization problem. The Lotka-Volterra system is at a steady state where both
derivatives vanish, so its conditions are its right side itself:

```@example init
r, s = SX.sym("r"), SX.sym("s")
prey = vertcat(r, s)
slope = vertcat(1.5r - r * s, -3s + r * s)
lotka = CasadiFunction("lotka", [prey], [slope])
steady = SciMLBase.ODEFunction(lotka; initialization = CasadiFunction("steady", [prey], [slope]))
solution = SciMLBase.solve(SciMLBase.ODEProblem(steady, [2.0, 1.0], (0.0, 10.0)), Tsit5();
                           initializealg = initialize, abstol = 1e-8, reltol = 1e-8)
solution.u[1], solution.u[end]
```

The system has two steady states, the origin and ``(3, 1.5)``, and the guess selects the
second. A condition can also hold an expression of the state that is not a variable of
the model: with the derivative of `r` at zero and `r + s = 2`, the start is ``(0.5, 1.5)``:

```@example init
observed = CasadiFunction("observed", [prey], [vertcat(slope[1], r + s - 2)])
solution = SciMLBase.solve(SciMLBase.ODEProblem(SciMLBase.ODEFunction(lotka; initialization = observed), [1.0, 1.0], (0.0, 10.0)),
                           Tsit5(); initializealg = initialize, abstol = 1e-8, reltol = 1e-8)
solution.u[1]
```
