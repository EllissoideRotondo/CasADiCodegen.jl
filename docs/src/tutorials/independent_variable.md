# Change the independent variable

A differential equation is usually written in the variable in which its laws are stated,
most often the time, but another variable can serve the purpose of a simulation better:
the horizontal distance of a projectile when the question is where it lands, or the
logarithm of a scale factor when the solution changes over many orders of magnitude. When
the derivative ``f_s`` of one state ``s`` of ``x' = f(x, t)`` has no zero on the interval
of the solution, ``s`` is strictly monotonic in time and can replace it, and the
chain rule gives the equations of the other states in ``s``,

```math
\frac{\mathrm{d}x_j}{\mathrm{d}s} = \frac{f_j(x, t)}{f_s(x, t)}, \qquad
\frac{\mathrm{d}t}{\mathrm{d}s} = \frac{1}{f_s(x, t)},
```

and the time becomes a state
whose equation follows from the rule for the derivative of an inverse function. In this
tutorial, we write this rewriting as a few operations on CasADi expressions, solve a
projectile in its horizontal distance, and then integrate a cosmological model in the
logarithm of its scale factor, where the interval of the original time is not known in
advance. The examples
need CasADiCodegen.jl, SciMLBase.jl and OrdinaryDiffEqTsit5.jl.

## The change of variable on expressions

A CasADi model holds its right side as an expression of the symbols of the state, so the
chain rule above is a division of expressions. The function below takes the state and the
right side as CasADi columns, the index `i` of the new independent variable in the state,
and the old independent variable `t`. It returns the pair
`(z, dz)`: the new state `z`, which is the state without its element `i`, followed by the
old variable `t`, and the derivative `dz` of `z` by the element `i`, which becomes the
independent variable. The symbol `t` must be the one that the right side uses; for an
autonomous model, a new symbol serves:

```@example ivar
using CasADiCodegen
using OrdinaryDiffEqTsit5: Tsit5
import SciMLBase

function change_independent_variable(state, rhs, i, t)
    n = size(state, 1)
    @assert size(rhs, 1) == n && 1 <= i <= n
    keep = [j for j in 1:n if j != i]
    return vertcat(state[keep], t), vertcat(rhs[keep] / rhs[i], 1 / rhs[i])
end
nothing # hide
```

The right side `dz` still contains the symbol `state[i]` where `rhs` contained it, so that
symbol is the third argument, the time, of the CasADi function that we give to
`ODEFunction`.

## A projectile in its horizontal distance

A ball is thrown at 45 degrees in the gravity ``g``, so that its horizontal and vertical
velocities are both ``v`` at the start. Its position ``(x, y)`` and its vertical velocity
``v_y`` follow

```math
\dot{x} = v, \qquad \dot{y} = v_y, \qquad \dot{v}_y = -g,
```

from ``x = y = 0`` and ``v_y(0) = v``. To know the height of the ball at a target
10 m away, we take ``x`` as the independent variable, which is possible because
``\dot{x} = v`` is never zero, and get the state ``(y, v_y, t)`` as a function of
``x``:

```@example ivar
x, y, vy, t = SX.sym("x"), SX.sym("y"), SX.sym("vy"), SX.sym("t")
g, v = SX.sym("g"), SX.sym("v")
z, dz = change_independent_variable(vertcat(x, y, vy), vertcat(v, vy, -g), 1, t)
ball = CasadiFunction("ball", [z, vertcat(g, v), x], [dz])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(ball), [0.0, 8.0, 0.0], (0.0, 10.0), [9.81, 8.0])
solution = SciMLBase.solve(problem, Tsit5())
solution[[:y, :t]][end]
```

The span of the problem is now a span of distance, from 0 to 10 m, so that the solve
stops at the target, and the solution interpolates the height at any distance in between,
for instance `solution(5.0)`. The exact values at 10 m are the time of flight
``t = x / v = 1.25`` s and the height ``y = v t - g t^2 / 2 = 2.3359375`` m, which the
solver reproduces to rounding, since the solution is a polynomial of second degree in
``x``.

In the time, the same question needs an event that stops the solver where ``x`` reaches
10 m. A `ContinuousCallback` of SciMLBase.jl with the condition ``x - 10`` and the affect
`terminate!` does this:

```@example ivar
in_time = CasadiFunction("ball_t", [vertcat(x, y, vy), vertcat(g, v), t], [vertcat(v, vy, -g)])
stop = SciMLBase.ContinuousCallback((u, t, integrator) -> u[1] - 10, SciMLBase.terminate!)
problem_t = SciMLBase.ODEProblem(SciMLBase.ODEFunction(in_time), [0.0, 0.0, 8.0], (0.0, 5.0), [9.81, 8.0])
solution_t = SciMLBase.solve(problem_t, Tsit5(); callback = stop)
solution_t.t[end], solution_t[:y][end]
```

Both routes give the same answer here. The change of variable avoids the root finding of
the event, and it gives the trajectory as a function of the distance, which suits a
question about a place, while the event suits a single crossing.

## A cosmological model in logarithmic time

In cosmology, the Friedmann equations describe the expansion of the universe through its
scale factor ``a``, the size of the universe relative to today, so that ``a = 1`` today
and ``a = 0`` at its beginning. Each species has a density ``\Omega_s``, in units
of the total density today, which varies as ``a^{-3(1 + w_s)}`` when the universe expands,
with ``w = 1/3`` for radiation, ``0`` for matter and ``-1`` for dark energy. In the
conformal time ``t``, whose increment is that of the cosmic time divided by ``a``, in units
of the inverse of today's rate of expansion, and with
``\Omega`` the sum of the densities,

```math
a' = \sqrt{\Omega}\, a^2, \qquad \Omega_s' = -3 (1 + w_s) \frac{a'}{a} \Omega_s .
```

Integrated backward from today, the scale factor reaches zero at a finite conformal time,
where the densities diverge, and this time is not known before the solve. An interval that
reaches past it has no solution at its end: the solver takes ever smaller steps as it
approaches the singularity, and stops with the code `Unstable` and a warning, which we
silence with the logger of the standard library Logging:

```@example ivar
using Logging: with_logger, NullLogger

Ωs, a = SX.sym("Ω", 3), SX.sym("a")
w = [1 / 3, 0, -1]
da = sqrt(sum(Ωs[k] for k in 1:3)) * a^2
dΩ = vertcat((-3 * (1 + w[k]) * da / a * Ωs[k] for k in 1:3)...)
friedmann = CasadiFunction("friedmann", [vertcat(Ωs, a)], [vertcat(dΩ, da)])
today = [5e-5, 0.3, 1 - 5e-5 - 0.3, 1.0]                  # Ω_r, Ω_m, Ω_Λ, a
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(friedmann), today, (0.0, -3.3))
solution = with_logger(() -> SciMLBase.solve(problem, Tsit5(); reltol = 1e-5), NullLogger())
solution.retcode, solution.t[end]
```

The scale factor grows monotonically with the time when ``\Omega > 0``, so it can replace
the time, and an interval of the scale factor, unlike one of the time, can be chosen to end
before the beginning of the universe. Its logarithm ``b = \ln a``, which spreads the
orders of magnitude evenly, can replace it in turn. We change from ``t`` to ``a`` with the function above, which keeps
the conformal time as the fourth state, and then from ``a`` to ``b``: with ``a = e^b`` and
``\mathrm{d}a / \mathrm{d}b = e^b``, the derivative by ``b`` is the derivative by ``a``
with ``a`` replaced by ``e^b``, times ``e^b``. `substitute` makes the replacement in the
expression:

```@example ivar
z, dz_a = change_independent_variable(vertcat(Ωs, a), vertcat(dΩ, da), 4, SX.sym("t"))
b = SX.sym("b")
dz_b = substitute(dz_a, a, exp(b)) * exp(b)
nothing # hide
```

The new right side depends on ``b`` itself, which is therefore the time argument of the
function. `ODEFunction` takes the time as the third argument, after the parameters, and
this model has none, so an empty column of parameters, `SX(0, 1)`, holds their place, and the
problem gets no parameters:

```@example ivar
friedmann_b = CasadiFunction("friedmann_b", [z, SX(0, 1), b], [dz_b])
problem_b = SciMLBase.ODEProblem(SciMLBase.ODEFunction(friedmann_b), [today[1:3]; 0.0], (0.0, -15.0))
solution_b = SciMLBase.solve(problem_b, Tsit5(); reltol = 1e-8)
solution_b.retcode, solution_b.t[end]
```

The solver reaches ``b = -15``, a scale factor of ``3 \times 10^{-7}``, since the interval ends before the singularity; we take a tight one to compare with the
exact solution. In ``b``, the equations of the densities are linear,
``\mathrm{d}\Omega_s / \mathrm{d}b = -3 (1 + w_s) \Omega_s``, with the exact solution
``\Omega_s(b) = \Omega_s(0)\, e^{-3 (1 + w_s) b}``, which explains why they vary so evenly. The largest relative difference of the
solution from it at the end is

```@example ivar
exact = [today[k] * exp(-3 * (1 + w[k]) * -15.0) for k in 1:3]
maximum(abs, solution_b.u[end][1:3] ./ exact .- 1)
```

The fourth state is the conformal time. It changes by 3 between ``b = -5`` and ``b = 0``,
but by less than 0.04 between ``b = -15`` and ``b = -9``, where radiation dominates, so
that this era, which the solve in ``b`` resolves in evenly spaced steps, occupies a tiny
interval of conformal time just after ``a = 0``, near ``t = -3.2584``, where the solve in
``t`` stopped:

```@example ivar
[(bk, solution_b(bk)[4]) for bk in (-15.0, -9.0, -5.0, 0.0)]
```
