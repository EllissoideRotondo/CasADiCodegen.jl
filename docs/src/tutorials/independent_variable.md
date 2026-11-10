# Change the independent variable

When the derivative ``f_s`` of one state ``s`` of ``x' = f(x, t)`` has no zero on the
interval of the solution, ``s`` can replace the time as the independent variable, and the
chain rule gives the equations in ``s``,

```math
\frac{\mathrm{d}x_j}{\mathrm{d}s} = \frac{f_j(x, t)}{f_s(x, t)}, \qquad
\frac{\mathrm{d}t}{\mathrm{d}s} = \frac{1}{f_s(x, t)},
```

where the time becomes a state. In this tutorial, we write this change on CasADi
expressions, solve a projectile in its horizontal distance, and integrate a cosmological
model in the logarithm of its scale factor.

## The change of variable on expressions

The chain rule is a division of CasADi expressions. The function below takes the state,
the right side, the index `i` of the new independent variable and the old one `t`, which
must be the symbol that the right side uses (a new symbol for an autonomous model). It
returns the new state `z`, the state without element `i` followed by `t`, and its
derivative `dz`:

```@example ivar
using CasADiCodegen
using OrdinaryDiffEqTsit5: Tsit5
import SciMLBase

function change_independent_variable(state, rhs, i, t)
    n = size(state, 1)
    @assert size(rhs, 1) == n && 1 <= i <= n
    keep = [j for j in 1:n if j != i]
    return vcat(state[keep], t), vcat(rhs[keep] / rhs[i], 1 / rhs[i])
end
nothing # hide
```

The symbol `state[i]` stays in `dz`, so it is the time argument of the CasADi function
given to `ODEFunction`.

## A projectile in its horizontal distance

A ball is thrown at 45 degrees with the velocity ``v`` in each direction, so that

```math
\dot{x} = v, \qquad \dot{y} = v_y, \qquad \dot{v}_y = -g,
```

from ``x = y = 0``. To find the height at a target 10 m away, we take ``x`` as the
independent variable, since ``\dot{x} = v`` is never zero:

```@example ivar
x, y, vy, t = SX.sym("x"), SX.sym("y"), SX.sym("vy"), SX.sym("t")
g, v = SX.sym("g"), SX.sym("v")
z, dz = change_independent_variable(vcat(x, y, vy), vcat(v, vy, -g), 1, t)
ball = CasadiFunction("ball", [z, vcat(g, v), x], [dz])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(ball), [0.0, 8.0, 0.0], (0.0, 10.0), [9.81, 8.0])
solution = SciMLBase.solve(problem, Tsit5())
solution[[:y, :t]][end]
```

The span of the problem is a distance, so the solve stops at the target and matches the
exact ``t = 1.25`` s and ``y = 2.3359375`` m to rounding. In the time, the same question
needs a `ContinuousCallback` that stops the solver where ``x`` reaches 10 m:

```@example ivar
in_time = CasadiFunction("ball_t", [vcat(x, y, vy), vcat(g, v), t], [vcat(v, vy, -g)])
stop = SciMLBase.ContinuousCallback((u, t, integrator) -> u[1] - 10, SciMLBase.terminate!)
problem_t = SciMLBase.ODEProblem(SciMLBase.ODEFunction(in_time), [0.0, 0.0, 8.0], (0.0, 5.0), [9.81, 8.0])
solution_t = SciMLBase.solve(problem_t, Tsit5(); callback = stop)
solution_t.t[end], solution_t[:y][end]
```

Both give the same answer. The change of variable avoids the root finding of the event
and gives the trajectory as a function of distance.

## A cosmological model in logarithmic time

The Friedmann equations describe the expansion of the universe through its scale factor
``a``, with ``a = 1`` today. Each species has a density ``\Omega_s`` that varies as
``a^{-3(1 + w_s)}``, with ``w = 1/3`` for radiation, ``0`` for matter and ``-1`` for dark
energy. In the conformal time ``t``, with ``\Omega`` the sum of the densities,

```math
a' = \sqrt{\Omega}\, a^2, \qquad \Omega_s' = -3 (1 + w_s) \frac{a'}{a} \Omega_s .
```

Integrated backward, ``a`` reaches zero at a conformal time not known before the solve.
An interval that reaches past it makes the solver stop with the code `Unstable` and a
warning, which we silence:

```@example ivar
using Logging: with_logger, NullLogger

Ωs, a = SX.sym("Ω", 3), SX.sym("a")
w = [1 / 3, 0, -1]
da = sqrt(sum(Ωs[k] for k in 1:3)) * a^2
dΩ = vcat((-3 * (1 + w[k]) * da / a * Ωs[k] for k in 1:3)...)
friedmann = CasadiFunction("friedmann", [vcat(Ωs, a)], [vcat(dΩ, da)])
today = [5e-5, 0.3, 1 - 5e-5 - 0.3, 1.0]                  # Ω_r, Ω_m, Ω_Λ, a
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(friedmann), today, (0.0, -3.3))
solution = with_logger(() -> SciMLBase.solve(problem, Tsit5(); reltol = 1e-5), NullLogger())
solution.retcode, solution.t[end]
```

For ``\Omega > 0``, ``a`` grows monotonically and can replace the time, and an interval
of ``a`` can end before the singularity. We change from ``t`` to ``a``, which keeps the
conformal time as the fourth state, and then to ``b = \ln a``, which spreads the orders of
magnitude evenly: the derivative by ``b`` is the derivative by ``a`` with ``a = e^b``,
times ``e^b``.

```@example ivar
z, dz_a = change_independent_variable(vcat(Ωs, a), vcat(dΩ, da), 4, SX.sym("t"))
b = SX.sym("b")
dz_b = substitute(dz_a, a, exp(b)) * exp(b)
nothing # hide
```

The right side depends on ``b``, the time argument. This model has no parameters, so an
empty column `SX(0, 1)` holds their place:

```@example ivar
friedmann_b = CasadiFunction("friedmann_b", [z, SX(0, 1), b], [dz_b])
problem_b = SciMLBase.ODEProblem(SciMLBase.ODEFunction(friedmann_b), [today[1:3]; 0.0], (0.0, -15.0))
solution_b = SciMLBase.solve(problem_b, Tsit5(); reltol = 1e-8)
solution_b.retcode, solution_b.t[end]
```

The solver reaches ``b = -15``, a scale factor of ``3 \times 10^{-7}``. In ``b``, the
densities follow ``\Omega_s(b) = \Omega_s(0)\, e^{-3 (1 + w_s) b}``, and the largest
relative difference from this exact solution is

```@example ivar
exact = [today[k] * exp(-3 * (1 + w[k]) * -15.0) for k in 1:3]
maximum(abs, solution_b.u[end][1:3] ./ exact .- 1)
```

The conformal time changes by 3 between ``b = -5`` and ``b = 0`` but by less than 0.04
between ``b = -15`` and ``b = -9``, so the radiation era, which the solve in ``b`` resolves
evenly, lies in a tiny interval near ``t = -3.2584``, where the solve in ``t`` stopped:

```@example ivar
[(bk, solution_b(bk)[4]) for bk in (-15.0, -9.0, -5.0, 0.0)]
```
