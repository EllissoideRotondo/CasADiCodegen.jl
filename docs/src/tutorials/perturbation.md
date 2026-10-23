# Perturbation expansions of an ODE

When a differential equation contains a small parameter ``\varepsilon`` and is simple at
``\varepsilon = 0``, its solution can be sought as a power series in ``\varepsilon``,

```math
x(t, \varepsilon) = x_0(t) + \varepsilon\, x_1(t) + \varepsilon^2 x_2(t) + \dots,
```

whose coefficients satisfy a hierarchy of equations: the equation of ``x_0`` is the
unperturbed one, and each further coefficient obeys a linear equation driven by the
coefficients before it. One solve of the hierarchy then gives the solution for every small
``\varepsilon`` at once, and the coefficients show how the parameter acts. The hierarchy
follows from inserting the series into the equation and collecting the terms of each
power of ``\varepsilon``, which is the Taylor expansion of the right side in
``\varepsilon``; done by hand, this is long and error-prone beyond the first order. In this
tutorial, we form the hierarchy with CasADi from the derivatives of the right side by
``\varepsilon``, solve it for a body in free fall and for a damped oscillator, and check
the coefficients against derivatives of the solution itself, which an integrator inside a
CasADi graph gives. The examples need CasADiCodegen.jl, SciMLBase.jl,
OrdinaryDiffEqTsit5.jl and OrdinaryDiffEqVerner.jl.

## The Taylor coefficients of an expression

The coefficient of ``\varepsilon^k`` in the Taylor series of an expression ``e`` about
``\varepsilon = 0`` is ``e^{(k)}(0) / k!``. The function below forms the derivatives one
after the other with `jacobian`, each divided by its order, and sets ``\varepsilon`` to
zero in each with `substitute`; it works element by element on a column:

```@example perturbation
using CasADiCodegen
using OrdinaryDiffEqTsit5: Tsit5
import SciMLBase

function taylor_coefficients(e, ε, order)
    @assert order >= 0
    terms = [e]
    for k in 1:order
        push!(terms, jacobian(terms[end], ε) / k)
    end
    return [substitute(c, ε, 0) for c in terms]
end
nothing # hide
```

For an equation of the second order ``\ddot{x} = F(x, \dot{x}, \varepsilon)``, the
coefficients of the position and of the velocity are the states, and the coefficient of
``\varepsilon^k`` of ``F`` evaluated on the two series is the acceleration of ``x_k``. The
function below builds the CasADi function of the hierarchy to the given order for any
``F``:

```@example perturbation
function perturbation_model(name, F, order)
    ε = SX.sym("ε")
    x, v = SX.sym("x", order + 1), SX.sym("v", order + 1)
    series(c) = sum(ε^k * c[k+1] for k in 0:order)
    accelerations = taylor_coefficients(F(series(x), series(v), ε), ε, order)
    return CasadiFunction(name, [vertcat(x, v)], [vertcat(v, accelerations...)]), accelerations
end
nothing # hide
```

The symbols `x` and `v` hold the coefficients ``x_0, \dots, x_n`` and
``\dot{x}_0, \dots, \dot{x}_n``, named `x_0` to `x_n` and `v_0` to `v_n`, since CasADi
numbers the elements of a vector symbol from 0 (`x[1]` in Julia is `x_0`). The result no
longer contains ``\varepsilon``, which `substitute` has set to zero. The function returns the CasADi function and, to inspect, the accelerations.

## Free fall in a varying gravitational field

A body thrown upward from the surface of a planet of radius ``R`` feels a gravity that
decreases with its altitude ``x``. In units of ``R`` and of the time
``\sqrt{R^3 / (G M)}``, its motion follows

```math
\ddot{x} = -\frac{1}{(1 + \varepsilon x)^2},
```

where ``\varepsilon = 1`` is the actual problem and ``\varepsilon = 0`` the constant
gravity of the surface, with the solution ``x_0 = t - t^2 / 2`` from ``x(0) = 0`` and
``\dot{x}(0) = 1``. The hierarchy to the second order has six states:

```@example perturbation
freefall, accelerations = perturbation_model("freefall", (x, v, ε) -> -(1 + ε * x)^(-2), 2)
accelerations
```

The equation of ``x_1`` is ``\ddot{x}_1 = 2 x_0``, the correction for the weaker gravity
along the unperturbed path, and that of ``x_2`` depends on ``x_0`` and ``x_1`` only, so
that the hierarchy is triangular: each equation is driven by the coefficients before it,
and here, since ``F`` does not depend on ``x`` at ``\varepsilon = 0``, ``x_k`` does not
appear in its own equation.
The state is ``(x_0, x_1, x_2, \dot{x}_0, \dot{x}_1, \dot{x}_2)``. Since the series must
meet the initial values for every ``\varepsilon``, ``x_0`` and ``\dot{x}_0`` take the
initial position and velocity, and the higher coefficients start at zero:

```@example perturbation
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(freefall), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], (0.0, 3.0))
hierarchy = SciMLBase.solve(problem, Tsit5(); abstol = 1e-10, reltol = 1e-10)
hierarchy[[:x_0, :x_1, :x_2]][end]
```

The equation has no ground, so the body passes ``x = 0`` and goes on falling; we keep the
interval of the ModelingToolkit.jl example, which ends at ``t = 3``. The first two
coefficients agree with the exact ones at ``t = 3``, ``x_0 = -1.5`` and
``x_1 = t^3 / 3 - t^4 / 12 = 2.25``. The sum of the series for any ``\varepsilon`` is now a
sum of three numbers, with no further solve. To see where the truncated series holds, we
compare it with solves of the full equation, with ``\varepsilon`` as a parameter, at
``t = 3``:

```@example perturbation
X, e, t = SX.sym("X", 2), SX.sym("e"), SX.sym("t")
full = CasadiFunction("full", [X, e, t], [vertcat(X[2], -(1 + e * X[1])^(-2))])
full_problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(full), [0.0, 1.0], (0.0, 3.0), [0.0])
series_at(ε) = sum(ε^k * hierarchy[Symbol("x_", k)][end] for k in 0:2)
direct_at(ε) = SciMLBase.solve(SciMLBase.remake(full_problem; p = [ε]), Tsit5(); abstol = 1e-10, reltol = 1e-10)[1, end]
[(ε, series_at(ε), direct_at(ε)) for ε in (0.01, 0.1, 0.3, 1.0)]
```

Each tuple gives ``\varepsilon``, the sum of the series and the solution of the full
equation. The difference is of the order of ``\varepsilon^3`` for small ``\varepsilon``, as
expected of a series truncated after the second order, and at ``\varepsilon = 1`` the series
is far off: gravity weakens so much with altitude that the body is still above the
surface at ``t = 3``, and three terms do not capture a correction of that size.

## The coefficients as derivatives of the solution

Since ``x_k(t)`` is the coefficient of ``\varepsilon^k`` in the solution, it is also the
derivative ``\partial^k x(t, \varepsilon) / \partial \varepsilon^k`` at
``\varepsilon = 0`` divided by ``k!``. An integrator inside a CasADi graph gives these
derivatives without the hierarchy, from the sensitivities of the full equation with respect
to its parameter (see [Integrate a differential equation](@ref)). An integrator is a CasADi
function that only an `MX` graph can call, so the symbols are `MX` here. We integrate the
full equation to ``t = 3`` with ``\varepsilon`` as the parameter and differentiate the final
position twice; the call of `flow` returns the final state, whose first element is the
position:

```@example perturbation
using OrdinaryDiffEqVerner: Vern9

Xm, em = MX.sym("X", 2), MX.sym("e")
ode = Dict("x" => Xm, "p" => em, "ode" => vertcat(Xm[2], -(1 + em * Xm[1])^(-2)))
flow = integrator("flow", Vern9(), ode, 0.0, 3.0; abstol = 1e-12, reltol = 1e-12)
position = flow(MX([0.0, 1.0]), em)[1]
first_derivative = jacobian(position, em)
derivatives = CasadiFunction("derivatives", [em], [position, first_derivative, jacobian(first_derivative, em) / 2])
coefficients = build_function(derivatives; expression = Val{false})[1]
coefficients([0.0])
```

The three numbers agree with ``x_0``, ``x_1`` and ``x_2`` of the hierarchy at ``t = 3``.
This route needs no symbolic rewriting and serves any order, but it gives the coefficients
at the final time of the integrator only, or at the times of a grid, and each order costs a
further integration of sensitivities; the hierarchy gives the coefficients as functions of
time in one solve.

## A weakly damped oscillator

Perturbation theory has a known weakness, which the damped harmonic oscillator shows:

```math
\ddot{x} + 2 \varepsilon \dot{x} + x = 0, \qquad x(0) = 0, \quad \dot{x}(0) = 1.
```

At ``\varepsilon = 0``, the solution is ``\sin t``, and the exact solution is
``x = e^{-\varepsilon t} \sin(\sqrt{1 - \varepsilon^2}\, t) / \sqrt{1 - \varepsilon^2}``.
The same function builds the hierarchy, and we solve it up to ``t = 50``:

```@example perturbation
oscillator, _ = perturbation_model("oscillator", (x, v, ε) -> -2ε * v - x, 2)
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(oscillator), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], (0.0, 50.0))
hierarchy = SciMLBase.solve(problem, Tsit5(); abstol = 1e-10, reltol = 1e-10)
exact(t, ε) = exp(-ε * t) * sin(sqrt(1 - ε^2) * t) / sqrt(1 - ε^2)
series_at(t, ε) = (u = hierarchy(t); u[1] + ε * u[2] + ε^2 * u[3])
[(t, series_at(t, 0.1), exact(t, 0.1)) for t in (pi / 2, 5pi / 2, 50.0)]
```

At the first maximum, the series with ``\varepsilon = 0.1`` is close to the exact solution,
but the two drift apart, and at ``t = 50`` the series is large where the exact solution has
decayed to almost zero. The coefficients grow with time, ``x_1`` like ``t`` and ``x_2``
like ``t^2``: these terms, called secular because they grow without bound, expand the
decay ``e^{-\varepsilon t}`` and the shift of the frequency in powers of ``\varepsilon t``,
which is small only for ``t \ll 1 / \varepsilon``. An approximation
that holds for all times needs an expansion in two time scales, ``t`` and
``\varepsilon t``, which no higher order of this series replaces.
