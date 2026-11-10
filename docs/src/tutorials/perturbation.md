# Perturbation expansions of an ODE

When a differential equation contains a small parameter ``\varepsilon``, its solution can
be sought as a power series

```math
x(t, \varepsilon) = x_0(t) + \varepsilon\, x_1(t) + \varepsilon^2 x_2(t) + \dots,
```

whose coefficients satisfy a hierarchy of linear equations, each driven by the
coefficients before it. In this tutorial, we form the hierarchy from the derivatives of
the right side by ``\varepsilon`` and solve it for a body in free fall and for a damped
oscillator.

## The Taylor coefficients of an expression

The coefficient of ``\varepsilon^k`` in the Taylor series of an expression ``e`` is
``e^{(k)}(0) / k!``. The function below forms these coefficients element by element with
`jacobian` and `substitute`:

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

For an equation ``\ddot{x} = F(x, \dot{x}, \varepsilon)``, the states are the
coefficients of position and velocity, and the coefficient of ``\varepsilon^k`` of ``F``
on the two series is the acceleration of ``x_k``:

```@example perturbation
function perturbation_model(name, F, order)
    ε = SX.sym("ε")
    x, v = SX.sym("x", order + 1), SX.sym("v", order + 1)
    series(c) = sum(ε^k * c[k+1] for k in 0:order)
    accelerations = taylor_coefficients(F(series(x), series(v), ε), ε, order)
    return CasadiFunction(name, [vcat(x, v)], [vcat(v, accelerations...)]), accelerations
end
nothing # hide
```

CasADi numbers the elements of a vector symbol from 0, so the states are named `x_0` to
`x_n` and `v_0` to `v_n`.

## Free fall in a varying gravitational field

A body thrown upward from a planet feels a gravity that decreases with its altitude
``x``. In units of the radius ``R`` and of the time ``\sqrt{R^3 / (G M)}``, its motion
follows

```math
\ddot{x} = -\frac{1}{(1 + \varepsilon x)^2},
```

where ``\varepsilon = 1`` is the actual problem and ``\varepsilon = 0`` the constant
gravity of the surface, with ``x_0 = t - t^2 / 2``. The hierarchy to the second order has
six states:

```@example perturbation
freefall, accelerations = perturbation_model("freefall", (x, v, ε) -> -(1 + ε * x)^(-2), 2)
accelerations
```

Each equation depends only on the coefficients before it. The initial
values go to ``x_0`` and ``\dot{x}_0``, and the higher coefficients start at zero:

```@example perturbation
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(freefall), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], (0.0, 3.0))
hierarchy = SciMLBase.solve(problem, Tsit5(); abstol = 1e-10, reltol = 1e-10)
hierarchy[[:x_0, :x_1, :x_2]][end]
```

The first two coefficients agree with the exact ``x_0 = -1.5`` and
``x_1 = t^3 / 3 - t^4 / 12 = 2.25`` at ``t = 3``. We compare the truncated series with
solves of the full equation, which has no ground, for several ``\varepsilon``:

```@example perturbation
X, e, t = SX.sym("X", 2), SX.sym("e"), SX.sym("t")
full = CasadiFunction("full", [X, e, t], [vcat(X[2], -(1 + e * X[1])^(-2))])
full_problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(full), [0.0, 1.0], (0.0, 3.0), [0.0])
series_at(ε) = sum(ε^k * hierarchy[Symbol("x_", k)][end] for k in 0:2)
direct_at(ε) = SciMLBase.solve(SciMLBase.remake(full_problem; p = [ε]), Tsit5(); abstol = 1e-10, reltol = 1e-10)[1, end]
[(ε, series_at(ε), direct_at(ε)) for ε in (0.01, 0.1, 0.3, 1.0)]
```

Each tuple gives ``\varepsilon``, the series and the full solution. The difference is of
order ``\varepsilon^3`` for small ``\varepsilon``, while at ``\varepsilon = 1`` three
terms do not capture the correction.

## The coefficients as derivatives of the solution

The coefficient ``x_k`` is also ``\partial^k x / \partial \varepsilon^k / k!`` at
``\varepsilon = 0``, which an integrator inside a CasADi graph gives from the
sensitivities of the full equation (see [Integrate a differential equation](@ref)). An
integrator can be called only from an `MX` graph, so the symbols are `MX`:

```@example perturbation
using OrdinaryDiffEqVerner: Vern9

Xm, em = MX.sym("X", 2), MX.sym("e")
ode = Dict("x" => Xm, "p" => em, "ode" => vcat(Xm[2], -(1 + em * Xm[1])^(-2)))
flow = integrator("flow", Vern9(), ode, 0.0, 3.0; abstol = 1e-12, reltol = 1e-12)
position = flow(MX([0.0, 1.0]), em)[1]
first_derivative = jacobian(position, em)
derivatives = CasadiFunction("derivatives", [em], [position, first_derivative, jacobian(first_derivative, em) / 2])
coefficients = build_function(derivatives; expression = Val{false})[1]
coefficients([0.0])
```

The three numbers agree with the hierarchy at ``t = 3``. This route serves any order
without symbolic rewriting, but it gives the coefficients only at the final time or on a
grid, and each order costs a further integration; the hierarchy gives them as functions of
time in one solve.

## A weakly damped oscillator

The damped harmonic oscillator shows a known weakness of the expansion:

```math
\ddot{x} + 2 \varepsilon \dot{x} + x = 0, \qquad x(0) = 0, \quad \dot{x}(0) = 1.
```

Its exact solution is
``x = e^{-\varepsilon t} \sin(\sqrt{1 - \varepsilon^2}\, t) / \sqrt{1 - \varepsilon^2}``.
We solve the hierarchy up to ``t = 50``:

```@example perturbation
oscillator, _ = perturbation_model("oscillator", (x, v, ε) -> -2ε * v - x, 2)
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(oscillator), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], (0.0, 50.0))
hierarchy = SciMLBase.solve(problem, Tsit5(); abstol = 1e-10, reltol = 1e-10)
exact(t, ε) = exp(-ε * t) * sin(sqrt(1 - ε^2) * t) / sqrt(1 - ε^2)
series_at(t, ε) = (u = hierarchy(t); u[1] + ε * u[2] + ε^2 * u[3])
[(t, series_at(t, 0.1), exact(t, 0.1)) for t in (pi / 2, 5pi / 2, 50.0)]
```

The series is close at the first maximum, but at ``t = 50`` it is large where the exact
solution has decayed. The coefficients ``x_1`` and ``x_2`` grow like ``t`` and ``t^2``:
these secular terms expand the decay and the frequency shift in powers of
``\varepsilon t``, so the series holds only for ``t \ll 1 / \varepsilon``, and a uniform
approximation needs an expansion in two time scales.
