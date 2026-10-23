# Bifurcation diagrams

The steady states of a model change with its parameters, and at some values of a
parameter they change in number or in stability: a steady state splits into three, or
loses its stability while a periodic orbit appears around it. A bifurcation diagram shows
these changes, and BifurcationKit.jl computes it by continuation, which follows a branch of
steady states as the parameter moves, detects the special points on it and starts the new
branches that leave them. The method needs the residual of the equations and its Jacobian
at each step, and derivatives of a higher order at the special points. In this tutorial, we give
BifurcationKit.jl a CasADi function, so that all these derivatives are generated code,
and we select the continued parameter and the quantities that the branches record by
their names. We compute the diagram of a pitchfork, then the Hopf point of a system of
differential equations and the branch of periodic orbits that starts there.

The examples need CasADiCodegen.jl and BifurcationKit.jl; `using BifurcationKit` also
loads SciMLBase.jl, on which the method of this tutorial depends.

```@example bifurcation
using CasADiCodegen
using BifurcationKit
nothing # hide
```

## The bifurcation problem of a CasADi function

As a first example, we take the steady states of a system of two states ``x`` and ``y``
and two parameters ``\mu`` and ``\alpha``:

```math
\begin{aligned}
0 &= \mu x - x^3 + \alpha y, \\
0 &= -y.
\end{aligned}
```

The state ``x = y = 0`` is a steady state for every ``\mu``, and for ``\mu > 0`` there are
two more, ``x = \pm\sqrt{\mu}`` with ``y = 0``. As for a `NonlinearFunction` (see
[Solve a system of nonlinear equations](@ref)), the CasADi function takes the state and
the parameters, in this order, and its first result is the residual. The states and the
parameters take the names of their scalar symbols, here `x`, `y`, `μ` and `α`; the elements
of a symbol of several elements, such as `MX.sym("u", 2)`, are `u_0` and `u_1`. Functions
of SX and of MX symbols both serve:

```@example bifurcation
x, y, μ, α = SX.sym("x"), SX.sym("y"), SX.sym("μ"), SX.sym("α")
pitchfork = CasadiFunction("pitchfork", [vertcat(x, y), vertcat(μ, α)], [vertcat(μ * x - x^3 + α * y, -y)])
nothing # hide
```

`BifurcationProblem` takes the function, a guess of a steady state, the values of the
parameters and the name of the parameter that the continuation varies, here ``\mu``, while
``\alpha`` keeps its value; `:α`, or its place `2`, would vary ``\alpha`` instead. The guess
and the values are vectors in the order of the symbols, or, as here, pairs of a name and a
value, one for each element. The guess need not be a steady state, since the continuation
first corrects it by Newton's method. The keyword `record_from_solution` names what each
point of a branch records, here the state ``x``:

```@example bifurcation
problem = BifurcationProblem(pitchfork, [:x => 1.0, :y => 1.0], [:μ => -1.0, :α => 1.0], :μ;
                             record_from_solution = :x)
nothing # hide
```

The problem holds generated code for the residual and its Jacobian, and for the
derivatives that BifurcationKit.jl uses at special points: the derivatives of the second
and third order by the state in given directions, and the derivative of the residual and
of its Jacobian by the continued parameter, besides the product of the Jacobian with a
vector. CasADi computes all of them from the graph of the residual. With `jet = false` the
problem holds the residual and the Jacobian only, which takes less code to generate and to
compile, and BifurcationKit.jl differentiates the generated residual itself, by automatic
differentiation and, for the derivative of the Jacobian by the parameter, by finite
differences.

## The diagram

The settings of the continuation are a `ContinuationPar`, which holds the interval of the
parameter and the number of eigenvalues that each step computes, from which the
continuation detects a change of stability. `bifurcationdiagram` follows the branch from
the guess, and from each branch point it starts the new branches, here to a depth of two
levels and on both sides of each point. Its fourth argument is a function that returns the
settings of each level, here the same ones. The result is a tree: `diagram.γ` is the first
branch, and `diagram.child` holds the diagrams of the branches that start on it:

```@example bifurcation
options = ContinuationPar(nev = 2, p_min = -4.0, p_max = 6.0)
diagram = bifurcationdiagram(problem, PALC(), 2, (args...) -> options; bothside = true)
[(point.type, point.param) for point in diagram.γ.specialpoint]
```

The first branch, ``x = 0``, has a branch point (`bp`) near ``\mu = 0`` between its two
end points. The continuation locates the point by bisection to the precision of its
settings, a few times ``10^{-4}`` here, which is why it is not exactly at zero. The branch
that starts at the point is ``x = \pm\sqrt{\mu}``, which the diagram finds twice, once
in each direction from the point. Since each side of it continues to the end of the
interval, ``\mu = 6``, and the branch holds its two sides one after the other, it runs
from one end, ``x = -\sqrt{6}`` or ``x = \sqrt{6}``, through the branch point to the other.
Each point records ``x`` as its named field:

```@example bifurcation
[(branch.γ.x[1], branch.γ.x[end]) for branch in diagram.child], sqrt(6)
```

BifurcationKit.jl draws a diagram with Plots.jl or Makie.jl, `plot(diagram)` once one of
them is loaded.

## A Hopf point and periodic orbits

A steady state of a system of differential equations can also lose its stability to a
periodic orbit. The system

```math
\begin{aligned}
x' &= \mu x - y - x (x^2 + y^2), \\
y' &= x + \mu y - y (x^2 + y^2)
\end{aligned}
```

has the steady state ``x = y = 0`` for every ``\mu``, which is stable for ``\mu < 0`` and
unstable for ``\mu > 0``; for ``\mu > 0`` the circle of radius ``\sqrt{\mu}`` is a periodic
orbit. We write its right side as a CasADi function `f(u, p, t)`, the form that an
`ODEFunction` takes, and give it to `BifurcationProblem`, which accepts the time as a third
argument when the residual does not depend on it; a residual that depends on the time has
no steady states, and `BifurcationProblem` rejects it with an `ArgumentError`. The second
result, the radius ``\sqrt{x^2 + y^2}``, is an observed variable, which the problem knows
by the name of the result, `radius`, while the states keep the names of their symbols,
`u` and `v`:

```@example bifurcation
u, v, ν, t = SX.sym("u"), SX.sym("v"), SX.sym("ν"), SX.sym("t")
r2 = u^2 + v^2
oscillator = CasadiFunction("oscillator", [vertcat(u, v), ν, t], [vertcat(ν * u - v - u * r2, u + ν * v - v * r2), sqrt(r2)],
                            ["state", "ν", "t"], ["rate", "radius"])
problem = BifurcationProblem(oscillator, [0.0, 0.0], [1.0], :ν; record_from_solution = (:u, :radius))
options = ContinuationPar(nev = 2, p_min = -3.0, p_max = 3.0)
branch = continuation(problem, PALC(), options; bothside = true)
[(point.type, point.param) for point in branch.specialpoint]
```

The record holds both names, `branch.u` and `branch.radius`, which are zero along the
whole branch, since the steady state stays at the origin. The continuation also counts the
unstable eigenvalues of the Jacobian at each point; the branch runs from ``\nu = -3`` to
``\nu = 3``, and it has two of them on the side of ``\nu > 0``:

```@example bifurcation
extrema(branch.radius), branch.n_unstable[1], branch.n_unstable[end]
```

From the Hopf point, the special point `2` in the list above, `continuation` starts the
branch of periodic orbits, which it computes by collocation (`Collocation` of
BifurcationKit.jl) on 20 intervals with polynomials of degree 5. `get_periodic_orbit`
returns the orbit at point 10 of the branch, whose row `i` holds state `i` at the times
`orbit.t`; the largest value of ``u`` is the radius of the circle:

```@example bifurcation
orbits = continuation(branch, 2, options, Collocation(20, 5))
orbit = get_periodic_orbit(orbits, 10)
maximum(orbit[1, :]), sqrt(orbits.param[10])
```

The two agree to about ``10^{-13}``.
