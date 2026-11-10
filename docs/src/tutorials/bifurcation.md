# Bifurcation diagrams

BifurcationKit.jl computes bifurcation diagrams by continuation, which needs the
Jacobian of the residual at each step and higher derivatives at the special points. In
this tutorial, we give it a CasADi function, so that all these derivatives are generated
code, and select parameters and recorded quantities by name. We compute the diagram of
a pitchfork, then a Hopf point and the periodic orbits that start there.

```@example bifurcation
using CasADiCodegen
using BifurcationKit
nothing # hide
```

## The bifurcation problem of a CasADi function

We compute the steady states of the system

```math
\begin{aligned}
0 &= \mu x - x^3 + \alpha y, \\
0 &= -y.
\end{aligned}
```

It has the steady state ``x = y = 0`` for every ``\mu`` and, for ``\mu > 0``, two more
at ``x = \pm\sqrt{\mu}``. The CasADi function takes the state and the parameters and
returns the residual first, as for a `NonlinearFunction`. States and parameters take
the names of their scalar symbols; the elements of `MX.sym("u", 2)` are `u_0` and `u_1`:

```@example bifurcation
x, y, μ, α = SX.sym("x"), SX.sym("y"), SX.sym("μ"), SX.sym("α")
pitchfork = CasadiFunction("pitchfork", [vcat(x, y), vcat(μ, α)], [vcat(μ * x - x^3 + α * y, -y)])
nothing # hide
```

`BifurcationProblem` takes the function, a guess of a steady state, the parameter
values and the continued parameter, by name (`:μ`) or by place (`1`). The guess and the
values are vectors in symbol order or, as here, pairs of name and value;
`record_from_solution` names what each point of a branch records:

```@example bifurcation
problem = BifurcationProblem(pitchfork, [:x => 1.0, :y => 1.0], [:μ => -1.0, :α => 1.0], :μ;
                             record_from_solution = :x)
nothing # hide
```

The problem holds generated code for the residual, its Jacobian, and the second and
third directional derivatives and parameter derivatives used at special points. With
`jet = false`, it holds only the residual and the Jacobian, which compiles faster, and
BifurcationKit.jl computes the rest by automatic differentiation and finite differences.

## The diagram

`bifurcationdiagram` follows the branch from the guess and starts new branches at each
branch point, here to a depth of two levels. The result is a tree: `diagram.γ` is the
first branch, and `diagram.child` holds the branches that start on it:

```@example bifurcation
options = ContinuationPar(nev = 2, p_min = -4.0, p_max = 6.0)
diagram = bifurcationdiagram(problem, PALC(), 2, (args...) -> options; bothside = Val(true))
[(point.type, point.param) for point in diagram.γ.specialpoint]
```

The branch ``x = 0`` has a branch point (`bp`) near ``\mu = 0``, located by bisection
to about ``10^{-4}``. The diagram finds the branch ``x = \pm\sqrt{\mu}`` twice, once in
each direction, and each runs from ``\mp\sqrt{6}`` to ``\pm\sqrt{6}``:

```@example bifurcation
[(branch.γ.x[1], branch.γ.x[end]) for branch in diagram.child], sqrt(6)
```

## A Hopf point and periodic orbits

Next, we take the system

```math
\begin{aligned}
x' &= \mu x - y - x (x^2 + y^2), \\
y' &= x + \mu y - y (x^2 + y^2)
\end{aligned}
```

It has a steady state at the origin that loses its stability at ``\mu = 0`` to a periodic
orbit of radius ``\sqrt{\mu}``. We write the right side as `f(u, p, t)`;
`BifurcationProblem` accepts the time argument when the residual does not depend on it,
and throws an `ArgumentError` when it does. The second result is an observed variable,
known by its name `radius`:

```@example bifurcation
u, v, ν, t = SX.sym("u"), SX.sym("v"), SX.sym("ν"), SX.sym("t")
r2 = u^2 + v^2
oscillator = CasadiFunction("oscillator", [vcat(u, v), ν, t], [vcat(ν * u - v - u * r2, u + ν * v - v * r2), sqrt(r2)],
                            ["state", "ν", "t"], ["rate", "radius"])
problem = BifurcationProblem(oscillator, [0.0, 0.0], [1.0], :ν; record_from_solution = (:u, :radius))
options = ContinuationPar(nev = 2, p_min = -3.0, p_max = 3.0)
branch = continuation(problem, PALC(), options; bothside = Val(true))
[(point.type, point.param) for point in branch.specialpoint]
```

The radius is zero along the branch, and two eigenvalues are unstable for ``\nu > 0``:

```@example bifurcation
extrema(branch.radius), branch.n_unstable[1], branch.n_unstable[end]
```

From the Hopf point, special point `2`, `continuation` computes the periodic orbits by
collocation; the largest value of ``u`` on an orbit matches the radius ``\sqrt{\nu}``:

```@example bifurcation
orbits = continuation(branch, 2, options, Collocation(20, 5))
orbit = get_periodic_orbit(orbits, 10)
maximum(orbit[1, :]), sqrt(orbits.param[10])
```
