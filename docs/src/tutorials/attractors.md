# Attractors and their basins

Attractors.jl finds the attractors of a dynamical system, their basins of attraction and
how both change with the parameters, which the local analysis of
[Bifurcation diagrams](@ref) cannot do. Each step integrates thousands of trajectories, so
the cost of one call of the right side sets the run time. In this tutorial, we give
Attractors.jl a problem of generated code, find the two chaotic attractors of a modified
Lorenz system and their basins, and continue them along a closed curve of the parameters.

```@example attractors
using CasADiCodegen
using Attractors
using StaticArrays: SVector
import SciMLBase
nothing # hide
```

## The dynamical system of a CasADi function

The system has two coexisting chaotic attractors:

```math
\begin{aligned}
x' &= y - x, \\
y' &= -x z + b |z|, \\
z' &= x y - a,
\end{aligned}
```

with ``a = 5`` and ``b = 0.1``. As in [Give a CasADi function to a SciML problem](@ref),
the states and the parameters take the names of their scalar symbols. With
`inplace = false`, the right side returns an `SVector` and the Jacobian an `SMatrix`, which
suits DynamicalSystemsBase.jl for a small system; this form holds at most 32 states with
the Jacobian (see [Static arrays](@ref)), and a larger model leaves out `inplace = false`
and takes a `Vector` state. `CoupledODEs` uses `Tsit5` by default and ignores the end of
the time span:

```@example attractors
x, y, z, a, b, t = SX.sym("x"), SX.sym("y"), SX.sym("z"), SX.sym("a"), SX.sym("b"), SX.sym("t")
rhs = CasadiFunction("modlorenz", [vcat(x, y, z), vcat(a, b), t],
                     [vcat(y - x, -x * z + b * abs(z), x * y - a)])
f = SciMLBase.ODEFunction(rhs; inplace = false)
problem = SciMLBase.ODEProblem(f, SVector(-4.0, 5.0, 0.0), (0.0, 1.0), [5.0, 0.1])
ds = CoupledODEs(problem)
nothing # hide
```

DynamicalSystemsBase.jl reads the names from the problem, so states and parameters can be
read or set by name, as in `set_parameter!(ds, :b, 0.2)`:

```@example attractors
observe_state(ds, :x), current_parameter(ds, :a)
```

A `TangentDynamicalSystem`, which integrates deviation vectors for Lyapunov exponents,
uses the generated Jacobian in `problem.f.jac` instead of ForwardDiff.jl:

```@example attractors
tangent = TangentDynamicalSystem(ds)
step!(tangent, 1.0)
current_deviations(tangent)
```

## Finding the attractors

The mapper `AttractorsViaRecurrences` maps an initial state to the label of the attractor
it reaches by following the trajectory over a grid of cells, one range per state in the
order of the states, since the grid takes no names. The default `consecutive_recurrences`
of 100 is too few for a chaotic attractor, so we raise it to 1000. We also fix the step
`Δt`, which the mapper otherwise estimates from random states, so that every run finds the
same attractors:

```@example attractors
grid = (range(-15.0, 15.0; length = 150),     # x
        range(-20.0, 20.0; length = 150),     # y
        range(-20.0, 20.0; length = 150))     # z
mapper = AttractorsViaRecurrences(ds, grid; consecutive_recurrences = 1000,
                                  consecutive_lost_steps = 100, Δt = 0.037)
nothing # hide
```

States that reach the same attractor get the same label:

```@example attractors
mapper([-4.0, 5.0, 0.0]), mapper([4.0, 2.0, 0.0]), mapper([1.0, 3.0, 2.0])
```

`extract_attractors` returns each attractor as a `StateSpaceSet`. Without the term
``b |z|`` the system is symmetric under ``x \to -x``, ``y \to -y``, and the two attractors
are close to mirror images:

```@example attractors
attractors = extract_attractors(mapper)
[(label, length(A), extrema(A[:, 1])) for (label, A) in attractors]
```

## Basins of attraction on a slice

We compute the basins on the slice ``y = 0`` with a 50 by 50 grid, giving the initial
states by name; `basins_fractions_labels` returns the fraction of states that reach each
attractor and the label of each state:

```@example attractors
xs, zs = range(-15.0, 15.0; length = 50), range(-20.0, 20.0; length = 50)
states = [Dict(:x => x, :y => 0.0, :z => z) for x in xs for z in zs]
fractions, labels = basins_fractions_labels(mapper, PrescribedICs(states); show_progress = false)
fractions
```

The map shows every second state, with ``z`` upward, `.` for attractor 1, `#` for
attractor 2 and `?` for any other label. The basins are interwoven down to the size of the grid:

```@example attractors
label_map = reshape(labels, length(zs), length(xs))
println(join([join(l == 1 ? '.' : l == 2 ? '#' : '?' for l in row) for row in eachrow(label_map[end:-2:1, 1:2:end])], "\n"))
```

## Global continuation

`AttractorSeedContinueMatch` follows all attractors along a curve of the parameters: at
each point it seeds trajectories from the previous attractors and from 100 random states,
and matches the new attractors with the previous ones by distance so that each keeps its
label. The curve is an ellipse in the plane of ``a`` and ``b``:

```@example attractors
continuation = AttractorSeedContinueMatch(mapper)
sampler = RandomICsSampler(100, grid, 1234)
θs = range(0, 2π; length = 41)
pcurve = [[:a => 5 + 0.5cos(θ), :b => 0.1 + 0.01sin(θ)] for θ in θs]
fractions_cont, attractors_cont = global_continuation(continuation, pcurve, sampler; show_progress = false)
nothing # hide
```

At every fifth point, we print ``a``, the basin fractions and the smallest ``x`` on each
attractor:

```@example attractors
[(round(5 + 0.5cos(θs[i]); digits = 3), fractions_cont[i], Dict(label => minimum(A[:, 1]) for (label, A) in attractors_cont[i]))
 for i in 1:5:length(θs)]
```

Attractor 1 vanishes between ``a = 5`` and ``a = 4.65`` and comes back near ``a = 5`` with
the label 3, because the default matcher compares only with the previous point. With
`use_vanished = true`, the returning attractor keeps its label:

```@example attractors
matcher = MatchBySSSetDistance(use_vanished = true)
fractions_cont, _ = global_continuation(AttractorSeedContinueMatch(mapper, matcher), pcurve,
                                        RandomICsSampler(100, grid, 1234); show_progress = false)
sort(unique(label for fs in fractions_cont for label in keys(fs)))
```

The continuation leaves the system at the parameters of the last point; `set_parameters!`
restores them:

```@example attractors
set_parameters!(ds, [:a => 5.0, :b => 0.1])
current_parameters(ds)
```
