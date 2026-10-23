# Attractors and their basins

A dissipative system can settle on more than one attractor, a steady state, a periodic
orbit or a chaotic set, and which one it reaches depends on where it starts. The set of
initial states that end on an attractor is its basin of attraction, and the fraction of a
region of the state space that each basin takes is a measure of how stable each attractor
is against a perturbation of finite size. The local analysis of [Bifurcation
diagrams](@ref) follows one steady state or periodic orbit at a time from a guess, and it
cannot find a chaotic attractor or the size of a basin. Attractors.jl integrates many
trajectories from a region of the state space, finds the attractors that they reach, maps
initial states to these attractors, and follows the attractors and their basins together
as the parameters change, which it calls a global continuation. Each of these steps
integrates thousands of trajectories, and each trajectory calls the right side at every
stage of every step, so that the cost of one call sets the run time. In this tutorial, we
give Attractors.jl a problem whose right side and Jacobian are generated code of a CasADi
function, address its states and parameters by name, find the two attractors
of a chaotic system and their basins, and continue them along a closed curve of the
parameters.

The examples need CasADiCodegen.jl, Attractors.jl, which brings DynamicalSystemsBase.jl
with it, StaticArrays.jl and SciMLBase.jl.

```@example attractors
using CasADiCodegen
using Attractors
using StaticArrays: SVector
import SciMLBase
nothing # hide
```

## The dynamical system of a CasADi function

The system is a modification of the Lorenz equations with two coexisting chaotic
attractors,

```math
\begin{aligned}
x' &= y - x, \\
y' &= -x z + b |z|, \\
z' &= x y - a,
\end{aligned}
```

with the parameters ``a = 5`` and ``b = 0.1``. As in [Give a CasADi function to a SciML
problem](@ref), the CasADi function takes the state, the parameters and the time, and the
states and the parameters take the names of their scalar symbols. DynamicalSystemsBase.jl
works best with an out-of-place right side of a static vector when the system is small, so
we make the `ODEFunction` with `inplace = false`, which returns an `SVector` for the right
side and an `SMatrix` for the Jacobian. This form holds at most 32 states with the
Jacobian (see [Static arrays](@ref)); for a larger model, leave out `inplace = false` and
give the initial state as a `Vector`, and the rest of this tutorial stays the same.
`CoupledODEs` turns the problem into the dynamical system that Attractors.jl takes, with
the solver `Tsit5` of OrdinaryDiffEq.jl by default; it keeps the start of the time span as
the initial time and ignores its end:

```@example attractors
x, y, z, a, b, t = SX.sym("x"), SX.sym("y"), SX.sym("z"), SX.sym("a"), SX.sym("b"), SX.sym("t")
rhs = CasadiFunction("modlorenz", [vertcat(x, y, z), vertcat(a, b), t],
                     [vertcat(y - x, -x * z + b * abs(z), x * y - a)])
f = SciMLBase.ODEFunction(rhs; inplace = false)
problem = SciMLBase.ODEProblem(f, SVector(-4.0, 5.0, 0.0), (0.0, 1.0), [5.0, 0.1])
ds = CoupledODEs(problem)
nothing # hide
```

The states of the problem are in the order of the symbols, here ``(x, y, z)``.
DynamicalSystemsBase.jl reads the names from the problem, so that a state or a parameter
can be read, and a parameter set, by its name; for example, `set_parameter!(ds, :b, 0.2)`
would change ``b`` before the attractors are searched:

```@example attractors
observe_state(ds, :x), current_parameter(ds, :a)
```

The explicit solver `Tsit5` does not use the Jacobian, but a `TangentDynamicalSystem`
does: it integrates the system together with deviation vectors, small perturbations of the
state that evolve by the Jacobian, from which the Lyapunov exponents follow (for example
with ChaosTools.jl). It finds the generated Jacobian in `problem.f.jac` and uses it instead
of differentiating the right side by ForwardDiff.jl. Starting from the identity matrix, the
deviation vectors after one unit of time are:

```@example attractors
tangent = TangentDynamicalSystem(ds)
step!(tangent, 1.0)
current_deviations(tangent)
```

## Finding the attractors

A mapper of Attractors.jl maps an initial state to the label of the attractor that its
trajectory reaches. The mapper `AttractorsViaRecurrences` covers a box of the state space
with a grid of cells, one range for each state in the order of the states (the grid takes
no names), and follows each trajectory from cell to cell. A trajectory that returns to
cells it has already visited `consecutive_recurrences` times in a row has found a new
attractor; the default, 100, is too few for a chaotic attractor, whose trajectory returns
to its cells irregularly, so we raise it to 1000. A trajectory that reaches the cells of a
known attractor gets its label, while one that stays outside the grid for
`consecutive_lost_steps` steps, 20 by default, gets the label `-1`. The mapper integrates in
steps of length `Δt`, which it otherwise estimates with
`Attractors.automatic_Δt_recurrences(ds, grid)` from the speed of the system at 5000
states drawn from the global random number generator; since these states differ from run
to run, two runs can find slightly different attractors, so we fix the step at 0.037, the
value of that estimate for this grid:

```@example attractors
grid = (range(-15.0, 15.0; length = 150),     # x
        range(-20.0, 20.0; length = 150),     # y
        range(-20.0, 20.0; length = 150))     # z
mapper = AttractorsViaRecurrences(ds, grid; consecutive_recurrences = 1000,
                                  consecutive_lost_steps = 100, Δt = 0.037)
nothing # hide
```

Called on an initial state, the mapper returns the label of the attractor that the
trajectory reaches; states that reach the same attractor get the same label, and a new
attractor gets the next one:

```@example attractors
mapper([-4.0, 5.0, 0.0]), mapper([4.0, 2.0, 0.0]), mapper([1.0, 3.0, 2.0])
```

`extract_attractors` returns the attractors found so far, a dictionary from the label to a
`StateSpaceSet`, a vector of states with one row per state, which holds one point of the
trajectory in each cell that it visited on the attractor. Without the term ``b |z|`` the
system is unchanged by ``x \to -x``, ``y \to -y``, and the ranges of ``x`` of the two
attractors are close to mirror images:

```@example attractors
attractors = extract_attractors(mapper)
[(label, length(A), extrema(A[:, 1])) for (label, A) in attractors]
```

## Basins of attraction on a slice

The basins of the whole grid take ``150^3`` trajectories, so we compute them on the slice
``y = 0`` instead, on a coarser grid of 50 by 50 states in ``x`` and ``z``. The initial
states can be given by name, as dictionaries, which DynamicalSystemsBase.jl puts in the
order of the problem. `PrescribedICs` gives the mapper these states, and
`basins_fractions_labels` returns the fraction of the states that reach each attractor,
together with the label of each state:

```@example attractors
xs, zs = range(-15.0, 15.0; length = 50), range(-20.0, 20.0; length = 50)
states = [Dict(:x => x, :y => 0.0, :z => z) for x in xs for z in zs]
fractions, labels = basins_fractions_labels(mapper, PrescribedICs(states); show_progress = false)
fractions
```

The same mapper keeps its attractors, so the labels are those of the previous section.
The map below shows every second state of the slice, with ``z`` upward and ``x`` to the
right, a dot for attractor 1, a hash for attractor 2 and a question mark for any other
label. The two basins are interwoven down to the size of the grid, so that a small change
of the initial state can change the attractor:

```@example attractors
label_map = reshape(labels, length(zs), length(xs))
println(join([join(l == 1 ? '.' : l == 2 ? '#' : '?' for l in row) for row in eachrow(label_map[end:-2:1, 1:2:end])], "\n"))
```

## Global continuation

The global continuation follows all the attractors along a curve of the parameters. At each
point of the curve, `AttractorSeedContinueMatch` starts trajectories from the attractors of
the previous point and from new random initial states, finds the attractors and the
fractions of their basins with the mapper, and matches the new attractors with the previous
ones by the distance of their centroids (the default matcher `MatchBySSSetDistance`), so
that an attractor keeps its label along the curve. The random initial states come from a
`RandomICsSampler` of 100 states per point, uniform in the box of the grid, with the seed
1234 of its random number generator. The curve is an ellipse in the plane of ``a`` and ``b``, given as pairs of
a parameter name and a value at each point:

```@example attractors
continuation = AttractorSeedContinueMatch(mapper)
sampler = RandomICsSampler(100, grid, 1234)
θs = range(0, 2π; length = 41)
pcurve = [[:a => 5 + 0.5cos(θ), :b => 0.1 + 0.01sin(θ)] for θ in θs]
fractions_cont, attractors_cont = global_continuation(continuation, pcurve, sampler; show_progress = false)
nothing # hide
```

`fractions_cont[i]` holds the fractions of the basins at the point `i` of the curve, and
`attractors_cont[i]` the attractors. At every fifth point, the parameter ``a``, the fractions
and the smallest value of ``x`` on each attractor are

```@example attractors
[(round(5 + 0.5cos(θs[i]); digits = 3), fractions_cont[i], Dict(label => minimum(A[:, 1]) for (label, A) in attractors_cont[i]))
 for i in 1:5:length(θs)]
```

The basin of attractor 1 holds about 70 percent of the sampled initial states while ``a`` decreases from 5.5 to 5, and
vanishes between ``a = 5`` and ``a = 4.65``, so that all initial states reach attractor 2
until the curve returns to ``a = 5``. There an attractor appears again where attractor 1 was, with a
similar extent in ``x`` and fraction; it gets the new label 3, since by default the
matching compares the attractors of a point with those of the previous point only, where
attractor 1 no longer existed. A matcher with `use_vanished = true` also compares them
with the attractors that have vanished before, and the attractor that comes back keeps its
label:

```@example attractors
matcher = MatchBySSSetDistance(use_vanished = true)
fractions_cont, _ = global_continuation(AttractorSeedContinueMatch(mapper, matcher), pcurve,
                                        RandomICsSampler(100, grid, 1234); show_progress = false)
sort(unique(label for fs in fractions_cont for label in keys(fs)))
```

The continuation leaves the dynamical system at the parameters of the last point of the
curve, here ``a = 5.5``; `set_parameters!` restores the parameters of the problem,
``a = 5`` and ``b = 0.1``:

```@example attractors
set_parameters!(ds, [:a => 5.0, :b => 0.1])
current_parameters(ds)
```
