# Model physical systems with differential-algebraic equations

In this tutorial, we write three physical models from their parts: an electric circuit, a
mass on a spring, and a pendulum held by a rod. A model written from parts has many
equations that only relate the variables of one part to those of another, so it is a
differential-algebraic equation (DAE) with far more unknowns than states, and the
pendulum, written as its physics states it, is a DAE on which the standard solvers
fail. A modeling language removes these difficulties for its user: it joins the
parts through connectors, eliminates the equations of the form `a = b`, sorts the
algebraic equations and substitutes their solutions into the differential equations
(tearing), and differentiates constraints until the DAE can be integrated (index
reduction). CasADiCodegen.jl sorts the equations and reduces the index of a DAE, but
provides no connectors and no tearing, so we write these steps in a few lines of Julia
and let CasADi compute the derivatives that they need.

The examples need CasADiCodegen.jl, SciMLBase.jl, the standard library LinearAlgebra and
the solver `Rodas5P` of OrdinaryDiffEqRosenbrock.jl, which integrates a DAE in mass-matrix
form; `BrownFullBasicInit` of OrdinaryDiffEqNonlinearSolve.jl computes consistent initial
values of its algebraic unknowns.

```@example dae
using CasADiCodegen, LinearAlgebra
import SciMLBase
C = CasADiCodegen.C    # the module of the CasADi front end
using OrdinaryDiffEqRosenbrock: Rodas5P
using OrdinaryDiffEqNonlinearSolve: BrownFullBasicInit
```

## Write the parts of a circuit

The circuit has a voltage source of 1 V, a resistor of 2 Ω and a capacitor of 1 F in a
loop, with the negative side of the source grounded. Each part touches the others through
pins, and a pin carries a voltage `v` and a current `i` that flows into the part. We make
each unknown a named scalar symbol and collect all of them in a list, so that the names
reach the solution later:

```@example dae
unknowns = SX[]
function unknown(name)
    s = SX.sym(name)
    push!(unknowns, s)
    s
end
pin(name) = (v = unknown("$(name)_v"), i = unknown("$(name)_i"))
nothing # hide
```

A part with two pins (a one-port) has the voltage across it and the current through it,
and three equations tie them to its pins. We write an equation as a residual, an
expression that is zero, and each part returns its residuals with its variables:

```@example dae
function oneport(name)
    p, n = pin("$(name)_p"), pin("$(name)_n")
    v, i = unknown("$(name)_v"), unknown("$(name)_i")
    (; p, n, v, i, equations = [v - (p.v - n.v), p.i + n.i, i - p.i])
end
nothing # hide
```

A connection of pins is what a modeling language makes from a connector: the pins share
one voltage and their currents sum to zero, by Kirchhoff's laws. Without connectors in
the package, it is one more Julia function:

```@example dae
connect(pins...) = [[pins[1].v - q.v for q in pins[2:end]]; sum(q.i for q in pins)]
nothing # hide
```

The resistance, the capacitance and the source voltage are parameters. The CasADi
function below takes the unknowns, the parameters and the time `t`, which the circuit
does not use. The resistor, the source and the ground
add one algebraic equation each, and the capacitor adds the only differential equation,
`C v' = i`:

```@example dae
R, Cap, V, t = SX.sym("R"), SX.sym("C"), SX.sym("V"), SX.sym("t")
source, resistor, capacitor = oneport("source"), oneport("resistor"), oneport("capacitor")
ground = pin("ground")

algebraic = [source.equations; resistor.equations; capacitor.equations
             resistor.v - R * resistor.i
             source.v - V
             ground.v
             connect(source.p, resistor.p)
             connect(resistor.n, capacitor.p)
             connect(capacitor.n, source.n, ground)]
differential = capacitor.i / Cap
(length(unknowns), length(algebraic))
```

The circuit has 20 unknowns and 19 algebraic equations, so with the differential
equation it is square.

## Solve the circuit as a DAE

A mass-matrix DAE `M u' = f(u, p, t)` with `M` diagonal puts the differential unknowns
first, with ones on the diagonal, and the algebraic unknowns after them, with zeros. The
capacitor voltage is the only differential unknown, so we move it to the front and build
the right side as a CasADi function:

```@example dae
others = filter(s -> C.name(s) != "capacitor_v", unknowns)
u = vertcat(capacitor.v, others...)
p = vertcat(R, Cap, V)
rc = CasadiFunction("rc", [u, p, t], [vertcat(differential, algebraic...)])
nothing # hide
```

[`ODEFunction`](@ref SciMLBase.ODEFunction(::CasADiCodegen.Function)) generates the code
of the right side and of its Jacobian, and passes the mass matrix to SciMLBase. It also
gives the problem the names of the scalar symbols of `u` and `p`, so that the solution
answers to them. We know only the initial capacitor voltage, which is zero, and start
every algebraic unknown from zero; `BrownFullBasicInit` then solves the algebraic
equations for consistent values before the first step:

```@example dae
mass = Diagonal([1.0; zeros(length(others))])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(rc; mass_matrix = mass), zeros(length(unknowns)), (0.0, 10.0), [2.0, 1.0, 1.0])
solution = SciMLBase.solve(problem, Rodas5P(); initializealg = BrownFullBasicInit())
solution[:capacitor_v][end], solution[:resistor_i][end]
```

The capacitor charges as `1 - exp(-t / 2)` and the current decays as `exp(-t / 2) / 2`,
which the solution follows at `t = 10`:

```@example dae
(1 - exp(-5), exp(-5) / 2)
```

## Reduce the circuit by hand

The solver above carries 19 algebraic unknowns through every step, although most of
them equal another unknown or its negative. The structure shows how to remove them: the
block-triangular form of the Jacobian of the algebraic equations by the algebraic
unknowns orders the equations so that each block can be solved once the earlier blocks
are known, and [`block_triangular_form`](@ref CasADiCodegen.block_triangular_form)
computes it from the sparsity pattern; its field `blocks` is the number of blocks:

```@example dae
z = vertcat(others...)
pattern = C.sparsity(jacobian(vertcat(algebraic...), z))
CasADiCodegen.block_triangular_form(pattern).blocks
```

There are 19 blocks for 19 unknowns, so each algebraic unknown follows from one equation
once the capacitor voltage is known, and no algebraic loop remains. We substitute these
solutions into the differential equation by hand: the resistor carries the source
voltage minus the capacitor voltage, so its current is `(V - v) / R`, and that current
charges the capacitor. The lists after the results name the arguments and the results;
the state and the parameters keep the names of their scalar symbols, and each result
after the first gives its name to an observed variable:

```@example dae
v = SX.sym("capacitor_v")
current = (V - v) / R
rc_reduced = CasadiFunction("rc_reduced", [v, p, t], [current / Cap, current], ["u", "p", "t"], ["du", "resistor_i"])
nothing # hide
```

The resistor current is no longer an unknown, but the solution computes it from the
state as the observed variable `resistor_i`:

```@example dae
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(rc_reduced), [0.0], (0.0, 10.0), [2.0, 1.0, 1.0])
solution = SciMLBase.solve(problem, Rodas5P(); abstol = 1e-8, reltol = 1e-8)
solution(5.0; idxs = :resistor_i)
```

The reduced problem is an ordinary differential equation of one state, and its result
agrees with the current
`exp(-5 / 2) / 2 ≈ 0.0410` at `t = 5`.

## Model a mass on a spring

A mass of 1 kg hangs from a spring of stiffness `10^4` N/m and rest length 1 m, whose
other end is fixed at the origin, and gravity pulls the mass down. The mass contributes
its position and its velocity as states, and the spring computes its force from the two
points that it joins. The spring has variables of its own, its length `x` and its
direction `dir`, but they are expressions of the position of the mass, so we write them
as expressions instead of unknowns, which is what the elimination of the equations
`dir = pos - anchor` and `x = norm(dir)` would give, with the anchor at the origin:

```@example dae
point_mass(name) = (pos = SX.sym("$(name)_pos", 2), v = SX.sym("$(name)_v", 2))

function spring(a, b, k, l)
    dir = a - b
    x = norm(dir)
    (; x, dir, force = -k * dir * (x - l) / x)
end

m, k, l = SX.sym("m"), SX.sym("k"), SX.sym("l")
body = point_mass("mass")
link = spring(body.pos, SX.zeros(2), k, l)
gravity = vertcat(0, -9.81)
nothing # hide
```

The derivative of the position is the velocity, and that of the velocity is the force
over the mass plus gravity. The function takes the time symbol `t` of the circuit, and the
length of the spring becomes a second result, so that it is an observed variable of the
solution:

```@example dae
rhs = vertcat(body.v, link.force / m + gravity)
spring_mass = CasadiFunction("spring_mass", [vertcat(body.pos, body.v), vertcat(m, k, l), t], [rhs, link.x],
                             ["u", "p", "t"], ["du", "spring_x"])
nothing # hide
```

The state has the names of the elements of `SX.sym("mass_pos", 2)` and
`SX.sym("mass_v", 2)`, which CasADi numbers from 0: `mass_pos_0`, `mass_pos_1`,
`mass_v_0` and `mass_v_1`. The mass starts at rest at `(1, -1)`, where the spring is
stretched by about 0.41 m, so it oscillates fast along the spring while it swings, and
the length of the spring stays between about 0.59 m and 1.41 m. The stiff spring makes
the problem stiff, so we solve it with `Rodas5P`:

```@example dae
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(spring_mass), [1.0, -1.0, 0.0, 0.0], (0.0, 3.0), [1.0, 1e4, 1.0])
solution = SciMLBase.solve(problem, Rodas5P())
extrema(solution[:spring_x])
```

## Reduce the index of a pendulum

A pendulum of unit mass on a rod of length `L` is described by its position `(x, y)`,
its velocity `(v_x, v_y)` and a multiplier `T` proportional to the tension of the rod,
which keeps the mass on the circle:

```math
x' = v_x, \quad v_x' = T x, \quad y' = v_y, \quad v_y' = T y - g, \quad 0 = x^2 + y^2 - L^2.
```

As a mass-matrix DAE these equations fail at the first step, since the constraint does
not contain `T`, so the Jacobian of the algebraic equation by the algebraic unknown is
zero and the Newton iterations of the solver cannot determine `T`. The DAE has index 3:
the constraint must be differentiated twice in time before it contains `T`, which gives
an equation of index 1 for `T`, and once more for an ordinary differential equation.
[`reduce_index`](@ref CasADiCodegen.reduce_index) finds the differentiations to index 1
from the sparsity pattern of the equations, by the algorithm of Pantelides, and lets
CasADi compute the derivatives. It takes the DAE as a dictionary in the form of the
CasADi integrators:

```@example dae
x, vx, y, vy, T = (SX.sym(name) for name in ("x", "vx", "y", "vy", "T"))
g, L = SX.sym("g"), SX.sym("L")
pendulum = Dict("x" => vertcat(x, vx, y, vy), "z" => T, "p" => vertcat(g, L),
                "ode" => vertcat(vx, T * x, vy, T * y - g), "alg" => x^2 + y^2 - L^2)
reduced = CasADiCodegen.reduce_index(pendulum)
reduced.index, reduced.differentiations
```

The algebraic equation of the reduced DAE is the second derivative of the constraint, in
which the derivatives of the states are replaced by the right side `"ode"`, so that it
contains `T` with the factor `2(x^2 + y^2)`, nonzero on the circle:

```@example dae
reduced.dae["alg"]
```

The constraint and its first derivative, `x v_x + y v_y = 0` up to a factor of 2, are
no longer equations of the model but its invariants: they stay zero along an exact
solution that starts where they are zero, and the initial values must satisfy them.
`initializealg = BrownFullBasicInit()` corrects a guess of the algebraic state `T`, but
it leaves the differential states as given, so these must satisfy the invariants; for a
model where that point is not obvious, a CasADi `rootfinder` or `nlpsol` on
`reduced.invariants` over the free states finds one. The pendulum starts at rest at
`(1, 0)`, which satisfies both invariants and where the reduced constraint gives
`T = 0`. Called directly, a CasADi integrator of the reduced dictionary with the plugin
`"idas"` integrates it with IDAS; here the integrator carries the model and the output
times to [`ODEProblem`](@ref SciMLBase.ODEProblem(::CasADiCodegen.Function, ::AbstractVector{<:Real}, ::AbstractVector{<:Real})),
whose state is `[x; z]` with a mass matrix, whose time span runs from 0 to the last
output time, and whose solution holds the states at the output times; `Rodas5P`
integrates it:

```@example dae
times = collect(0.1:0.1:10.0)
model = C.integrator("pendulum", "idas", reduced.dae, 0.0, times)
problem = SciMLBase.ODEProblem(model, [1.0, 0.0, 0.0, 0.0], [9.8, 1.0]; z0 = [0.0])
solution = SciMLBase.solve(problem, Rodas5P(); abstol = 1e-8, reltol = 1e-8)
invariants = CasadiFunction("invariants", [reduced.dae["x"], reduced.dae["z"], reduced.dae["p"]], [reduced.invariants])
nx = size(reduced.dae["x"], 1)
drift(solution) = maximum(u -> maximum(abs, Matrix(invariants(u[1:nx], u[nx+1:end], solution.prob.p))), solution.u)
drift(solution)
```

The reduced DAE holds the second derivative of the constraint at zero, not the
constraint itself, so the position leaves the circle slowly as the errors of the
integration accumulate; with the tolerances `1e-8`, the largest residual of the
invariants over the ten seconds is about `2e-8`. The reduced model keeps every state and
enforces only the highest derivative of each constraint, since `reduce_index` does not
select dummy derivatives, which would hold the constraint itself. Baumgarte's
stabilization instead replaces the second derivative by a combination of it
and the lower derivatives, `g'' - 2γ g' + γ² g = 0` for the pole `γ < 0` of the keyword
`baumgarte_pole`, so that a deviation from the circle decays as `exp(γ t)`:

```@example dae
stabilized = CasADiCodegen.reduce_index(pendulum; baumgarte_pole = -5.0)
model = C.integrator("pendulum", "idas", stabilized.dae, 0.0, times)
problem = SciMLBase.ODEProblem(model, [1.0, 0.0, 0.0, 0.0], [9.8, 1.0]; z0 = [0.0])
drift(SciMLBase.solve(problem, Rodas5P(); abstol = 1e-8, reltol = 1e-8))
```

The invariants of `stabilized` are those of `reduced`, so the same function measures
the drift, which the stabilization makes a few times smaller over these ten seconds.
The gain grows with the time: over 100 seconds, the residual reaches about `5e-6`
without the stabilization and stays below `1e-8` with it. A larger
`|γ|` pulls the solution back faster but makes the equations stiffer.
