# Model physical systems with differential-algebraic equations

In this tutorial, we write three physical models from their parts: an electric circuit, a
mass on a spring, and a pendulum held by a rod. A model written from parts has many
equations that only relate the variables of one part to those of another, so it is a
differential-algebraic equation (DAE) with far more variables than states, and the
pendulum, written as its physics states it, is a DAE on which the standard solvers
fail. [`System`](@ref CasADiCodegen.Models.System) of the submodule
`CasADiCodegen.Models` removes these difficulties: it eliminates the equations of the form
`a = b` and the other algebraic equations that it can solve, substitutes their solutions
into the differential equations, and differentiates constraints until the DAE can be
integrated (index reduction), with derivatives from CasADi. The connections of the parts
are a few lines of Julia that we write ourselves; [Write a model as equations](@ref)
describes the elements of the submodule.

The examples need CasADiCodegen.jl, SciMLBase.jl, the standard library LinearAlgebra and
the solver `Rodas5P` of OrdinaryDiffEqRosenbrock.jl, which integrates a DAE in mass-matrix
form.

```@example dae
using CasADiCodegen, CasADiCodegen.Models, LinearAlgebra
import SciMLBase
C = CasADiCodegen.C    # the module of the CasADi front end
using OrdinaryDiffEqRosenbrock: Rodas5P
```

## Write the parts of a circuit

The circuit has a voltage source of 1 V, a resistor of 2 Ω and a capacitor of 1 F in a
loop, with the negative side of the source grounded. Each part touches the others through
pins, and a pin carries a voltage `v` and a current `i` that flows into the part. Each
variable is a named scalar symbol, so that the names reach the solution later:

```@example dae
pin(name) = (v = SX.sym("$(name)_v"), i = SX.sym("$(name)_i"))
nothing # hide
```

A part with two pins (a one-port) has the voltage across it and the current through it,
and three equations tie them to its pins. Each part returns its equations, written with
`~`, together with its variables:

```@example dae
function oneport(name)
    p, n = pin("$(name)_p"), pin("$(name)_n")
    v, i = SX.sym("$(name)_v"), SX.sym("$(name)_i")
    (; p, n, v, i, equations = [v ~ p.v - n.v, 0 ~ p.i + n.i, i ~ p.i])
end
nothing # hide
```

A connection of pins is what a modeling language makes from a connector: the pins share
one voltage and their currents sum to zero, by Kirchhoff's laws. Here it is one more Julia
function:

```@example dae
connect(a, others...) = [[a.v ~ q.v for q in others]; 0 ~ a.i + sum(q.i for q in others)]
nothing # hide
```

The resistance, the capacitance and the source voltage are parameters. The resistor, the
source and the ground add one algebraic equation each, and the capacitor adds the only
differential equation, `Cap v' = i`, in which `D(capacitor.v)` is the derivative of the
capacitor voltage by the time `t`:

```@example dae
R, Cap, V, t = SX.sym("R"), SX.sym("C"), SX.sym("V"), SX.sym("t")
D = Differential(t)
source, resistor, capacitor = oneport("source"), oneport("resistor"), oneport("capacitor")
ground = pin("ground")

circuit = [source.equations; resistor.equations; capacitor.equations
           resistor.v ~ R * resistor.i
           Cap * D(capacitor.v) ~ capacitor.i
           source.v ~ V
           ground.v ~ 0
           connect(source.p, resistor.p)
           connect(resistor.n, capacitor.p)
           connect(capacitor.n, source.n, ground)]
length(circuit)
```

The circuit has 20 variables, each a voltage or a current: two for each of the six pins of
the parts, two for each of the three parts, and two for the ground. It has as many scalar
equations.

## Compile and solve the circuit

`System` takes the equations, the time and the parameters; every other symbol of the
equations is a variable of the model, and [`unknowns`](@ref CasADiCodegen.Models.unknowns)
gives the states that remain after the compilation:

```@example dae
rc = System(circuit, t; parameters = [R, Cap, V], name = "rc")
unknowns(rc)
```

Of the 20 variables, only the capacitor voltage remains. Every algebraic equation of the
circuit contains some variable linearly, with a coefficient that is a number or the
resistance, so `System` solves it for that variable and substitutes the solution into the
other equations, until no algebraic equation remains. It then solves the differential
equation for the derivative `dcapacitor_v`. What is left is an ordinary differential
equation of one state:

```@example dae
equations(rc)
```

The 19 eliminated variables and the derivative are observed variables, which the solution
computes from the state; [`observed`](@ref CasADiCodegen.Models.observed) gives each with
its expression, the resistor current among them:

```@example dae
filter(o -> C.name(first(o)) == "resistor_i", observed(rc))
```

`SciMLBase.ODEProblem` takes the initial values and the parameters as
pairs from a symbol, or its name, to its value:

```@example dae
problem = SciMLBase.ODEProblem(rc, [capacitor.v => 0.0], (0.0, 10.0), [R => 2.0, Cap => 1.0, V => 1.0])
solution = SciMLBase.solve(problem, Rodas5P(); abstol = 1e-8, reltol = 1e-8)
solution[:capacitor_v][end], solution[:resistor_i][end]
```

The capacitor charges as `1 - exp(-t / 2)` and the current decays as `exp(-t / 2) / 2`,
which the solution follows at `t = 10`:

```@example dae
(1 - exp(-5), exp(-5) / 2)
```

## Model a mass on a spring

A mass of 1 kg hangs from a spring of stiffness `10^4` N/m and rest length 1 m, whose
other end is fixed at the origin, and gravity pulls the mass down. The mass contributes
its position, a vector symbol of two elements, and the spring computes its force from the
two points that it joins. The direction of the spring and its length are expressions of
the position, which the function `spring` returns with the force:

```@example dae
function spring(a, b, k, l)
    dir = a - b
    x = norm(dir)
    (; x, dir, force = -k * dir * (x - l) / x)
end

m, k, l = SX.sym("m"), SX.sym("k"), SX.sym("l")
pos = SX.sym("mass_pos", 2)
link = spring(pos, SX.zeros(2), k, l)
gravity = vcat(0, -9.81)
nothing # hide
```

The second derivative of the position is the force over the mass plus gravity, one vector
equation of two elements. To read the length of the spring from the solution, we give it a
symbol `spring_x` and an equation; `System` eliminates that equation again, so that the
length becomes an observed variable:

```@example dae
spring_x = SX.sym("spring_x")
spring_mass = System([D(D(pos)) ~ link.force / m + gravity, spring_x ~ link.x], t;
                     parameters = [m, k, l], name = "spring_mass")
unknowns(spring_mass)
```

The states are the elements of `pos`, named `mass_pos_0` and `mass_pos_1`, since CasADi
numbers the elements from 0, while the Julia index `pos[1]` is the first, and their
derivatives. The mass starts at rest at `(1, -1)`, where the spring is stretched by
about 0.41 m, so it oscillates fast along the spring while it swings, and the length of
the spring stays between about 0.59 m and 1.41 m. The stiff spring makes the problem
stiff, so we solve it with `Rodas5P`:

```@example dae
start = [pos[1] => 1.0, pos[2] => -1.0, D(pos[1]) => 0.0, D(pos[2]) => 0.0]
problem = SciMLBase.ODEProblem(spring_mass, start, (0.0, 3.0), [m => 1.0, k => 1e4, l => 1.0])
solution = SciMLBase.solve(problem, Rodas5P())
extrema(solution[:spring_x])
```

## Reduce the index of a pendulum

A pendulum of unit mass on a rod of length `L` is described by its position `(x, y)` and
a multiplier `T` proportional to the tension of the rod, which keeps the mass on the
circle:

```math
x'' = T x, \quad y'' = T y - g, \quad 0 = x^2 + y^2 - L^2.
```

As a mass-matrix DAE these equations fail at the first step, since the constraint does
not contain `T`, so the Jacobian of the algebraic equation by the algebraic state is
zero and the Newton iterations of the solver cannot determine `T`. The DAE has index 3:
the constraint must be differentiated twice in time before it contains `T`, which gives
an equation of index 1 for `T`, and once more for an ordinary differential equation.
`System` reduces the index whenever its algebraic equations do not determine its
algebraic states: it finds the differentiations to index 1 from the sparsity pattern of the equations,
by the algorithm of Pantelides in [`reduce_index`](@ref CasADiCodegen.reduce_index), and
lets CasADi compute the derivatives:

```@example dae
x, y, T = SX.sym("x"), SX.sym("y"), SX.sym("T")
g, L = SX.sym("g"), SX.sym("L")
pendulum = System([D(D(x)) ~ T * x, D(D(y)) ~ T * y - g, x^2 + y^2 ~ L^2], t; parameters = [g, L], name = "pendulum")
differentiation_index(pendulum), unknowns(pendulum)
```

[`differentiation_index`](@ref CasADiCodegen.Models.differentiation_index) gives the
index of the model as written, 3.
The states are the positions and the velocities `dx` and `dy`, and `T` is the algebraic
state. The algebraic equation of the compiled model is the second derivative of the
constraint, in which the derivatives of the states are replaced by the right side, so that
it contains `T` with the factor `2(x^2 + y^2)`, nonzero on the circle:

```@example dae
equations(pendulum)[end]
```

The constraint and its first derivative, `x dx + y dy = 0` up to a factor of 2, are no
longer equations of the model but its [`invariants`](@ref CasADiCodegen.Models.invariants):
they stay zero along an exact solution that starts where they are zero, and the initial
values must satisfy them. `initializealg = BrownFullBasicInit()` of
OrdinaryDiffEqNonlinearSolve.jl corrects a guess of the algebraic state `T`, but it leaves
the differential states as given, so these must satisfy the invariants. The pendulum starts at rest at `(1, 0)`, which satisfies both
invariants and where the reduced constraint gives `T = 0`:

```@example dae
start = [x => 1.0, D(x) => 0.0, y => 0.0, D(y) => 0.0, T => 0.0]
problem = SciMLBase.ODEProblem(pendulum, start, (0.0, 10.0), [g => 9.8, L => 1.0])
solution = SciMLBase.solve(problem, Rodas5P(); abstol = 1e-8, reltol = 1e-8)
column(v) = vcat(v...)
residuals = CasadiFunction("residuals", [column(unknowns(pendulum)), column(parameters(pendulum))], [invariants(pendulum)])
drift(solution) = maximum(u -> maximum(abs, Matrix(residuals(u, solution.prob.p))), solution.u)
drift(solution)
```

The function `residuals` evaluates the invariants on the states, in the order of
`unknowns(pendulum)`, and the parameters, in the order of `parameters(pendulum)`. The
compiled model holds the second derivative of the constraint at zero, not the
constraint itself, so the position leaves the circle slowly as the errors of the
integration accumulate; with the tolerances `1e-8`, the largest residual of the
invariants over the ten seconds is about `2e-8`. The compiled model keeps every state and
enforces only the highest derivative of each constraint, since `reduce_index` does not
select dummy derivatives, which would hold the constraint itself. Baumgarte's
stabilization instead replaces the second derivative by a combination of it
and the lower derivatives, `c'' - 2γ c' + γ² c = 0` for the constraint `c` and the pole
`γ < 0` of the keyword
`baumgarte_pole`, so that a deviation from the circle decays as `exp(γ t)`:

```@example dae
stabilized = System([D(D(x)) ~ T * x, D(D(y)) ~ T * y - g, x^2 + y^2 ~ L^2], t; parameters = [g, L],
                    name = "pendulum", baumgarte_pole = -5.0)
problem = SciMLBase.ODEProblem(stabilized, start, (0.0, 10.0), [g => 9.8, L => 1.0])
drift(SciMLBase.solve(problem, Rodas5P(); abstol = 1e-8, reltol = 1e-8))
```

The invariants of `stabilized` are those of `pendulum`, so the same function measures
the drift, which the stabilization makes a few times smaller over these ten seconds.
The gain grows with the time: over 100 seconds, the residual reaches about `5e-6`
without the stabilization and stays below `1e-8` with it. A larger
`|γ|` pulls the solution back faster but makes the equations stiffer.
