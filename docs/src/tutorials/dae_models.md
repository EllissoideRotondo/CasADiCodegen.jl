# Model physical systems with differential-algebraic equations

In this tutorial, we write an electric circuit, a mass on a spring and a pendulum from
their parts. [`System`](@ref CasADiCodegen.Models.System) of the submodule
`CasADiCodegen.Models` eliminates the algebraic equations that it can solve and reduces
the index of the pendulum, with derivatives from CasADi;
[Write a model as equations](@ref) describes the submodule. `Rodas5P` integrates the mass-matrix form.

```@example dae
using CasADiCodegen, CasADiCodegen.Models, LinearAlgebra
import SciMLBase
using OrdinaryDiffEqRosenbrock: Rodas5P
```

## Write the parts of a circuit

The circuit is a source of 1 V, a resistor of 2 Ω and a capacitor of 1 F in a loop,
with the negative side of the source grounded. A pin carries a voltage `v` and a current
`i` into the part, each a named scalar symbol:

```@example dae
pin(name) = (v = SX.sym("$(name)_v"), i = SX.sym("$(name)_i"))
nothing # hide
```

A one-port has the voltage across it and the current through it, tied to its two pins by
three equations written with `~`:

```@example dae
function oneport(name)
    p, n = pin("$(name)_p"), pin("$(name)_n")
    v, i = SX.sym("$(name)_v"), SX.sym("$(name)_i")
    (; p, n, v, i, equations = [v ~ p.v - n.v, 0 ~ p.i + n.i, i ~ p.i])
end
nothing # hide
```

Connected pins share one voltage, and their currents sum to zero:

```@example dae
connect(a, others...) = [[a.v ~ q.v for q in others]; 0 ~ a.i + sum(q.i for q in others)]
nothing # hide
```

The capacitor adds the only differential equation, `Cap v' = i`, where `D(capacitor.v)`
is the derivative by `t`:

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

The circuit has 20 voltages and currents and as many scalar equations.

## Compile and solve the circuit

`System` takes the equations, the time and the parameters; every other symbol is a
variable. [`unknowns`](@ref CasADiCodegen.Models.unknowns) gives the states that remain:

```@example dae
rc = System(circuit, t; parameters = [R, Cap, V], name = "rc")
unknowns(rc)
```

Only the capacitor voltage remains: each algebraic equation is linear in some variable,
so `System` solves it and substitutes the solution. What is left is an ODE of one state:

```@example dae
equations(rc)
```

The eliminated variables are observed variables; [`observed`](@ref
CasADiCodegen.Models.observed) gives each with its expression:

```@example dae
filter(o -> casadi.name(first(o)) == "resistor_i", observed(rc))
```

`SciMLBase.ODEProblem` takes the initial values and the parameters as pairs from a
symbol, or its name, to a value:

```@example dae
problem = SciMLBase.ODEProblem(rc, [capacitor.v => 0.0], (0.0, 10.0), [R => 2.0, Cap => 1.0, V => 1.0])
solution = SciMLBase.solve(problem, Rodas5P(); abstol = 1e-8, reltol = 1e-8)
solution[:capacitor_v][end], solution[:resistor_i][end]
```

The exact solution is `1 - exp(-t / 2)` for the voltage and `exp(-t / 2) / 2` for the
current:

```@example dae
(1 - exp(-5), exp(-5) / 2)
```

## Model a mass on a spring

A mass of 1 kg hangs under gravity from a spring of stiffness `10^4` N/m and rest
length 1 m, fixed at the origin. The function `spring` returns the length, the direction
and the force of the spring from the two points that it joins:

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

We give the length its own symbol and equation, which `System` eliminates, so that the
length becomes an observed variable:

```@example dae
spring_x = SX.sym("spring_x")
spring_mass = System([D(D(pos)) ~ link.force / m + gravity, spring_x ~ link.x], t;
                     parameters = [m, k, l], name = "spring_mass")
unknowns(spring_mass)
```

The states are the elements of `pos`, named `mass_pos_0` and `mass_pos_1` since CasADi
counts from 0, and their derivatives. The mass starts at rest at `(1, -1)`, and the stiff
spring makes the problem stiff:

```@example dae
start = [pos[1] => 1.0, pos[2] => -1.0, D(pos[1]) => 0.0, D(pos[2]) => 0.0]
problem = SciMLBase.ODEProblem(spring_mass, start, (0.0, 3.0), [m => 1.0, k => 1e4, l => 1.0])
solution = SciMLBase.solve(problem, Rodas5P())
extrema(solution[:spring_x])
```

## Reduce the index of a pendulum

A pendulum of unit mass on a rod of length `L` has the position `(x, y)` and a
multiplier `T` proportional to the tension:

```math
x'' = T x, \quad y'' = T y - g, \quad 0 = x^2 + y^2 - L^2.
```

The constraint does not contain `T`, so a mass-matrix solver fails at the first step.
The DAE has index 3, and `System` reduces it to index 1 by the algorithm of Pantelides in
[`reduce_index`](@ref CasADiCodegen.reduce_index), differentiating the constraint with
CasADi:

```@example dae
x, y, T = SX.sym("x"), SX.sym("y"), SX.sym("T")
g, L = SX.sym("g"), SX.sym("L")
pendulum = System([D(D(x)) ~ T * x, D(D(y)) ~ T * y - g, x^2 + y^2 ~ L^2], t; parameters = [g, L], name = "pendulum")
differentiation_index(pendulum), unknowns(pendulum)
```

[`differentiation_index`](@ref CasADiCodegen.Models.differentiation_index) gives the
index as written. The algebraic equation of the compiled model is the second derivative of
the constraint, which contains `T` with the factor `2(x^2 + y^2)`:

```@example dae
equations(pendulum)[end]
```

The constraint and its first derivative become [`invariants`](@ref
CasADiCodegen.Models.invariants), which the initial values must satisfy:
`BrownFullBasicInit()` of OrdinaryDiffEqNonlinearSolve.jl corrects the algebraic state
`T` but not the differential states.
The pendulum starts at rest at `(1, 0)`, where `T = 0`. The function `residuals`
measures how far the solution drifts from the invariants:

```@example dae
start = [x => 1.0, D(x) => 0.0, y => 0.0, D(y) => 0.0, T => 0.0]
problem = SciMLBase.ODEProblem(pendulum, start, (0.0, 10.0), [g => 9.8, L => 1.0])
solution = SciMLBase.solve(problem, Rodas5P(); abstol = 1e-8, reltol = 1e-8)
column(v) = vcat(v...)
residuals = CasadiFunction("residuals", [column(unknowns(pendulum)), column(parameters(pendulum))], [invariants(pendulum)])
drift(solution) = maximum(u -> maximum(abs, Matrix(residuals(u, solution.prob.p))), solution.u)
drift(solution)
```

The compiled model holds only the second derivative of the constraint at zero, since
`reduce_index` selects no dummy derivatives, so the position leaves the circle slowly;
the drift above is about `2e-8`. With the keyword `baumgarte_pole = γ < 0`, Baumgarte's
stabilization enforces `c'' - 2γ c' + γ² c = 0` for the constraint `c`, so that a
deviation decays as `exp(γ t)`:

```@example dae
stabilized = System([D(D(x)) ~ T * x, D(D(y)) ~ T * y - g, x^2 + y^2 ~ L^2], t; parameters = [g, L],
                    name = "pendulum", baumgarte_pole = -5.0)
problem = SciMLBase.ODEProblem(stabilized, start, (0.0, 10.0), [g => 9.8, L => 1.0])
drift(SciMLBase.solve(problem, Rodas5P(); abstol = 1e-8, reltol = 1e-8))
```

The drift is a few times smaller over ten seconds; over 100 seconds it is about `5e-6`
without the stabilization and below `1e-8` with it. A larger `|γ|` corrects faster but
makes the equations stiffer.
