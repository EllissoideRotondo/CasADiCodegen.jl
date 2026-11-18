# Model a circuit and a pendulum as DAEs

In this tutorial, we write an electric circuit from its parts, and a pendulum.
[`System`](@ref CasADiCodegen.Models.System) of the submodule `CasADiCodegen.Models`
eliminates the algebraic equations that it can solve and reduces the index of the
pendulum, with derivatives from CasADi; [Write a model as equations](@ref) describes the
submodule.

```@example dae
using CasADiCodegen, CasADiCodegen.Models, LinearAlgebra
import SciMLBase
using OrdinaryDiffEqRosenbrock: Rodas5P
```

## Write the source, resistor and capacitor

The circuit is a source of 1 V, a resistor of 2 Ω and a capacitor of 1 F in a loop,
with the negative side of the source grounded. A pin carries a voltage `v` and a current
`i` into the part:

```@example dae
pin(name) = (v = SX.sym("$(name)_v"), i = SX.sym("$(name)_i"))
nothing # hide
```

A one-port has a voltage across it and a current through it, tied to its two pins by
three equations:

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

The capacitor adds the only differential equation, `Cap v' = i`:

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

The circuit has 20 voltages and currents and as many equations.

## Reduce the circuit to one ODE and solve it

`System` compiles the equations, and [`unknowns`](@ref CasADiCodegen.Models.unknowns)
gives the states that remain:

```@example dae
rc = System(circuit, t; parameters = [R, Cap, V], name = "rc")
unknowns(rc)
```

Each algebraic equation is linear in some variable, so `System` solves it and
substitutes the solution, which leaves an ODE of one state:

```@example dae
equations(rc)
```

[`observed`](@ref CasADiCodegen.Models.observed) gives each eliminated variable with its
expression:

```@example dae
filter(o -> casadi.name(first(o)) == "resistor_i", observed(rc))
```

`SciMLBase.ODEProblem` takes the initial values and the parameters as pairs from a
symbol, or its name, to a value:

```@example dae
v0 = [capacitor.v => 0.0]
tspan = (0.0, 10.0)
rc_values = [R => 2.0, Cap => 1.0, V => 1.0]
problem = SciMLBase.ODEProblem(rc, v0, tspan, rc_values)
solution = SciMLBase.solve(problem, Rodas5P(); abstol = 1e-8, reltol = 1e-8)
solution[:capacitor_v][end], solution[:resistor_i][end]
```

The exact values at `t = 10` are `1 - exp(-t / 2)` and `exp(-t / 2) / 2`:

```@example dae
(1 - exp(-5), exp(-5) / 2)
```

## Reduce the index of a pendulum

A pendulum of unit mass on a rod of length `L` has a multiplier `T` proportional to the
tension:

```math
x'' = T x, \quad y'' = T y - g, \quad 0 = x^2 + y^2 - L^2.
```

The constraint does not contain `T`, so a mass-matrix solver fails at the first step.
`System` reduces the index from 3 to 1 by the algorithm of Pantelides in
[`reduce_index`](@ref CasADiCodegen.reduce_index):

```@example dae
x, y, T = SX.sym("x"), SX.sym("y"), SX.sym("T")
g, L = SX.sym("g"), SX.sym("L")
pendulum_equations = [D(D(x)) ~ T * x, D(D(y)) ~ T * y - g, x^2 + y^2 ~ L^2]
pendulum = System(pendulum_equations, t; parameters = [g, L], name = "pendulum")
differentiation_index(pendulum), unknowns(pendulum)
```

[`differentiation_index`](@ref CasADiCodegen.Models.differentiation_index) gives the
index as written. The algebraic equation of the compiled model is the second derivative of
the constraint, which contains `T` with the factor `2(x^2 + y^2)`:

```@example dae
equations(pendulum)[end]
```

The constraint and its first derivative become [`invariants`](@ref
CasADiCodegen.Models.invariants), which the initial values must satisfy (see
[Initialize differential-algebraic equations](@ref)). The pendulum starts at rest at
`(1, 0)`, where `T = 0`, and `residuals` measures the drift from the invariants:

```@example dae
start = [x => 1.0, D(x) => 0.0, y => 0.0, D(y) => 0.0, T => 0.0]
pendulum_values = [g => 9.8, L => 1.0]
problem = SciMLBase.ODEProblem(pendulum, start, tspan, pendulum_values)
solution = SciMLBase.solve(problem, Rodas5P(); abstol = 1e-8, reltol = 1e-8)

column(v) = vcat(v...)
states_in, parameters_in = column(unknowns(pendulum)), column(parameters(pendulum))
residuals = CasadiFunction("residuals", [states_in, parameters_in], [invariants(pendulum)])
function drift(solution)
    p = solution.prob.p
    maximum(u -> maximum(abs, Matrix(residuals(u, p))), solution.u)
end
drift(solution)
```

The compiled model holds only the second derivative of the constraint at zero, since
`reduce_index` selects no dummy derivatives, so the position leaves the circle slowly.
With the keyword `baumgarte_pole = γ < 0`, Baumgarte's stabilization enforces
`c'' - 2γ c' + γ² c = 0` for the constraint `c`, so that a deviation decays as
`exp(γ t)`:

```@example dae
stabilized = System(pendulum_equations, t; parameters = [g, L], name = "pendulum",
                    baumgarte_pole = -5.0)
problem = SciMLBase.ODEProblem(stabilized, start, tspan, pendulum_values)
drift(SciMLBase.solve(problem, Rodas5P(); abstol = 1e-8, reltol = 1e-8))
```

A larger `|γ|` corrects faster but makes the equations stiffer.
