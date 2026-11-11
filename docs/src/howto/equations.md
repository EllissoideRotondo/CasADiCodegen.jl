# Write a model as equations

The submodule `CasADiCodegen.Models` compiles equations among the variables of a model,
with derivatives on either side, into the semi-explicit form of the CasADi integrators:
differential states `x` with `x' = ode(x, z, p, t)` and algebraic states `z` with
`0 = alg(x, z, p, t)`. Both SciML and CasADi then integrate the model.

## Write the equations

Load the submodule, make a scalar SX symbol for the time, and take its `Differential`:

```@example equations
using CasADiCodegen, CasADiCodegen.Models
import SciMLBase
using OrdinaryDiffEqTsit5: Tsit5

t = SX.sym("t")
D = Differential(t)
nothing # hide
```

`D(x)` is the derivative symbol `dx`, and `D(D(x))` is `ddx`; for `SX.sym("q", 2)` the
symbols are `dq_0` and `dq_1`. `D` of an expression applies the chain rule, and `System`
later sets the derivatives of the parameters to 0:

```@example equations
x, v, m, k, c = SX.sym("x"), SX.sym("v"), SX.sym("m"), SX.sym("k"), SX.sym("c")
D(m * x^2 + sin(t))
```

Write each equation with `~` between two SX expressions or numbers; an equation of two
vectors of one size stands for one equation per element. A damped oscillator:

```@example equations
laws = [D(x) ~ v, m * D(v) ~ -k * x - c * v]
```

## Compile the equations

Give the equations, the time and the parameters to [`System`](@ref
CasADiCodegen.Models.System):

```@example equations
oscillator = System(laws, t; parameters = [m, k, c], name = "oscillator")
```

Every other symbol is a variable, counted once with its derivatives, and the model needs
one scalar equation per variable. `System` throws an `ArgumentError` when the counts
differ, when two symbols share a name, or when elimination leaves an equation without
variables.

`System` solves for the highest derivatives when they enter linearly with a nonsingular
coefficient; otherwise they become algebraic states, as in `exp(D(x)) ~ x`, and need
initial values. [`equations`](@ref CasADiCodegen.Models.equations) gives the compiled
model, [`unknowns`](@ref CasADiCodegen.Models.unknowns) its states,
[`parameters`](@ref CasADiCodegen.Models.parameters) its parameters, and
[`observed`](@ref CasADiCodegen.Models.observed) the removed variables with their
expressions:

```@example equations
equations(oscillator)
```

In a second-order equation, the derivative below the highest, `dx`, becomes a state:

```@example equations
second_order = System([m * D(D(x)) ~ -k * x - c * D(x)], t; parameters = [m, k, c])
unknowns(second_order)
```

The states come in the order in which the variables first appear, each followed by its
lower derivatives, and then the algebraic states. `System` also eliminates each algebraic
equation that is linear in a variable with a constant or parametric coefficient, such as
`a ~ b`, and keeps the eliminated variable as observed; the other algebraic equations
stay, and `System` differentiates them when needed (index reduction), with the keyword
`baumgarte_pole` for stabilization. The docstring of
[`System`](@ref CasADiCodegen.Models.System) lists the steps, and
[Model physical systems with differential-algebraic equations](@ref) applies them.

## Solve the model

Give the initial values and the parameters to `SciMLBase.ODEProblem` as pairs from a
symbol, or its name, to a value; `dx` is a state here:

```@example equations
problem = SciMLBase.ODEProblem(second_order, [x => 1.0, :dx => 0.0], (0.0, 10.0), [m => 1.0, k => 4.0, c => 0.5])
solution = SciMLBase.solve(problem, Tsit5())
solution[:x][end], solution[:ddx][end]
```

A differential state or a parameter without a value makes `ODEProblem` throw an
`ArgumentError`, and observed variables such as `ddx` are indexed by name.

A model with algebraic states has a mass matrix and needs a solver such as `Rodas5P` of
OrdinaryDiffEqRosenbrock.jl. An algebraic state without a value starts from 0; give a
consistent value, or pass `initializealg = BrownFullBasicInit()` of
OrdinaryDiffEqNonlinearSolve.jl to `solve`. After index reduction, the initial states must
also satisfy the [`invariants`](@ref CasADiCodegen.Models.invariants).

## Use the model with CasADi

`Dict(sys)` returns the dictionary of a CasADi integrator, with `"x"`, `"z"`, `"p"`,
`"t"`, `"ode"` and `"alg"`, without the entries that the model lacks:

```@example equations
F = casadi.integrator("F", "cvodes", Dict(second_order), 0.0, 10.0)
result = F(Dict("x0" => casadi.DM([1.0, 0.0]), "p" => casadi.DM([1.0, 4.0, 0.5])))
result["xf"]
```

`"x0"` and `"z0"` take the states in the order of `unknowns`, and `"p"` the parameters in
the order of `parameters`. A model with algebraic states needs a DAE plugin such as
`"idas"`.
