# Write a model as equations

A CasADi function of a right side states the derivative of each state explicitly. A model
taken from physics is often easier to state as equations among its variables, in which
derivatives appear on either side, some variables only repeat others, and some equations
hold no derivative at all. The submodule `CasADiCodegen.Models` takes such equations and
compiles them into the semi-explicit form of the CasADi integrators: differential states
`x` with `x' = ode(x, z, p, t)`, and algebraic states `z`, which enter no derivative and
are fixed by the algebraic equations `0 = alg(x, z, p, t)`. The SciML solvers and CasADi
itself integrate this form, with derivatives from CasADi; a model with algebraic states
needs a solver for a mass matrix.

The examples need CasADiCodegen.jl, SciMLBase.jl and the solver `Tsit5` of
OrdinaryDiffEqTsit5.jl.

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

`D(x)` of a symbol `x` is a derivative symbol named `"d"` followed by the name of `x`,
which stands for the time derivative of `x`; each call returns the same symbol, and
`D(D(x))` is `ddx`. `D` of a vector symbol gives the vector of the derivative symbols of
its elements, `dq_0` and `dq_1` for `SX.sym("q", 2)`. `D` of an expression applies the
chain rule, with `D(t) = 1`; it makes a derivative symbol for each symbol of the
expression, and `System` later replaces the derivative symbol of each parameter by 0:

```@example equations
x, v, m, k, c = SX.sym("x"), SX.sym("v"), SX.sym("m"), SX.sym("k"), SX.sym("c")
D(m * x^2 + sin(t))
```

Write each equation with `~` between its two sides. A side is an SX expression, scalar or
vector, or a number, and an equation of two vector expressions of one size stands for one
equation per element; the elements of `SX.sym("q", 2)` are named `q_0` and `q_1`. A damped
oscillator, written with its velocity, is:

```@example equations
laws = [D(x) ~ v, m * D(v) ~ -k * x - c * v]
```

## Compile the equations

Give the equations, the time and the parameters to [`System`](@ref
CasADiCodegen.Models.System):

```@example equations
oscillator = System(laws, t; parameters = [m, k, c], name = "oscillator")
```

Every symbol of the equations other than the time and the parameters is a variable of the
model, and a variable counts once with all its derivatives. The model needs one scalar
equation per variable. `System` throws an `ArgumentError` when the counts differ, when two
symbols of the model share a name, since the solution looks a variable up by its name, and
when the elimination below leaves an equation without variables; add the missing
equation, or rename the symbol.

When the equations that contain the derivatives of highest order, here `dv`, are linear in
them with a nonsingular coefficient, `System` solves them for these derivatives, which
here needs the division by `m`; otherwise the derivatives become algebraic states, as in
`exp(D(x)) ~ x`, whose initial values the problem then needs too. [`equations`](@ref CasADiCodegen.Models.equations) gives the
compiled model, [`unknowns`](@ref CasADiCodegen.Models.unknowns) its states,
[`parameters`](@ref CasADiCodegen.Models.parameters) its parameters, and
[`observed`](@ref CasADiCodegen.Models.observed) the pairs of each variable that the
compilation removed and its expression in the states:

```@example equations
equations(oscillator)
```

The same model has a shorter form as one equation of the second order. `System` then
makes `dx`, the derivative below the highest, a state of its own:

```@example equations
second_order = System([m * D(D(x)) ~ -k * x - c * D(x)], t; parameters = [m, k, c])
unknowns(second_order)
```

The states are the differential states, each variable followed by its derivatives below
the highest, in the order in which the variables first appear in the equations, and then
the algebraic states.

`System` also removes each algebraic equation that is linear in a variable with a
coefficient that is a number or an expression of the parameters, such as the equations
`a ~ b` that a connection of parts makes: it solves the equation for that variable,
substitutes the solution into the other equations, and keeps the solution, so that the
eliminated variable is an observed variable. An algebraic equation that it cannot solve this way, such as one of a set of
nonlinear equations that hold several variables together, stays an algebraic equation,
and its variables become algebraic states. When the algebraic equations do not determine
the algebraic states, `System` differentiates them until they do (index reduction), with
the keyword `baumgarte_pole` for a stabilization of the differentiated constraints. The docstring of [`System`](@ref CasADiCodegen.Models.System) lists the steps,
and the tutorial [Model physical systems with differential-algebraic equations](@ref)
applies them to a circuit and a pendulum.

## Solve the model

Give the initial values and the parameters to `SciMLBase.ODEProblem` as pairs, each from a
symbol or its name to a value. A derivative below the highest is a state, so give it with
its symbol `D(x)` or its name `:dx`:

```@example equations
problem = SciMLBase.ODEProblem(second_order, [x => 1.0, :dx => 0.0], (0.0, 10.0), [m => 1.0, k => 4.0, c => 0.5])
solution = SciMLBase.solve(problem, Tsit5())
solution[:x][end], solution[:ddx][end]
```

Each differential state and each parameter needs a value, or `ODEProblem` throws an
`ArgumentError` that names it. The solution also answers to the names of the observed
variables, such as the acceleration `ddx`, which `System` solved for and removed.

A model with algebraic states has a mass matrix, a diagonal matrix with zeros in the rows
of the algebraic equations, so solve it with a solver for such models, such as `Rodas5P`
of OrdinaryDiffEqRosenbrock.jl. An algebraic state without a value starts from 0; give it a
value that satisfies the algebraic equations, or pass
`initializealg = BrownFullBasicInit()` of OrdinaryDiffEqNonlinearSolve.jl to `solve`, which
corrects it before the first step. A model whose index `System` reduced also has
[`invariants`](@ref CasADiCodegen.Models.invariants), the constraints that the reduction
differentiated, which the differential states must satisfy at the start.

## Use the model with CasADi

`Dict(sys)` returns the model in the form of the CasADi integrators: the differential
states `"x"`, the algebraic states `"z"`, the parameters `"p"` and the time `"t"`, each a
column of symbols, and the expressions `"ode"` and `"alg"`. `"z"` and `"alg"` are absent
for a model without algebraic states, and `"p"` for one without parameters:

```@example equations
F = CasADiCodegen.C.integrator("F", "cvodes", Dict(second_order), 0.0, 10.0)
result = F(Dict("x0" => CasADiCodegen.C.DM([1.0, 0.0]), "p" => CasADiCodegen.C.DM([1.0, 4.0, 0.5])))
result["xf"]
```

`"x0"` and `"z0"` take the states in the order of `unknowns(second_order)`, and `"p"` the
parameters in the order of `parameters(second_order)`, which is that of the keyword
`parameters`. A model with algebraic states needs a DAE plugin such as `"idas"`.
