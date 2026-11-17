# Use the generated code with SciML packages

With SciMLBase.jl loaded, a CasADi function becomes the function of a SciML problem, with
generated code for the function and for the derivatives that the solvers ask for. The
CasADi function takes the unknowns, then optionally the parameters, and for an ODE then the
time. The result is an ordinary SciML problem, so any package that takes one takes it.

## Solve an ODE

```@example sciml
using CasADiCodegen
using OrdinaryDiffEqVerner: Vern9
import SciMLBase

u, k, t = SX.sym("u", 2), SX.sym("k"), SX.sym("t")
rhs = CasadiFunction("rhs", [u, k, t], [vcat(u[2], -k * u[1])])     # also f(u) and f(u, p)
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(rhs), [1.0, 0.0], (0.0, pi), [4.0])
SciMLBase.solve(problem, Vern9(); abstol = 1e-12, reltol = 1e-12).u[end]
```

A function of the time without parameters takes an empty argument, `f(u, SX(0, 1), t)`.
The time argument is whatever variable the model is integrated in: a model rewritten in
another independent variable, for example with `substitute`, passes that variable there.

## Names and observed variables

All constructors except `OptimizationFunction` take the names of the scalar symbols, so a
solution and a problem are indexed by name; the elements of `SX.sym("u", 2)` are `u_0` and
`u_1`. Each result after the first is an observed variable:

```@example sciml
x, v = SX.sym("x"), SX.sym("v")
oscillator = CasadiFunction("oscillator", [vcat(x, v), k, t], [vcat(v, -k * x), (v^2 + k * x^2) / 2],
                            ["u", "k", "t"], ["du", "energy"])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(oscillator), [1.0, 0.0], (0.0, pi), [4.0])
solution = SciMLBase.solve(problem, Vern9(); abstol = 1e-12, reltol = 1e-12)
solution[:x][end], solution[:energy][end], SciMLBase.remake(problem; p = [:k => 9.0]).p
```

The parameters are an argument of the generated code, so a `remake` with new values
compiles nothing. A `remake` without `p` shares the parameter vector of the original, so
give the copy its own vector, `remake(problem; p = copy(problem.p))`, before you change it
in place. A function whose arguments are not `vcat`s of named symbols, or that repeats a
name, gets no names, and a second result is then an error; a `sys` given to the constructor
replaces the names.

## Data as parameters

A vector symbol keeps its own name besides those of its elements, so a table of data can be
one parameter. Here a linear interpolant takes its values as an argument; the option
`"inline" => true` makes it differentiable by the values, without which CasADi gives zero
as that derivative and the sensitivities to the data are zero with no error:

```@example sciml
using OrdinaryDiffEqTsit5: Tsit5

grid = collect(0.0:0.1:10.0)
table = casadi.interpolant("table", "linear", [grid], 1, Dict("inline" => true))
y, s, data = MX.sym("y"), MX.sym("s"), MX.sym("data", length(grid))
area = CasadiFunction("area", [y, data, s], [table(s, data)])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(area), [0.0], (0.0, 1.0), grid .^ 2)
cubic = SciMLBase.remake(problem; p = [:data => grid .^ 3])
cubic.f === problem.f, SciMLBase.solve(cubic, Tsit5(); tstops = grid).u[end]
```

The new values need no new code, and `paramjac` is the derivative by them. An interpolant
can only be called on MX symbols.

## Sparse Jacobians and static arrays

With SparseArrays.jl loaded, `sparse = true` gives the function a `jac_prototype` with the
sparsity pattern of the Jacobian:

```@example sciml
using SparseArrays

z = SX.sym("z", 5)
chain = CasadiFunction("chain", [z], [vcat([z[i] - (i > 1 ? z[i-1]^2 : 0.0) for i in 1:5]...)])
SciMLBase.NonlinearFunction(chain; sparse = true).jac_prototype
```

For a small system, `inplace = false` gives functions that return an `SVector`, and an
`SMatrix` for each Jacobian. Each array holds at most 1024 elements, so a system with a
Jacobian has at most 32 states, and a larger one throws an `ArgumentError`:

```@example sciml
using StaticArrays: SVector

f = SciMLBase.ODEFunction(rhs; inplace = false)
problem = SciMLBase.ODEProblem(f, SVector(1.0, 0.0), (0.0, pi), SVector(4.0))
f.jac(problem.u0, problem.p, 0.0)
```

## Other problems

| Constructor | Generated functions |
|---|---|
| `ODEFunction(f)` | Right side, `jac`, and `paramjac` when `f` has parameters |
| `SDEFunction(f, g)` | Drift, noise `g`, `jac` and `paramjac` of the drift |
| `NonlinearFunction(f)` | Residual and `jac`; a `resid_prototype` for a residual of another size |
| `DiscreteFunction(f)` | Map to the next state |
| `ImplicitDiscreteFunction(f)` | Residual of the implicit map |
| `OptimizationFunction(objective, constraints)` | Objective, `grad`, `hess`, `cons`, `cons_j`, and `cons_h` or, with `sparse = true`, `lag_h` |

The keywords of each are in [SciML problems and functions](@ref). `chunk_size`,
`split_above`, `optlevel` and `cache` go to [`build_function`](@ref).

A **nonlinear problem** takes the residual and the parameters:

```@example sciml
using SimpleNonlinearSolve: SimpleNewtonRaphson

w, q = SX.sym("w", 2), SX.sym("q")
residual = CasadiFunction("r", [w, q], [vcat(w[1]^2 + w[2]^2 - q, w[1] - w[2])])   # also f(w)
problem = SciMLBase.NonlinearProblem(SciMLBase.NonlinearFunction(residual), [1.0, 0.5], [2.0])
SciMLBase.solve(problem, SimpleNewtonRaphson(); abstol = 1e-12).u
```

An **optimization problem** takes the objective and, optionally, a column of constraints
with the same arguments, whose bounds go to `lcons` and `ucons`. The problem needs no
`adtype`: the solver gets the generated derivatives. `hess = false` leaves out the
Hessians, whose code can take long to build, for a first-order solver such as L-BFGS:

```@example sciml
using OptimizationIpopt: IpoptOptimizer

objective = CasadiFunction("J", [w, q], [(w[1] - q)^2 + w[2]^2])     # also f(w)
constraints = CasadiFunction("c", [w, q], [w[1] + w[2]])
f = SciMLBase.OptimizationFunction(objective, constraints; sparse = true)
problem = SciMLBase.OptimizationProblem(f, [0.0, 0.0], [1.0]; lcons = [1.0], ucons = [Inf])
ipopt = IpoptOptimizer(additional_options = Dict{String,Any}("print_level" => 0, "sb" => "yes"))
SciMLBase.solve(problem, ipopt).u
```

With `sparse = true`, `lag_h` holds the upper triangle of the Hessian of the Lagrange
function in place of a dense Hessian per constraint:

```@example sciml
f.lag_hess_prototype
```

A transcribed optimal control problem, with thousands of unknowns, solves the same way.

`OptimizationProblem(opti)` converts a problem of the `Opti` stack of CasADi with its
initial values, parameter values and bounds. A bound on a variable stays a constraint, so
the problem has no `lb` and `ub`. Write a two-sided bound with `bounded(lb, w, ub)` (see
[The CasADi front end](@ref)):

```@example sciml
opti = Opti()
p, a = variable(opti, 2), parameter(opti)
minimize(opti, (p[1] - a)^2 + p[2]^2)
subject_to(opti, p[1] + p[2] >= 1)
subject_to(opti, bounded([-2.0, -2.0], p, [2.0, 2.0]))
casadi.set_value(opti, a, 0.25)
problem = SciMLBase.OptimizationProblem(opti; sparse = true)
SciMLBase.solve(problem, ipopt).u, length(problem.lcons)
```

A **stochastic problem** takes the drift and the noise, with the same arguments. A noise
that returns a column is diagonal, unless the problem gets a scalar Wiener process; a noise
that returns a matrix, one column per Wiener process, sets the `noise_rate_prototype`, with
the structural zeros of the matrix:

```@example sciml
using StochasticDiffEqHighOrder: SRIW1
using DiffEqNoiseProcess: WienerProcess

r = SX.sym("r", 3)
drift = CasadiFunction("drift", [r, k, t], [vcat(10 * (r[2] - r[1]), r[1] * (28 - r[3]) - r[2], r[1] * r[2] - k * r[3])])
noise = CasadiFunction("noise", [r, k, t], [0.3 * r])
diagonal = SciMLBase.SDEProblem(drift, noise, [5.0, 5.0, 1.0], (0.0, 1.0), [8 / 3])
scalar = SciMLBase.SDEProblem(drift, noise, [5.0, 5.0, 1.0], (0.0, 1.0), [8 / 3]; noise = WienerProcess(0.0, 0.0, 0.0))
o = SX(1, 1)
mixed = CasadiFunction("mixed", [r, k, t], [[0.3r[1] o; 0.3r[2] 0.1; o 0.3r[3]]])
SciMLBase.solve(diagonal, SRIW1(); seed = 1).retcode,
    size(SciMLBase.SDEProblem(drift, mixed, [5.0, 5.0, 1.0], (0.0, 1.0), [8 / 3]).noise_rate_prototype)
```

A **discrete problem** takes the map to the next state, which receives the time of the new
state and no step length; a step length is a parameter. `ImplicitDiscreteProblem` takes the
residual `F(u⁺, u, p, t)`, which `IDSolve` differentiates with dual numbers through the
generated code, in place:

```@example sciml
using OrdinaryDiffEqFunctionMap: FunctionMap
using ImplicitDiscreteSolve: IDSolve

n = SX.sym("n")
growth = CasadiFunction("growth", [n, k], [n + k * n * (1 - n)])
SciMLBase.solve(SciMLBase.DiscreteProblem(growth, [0.1], (0.0, 20.0), [0.5]), FunctionMap()).u[end]
```

```@example sciml
e, e_prev, h = SX.sym("e"), SX.sym("e_prev"), SX.sym("h")
euler = CasadiFunction("euler", [e, e_prev, vcat(k, h), t], [e - e_prev + h * k * (e - cos(t))])
implicit = SciMLBase.ImplicitDiscreteProblem(euler, [0.0], (0.0, 2.0), [50.0, 0.1])
path = SciMLBase.solve(implicit, IDSolve(); dt = 0.1, adaptive = false, initializealg = SciMLBase.NoInit())
path.u[end]
```

A **bifurcation problem** of BifurcationKit.jl takes the residual, the guess, the parameter
values and the continued parameter, by name or by place. It holds generated code for the
residual, its Jacobian, and the higher derivatives that BifurcationKit uses at special
points; `jet = false` keeps the Jacobian only, which compiles faster:

```@example sciml
using BifurcationKit: BifurcationProblem, PALC, ContinuationPar, continuation

b, μ = SX.sym("b"), SX.sym("μ")
pitchfork = CasadiFunction("pitchfork", [b, μ], [μ * b - b^3])
problem = BifurcationProblem(pitchfork, [:b => 1.0], [:μ => -1.0], :μ; record_from_solution = :b)
branch = continuation(problem, PALC(), ContinuationPar(p_min = -2.0, p_max = 2.0); bothside = Val(true))
[(point.type, round(point.param; digits = 2)) for point in branch.specialpoint]
```

Packages that read the names of a problem, such as DynamicalSystemsBase.jl and
Attractors.jl, take the states and parameters by name:

```@example sciml
using Attractors: CoupledODEs, observe_state, current_parameter

lorenz = SciMLBase.ODEFunction(drift; inplace = false)
ds = CoupledODEs(SciMLBase.ODEProblem(lorenz, SVector(5.0, 5.0, 1.0), (0.0, 1.0), [8 / 3]))
observe_state(ds, :r_0), current_parameter(ds, :k)
```

For the problem of a solver object of CasADi (`cvodes`, `kinsol`, `nlpsol`), see
[Solve the problem in Julia](@ref).
