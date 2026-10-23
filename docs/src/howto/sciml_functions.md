# Give a CasADi function to a SciML problem

With SciMLBase.jl loaded, a CasADi function becomes the function of a SciML
problem. The problem holds generated code for the function and for the derivatives
that the solvers ask for, so no automatic differentiation runs at solve time, and by
default each generated function works in place and allocates no memory. Each CasADi
function takes the unknowns first and then, optionally, the parameters, and for
`ODEFunction` then the time. The examples also need OrdinaryDiffEqVerner.jl,
SimpleNonlinearSolve.jl, StaticArrays.jl and the standard library SparseArrays.

## Solve an ODE

We write the right side of a harmonic oscillator as a CasADi function of the state,
the parameter and the time, and give it to `ODEFunction`:

```@example sciml
using CasADiCodegen
using OrdinaryDiffEqVerner: Vern9
import SciMLBase

u, k, t = SX.sym("u", 2), SX.sym("k"), SX.sym("t")
rhs = CasadiFunction("rhs", [u, k, t], [vertcat(u[2], -k * u[1])])     # also f(u) and f(u, p)
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(rhs), [1.0, 0.0], (0.0, pi), [4.0])
SciMLBase.solve(problem, Vern9(); abstol = 1e-12, reltol = 1e-12).u[end]
```

## Other problems

The constructors of nonlinear and optimization problems follow the same pattern.
We solve two equations in `z` with the parameter `q`:

```@example sciml
using SimpleNonlinearSolve: SimpleNewtonRaphson

z, q = SX.sym("z", 2), SX.sym("q")
residual = CasadiFunction("r", [z, q], [vertcat(z[1]^2 + z[2]^2 - q, z[1] - z[2])])   # also f(z)
problem = SciMLBase.NonlinearProblem(SciMLBase.NonlinearFunction(residual), [1.0, 0.5], [2.0])
SciMLBase.solve(problem, SimpleNewtonRaphson(); abstol = 1e-12).u
```

For an optimization problem, the objective is one number and the constraints are one
vector, with the same arguments. The constraints are optional; with them, the problem
takes their bounds `lcons` and `ucons`. The function holds its derivatives, so the
problem needs no `adtype`:

```@example sciml
v = SX.sym("v", 2)
objective = CasadiFunction("J", [v, q], [(v[1] - q)^2 + v[2]^2])     # also f(v)
constraints = CasadiFunction("c", [v, q], [vertcat(v[1] + v[2])])
f = SciMLBase.OptimizationFunction(objective, constraints)
problem = SciMLBase.OptimizationProblem(f, [0.0, 0.0], [1.0]; lcons = [1.0], ucons = [Inf])
nothing # hide
```

A solver of Optimization.jl, such as Ipopt through OptimizationMOI.jl, then solves
`problem`. A problem written with the `Opti` stack of CasADi becomes a SciML problem in
one call, `OptimizationProblem(opti)`, with its initial values, the values of its
parameters and the bounds of its constraints:

```@example sciml
C = CasADiCodegen.C
opti = C.Opti()
w, a = C.variable(opti, 2), C.parameter(opti)
C.minimize(opti, (w[1] - a)^2 + w[2]^2)
C.subject_to(opti, w[1] + w[2] >= 1)
C.set_value(opti, a, C.DM(0.25))
problem = SciMLBase.OptimizationProblem(opti; sparse = true)
problem.lcons, problem.ucons
```

| Constructor | Generated functions |
|---|---|
| `ODEFunction(f; jac = true, paramjac = true, sparse = false, inplace = true, specialize = FullSpecialize)` | Right side, `jac`, and `paramjac` when `f` has parameters |
| `SDEFunction(f, g; jac = true, paramjac = true, sparse = false, inplace = true, specialize = FullSpecialize)` | Drift, noise `g`, `jac` and `paramjac` of the drift |
| `DiscreteFunction(f; inplace = true, specialize = FullSpecialize)` | Map to the next state |
| `ImplicitDiscreteFunction(f; inplace = true, specialize = FullSpecialize)` | Residual of the implicit map |
| `NonlinearFunction(f; jac = true, sparse = false, inplace = true, specialize = FullSpecialize)` | Residual and `jac` |
| `OptimizationFunction(objective, constraints; hess = true, sparse = false, lag_h = sparse && hess)` | Objective, `grad`, `hess`, `cons`, `cons_j`, and either `cons_h`, the Hessian of each constraint, or `lag_h`, the Hessian of the Lagrange function |

- The matrices are dense unless `sparse = true`.
- `hess = false` leaves out `hess`, `cons_h` and `lag_h`, whose code can take long to
  build, for a solver that uses first derivatives only, such as L-BFGS.
- The keywords `chunk_size`, `split_above`, `optlevel` and `cache` go to
  [`build_function`](@ref) for each generated function; see
  [Large functions and the cache on disk](@ref). With `cache`, the generated functions
  are defined while the constructor runs, so a Julia function that makes the SciML
  function and then solves its problem must call `solve` through `Base.invokelatest`.
- The other keywords go to the SciML constructor.
- A function that has no parameters ignores the parameters of the problem. A function of
  the time without parameters takes an empty parameter argument, `f(u, SX(0, 1), t)` (or
  `MX(0, 1)`), and its problem needs no parameters.
- `ODEFunction(f; initialization = c)` adds conditions `c(u, p, t) = 0` on the initial
  values, which the solver solves together with the algebraic equations before its first
  step, and `solved_parameters` makes parameters unknowns of that problem; see
  [Initialize differential-algebraic equations](@ref).
- The noise `g` of `SDEFunction` has the arguments of the drift `f` and returns a column,
  for diagonal or scalar noise, or a matrix with a column per Wiener process.
  `SDEProblem(f, g, u0, tspan, p)` makes the function and gives a matrix noise its
  `noise_rate_prototype`, sparse with `sparse = true`; see
  [Add noise to a differential equation](@ref).
- The map `f(u, p, t)` of `DiscreteFunction` returns the next state, and the residual
  `f(u⁺, u, p, t)` of `ImplicitDiscreteFunction` takes the next and the current state;
  `DiscreteProblem(f, u0, tspan, p)` and `ImplicitDiscreteProblem(f, u0, tspan, p)` make
  the function and the problem. See [Discrete-time systems](@ref).
- A residual of `NonlinearFunction` of another size than its unknowns gives the function
  a `resid_prototype`, for a `NonlinearLeastSquaresProblem`.

For a large system whose Jacobian has few nonzeros, `sparse = true` gives the function
a `jac_prototype`, a `SparseMatrixCSC` with the sparsity pattern of the Jacobian, into
whose stored values `jac` writes; it needs the standard library SparseArrays loaded. For a chain of
equations, each of which depends on its unknown and on the one before, the prototype
has two diagonals:

```@example sciml
using SparseArrays

y = SX.sym("y", 5)
chain = CasadiFunction("chain", [y], [vertcat([y[i] - (i > 1 ? y[i-1]^2 : 0.0) for i in 1:5]...)])
SciMLBase.NonlinearFunction(chain; sparse = true).jac_prototype
```

The dense form of `OptimizationFunction` writes one n×n matrix per constraint in
`cons_h`, so that a transcribed optimal control problem of 2604 unknowns and 2402
constraints needs about 130 GB for these matrices alone. With `sparse = true`, `hess` and
`cons_j` write the stored values of the prototypes `hess_prototype` and
`cons_jac_prototype`, and by default the function holds `lag_h(h, x, σ, μ, p)` in place of
`cons_h`: the upper triangle of the Hessian of the Lagrange function `σ f + μ' g`, whose
pattern is `lag_hess_prototype`, so that Ipopt gets its second derivatives from one call.
On a chained Rosenbrock problem of 100 unknowns and 98 constraints, this call took a fifth
of the time of the Hessian of the objective and those of the constraints. With
`lag_h = false`, `cons_h` writes one sparse matrix per constraint, whose patterns are the
`cons_hess_prototype`. The same keywords reach the function of
`OptimizationProblem(S, x0, p)` for a solver object `S` of `nlpsol` (see
[Solve the problem in Julia](@ref)):

```@example sciml
f = SciMLBase.OptimizationFunction(objective, constraints; sparse = true)
f.lag_hess_prototype
```

## Names and observed variables

`ODEFunction`, `SDEFunction`, `DiscreteFunction`, `ImplicitDiscreteFunction` and
`NonlinearFunction` give the SciML function the names of the symbols of the CasADi
function, so that a solution and a problem can be indexed by name, as in
`solution[:x]` or `remake(problem; p = [:k => 9.0])`. A state built from scalar symbols has their names; the
elements of one symbol of several elements, such as `SX.sym("u", 2)`, are `u_0` and
`u_1`, the names that CasADi gives them. A parameter symbol of several elements, such as
`MX.sym("w", 2)`, is also one parameter under its own name, the vector of its elements, so
that `remake(problem; p = [:w => [1.0, 2.0]])` replaces all of them (see
[Data as parameters](@ref)). Each result of the CasADi function after the
first one is an observed variable, which the solution computes from the state with
generated code, under the name of that result:

```@example sciml
x, v = SX.sym("x"), SX.sym("v")
oscillator = CasadiFunction("oscillator", [vertcat(x, v), k, t], [vertcat(v, -k * x), (v^2 + k * x^2) / 2],
                            ["u", "k", "t"], ["du", "energy"])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(oscillator), [1.0, 0.0], (0.0, pi), [4.0])
solution = SciMLBase.solve(problem, Vern9(); abstol = 1e-12, reltol = 1e-12)
solution[:x][end], solution[:energy][end], SciMLBase.remake(problem; p = [:k => 9.0]).p
```

An argument may join named symbols with `vertcat`. A function whose arguments are not made of named symbols, or in which a name of a symbol
or of a result occurs twice, gets no names, and a second result is then an error. The residual of
`ImplicitDiscreteFunction` has one result and no observed variables. A `sys` given to the constructor
replaces the names and the observed variables. Packages that read a problem through
SymbolicIndexingInterface.jl find the same names: for the name of a state, a parameter or
an observed variable, `SymbolicIndexingInterface.observed(problem, name)` returns a
function `(u, p, t) -> value`, which `observe_state` of DynamicalSystemsBase.jl calls for a
state (see [Attractors and their basins](@ref)).

## Static arrays

For a small system, a state of static arrays keeps the solver off the heap. With
`inplace = false`, `ODEFunction`, `SDEFunction`, `NonlinearFunction` and `DiscreteFunction`
hold out-of-place functions that return an `SVector` for the right side or the map and an
`SMatrix` for each Jacobian, which keeps the shape of a matrix also for one state or one
parameter, as the implicit solvers need. Each of these arrays holds at most 1024
elements, so that a Jacobian limits the system to 32 states, a map without Jacobian to
1024, and a larger system throws an `ArgumentError`:

```@example sciml
using StaticArrays: SVector

f = SciMLBase.ODEFunction(rhs; inplace = false)
problem = SciMLBase.ODEProblem(f, SVector(1.0, 0.0), (0.0, pi), SVector(4.0))
f.jac(problem.u0, problem.p, 0.0)
```

## Compilation for each new model

Each generated function has a type of its own, so with the default
`specialize = SciMLBase.FullSpecialize` of these constructors, OrdinaryDiffEq.jl compiles its solver again for
each new model, which takes a few seconds at the first solve. With
`specialize = SciMLBase.AutoSpecialize`, the solver wraps the in-place functions of a
`Vector{Float64}` state in function wrappers of one type, so the code that it compiled for
one model serves the next one at the cost of an indirect call per evaluation. A solver
that uses the Jacobian, such as `Rodas5P()`, reuses its code with the default `jac = true`;
an explicit solver, such as `Tsit5()`, reuses it only with `jac = false`. In one session
on Lotka-Volterra and then the Lorenz system, this specialization changed the first solve
of the second model as the table shows, and made each later solve 5 to 20 percent slower.

| first solve of the second model | `FullSpecialize` | `AutoSpecialize` |
|---|---|---|
| `Rodas5P()` | 2.55 s | 0.15 s |
| `Tsit5()` | 0.89 s | 0.08 s, with `jac = false` |

For the problem of a solver object of CasADi (`cvodes`, `kinsol`, `nlpsol`), see
[Solve the problem in Julia](@ref).
