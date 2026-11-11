# Give a CasADi function to a SciML problem

With SciMLBase.jl loaded, a CasADi function becomes the function of a SciML problem,
with generated code for the function and for the derivatives that the solvers ask for.
The CasADi function takes the unknowns, then optionally the parameters, and for
`ODEFunction` then the time.

## Solve an ODE

Give the right side of a harmonic oscillator to `ODEFunction`:

```@example sciml
using CasADiCodegen
using OrdinaryDiffEqVerner: Vern9
import SciMLBase

u, k, t = SX.sym("u", 2), SX.sym("k"), SX.sym("t")
rhs = CasadiFunction("rhs", [u, k, t], [vcat(u[2], -k * u[1])])     # also f(u) and f(u, p)
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(rhs), [1.0, 0.0], (0.0, pi), [4.0])
SciMLBase.solve(problem, Vern9(); abstol = 1e-12, reltol = 1e-12).u[end]
```

## Other problems

Nonlinear problems follow the same pattern:

```@example sciml
using SimpleNonlinearSolve: SimpleNewtonRaphson

z, q = SX.sym("z", 2), SX.sym("q")
residual = CasadiFunction("r", [z, q], [vcat(z[1]^2 + z[2]^2 - q, z[1] - z[2])])   # also f(z)
problem = SciMLBase.NonlinearProblem(SciMLBase.NonlinearFunction(residual), [1.0, 0.5], [2.0])
SciMLBase.solve(problem, SimpleNewtonRaphson(); abstol = 1e-12).u
```

For an optimization problem, give the objective and, optionally, a vector of
constraints with the same arguments, whose bounds go to `lcons` and `ucons`. The problem
needs no `adtype`, and Ipopt uses the generated gradient, Jacobian and Hessians:

```@example sciml
v = SX.sym("v", 2)
objective = CasadiFunction("J", [v, q], [(v[1] - q)^2 + v[2]^2])     # also f(v)
constraints = CasadiFunction("c", [v, q], [v[1] + v[2]])
f = SciMLBase.OptimizationFunction(objective, constraints)
problem = SciMLBase.OptimizationProblem(f, [0.0, 0.0], [1.0]; lcons = [1.0], ucons = [Inf])

using OptimizationIpopt: IpoptOptimizer

ipopt = IpoptOptimizer(additional_options = Dict{String,Any}("print_level" => 0, "sb" => "yes"))
SciMLBase.solve(problem, ipopt).u
```

`OptimizationProblem(opti)` converts a problem of the `Opti` stack of CasADi, with its
initial values, parameter values and constraint bounds:

```@example sciml
opti = Opti()
w, a = variable(opti, 2), parameter(opti)
minimize(opti, (w[1] - a)^2 + w[2]^2)
subject_to(opti, w[1] + w[2] >= 1)
casadi.set_value(opti, a, 0.25)
problem = SciMLBase.OptimizationProblem(opti; sparse = true)
SciMLBase.solve(problem, ipopt).u
```

| Constructor | Generated functions |
|---|---|
| `ODEFunction(f; jac = true, paramjac = true, sparse = false, inplace = true, specialize = FullSpecialize)` | Right side, `jac`, and `paramjac` when `f` has parameters |
| `SDEFunction(f, g; jac = true, paramjac = true, sparse = false, inplace = true, specialize = FullSpecialize)` | Drift, noise `g`, `jac` and `paramjac` of the drift |
| `DiscreteFunction(f; inplace = true, specialize = FullSpecialize)` | Map to the next state |
| `ImplicitDiscreteFunction(f; inplace = true, specialize = FullSpecialize)` | Residual of the implicit map |
| `NonlinearFunction(f; jac = true, sparse = false, inplace = true, specialize = FullSpecialize)` | Residual and `jac` |
| `OptimizationFunction(objective, constraints; hess = true, sparse = false, lag_h = sparse && hess)` | Objective, `grad`, `hess`, `cons`, `cons_j`, and either `cons_h`, the Hessian of each constraint, or `lag_h`, the Hessian of the Lagrange function |

- `hess = false` leaves out `hess`, `cons_h` and `lag_h`, whose code can take long to
  build, for a first-order solver such as L-BFGS.
- `chunk_size`, `split_above`, `optlevel` and `cache` go to [`build_function`](@ref)
  (see [Large functions and the cache on disk](@ref)), and the other keywords to the SciML
  constructor. With `cache`, a Julia function that builds and then solves must call
  `solve` through `Base.invokelatest`.
- A function of the time without parameters takes an empty argument, `f(u, SX(0, 1), t)`.
- `ODEFunction(f; initialization = c)` adds conditions `c(u, p, t) = 0` on the initial
  values, and `solved_parameters` makes parameters unknowns of them; see
  [Initialize differential-algebraic equations](@ref).
- The noise `g` of `SDEFunction` takes the arguments of the drift and returns a column, or
  a matrix with a column per Wiener process; see
  [Add noise to a differential equation](@ref).
- The map `f(u, p, t)` of `DiscreteFunction` returns the next state, and the residual
  `f(u⁺, u, p, t)` of `ImplicitDiscreteFunction` takes the next and the current state; see
  [Discrete-time systems](@ref).
- A residual of `NonlinearFunction` of another size than its unknowns gets a
  `resid_prototype`, for a `NonlinearLeastSquaresProblem`.

For a Jacobian with few nonzeros, load SparseArrays and pass `sparse = true`, which
gives the function a `jac_prototype` with the sparsity pattern:

```@example sciml
using SparseArrays

y = SX.sym("y", 5)
chain = CasadiFunction("chain", [y], [vcat([y[i] - (i > 1 ? y[i-1]^2 : 0.0) for i in 1:5]...)])
SciMLBase.NonlinearFunction(chain; sparse = true).jac_prototype
```

Pass `sparse = true` to a large `OptimizationFunction` too, whose dense `cons_h` holds
an n×n matrix per constraint. `lag_h(h, x, σ, μ, p)` then replaces `cons_h` with the upper
triangle of the Hessian of the Lagrange function `σ f + μ' g`, unless `lag_h = false`:

```@example sciml
f = SciMLBase.OptimizationFunction(objective, constraints; sparse = true)
f.lag_hess_prototype
```

## Names and observed variables

All constructors except `OptimizationFunction` take the names of the symbols, so you can
index a solution and a problem by name; the elements of `SX.sym("u", 2)` are `u_0` and
`u_1` (for vector parameters see [Data as parameters](@ref)). Each result after the first
is an observed variable:

```@example sciml
x, v = SX.sym("x"), SX.sym("v")
oscillator = CasadiFunction("oscillator", [vcat(x, v), k, t], [vcat(v, -k * x), (v^2 + k * x^2) / 2],
                            ["u", "k", "t"], ["du", "energy"])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(oscillator), [1.0, 0.0], (0.0, pi), [4.0])
solution = SciMLBase.solve(problem, Vern9(); abstol = 1e-12, reltol = 1e-12)
solution[:x][end], solution[:energy][end], SciMLBase.remake(problem; p = [:k => 9.0]).p
```

A function whose arguments are not `vcat`s of named symbols, or that repeats a name,
gets no names, and a second result is then an error; `ImplicitDiscreteFunction` takes no
observed variables, and a `sys` given to the constructor replaces the names. The same names
serve SymbolicIndexingInterface.jl, as in `observe_state` of DynamicalSystemsBase.jl (see
[Attractors and their basins](@ref)).

## Static arrays

For a small system, pass `inplace = false` to `ODEFunction`, `SDEFunction`,
`NonlinearFunction` or `DiscreteFunction`, whose functions then return an `SVector` and an
`SMatrix` for each Jacobian. Each array holds at most 1024 elements, so a system with a
Jacobian has at most 32 states, and a larger one throws an `ArgumentError` (see
[Avoid allocation with static arrays](@ref)):

```@example sciml
using StaticArrays: SVector

f = SciMLBase.ODEFunction(rhs; inplace = false)
problem = SciMLBase.ODEProblem(f, SVector(1.0, 0.0), (0.0, pi), SVector(4.0))
f.jac(problem.u0, problem.p, 0.0)
```

## Compilation for each new model

With the default `specialize = SciMLBase.FullSpecialize`, OrdinaryDiffEq.jl compiles
its solver again for each new model. Pass `specialize = SciMLBase.AutoSpecialize` to reuse
it for a `Vector{Float64}` state; an explicit solver such as `Tsit5()` reuses it only with
`jac = false`. On Lotka-Volterra and then the Lorenz system, the first solve of the second
model took the times below, and each later solve was 5 to 20 percent slower:

| first solve of the second model | `FullSpecialize` | `AutoSpecialize` |
|---|---|---|
| `Rodas5P()` | 2.55 s | 0.15 s |
| `Tsit5()` | 0.89 s | 0.08 s, with `jac = false` |

For the problem of a solver object of CasADi (`cvodes`, `kinsol`, `nlpsol`), see
[Solve the problem in Julia](@ref).
