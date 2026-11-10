# Scope and differences from CasADi

This page lists what the package supports, what it refuses at build time, and where
its results differ from those of CasADi.

## What is supported

| Item | State |
|---|---|
| Scalar (SX) functions | Every scalar operation of CasADi 3.8.2 (below) |
| Matrix (MX) functions, not expanded | Elementwise operations, slices, joins, splits, transpose, reshape, projection (change of sparsity pattern), assignment to chosen nonzeros, reads at computed indices, interval search in a grid, `monitor`, calls of SX and MX functions, `map`, `mapaccum`, `fold`; `if_else` and `conditional` with `short_circuit = true` run only the selected branch |
| Functions and nodes of CasADi without instructions | Interpolants `linear` and `bspline`, the B-spline node, `einstein`, `kron_contract`, `repsum`, `convexify`, `attachAssert`, `printme`, `dump`, `lift`, `find`, `map` (serial, threads or OpenMP) as one loop, external functions (`Float64` only), rootfinder `newton`, integrators `rk` and `collocation` with first- and second-order sensitivities |
| Solver objects of CasADi | As SciML problems: `ODEProblem`, `NonlinearProblem`, `OptimizationProblem` (also of an `Opti` stack). In generated code through `solvers = CasADiCodegen.Solvers(...)`: `cvodes`, `idas`, `kinsol`, `fast_newton` with forward and reverse derivatives; `nlpsol` solvers, `Float64` only, forward and reverse; integrators with event functions, also with algebraic states, `Float64` only, forward only; `conic` solvers of quadratic problems |
| Linear algebra, not expanded | Matrix product, inner product, bilinear form, rank-one update, Frobenius, 1- and infinity norms, smallest and largest element, Kronecker product, `logsumexp`, determinant, inverse, linear solve |
| Julia functions | `callback(f, name, sizes_in, sizes_out; jacobian)`, derivatives of any order from Julia functions or a backend |
| Julia models traced into CasADi | `traced` and `untraced` on `Traced` numbers; `traced_function` for the function of an `ODEProblem` or `NonlinearProblem`, in or out of place; `traced_problem` for the whole problem with an analytic Jacobian |
| Functions of SciML problems | `ODEFunction`, `NonlinearFunction`, `OptimizationFunction` with generated derivatives, dense by default; `specialize`; the `build_function` keywords `chunk_size`, `split_above`, `optlevel`, `cache` |
| `ODEFunction` and `NonlinearFunction` | `sparse = true`: sparse Jacobian with `jac_prototype`; `inplace = false`: static arrays; in-place Jacobians also as `jac(u, p, t)`; symbol names as `sys`, further results as observed variables |
| `OptimizationFunction` and `OptimizationProblem(S::nlpsol)` | `hess = false`: no Hessian; `sparse = true`: sparse `hess`, `cons_j`, `cons_h` and Lagrangian Hessian `lag_h`, with prototypes; `grad`, `hess`, `cons`, `cons_j`, `cons_h` also out of place. Optimization.jl takes the derivatives as declared, without automatic differentiation or warning |
| Models written as equations | `Differential(t)` and `lhs ~ rhs` of `CasADiCodegen.Models` on SX. `System` eliminates algebraic equations linear in an unknown whose coefficient holds only numbers and parameters, makes states of lower derivatives, solves for the highest derivatives where they enter linearly (otherwise keeps them algebraic), then applies `reduce_index`; `baumgarte_pole` stabilizes the differentiated constraints. `Dict(sys)` serves the CasADi integrators; `ODEProblem(sys, u0, tspan, p)` carries names and observed variables. Not supported: MX, connector types (connections are equations in Julia), rootfinding of nonlinear algebraic systems (they stay algebraic), dummy derivatives, events, consistent initial differential states (the user gives them; `BrownFullBasicInit` of OrdinaryDiffEqNonlinearSolve.jl corrects the algebraic states) |
| Models of a `DaeBuilder` | `ODEProblem(dae, tspan)` with start values, scalar variable names, algebraic states as a mass matrix, inputs as constant parameters; `der(dae, x)` declares a differential state. No quadratures, no `when` equations |
| Linearization, order of equations, index reduction | `linearization(f, h)`, `linearization(f, g, h)`: `A`, `B`, `C`, `D` of `x' = f(x, u, p)`, `y = h(x, u, p)`, optionally with algebraic states of `0 = g(x, z, u, p)`, dense or `sparse = true`. `block_triangular_form(sparsity)`: permutations and blocks of `Sparsity::btf`. `reduce_index(dae)`: index-1 form of a semi-explicit DAE and its invariants by the Pantelides algorithm, optionally with Baumgarte stabilization, in place of `dae_reduce_index` and `dae_map_semi_expl`, which the front end does not wrap; it enforces only the highest derivative of each constraint, so the invariants drift unless stabilized |
| Roots of equations | `rootfinder(name, alg, f)`, dense unknowns and arguments, derivatives of any order; static arrays up to 16 unknowns, vectors beyond |
| Ordinary differential equations | `integrator(name, alg, dae, t0, tf)`, with free times as `integrator(name, alg, dae)` or with a vector `tf`; quadratures, forward sensitivities, adjoint sensitivities with `adjoint = true`. With free times, events of one or several event functions, each with a state transition and a fixed or bounded count. Algebraic states with `mass`, with sensitivities and events but not with a vector of times |
| Discontinuous right sides | A warning, or `ArgumentError` with `nonsmooth = :error`, for a switch on the state, or on time and a parameter together, in the integrators of this package and in code that replaces a CasADi integrator. Switches as events with `modes = m => S` and a bound `switches`, at fixed, free or vector times, with the switch times as results. A sliding arc or two switches in one step of the event search throws (see [Discontinuous models](@ref discontinuities)) |
| Smoothing of switches | `smooth_step`, `smooth_sign`, `smooth_abs`, `smooth_max`, `smooth_min` on numbers, SX and MX. `smoothed` of an SX or MX expression or function, also through calls, `map` and `conditional`; it leaves equality tests, `floor`, `ceil`, `fmod`, `remainder`, the index of `conditional`, `linear` interpolants, Julia callbacks, external functions and solvers unchanged. An SX function that calls another function is refused; its MX form works. `natural_continuation` in the width |
| Rules for automatic differentiation | `CasADiCodegen.differentiable(f)`: forward and reverse rules for ChainRulesCore.jl |
| Julia functions on symbols | `inv(A, lsolver)` with a CasADi linear solver, `log2`, `exp2`, `abs2`, `SpecialFunctions.erf`, `SpecialFunctions.erfinv`; `LinearAlgebra.norm(x, p)` of a symbolic matrix or an array of symbols, elementwise, for every `p`; `LinearAlgebra.opnorm` with `p = 1`, `p = Inf`, and `p = 2` of a row or column |
| Julia array code on symbols | `sum` with `dims` and `init`; iteration over a symbolic matrix and over a `DM`; concatenation with arrays of numbers and symbols (`[x; v]` is one column, `[x, y]` a list of matrices); broadcasting, also in place, over symbolic matrices, arrays of symbols and numbers of different shapes; `+`, `-`, `*`, `\` and `LinearAlgebra.dot` with arrays of symbols; numbers promote in `[x, 1.0]`. Concatenating `SX` with `MX` throws `ArgumentError`. `lb <= x <= ub` fails because `&&` takes a `Bool`; `bounded(lb, x, ub)` gives the constraint |
| Forms of a generated function | Out of place and in place on `Array`, `SArray`, `MArray` in matrix shape; in place on `SparseMatrixCSC`; in place on vectors of nonzeros with `nonzeros = true` |
| Number types | `Float64`, `Float32`, ForwardDiff dual numbers; any real type with `nanmath = false` of `build_function` and `generate_julia` |
| Platforms | Tested on Linux; macOS and Windows, where CasADiNative.jl loads, untested |

The scalar operations are:

| Kind | Operations |
|---|---|
| Arithmetic | `+`, `-`, `*`, `/`, negation, reciprocal, square, twice (the product `2x`), `^` with any exponent, `sqrt`, `abs`, `sign`, `copysign`, `hypot` |
| Exponential | `exp`, `log`, `expm1`, `log1p` |
| Trigonometric | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` |
| Rounding | `floor`, `ceil`, `fmod`, `remainder` |
| Comparison and logic | `<`, `<=`, `==`, `!=`, `not`, `and`, `or`, `if_else_zero`, `fmin`, `fmax` |
| Special | `erf`, `erfinv`, `printme` |

## What is refused

Generated code never calls CasADi, so a CasADi function without a method in the
package is refused at build time with a message that names the function and the remedy.

| Function of CasADi in the graph | Remedy |
|---|---|
| Integrator `cvodes` or `idas` | A SciML solver through the option `solvers` (see [Replace the solvers of CasADi](@ref)) |
| Rootfinder `kinsol` or `fast_newton` | A solver through `solvers` |
| Solver of `nlpsol` | A solver through `solvers` |
| Integrator `rk` or `collocation` with an event function | A solver through `Solvers(overrides = ...)` |
| Subtype of `Callback` (a callback written as a Julia type, whose object CasADi keeps to itself) | `callback` with a Julia function |
| Solver of `conic` (quadratic problem) | A solver through `solvers`, field `optimization`; CasADi has no derivatives of it either |
| `conic` with cone constraints; other classes such as a Functional Mock-up Unit (FMU) | None |
| Matrix 2-norm without expansion | None: CasADi cannot evaluate it either |

## Differences from CasADi

- The `newton` rootfinder ignores the option `constraints`, which the Newton method of
  CasADi does not use either. It solves each linear system by Gaussian elimination
  with row exchanges, or by sparse LU for a Jacobian of 16 rows or more with at most a
  quarter of its elements stored, where CasADi uses QR; the roots agree to rounding
  error.
- An operation outside its domain gives `NaN` through NaNMath.jl, as in CasADi;
  `nanmath = false` selects the Base functions, which throw. `erfinv` is
  `SpecialFunctions.erfinv`, whereas that of CasADi has a relative error up to 1e-4.
- Determinant, inverse and linear solve use dense LU where CasADi uses sparse QR, so
  results agree to rounding error and a singular matrix gives `Inf` or `NaN` without
  an error. A linear solve with a sparse matrix of 8 rows or more uses the sparse QR of
  CasADi when its factors have fewer nonzeros than half the full matrix; for a
  tridiagonal matrix, measured once:

  | Rows | Sparse QR | Dense LU | CasADi virtual machine |
  |---|---|---|---|
  | 50 | 1.9 µs | 10.3 µs | 6.6 µs |
  | 200 | 7.8 µs | 143.7 µs | 12.3 µs |
- CasADi 3.8.2 returns a wrong value, read from other memory, for the sum into nonzeros
  at a fixed slice plus computed indices, the reverse derivative of a read such as
  `get_nz(v, false, Slice(1, 5, 2), i)`; the generated code gives the correct sum.
- CasADi 3.8.2 misses a `cvodes` event that falls on an output time, so the state does
  not jump there or later. The replacing code (see [Integrators with events](@ref))
  applies the event and returns the state after it at that output time.
