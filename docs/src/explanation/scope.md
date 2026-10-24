# Scope and differences from CasADi

This page lists what the package supports, what it refuses when it builds the code,
and where the generated code or the front end behaves differently from CasADi or
from Julia. It is meant for a reader who checks whether a model fits the package
before writing it, or who meets an error at build time.

## What is supported

| Item | State |
|---|---|
| Scalar (SX) functions | Each scalar operation of CasADi 3.8.1, listed below |
| Matrix (MX) functions, not expanded | Elementwise operations, slices, joins, splits, transpose, reshape, projection (a change of the sparsity pattern of a matrix), assignment to chosen nonzeros of a matrix, reading of nonzeros at indices that the model computes, the search of an interval in a grid, `monitor`, calls to SX and MX functions, `map`, `mapaccum` and `fold`, `if_else` and `conditional` with `short_circuit = true`, which run only the selected branch |
| Functions and nodes of CasADi without instructions | Julia code for the interpolants `linear` and `bspline`, the B-spline node, `einstein`, `kron_contract`, `repsum`, `convexify`, `attachAssert`, `printme`, `dump`, `lift`, `find`, `map` (serial, on threads or with OpenMP) as one loop, external functions (`Float64` only), the rootfinder `newton`, and the integrators `rk` and `collocation` with sensitivities of the first and the second order |
| Solver objects of CasADi | As SciML problems: `ODEProblem`, `NonlinearProblem` and `OptimizationProblem`, also of an `Opti` stack. In generated code with `solvers = CasADiCodegen.Solvers(...)`: `cvodes`, `idas`, `kinsol` and `fast_newton`, with forward and reverse derivatives; the solvers of `nlpsol`, with `Float64` only and forward and reverse derivatives; the integrators with event functions, with or without algebraic states, with `Float64` only and forward derivatives only; the solvers of `conic` for quadratic problems |
| Linear algebra, not expanded | Matrix product, inner product, bilinear form, rank one update, Frobenius, 1 and infinity norms, smallest and largest element, Kronecker product, `logsumexp`, determinant, inverse, linear solve |
| Julia functions | `callback(f, name, sizes_in, sizes_out; jacobian)`, with derivatives of any order from Julia functions or from a backend |
| Julia models traced into CasADi | Generic Julia code on `Traced` numbers through `traced` and `untraced`; the function of an `ODEProblem` or a `NonlinearProblem`, in place or out of place, through `traced_function`; the whole problem, as a problem of generated code with an analytic Jacobian, through `traced_problem` |
| Functions of SciML problems | `ODEFunction`, `NonlinearFunction` and `OptimizationFunction` from CasADi functions, with generated derivatives, dense by default; the specialization with `specialize`; the keywords `chunk_size`, `split_above`, `optlevel` and `cache` of `build_function` |
| `ODEFunction` and `NonlinearFunction` | A sparse Jacobian with its `jac_prototype` with `sparse = true`; out-of-place functions of static arrays with `inplace = false`; in place, Jacobians that also return a new matrix, as `jac(u, p, t)`; the names of the symbols as `sys`, and further results as observed variables |
| `OptimizationFunction` and `OptimizationProblem(S::nlpsol)` | No Hessian with `hess = false`; with `sparse = true`, sparse `hess`, `cons_j` and `cons_h` with their prototypes, and the Hessian of the Lagrange function `lag_h` with `lag_hess_prototype`; `grad`, `hess`, `cons`, `cons_j` and `cons_h` also out of place; Optimization.jl sees the derivatives as declared, with no automatic differentiation and no warning |
| Models of a `DaeBuilder` | `ODEProblem(dae, tspan)`, with the start values, the names of scalar variables, algebraic states as a mass matrix, and inputs as constant parameters; no quadratures and no `when` equations. `der(dae, x)` declares a differential state |
| Linearization, order of equations and index reduction | `linearization(f, h)` and `linearization(f, g, h)`: generated code for the matrices `A`, `B`, `C` and `D` of `x' = f(x, u, p)`, `y = h(x, u, p)`, also with the algebraic states of `0 = g(x, z, u, p)`, dense or with `sparse = true`; `block_triangular_form(sparsity)`: the permutations and blocks of `Sparsity::btf`; `reduce_index(dae)`: the index-1 form of a semi-explicit DAE and its invariants by the structural algorithm of Pantelides, with Baumgarte's stabilization as an option, in place of `dae_reduce_index` and `dae_map_semi_expl` of CasADi, which the front end does not wrap. The reduced model keeps every state and enforces only the highest derivative of each constraint, so the invariants drift unless they are stabilized |
| Roots of equations | `rootfinder(name, alg, f)`; dense unknowns and arguments, derivatives of any order; static arrays up to 16 unknowns and vectors beyond |
| Ordinary differential equations | `integrator(name, alg, dae, t0, tf)`, with free times `integrator(name, alg, dae)`, and with a vector of times for `tf`; quadratures, forward sensitivities, adjoint sensitivities with `adjoint = true`, and with free times one or more events of one or several event functions, each with a transition of the state, a fixed number of events or at most a number with their count; algebraic states with `mass`, with forward and adjoint sensitivities and with events, but not with a vector of times |
| Discontinuous right sides | A warning, or `ArgumentError` with `nonsmooth = :error`, for a switch on the state, or on the time and a parameter together, in the integrators of this package, of CasADi and their replacements. Switches as events with `modes = m => S` and a bound `switches`, at fixed times, free times and a vector of times, with the switch times as results. A call throws for a sliding arc and for two switches in one step of the event search (see [Discontinuous models](@ref discontinuities)) |
| Smoothing of switches | `smooth_step`, `smooth_sign`, `smooth_abs`, `smooth_max` and `smooth_min` on numbers, SX and MX. `smoothed` of an SX or MX expression or function, also through calls, `map` and `conditional`; it keeps equality tests, `floor`, `ceil`, `fmod`, `remainder`, the index of `conditional`, a `linear` interpolant, and Julia callbacks, external functions and solvers. An SX function that calls another function is refused; its MX form works. `natural_continuation` in the width |
| Rules for automatic differentiation | `CasADiCodegen.differentiable(f)`, with a forward and a reverse rule for ChainRulesCore.jl |
| Julia functions on symbols | Methods that the front end leaves undefined: `inv` as the matrix inverse, `min`, `max`, `ifelse`, `atan(y, x)`, `hypot`, `rem`, `exp2`, `!=`, `&`, `LinearAlgebra.dot`, `LinearAlgebra.norm(x, p)` of a symbolic matrix and of a Julia array of symbols, as the norm of its elements for every `p`, `LinearAlgebra.opnorm` with `p = 1` and `p = Inf`, and with `p = 2` of a row or a column, `SpecialFunctions.erf`, `zero(x)` as the zero matrix of the shape of `x` and `one(x)` as the identity of a square `x`, and `bounded(lb, x, ub)` with Julia arrays as bounds |
| Julia array code on symbols | `sum`, with `dims` and `init`, `maximum`, `minimum` and iteration over the elements of a symbolic matrix, and over the numbers of a `DM`; concatenation that starts with a CasADi matrix and goes on with Julia arrays of numbers and symbols, so that `[x; v]` is one column (`[v; x]` keeps the Julia meaning, a vector, and `[v...; x]` gives the column), while `[x, y]` with commas is a list of matrices; broadcasting, in place too, that mixes symbolic matrices, Julia arrays of symbols and numbers, with a function that CasADi does not apply elementwise applied to each element; `+`, `-`, `*` and `\` with Julia arrays; indexing by a vector of indices; numbers promote to symbols in `[x, 1.0]`; a concatenation of `SX` and `MX` throws `ArgumentError`, because CasADi cannot combine the two in one expression. A chained comparison `lb <= x <= ub` cannot work, because Julia evaluates it as `lb <= x && x <= ub` and `&&` takes a `Bool` only |
| Forms of a generated function | Out of place and in place on `Array`, `SArray` and `MArray` in matrix shape; in place on `SparseMatrixCSC`; in place on vectors of nonzeros with `nonzeros = true` |
| Number types | `Float64`, `Float32` and ForwardDiff dual numbers; any real type with the keyword `nanmath = false` of `build_function` and `generate_julia` |
| Platforms | Linux, where the package is tested; macOS is not tested. The package does not load on Windows, because the Windows build of CasADiSWIG_jll 3.8.1 does not export the functions that the front end needs; [CasADiNative.jl](https://github.com/casadi/CasADiNative.jl), the official Julia package of CasADi, wraps the same front end and does not support Windows either |

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

Generated code never calls CasADi, so a function of CasADi for which the package has
no method is refused when the code is built. The message names the function and
says what the user can do, as the table below shows.

| Function of CasADi in the graph | What the user can do |
|---|---|
| Integrator `cvodes` or `idas` | Give a solver of the SciML ecosystem with the option `solvers` (see [Replace the solvers of CasADi](@ref)) |
| Rootfinder `kinsol` or `fast_newton` | Give a solver with the option `solvers` |
| Solver of `nlpsol` | Give a solver with the option `solvers` |
| Integrator `rk` or `collocation` with an event function | Give a solver with an override, `Solvers(overrides = ...)` |
| Callback of a subtype of `CasadiCallback`, the abstract type of the front end for a callback written as a Julia type, whose Julia object CasADi keeps to itself | Make it with `callback` and a Julia function |
| Solver of `conic` (quadratic problem) | Give a solver with the option `solvers`, field `optimization`; CasADi itself has no derivatives of such a solver |
| Solver of `conic` with cone constraints; other classes, such as a Functional Mock-up Unit (FMU) | Nothing: they are not supported |
| Matrix 2-norm without expansion | Nothing: CasADi itself cannot evaluate it |

## Differences from CasADi

### Numbers in a Dict of options

!!! warning "Integer options before version 0.26.0"
    The Julia front end of CasADi 3.8.1 (CasADiSWIG_jll) gives an integer, or a
    `Float64` with a whole value, in a Dict of options to CasADi as `true`, that is
    1. Before version 0.26.0, CasADiCodegen.jl passed these values unchanged, so
    `Dict("number_of_finite_elements" => 7)` made an integrator with 1 step, and
    `"max_iter" => 50` allowed 1 iteration, in a nested Dict too. CasADi gave no
    error, because it accepts `true` for an integer or a real option. The values 0
    and 1 were right by chance; any other whole number, negative too, was 1. A model
    with such an option gives other numbers from version 0.26.0 on, so any result
    that came from it must be computed again. The code of the model needs no change,
    except in the two cases that the end of this section names.

The cause is in the Julia front end of CasADi 3.8.1, which converts a number to the
boolean `true` before it tries an integer or a double; a vector of such numbers becomes
a vector of `bool`, which CasADi refuses with the error "type mismatch", so a vector
gave an error and no wrong result. The defect is reported as
[casadi/casadi#4445](https://github.com/casadi/casadi/issues/4445), for CasADi 3.8.2.
CasADiCodegen.jl converts the Dict before each call of the front end that takes
options, so that an integer reaches CasADi as an integer and any other real number as
a double, also in a vector and in a nested Dict.

Two cases keep the defect: a value in a `Vector{Any}` or in another container that is
not a Dict, and the module of CasADiSWIG_jll used directly, without CasADiCodegen.jl.
The first case is avoided with a vector of a concrete element type, such as `[1, 2]`
or `Float64[1, 2]`, and with nested options given as a Dict. The function
`change_option(f, name, value)` is not affected: it takes a `GenericType` only, and the
front end has no way to make one from a number, so it cannot set a numeric option.

### Other differences

- The `newton` rootfinder does not read the option `constraints`, because the Newton
  method of CasADi does not use it. It solves each linear system by elimination with
  row exchanges, dense or, for a Jacobian of 16 rows or more with at most a quarter of
  its elements stored, sparse LU in the order of the unknowns, where CasADi uses a QR
  factorization, so the roots agree to rounding error only.
- An operation that leaves its domain gives `NaN`, through NaNMath.jl, as in CasADi;
  `nanmath = false` selects the functions of Base, which throw. `erfinv` is the
  accurate `SpecialFunctions.erfinv`, where that of CasADi has a relative error up
  to 1e-4.
- Determinant and inverse use a dense LU factorization, where CasADi uses a sparse QR
  factorization, so the results agree to rounding error only, and a singular matrix
  gives `Inf` or `NaN` and no error. A linear solve does too, except that a solve with
  a sparse matrix of 8 rows or more uses the sparse QR factorization of CasADi when its factors have
  fewer nonzeros than half of the full matrix; for a tridiagonal matrix, measured once:

  | Rows | Sparse QR | Dense LU | CasADi virtual machine |
  |---|---|---|---|
  | 50 | 1.9 µs | 10.3 µs | 6.6 µs |
  | 200 | 7.8 µs | 143.7 µs | 12.3 µs |
- CasADi 3.8.1 computes a wrong value, read from other memory, for one node: the sum
  into the nonzeros at a fixed slice plus indices that are values of the model. This
  node is the reverse derivative of a read such as `get_nz(v, false, Slice(1, 5, 2), i)`,
  in which the `Slice` of CasADi counts from 0 and so selects the second and the fourth
  nonzero. The generated code gives the correct sum, so its result differs there from
  that of the virtual machine of CasADi, the interpreter with which CasADi evaluates a
  function on numbers.
- CasADi 3.8.1 gives an integrator with an event function the sparsity pattern of the
  model without events, so its Jacobian loses the effect of the transitions, with no
  error, in the virtual machine of CasADi too. In this package, `jacobian(ex, arg)` of
  an expression that calls such an integrator gives the dense Jacobian of one forward
  direction per element of `arg`, and `CasADiCodegen.C.jacobian(F)` of a function `F`
  that holds one throws an error that names `jtimes`.
- CasADi 3.8.1 misses an event of a `cvodes` integrator that falls exactly on one of its
  output times, so the state does not jump there or later. The code that replaces such an
  integrator (see [Integrators with events](@ref)) applies the event and returns the state
  after it at that output time, which is the solution of the model; with an event
  function `10 - t`, a jump of 50 and the output times 5, 10, 10.5 and 15, the two
  results differ by up to 50 after `t = 10`, while with the event at 10.25 they agree.
- The front end of CasADi 3.8.1 calls the const overload of `DaeBuilder::der` for
  `der(dae, x)`, which throws "Variable x has no time derivative" for a variable without
  a derivative. In this package, `der(dae, x)` calls the overload that Python and C++
  call, which makes the derivative, so that `eq(dae, der(dae, x), rhs)` declares the
  differential state `x`.
- `inv(A)` is the matrix inverse for SX and for MX, as in Julia, and `inv(A, "qr")`
  names the linear solver of an MX graph. The function `CasADiCodegen.C.inv(A)` of the
  front end of CasADi 3.8.1 is the reciprocal of each element for an MX matrix.
