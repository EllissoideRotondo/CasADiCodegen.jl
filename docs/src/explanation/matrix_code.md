# Matrix code and functions without instructions

The package generates two kinds of code from a matrix (MX) function, expanded scalar
code and matrix code. It also translates the functions that CasADi runs as its own
compiled code, such as interpolants, integrators and rootfinders, so that the generated
code never calls CasADi.

## Expanded code and matrix code

The default `expand = true` expands an MX function to scalar operations. This gives the
fastest code for a model of moderate size, but repeats the body of a called function at
each call (see [Cost of the two kinds of code](@ref)). When the repetition would give
more than 20000 scalar operations and more than 10 times the operations of the functions
involved, each counted once, the function keeps its matrix operations: an integrator or a
`fold` of 1000 steps calls its step rather than writing it 1000 times. A `map` counts as
its copies, since the code does that work anyway, so it is still expanded.

With `expand = false` the code keeps the matrix operations, compiles each called function
once, and shares the code of functions with the same content. A function that calls no
other function and has more argument nonzeros than operations, such as an `inline`
interpolant with its table as an argument, is inlined.

```julia
g, g! = build_function(f; expression = Val{false}, expand = false)   # f is an MX function
```

Matrix code reads constants, and arguments of an unsplit function, without a copy, so a
table lookup costs the same for any table size. When a function calls Julia functions at
its top level, `expand = true` expands the operations between the calls (see
[Call a Julia function from the model](@ref)); the code keeps the matrix operations only
when CasADi cannot expand those parts, or when a called function calls a Julia function.

Expanded code computes every branch of `if_else` and `conditional`, even with
`short_circuit = true`, so an expensive branch needs `expand = false`. A branch that
CasADi cannot expand, such as a call of a Julia function, gives matrix code with either
setting.

Matrix code writes a `map`, and a strided copy of nonzeros as in a transpose, as one loop
over blocks. The nested functions of `mapaccum` and `fold` give code that grows with the
logarithm of the number of steps, where expanded code repeats the body for each column or
step.

## Functions that CasADi runs as compiled code

CasADi does not expose the data of these functions through its interface. The package
reads it from the C code that CasADi generates (tables, index lists) and from the output
of `serialize` (solver options), and emits Julia code that accepts dual numbers. At build
time it compares the results with those of CasADi at fixed test points; a difference is
an error, although agreement there does not prove agreement at every input.

| Function or node of CasADi | Julia code |
|---|---|
| Interpolant `linear` or `bspline` with fixed values or values as an argument, for `linear` also the grid as an argument; also with `inline`, which lets CasADi differentiate by the values | The grid search and B-spline recursion of CasADi, with every `lookup_mode`: `linear` (scan), `exact` (equidistant grid), `binary` (bisection) |
| B-spline node; `einstein` (sum of tensor products over repeated indices); `kron_contract` (block products without forming the Kronecker product); `repsum` (sum of equal blocks); `find` (index of the first nonzero) | The loops of CasADi over tables read from its C code |
| `convexify`, which makes a symmetric matrix positive definite | The routines of CasADi 3.8.2 for the three strategies, statement by statement |
| `attachAssert`, `monitor`, `printme`, `dump`, `lift` (a mark for the lifted Newton method) | The check, print or file of CasADi; the value passes through |
| `map`, serial, on threads or with OpenMP | One loop over blocks of columns in one thread; a map of a Julia function is written out, one call per block |
| External function from a compiled library | A direct call of the library, `Float64` only |
| Rootfinder `newton` | The iteration of CasADi 3.8.2 with its line search and options; as in CasADi, a residual of NaN only counts as converged |
| Integrators `rk` and `collocation` with forward and adjoint sensitivities and second derivatives | A loop over the step function of CasADi; a collocation step is a Newton rootfinder |

One call with `Float64`, measured once, against the CasADi virtual machine (its
interpreter on numbers). The interpolants have 9 points and return the value with its
derivative; the integrators have 2 states and 20 steps and return the final state with
its Jacobian; the rootfinder has 2 unknowns and returns the root with its Jacobian by 3
parameters:

| Function | Generated code | Memory per call | CasADi |
|---|---|---|---|
| Interpolant `linear` | 0.05 µs | 0 B | 4.7 µs |
| Interpolant `bspline` | 0.11 µs | 0 B | 4.9 µs |
| `convexify` of a 4 by 4 matrix, `eigen-clip` | 0.52 µs | 0 B | 4.3 µs |
| Rootfinder `newton` | 3.6 µs | 304 B | 49 µs |
| Integrator `rk` | 6.7 µs | 0 B | 262 µs |
| Integrator `collocation` | 112 µs | 11 kB | 1570 µs |
