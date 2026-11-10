# Matrix code and functions without instructions

The package generates two kinds of code from a matrix (MX) function, expanded scalar
code and matrix code, and translates the functions that CasADi evaluates with its own
compiled code, such as interpolants, integrators and rootfinders, so that the
generated code never calls CasADi.

## Expanded code and matrix code

The default `expand = true` expands an MX function to scalar operations, which gives
the fastest code for a model of moderate size but repeats the body of a called
function at each call (see [Cost of the two kinds of code](@ref)). With
`expand = false` the code keeps the matrix operations and compiles each called
function once, except a function with more argument nonzeros than operations, such
as an `inline` interpolant with its table as an argument, which is inlined.

```julia
g, g! = build_function(f; expression = Val{false}, expand = false)   # f is an MX function
```

Matrix code reads constants, and arguments of an unsplit function, without a copy,
so a table lookup costs the same for any table size. When a function calls Julia
functions at its top level, `expand = true` expands the operations between the calls
(see [Call a Julia function from the model](@ref)); the code keeps the matrix
operations only when CasADi cannot expand those parts, or when a Julia function is
called inside a called function.

With `short_circuit = true`, `if_else(c, a, b, true)` and
`conditional(k, cases, default, true)` run only the selected branch in matrix code,
while expanded code computes every branch, so an expensive branch needs
`expand = false`. A branch that CasADi cannot expand, such as a call of a Julia
function, gives matrix code with either setting.

Matrix code writes a `map`, and a strided copy of nonzeros as in a transpose, as one
loop over blocks, and the nested functions of `mapaccum` and `fold` make code that
grows with the logarithm of the number of steps; expanded code repeats the body for
each column or step. An index computed by the model, as `v[i]` with a symbolic `i`,
and the interval search of an `inline` linear interpolant become ordinary Julia code,
and `monitor` prints its operand at each call.

## Functions and nodes of CasADi without instructions

CasADi does not expose the data of these functions through its interface. The
package reads it from the C code that CasADi generates (tables, index lists) and from
the output of `serialize` (solver options), emits Julia code that accepts dual
numbers, and at build time compares the results with those of CasADi at fixed test
points; a difference is an error, although agreement there does not prove agreement
at every input.

| Function or node of CasADi | Julia code |
|---|---|
| Interpolant `linear` or `bspline` with fixed values or values as an argument, for `linear` also the grid as an argument; also with `inline`, which lets CasADi differentiate by the values | The grid search and B-spline recursion of CasADi, with every `lookup_mode`: `linear` (scan), `exact` (equidistant grid), `binary` (bisection) |
| B-spline node; `einstein` (sum of tensor products over repeated indices); `kron_contract` (block products without forming the Kronecker product); `repsum` (sum of equal blocks); `find` (index of the first nonzero) | The loops of CasADi over tables read from its C code |
| `convexify`, which makes a symmetric matrix positive definite | The routines of CasADi 3.8.2 for the three strategies, statement by statement |
| `attachAssert`, `monitor`, `printme`, `dump`, `lift` (a mark for the lifted Newton method) | The check, print or file of CasADi; the value passes through |
| `map`, serial, on threads or with OpenMP | One loop over blocks of columns in one thread; a map of a Julia function is written out, one call per block |
| External function from a compiled library | A direct call of the library, `Float64` only |
| Rootfinder `newton` | The iteration of CasADi 3.8.2 with its line search and options |
| Integrators `rk` and `collocation` with forward and adjoint sensitivities and second derivatives | A loop over the step function of CasADi; a collocation step is a Newton rootfinder |

The interpolants below have 9 points and return the value with its derivative, the
integrators have 2 states and 20 steps and return the final state with its Jacobian,
and the rootfinder has 2 unknowns and returns the root with its Jacobian by 3
parameters. One call with `Float64`, measured once, against the CasADi virtual
machine (its interpreter on numbers):

| Function | Generated code | Memory per call | CasADi |
|---|---|---|---|
| Interpolant `linear` | 0.05 µs | 0 B | 4.7 µs |
| Interpolant `bspline` | 0.11 µs | 0 B | 4.9 µs |
| `convexify` of a 4 by 4 matrix, `eigen-clip` | 0.52 µs | 0 B | 4.3 µs |
| Rootfinder `newton` | 3.6 µs | 304 B | 49 µs |
| Integrator `rk` | 6.7 µs | 0 B | 262 µs |
| Integrator `collocation` | 112 µs | 11 kB | 1570 µs |

## Cost of the two kinds of code

A model that calls a scalar function of 710 instructions 20 times in a chain, and
returns the result with its Jacobian, shows the cost of expansion: expanded code
repeats the 710 instructions at each call, so it builds and compiles longer and here
also runs slower than matrix code, which compiles the called function once. Both are
faster than the CasADi virtual machine; one measurement of each:

| Form | Build | First call | Later calls |
|---|---|---|---|
| `expand = true` | 4.8 s | 86 s | 183 µs |
| `expand = false` | 0.3 s | 13 s | 134 µs |
| CasADi virtual machine | | | 950 µs |
