# Matrix code and functions without instructions

This page explains the two kinds of Julia code that the package generates from a
matrix (MX) function of CasADi, and how it generates code for the functions and
nodes of CasADi without instructions: an interpolant, an integrator, a rootfinder
and others, which CasADi evaluates with its own compiled code instead of a list of
instructions that the package could read. It is meant for a reader who
chooses between `expand = true` and `expand = false`, or who wants to know why a
model with an interpolant, an integrator or a rootfinder of CasADi gives Julia code
that does not call CasADi.

## Expanded code and matrix code

By default a matrix (MX) function is expanded to scalar operations, which gives the
fastest code for a model of moderate size but repeats the body of a called function
at each call, so a model with many calls of a large function compiles slowly and can
run slower (see [Cost of the two kinds of code](@ref)). With `expand = false` the
generated code keeps the matrix operations, and each called function becomes one
Julia function that is compiled once. With the default `expand = true`, the package
first asks CasADi to expand the whole function; when the function calls Julia
functions at its top level, it expands the operations before, between and after
those calls instead, which still gives scalar code (see
[Call a Julia function from the model](@ref)); and only when CasADi cannot expand
what lies between the calls, or a call of a Julia function is inside a called
function, does the code keep the matrix operations.

`if_else(c, a, b, true)` and `conditional(k, cases, default, true)` of CasADi, whose
last argument `short_circuit = true` asks for a switch between functions, make code
that needs only one branch. Expanded code computes each branch and selects one value,
as CasADi does when it expands, while code that keeps the matrix operations runs
only the branch that is selected. The package does not choose between the two from
the cost of a branch: when CasADi can expand every branch, the default
`expand = true` gives expanded code that computes all of them, so an expensive branch
needs `expand = false`. A branch that CasADi cannot expand, such as a call of a Julia
function, keeps the code as matrix code with either setting, because the switch calls each branch as a separate function and a call of a Julia function
inside a called function is the last case of the rule above.

```julia
g, g! = build_function(f; expression = Val{false}, expand = false)   # f is an MX function
```

Matrix code writes a `map` of CasADi, serial, on threads or with OpenMP, as one loop
that calls the mapped function on each block of columns in one thread, so that its
size does not grow with the number of columns; a copy of nonzeros that moves blocks
by a fixed stride, as a transpose does, is likewise one loop over the blocks.
Expanded code, in contrast, repeats the body of the mapped function for each column.
CasADi builds `mapaccum` and `fold` as nested functions, each of which calls the next
one a fixed number of times, so that their matrix code grows with the logarithm of
the number of steps.

Code that keeps the matrix operations also contains the operations that CasADi
cannot expand to scalar code. An index that is a value of the model, as in `v[i]`
with a symbolic `i`, and the search of an interval in a grid, which a linear
interpolant with the option `inline` uses, become ordinary Julia code, while
`monitor` prints its operand with its text at each call.

## Functions and nodes of CasADi without instructions

Some functions and nodes of CasADi have no list of instructions, and CasADi does not
give their data through its interface. The package reads that data from two texts
that CasADi can write for any function: the C code that CasADi generates, which
holds each table and each index list, and the text that `serialize` gives with the
names of the fields, which holds the options of a solver. From that data the package
emits Julia code that does not call CasADi and that accepts dual numbers. When it
builds the code, the package evaluates it at test points, which are fixed values of
the inputs that the package chooses, and compares the results with those of CasADi
within a small tolerance. A difference at these points is an error at build time;
the check finds a wrong translation at the test points, but it does not prove that
the code agrees with CasADi at every input.

| Function or node of CasADi | Julia code |
|---|---|
| Interpolant, `linear` or `bspline`, with fixed values or with the values as an argument, and for `linear` also with the grid as an argument; also with the option `inline`, with which CasADi differentiates them by the values | The search in the grid and the B-spline recursion of CasADi, with the three values of the option `lookup_mode`: `linear` (a scan of the grid), `exact` (a direct computation on an equidistant grid) and `binary` (a bisection) |
| B-spline node; `einstein`, a sum of products of tensor elements over repeated indices; the contracted Kronecker products (`kron_contract`), which multiply each block of a block matrix with a second operand without forming the Kronecker product; `repsum`, the sum of the equal blocks of a matrix; `find`, the index of the first nonzero element | The loops of CasADi over tables that are read from its C code |
| `convexify`, which makes a symmetric matrix positive definite | The routines of CasADi 3.8.1, statement by statement, for the three strategies |
| `attachAssert`, `monitor`, `printme`, `dump`, `lift` (which marks an expression for the lifted Newton method of CasADi) | The check, the print or the file of CasADi; the value passes |
| `map`, serial, on threads or with OpenMP | One loop over the blocks of columns, in one thread, that calls the mapped function; a map of a Julia function calls it once per block, written out |
| External function from a compiled library | A direct call of the library, with `Float64` only |
| Rootfinder `newton` | The iteration of CasADi 3.8.1 with its line search and its options |
| Integrators `rk` and `collocation`, with forward and adjoint sensitivities and with derivatives of the second order | The loop of steps of CasADi, as a loop over the step function of CasADi; a collocation step is a Newton rootfinder |

The table below gives the time of one call of the generated code with `Float64`,
and of the same function evaluated by CasADi through the front end, that is, by
the virtual machine of CasADi, its interpreter of a function on numbers, measured
once.
The interpolants have 9 points and return the value with its derivative, the
integrators have 2 states and 20 steps and return the final state with its
Jacobian, and the rootfinder has 2 unknowns and returns the root with its Jacobian
by 3 parameters.

| Function | Generated code | Memory per call | CasADi |
|---|---|---|---|
| Interpolant `linear` | 0.05 µs | 0 B | 4.7 µs |
| Interpolant `bspline` | 0.11 µs | 0 B | 4.9 µs |
| `convexify` of a 4 by 4 matrix, `eigen-clip` | 0.52 µs | 0 B | 4.3 µs |
| Rootfinder `newton` | 3.6 µs | 304 B | 49 µs |
| Integrator `rk` | 6.7 µs | 0 B | 262 µs |
| Integrator `collocation` | 112 µs | 11 kB | 1570 µs |

## Cost of the two kinds of code

The cost of expansion shows in a model that calls one scalar function of 710
instructions 20 times in a chain and returns the result with its Jacobian. Expanded
code repeats the 710 instructions at each of the 20 calls, so it takes longer to
build and to compile, and in this model it is also slower at later calls than the
matrix code, which compiles the called function once; both are faster than the
virtual machine of CasADi. The table gives one measurement of each form.

| Form | Build | First call | Later calls |
|---|---|---|---|
| `expand = true` | 4.8 s | 86 s | 183 µs |
| `expand = false` | 0.3 s | 13 s | 134 µs |
| CasADi virtual machine | | | 950 µs |
