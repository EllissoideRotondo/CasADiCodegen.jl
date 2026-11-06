# Call a Julia function from the model

A model sometimes needs code that CasADi cannot see, such as an ephemeris, a table
or any other Julia function. [`CasADiCodegen.callback`](@ref CasADiCodegen.callback(::Any, ::AbstractString, ::Tuple, ::Tuple)) makes such a function a
CasADi function that a matrix (MX) graph can call, and the generated code then calls
the Julia function directly.

We wrap a drag force and its Jacobian, call it in a model, and generate code:

```@example cb
using CasADiCodegen

drag(v) = -0.5 * sqrt(v[1]^2 + v[2]^2) .* v                  # any Julia code
drag_jacobian(v) = (n = sqrt(v[1]^2 + v[2]^2); -0.5 * (n * [1 0; 0 1] + v * v' / n))
force = callback(drag, "drag", (2,), (2,); jacobian = drag_jacobian)

v = MX.sym("v", 2)
f = CasadiFunction("f", [v], [force(2v), jacobian(force(2v), v)])
g, g! = build_function(f; expression = Val{false})
g([3.0, 4.0])[2]
```

The two tuples hold one size for each argument and one for each result. A size is
an integer for a vector, or a tuple `(rows, columns)` for a matrix. The Julia
function receives one array for each argument, here the vector `v`, and returns one
array for each result. A function that returns static arrays, such as an `SVector`
of StaticArrays.jl, gives a call that allocates no memory.

## Give the derivatives

The keyword `jacobian` tells CasADi how to differentiate the function. It takes one
of three kinds of value.

### A Julia function, for the first order

The function takes the arguments of the callback and returns the Jacobian matrix,
with one row for each element of the result and one column for each element of the
argument. With several results or arguments, it returns a tuple with one matrix for
each pair of result and argument. Any array with the elements of the matrix in
column-major order is accepted, so a vector serves for a Jacobian of one column.

### A tuple of Julia functions, for higher orders

Each function returns the Jacobian of the one before it, so the tuple gives one order
of derivatives for each function. A matrix result counts as the vector of its
elements in column-major order.

### An ADTypes.jl backend, for any order

A backend such as `AutoForwardDiff()` differentiates the Julia function through
DifferentiationInterface.jl, to any order; DifferentiationInterface.jl and the
package of the backend must be loaded. A backend needs no code for the derivatives,
but it is slower than Julia functions that compute them.

## How the code reaches the function

CasADi cannot expand a call to a Julia function, but the package expands the
operations before, between and after the calls, so that the generated code is
scalar code in one function that passes static arrays to the Julia functions and
reads their results as static arrays or as ordinary arrays; a call whose results no
output needs is left out. The code stays matrix code, which keeps the matrix
operations of CasADi instead of scalar ones (see
[Matrix code and functions without instructions](@ref)), in three cases: with the
keyword `expand = false` of `build_function`, which turns the expansion off; when
the code is so large that it must be split (see
[Large functions and the cache on disk](@ref)); and when a call is inside a called
function that CasADi does not inline, that is, a function that the graph calls as a
separate function instead of copying its operations into the caller. Four rules
follow from how the code reaches the function.

- A function that `build_function` makes for the session holds the Julia function
  itself, so a closure works, and the data that the closure
  captures stay alive with it.
- `generate_julia` writes each call by the full name of the function and adds an
  `import` of its package to the module. It therefore needs named functions of a
  module, whose data are constants of that module; it rejects closures and a
  Jacobian from a backend, and the package must be in the environment that loads
  the file.
- The functions must accept the number type of the call, for example the dual
  numbers of ForwardDiff.jl when that package differentiates the generated code.
- The package keeps each Julia function that `callback`, `integrator` or
  `rootfinder` wraps in a table for the life of the session, because CasADi can call
  it at any later time. A program that makes such functions in a loop therefore uses
  more memory at each pass, so it must make them once and reuse them.
