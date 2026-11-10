# Call a Julia function from the model

[`CasADiCodegen.callback`](@ref CasADiCodegen.callback(::Any, ::AbstractString, ::Tuple, ::Tuple))
wraps a Julia function, such as an ephemeris, as a CasADi function that an MX graph
can call. Wrap a drag force and its Jacobian, call it in a model, and generate code:

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

The two tuples hold the size of each argument and of each result, an integer or
`(rows, columns)`. A function that returns static arrays gives a call without allocation.

## Give the derivatives

The keyword `jacobian` takes one of three kinds of value.

### A Julia function, for the first order

The function returns the Jacobian, or with several results or arguments a tuple of one
matrix per pair; any array with the elements in column-major order is accepted.

### A tuple of Julia functions, for higher orders

Each function returns the Jacobian of the one before it.

### An ADTypes.jl backend, for any order

A backend such as `AutoForwardDiff()` differentiates the function through
DifferentiationInterface.jl, which must be loaded with the backend package; it is
slower than hand-written derivatives.

## How the code reaches the function

The package expands the operations around the calls into scalar code that passes
static arrays to the Julia functions. The code stays matrix code (see
[Matrix code and functions without instructions](@ref)) with `expand = false`, when it
must be split (see [Large functions and the cache on disk](@ref)), and when a call is
inside a function that CasADi does not inline.

- A function that `build_function` makes for the session holds the Julia function
  itself, so a closure works.
- `generate_julia` writes each call by its full name and imports its package, so it
  rejects closures and a Jacobian from a backend; the package must be in the
  environment that loads the file.
- The functions must accept the number type of the call, such as dual numbers.
- Each wrapped Julia function stays in memory for the life of the session, so make
  callbacks, integrators and rootfinders once, not in a loop.
