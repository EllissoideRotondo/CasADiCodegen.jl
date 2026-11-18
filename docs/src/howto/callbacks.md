# Call a Julia function from the model

[`CasADiCodegen.callback`](@ref CasADiCodegen.callback(::Any, ::AbstractString, ::Tuple, ::Tuple))
wraps a Julia function, such as an ephemeris, as a CasADi function that an MX graph
can call. Wrap a drag force and its Jacobian, call it in a model, and generate code:

```@example cb
using CasADiCodegen

drag(v) = -0.5 * sqrt(v[1]^2 + v[2]^2) .* v                  # any Julia code
function drag_jacobian(v)
    n = sqrt(v[1]^2 + v[2]^2)
    return -0.5 * (n * [1 0; 0 1] + v * v' / n)
end
force = callback(drag, "drag", (2,), (2,); jacobian = drag_jacobian)

v = MX.sym("v", 2)
drag_force = force(2v)
f = CasadiFunction("f", [v], [drag_force, jacobian(drag_force, v)])
g, g! = build_function(f; expression = Val{false})
g([3.0, 4.0])[2]
```

The two tuples hold the size of each argument and of each result, an integer or
`(rows, columns)`. A function that returns static arrays gives a call that does not
allocate.

## Give the derivatives of the Julia function

The keyword `jacobian` takes one of three kinds of value.

### Give the Jacobian as a Julia function

For the first derivatives, the function returns the Jacobian, or with several results or
arguments a tuple of one matrix per pair; any array with the elements in column-major
order is accepted.

### Give higher derivatives as a tuple of Julia functions

Each function returns the Jacobian of the one before it.

### Let an AD package compute the derivatives

A backend such as `AutoForwardDiff()` differentiates the function to any order through
DifferentiationInterface.jl, which must be loaded with the backend package; it is
slower than hand-written derivatives.

## Requirements on the Julia function

The package expands the operations around the calls into scalar code that passes
static arrays to the Julia functions. The code stays matrix code (see
[Scalar code, matrix code and special CasADi functions](@ref)) with `expand = false`, when
it
must be split (see [Speed up compilation of large models](@ref)), and when a call is
inside a function that CasADi does not inline.

- A function that `build_function` makes for the session holds the Julia function
  itself, so a closure works.
- `generate_julia` writes each call by its full name and imports its package, so it
  rejects closures and a Jacobian from a backend; the package must be in the
  environment that loads the file.
- The functions must accept the number type of the call, such as dual numbers.
- Each wrapped Julia function stays in memory for the life of the session, so make
  callbacks, integrators and rootfinders once, not in a loop.
