# Trace a model written in Julia

A model that exists as generic Julia code, written for any subtype of `Real`, can
enter a CasADi graph without being rewritten. [`CasADiCodegen.Traced`](@ref) is a
real number that records each operation on it as a CasADi expression, as the `Num`
of Symbolics.jl does, so that calling the Julia function on traced values gives the
graph of the function, which CasADi then differentiates and CasADiCodegen.jl emits.

We trace the two-body acceleration, written with a static vector and `norm`, and
generate its value and its Jacobian:

```@example traced
using CasADiCodegen, LinearAlgebra, StaticArrays

acceleration(r, μ) = -μ * r / norm(r)^3          # generic Julia code

r = SX.sym("r", 3)
a = CasADiCodegen.untraced(acceleration(SVector{3}(CasADiCodegen.traced(r)), 398600.4418))
f = CasadiFunction("a", [r], [a, jacobian(a, r)])
g, g! = build_function(f; expression = Val{false})
value, J = g([7000.0, -1200.0, 300.0])
J
```

[`CasADiCodegen.traced`](@ref) turns a symbolic column vector into a vector of traced
values, and [`CasADiCodegen.untraced`](@ref) turns the result back into a column
vector of CasADi, on which `jacobian`, `hessian` and the other functions of CasADi
work. A plain number in the code becomes a constant of the graph, and a loop with a
fixed number of steps is unrolled into it.

## The function of a SciML problem

The function `f(du, u, p, t)` of a SciML problem, written in place for any number
type, becomes a CasADi function in one call. [`CasADiCodegen.traced_function`](@ref)
calls `f` on traced arrays with the shapes of the initial state and of the parameters,
so a state stored as a matrix keeps its shape inside `f`. The CasADi function
`rhs(u, p, t)` takes the state as the column vector `vec(u0)`, and its generated code
reads and writes a matrix in that order, so the problem may keep the matrix `u0`:

```@example traced
import SciMLBase

function diffusion!(du, u, p, t)
    n = size(u, 1)
    for j in axes(u, 2), i in 1:n
        left, right = u[max(i - 1, 1), j], u[min(i + 1, n), j]
        du[i, j] = p[1] * (left - 2u[i, j] + right) - p[2] * u[i, j]^2
    end
end
u0 = rand(4, 3)
original = SciMLBase.ODEProblem(diffusion!, u0, (0.0, 1.0), [0.5, 2.0])
rhs = CasADiCodegen.traced_function("diffusion", original)
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(rhs), u0, (0.0, 1.0), [0.5, 2.0])
nothing # hide
```

The problem then has the generated right side and its Jacobian, or with
`ODEFunction(rhs; sparse = true)` a sparse Jacobian (see
[Give a CasADi function to a SciML problem](@ref)). A `NonlinearProblem` gives a
function without the time, and a function out of place works as well. The parameters may
also be a tuple or a named tuple of numbers, and the keywords `states` and `parameters`
name the elements. [`CasADiCodegen.traced_problem`](@ref CasADiCodegen.traced_problem(::AbstractString, ::SciMLBase.ODEProblem))
makes the new problem in one call, with the initial state, the time span, the mass matrix
and the keywords of the original one; [From a Julia problem to generated code](@ref)
walks through it. In `diffusion!`,
`max` and `min` act on integer indices, not on traced values, so they need no
`ifelse` (see the next section).

## Choices that depend on the input

A comparison of traced values gives a traced condition, not a `Bool`. An `if`, a
`&&` or a `? :` on such a condition therefore throws a `TypeError`, instead of
following the branch that the values at the time of the trace select and
differentiating only that branch. A choice that depends on the input is written
with `ifelse`, `min`, `max` or `clamp`, which become the operations `if_else`,
`fmin` and `fmax` of CasADi and keep both branches in the graph:

```@example traced
x = SX.sym("x")
t = only(CasADiCodegen.traced(x))
thrust = CasADiCodegen.untraced(ifelse(t > 0, clamp(2t, 0, 1), 0.0))
h, _ = build_function(CasadiFunction("thrust", [x], [thrust]); expression = Val{false})
h([0.2]), h([0.8]), h([-1.0])
```

A Julia function with branches that cannot be written this way, or one that calls
code that accepts only `Float64`, enters the graph through
[`CasADiCodegen.callback`](@ref) instead (see [Call a Julia function from the model](@ref)).

## Matrix graphs

`traced` also takes a column vector of a matrix (MX) graph. The traced values are
then `Traced{MX}`, and the generic code can work on the results of a callback or of
any other MX expression, while the graph stays a matrix graph.
