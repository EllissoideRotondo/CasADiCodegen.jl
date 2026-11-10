# Trace a model written in Julia

[`CasADiCodegen.Traced`](@ref) is a real number that records each operation on it as a
CasADi expression, as the `Num` of Symbolics.jl does, so generic Julia code called on
traced values gives a CasADi graph. Trace the two-body acceleration and generate its
value and Jacobian:

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

[`CasADiCodegen.traced`](@ref) turns a symbolic column vector into traced values, and
[`CasADiCodegen.untraced`](@ref) turns the result back into a CasADi column vector. A
plain number becomes a constant of the graph, and a loop is unrolled.

## The function of a SciML problem

[`CasADiCodegen.traced_function`](@ref) traces `f(du, u, p, t)` of a SciML problem on
arrays with the shapes of `u0` and `p`; the CasADi function `rhs(u, p, t)` takes the
state as `vec(u0)`, so the problem may keep a matrix `u0`:

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

The problem has the generated right side and its Jacobian, sparse with
`ODEFunction(rhs; sparse = true)` (see [Give a CasADi function to a SciML problem](@ref)).
A `NonlinearProblem` and an out-of-place function work as well; the parameters may be
a tuple or a named tuple, and the keywords `states` and `parameters` name the elements.
[`CasADiCodegen.traced_problem`](@ref CasADiCodegen.traced_problem(::AbstractString, ::SciMLBase.ODEProblem))
makes the new problem in one call (see [From a Julia problem to generated code](@ref)).

## Choices that depend on the input

A comparison of traced values gives a traced condition, so an `if`, `&&` or `? :` on
it throws a `TypeError`. Write the choice with `ifelse`, `min`, `max` or `clamp`, which
keep both branches in the graph:

```@example traced
x = SX.sym("x")
t = only(CasADiCodegen.traced(x))
thrust = CasADiCodegen.untraced(ifelse(t > 0, clamp(2t, 0, 1), 0.0))
h, _ = build_function(CasadiFunction("thrust", [x], [thrust]); expression = Val{false})
h([0.2]), h([0.8]), h([-1.0])
```

Code that cannot be written this way, or that accepts only `Float64`, enters the graph
through a callback (see [Call a Julia function from the model](@ref)).

## Matrix graphs

`traced` also takes an MX column vector and gives `Traced{MX}` values, so the generic
code can work on the result of a callback and the graph stays a matrix graph.
