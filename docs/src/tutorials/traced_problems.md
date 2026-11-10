# From a Julia problem to generated code

Tracing calls a plain-Julia SciML function once on numbers that record each operation,
which turns it into a CasADi graph; CasADiCodegen.jl then generates code for the function
and its analytic, sparse Jacobian. In this tutorial, we trace the Robertson problem, a
reaction-diffusion model, a pendulum with a mass matrix and a nonlinear system. The rules
for the traced number type and for branches are in [Trace a model written in Julia](@ref).

```@example traced_problems
using CasADiCodegen, LinearAlgebra, SparseArrays
import SciMLBase
using OrdinaryDiffEqRosenbrock: Rodas5P
using OrdinaryDiffEqBDF: FBDF
nothing # hide
```

## A problem written for OrdinaryDiffEq.jl

The Robertson equations are a standard stiff test problem:

```math
\begin{aligned}
y_1' &= -k_1 y_1 + k_3 y_2 y_3, \\
y_2' &= k_1 y_1 - k_2 y_2^2 - k_3 y_2 y_3, \\
y_3' &= k_2 y_2^2.
\end{aligned}
```

We write the right side for OrdinaryDiffEq.jl, with no reference to CasADi:

```@example traced_problems
function rober!(du, u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    du[1] = -k₁ * y₁ + k₃ * y₂ * y₃
    du[2] = k₁ * y₁ - k₂ * y₂^2 - k₃ * y₂ * y₃
    du[3] = k₂ * y₂^2
    nothing
end
original = SciMLBase.ODEProblem(rober!, [1.0, 0.0, 0.0], (0.0, 1e5), (0.04, 3e7, 1e4))
nothing # hide
```

## The traced problem

[`CasADiCodegen.traced_problem`](@ref CasADiCodegen.traced_problem(::AbstractString, ::SciMLBase.ODEProblem))
traces `rober!` and returns an `ODEProblem` whose
[`ODEFunction`](@ref SciMLBase.ODEFunction(::CasADiCodegen.Function)) holds generated code
for the right side and its Jacobian;
[`CasADiCodegen.traced_function`](@ref CasADiCodegen.traced_function(::AbstractString, ::Union{SciMLBase.ODEProblem,SciMLBase.NonlinearProblem}))
returns the CasADi function alone.

The traced problem keeps the initial state, the time span, the mass matrix and the keywords
of the original, such as `saveat` or `callback`, but not other fields of its `ODEFunction`,
such as a hand-written `jac`. The parameters become a vector in the order of the tuple, so
a kept callback must read them by place, `p[1]`, or by name; a tuple of parameters must hold
numbers, not arrays:

```@example traced_problems
problem = CasADiCodegen.traced_problem("rober", original)
problem.p
```

The solvers use this analytic Jacobian instead of automatic differentiation:

```@example traced_problems
problem.f.jac(problem.u0, problem.p, 0.0)
```

Rodas5P gives the same final state for the traced and the original problem, since both
Jacobians are exact:

```@example traced_problems
solution = SciMLBase.solve(problem, Rodas5P(); abstol = 1e-10, reltol = 1e-10)
reference = SciMLBase.solve(original, Rodas5P(); abstol = 1e-10, reltol = 1e-10)
solution.u[end], maximum(abs, solution.u[end] - reference.u[end])
```

A change of a parameter value needs no new trace, because the values enter the generated
code as arguments; a value that fixes the structure, such as a size or the number of steps
of a loop, is read once during the trace and needs a new trace when it changes.

## Names of the states and of the parameters

The traced problem can be indexed by name. By default the states are `u_0`, `u_1`, `u_2`
and the parameters `p_0`, `p_1`, `p_2`; the keywords `states` and `parameters` give one name
per element, in the order of `vec(u0)` and of the parameters:

```@example traced_problems
named = CasADiCodegen.traced_problem("rober", original; states = [:y₁, :y₂, :y₃], parameters = [:k₁, :k₂, :k₃])
solution = SciMLBase.solve(named, Rodas5P(); abstol = 1e-10, reltol = 1e-10)
solution[:y₂][end], SciMLBase.remake(named; p = [:k₁ => 0.08]).p
```

Without these keywords, the names come from the symbolic system of the original function,
such as a `SciMLBase.SymbolCache`, and the parameter names also from the keys of a named
tuple.

## A model with loops, views and a sparse Jacobian

A discretized partial differential equation has a large, sparse Jacobian. We take the
Gray-Scott equations in one dimension, written with a matrix state, views of its columns,
a loop for the diffusion and `@.` for the reaction:

```@example traced_problems
function grayscott!(du, u, p, t)
    Du, Dv, F, k, h = p
    U, V = @view(u[:, 1]), @view(u[:, 2])
    dU, dV = @view(du[:, 1]), @view(du[:, 2])
    n = size(u, 1)
    for i in 1:n
        l = i == 1 ? 2 : i - 1
        r = i == n ? n - 1 : i + 1
        dU[i] = Du * (U[l] - 2U[i] + U[r]) / h^2
        dV[i] = Dv * (V[l] - 2V[i] + V[r]) / h^2
    end
    @. dU += -U * V^2 + F * (1 - U)
    @. dV += U * V^2 - (F + k) * V
    nothing
end
n = 256
x = range(0.0, 1.0; length = n)
bump = @. exp(-200 * (x - 0.5)^2)
u0 = [1 .- 0.5 .* bump 0.25 .* bump]
original = SciMLBase.ODEProblem(grayscott!, u0, (0.0, 100.0), (1e-3, 5e-4, 0.04, 0.06, step(x)))
nothing # hide
```

The loop is unrolled during the trace, and the branches on integer indices are taken once.
A branch on a state, a parameter or the time throws a `TypeError` and must use `ifelse`
(see [Choices that depend on the input](@ref)). With `sparse = true`, which needs
SparseArrays loaded, the `ODEFunction` holds a `jac_prototype` with the pattern that CasADi
derives; each row has at most four nonzeros:

```@example traced_problems
problem = CasADiCodegen.traced_problem("grayscott", original; sparse = true)
size(problem.f.jac_prototype), nnz(problem.f.jac_prototype)
```

FBDF factorizes this sparse matrix. The traced problem keeps the matrix shape of the state:

```@example traced_problems
solution = SciMLBase.solve(problem, FBDF(); abstol = 1e-8, reltol = 1e-8, save_everystep = false)
solution.retcode, size(solution.u[end])
```

A graph of tens of thousands of instructions takes several seconds to compile at the first
call. The keywords `chunk_size`, `optlevel` and `cache` of
[`ODEFunction`](@ref SciMLBase.ODEFunction(::CasADiCodegen.Function)), which
`traced_problem` passes on, control that cost (see
[Large functions and the cache on disk](@ref)).

## A mass matrix

The traced problem keeps the mass matrix of a differential-algebraic equation. We take a
pendulum in Cartesian coordinates whose tension ``T`` is an algebraic state; the last
equation is the length constraint differentiated twice, since tracing records the equations
as written and does not reduce the index:

```@example traced_problems
function pendulum!(du, u, p, t)
    x, dx, y, dy, T = u
    g, L = p
    du[1] = dx
    du[2] = T * x
    du[3] = dy
    du[4] = T * y - g
    du[5] = 2 * (dx^2 + dy^2 + y * (y * T - g) + T * x^2)
    nothing
end
f = SciMLBase.ODEFunction(pendulum!; mass_matrix = Diagonal([1.0, 1, 1, 1, 0]))
original = SciMLBase.ODEProblem(f, [1.0, 0, 0, 0, 0], (0.0, 10.0), [9.8, 1.0])
problem = CasADiCodegen.traced_problem("pendulum", original; states = [:x, :dx, :y, :dy, :T])
solution = SciMLBase.solve(problem, Rodas5P(); abstol = 1e-8, reltol = 1e-8)
problem.f.mass_matrix == f.mass_matrix, solution[:x][end]^2 + solution[:y][end]^2
```

The initial state is consistent, since the last equation gives ``T = 0`` at rest at ``x = L``.

## Nonlinear problems

A `NonlinearProblem` traces in the same way and gets the analytic Jacobian of its residual.
Here the residual, the intersection of a circle with the line ``u_1 = u_2``, is written out
of place:

```@example traced_problems
using NonlinearSolve: NewtonRaphson

circle(u, p) = [u[1]^2 + u[2]^2 - p[1], u[1] - u[2]]
original = SciMLBase.NonlinearProblem(circle, [1.0, 0.5], [2.0])
problem = CasADiCodegen.traced_problem("circle", original)
SciMLBase.solve(problem, NewtonRaphson()).u
```

The generated functions are in place whatever the form of the original, unless the initial
state is a static vector, which gives functions that return static arrays.
