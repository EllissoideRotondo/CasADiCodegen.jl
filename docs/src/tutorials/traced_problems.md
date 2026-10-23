# From a Julia problem to generated code

Many models exist already as SciML problems written in plain Julia: a function
`f!(du, u, p, t)` with loops, views and broadcasting, and an `ODEProblem` around it. An
implicit solver needs the Jacobian of `f!` at each factorization, and without one it
computes it by automatic differentiation or by finite differences, which evaluate `f!`
once per chunk of states or per color of the sparsity pattern and do not know which entries are
zero unless the user supplies a pattern. Tracing calls `f!` once on numbers that record
each operation, which turns the function into a CasADi expression graph. CasADi forms the
analytic Jacobian and its sparsity pattern from that graph, and CasADiCodegen.jl generates
Julia code for both. In this tutorial, we turn the Robertson problem, as written for
OrdinaryDiffEq.jl, into a problem of generated code with an analytic Jacobian, name its
variables, and then trace a reaction-diffusion model with a matrix state and a sparse
Jacobian, a differential-algebraic equation with a mass matrix, and a nonlinear system.
The number type that records the operations, and the rules for branches, are described in
[Trace a model written in Julia](@ref); this tutorial does not repeat them.

The examples need CasADiCodegen.jl, SciMLBase.jl, OrdinaryDiffEqRosenbrock.jl,
OrdinaryDiffEqBDF.jl, NonlinearSolve.jl and the standard libraries LinearAlgebra and
SparseArrays.

```@example traced_problems
using CasADiCodegen, LinearAlgebra, SparseArrays
import SciMLBase
using OrdinaryDiffEqRosenbrock: Rodas5P
using OrdinaryDiffEqBDF: FBDF
nothing # hide
```

## A problem written for OrdinaryDiffEq.jl

The Robertson equations describe three species of a chemical reaction whose rates
differ by nine orders of magnitude, which makes them a standard stiff test problem:

```math
\begin{aligned}
y_1' &= -k_1 y_1 + k_3 y_2 y_3, \\
y_2' &= k_1 y_1 - k_2 y_2^2 - k_3 y_2 y_3, \\
y_3' &= k_2 y_2^2.
\end{aligned}
```

We write the right side as an in-place Julia function, with the parameters as a tuple,
as for OrdinaryDiffEq.jl alone; nothing in it refers to CasADi:

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
takes an `ODEProblem` or a `NonlinearProblem`. It calls `rober!` on arrays of traced
numbers with the shapes of the initial state and of the parameters, which gives the CasADi
function `rober(u, p, t)`, and passes that function to
[`ODEFunction`](@ref SciMLBase.ODEFunction(::CasADiCodegen.Function)), which holds
generated code for the right side and for its Jacobian by the state, and then to a new
`ODEProblem`. [`CasADiCodegen.traced_function`](@ref CasADiCodegen.traced_function(::AbstractString, ::Union{SciMLBase.ODEProblem,SciMLBase.NonlinearProblem}))
returns the CasADi function alone, for other uses such as an integrator inside a larger
CasADi graph.

The new problem keeps the initial state, the time span, the mass matrix and the keywords
of the original problem, such as `saveat` or `callback`; the other fields of the original
`ODEFunction`, such as a hand-written `jac`, are not kept. The parameters become a vector
of the values of the tuple, in their order, since the generated code takes them as one
vector, so that a kept callback that reads `p` by field must read it by place, `p[1]`, or by
name, `integrator.ps[:k₁]`, with the names of the next section. A
tuple or a named tuple of parameters must hold numbers; one that holds arrays is not
taken:

```@example traced_problems
problem = CasADiCodegen.traced_problem("rober", original)
problem.p
```

The function has this analytic Jacobian, which the solvers use instead of automatic
differentiation; at the initial state, where ``y_2 = y_3 = 0``, only its column of the
derivatives by ``y_1`` is nonzero:

```@example traced_problems
problem.f.jac(problem.u0, problem.p, 0.0)
```

The new problem is solved as any other, here with Rodas5P, a Rosenbrock method that uses
the Jacobian at each step. The original problem, solved by the same method with a
Jacobian from automatic differentiation of `rober!`, gives the same final state; the two
Jacobians are both exact, so the two solutions differ by rounding at most, and here not at all:

```@example traced_problems
solution = SciMLBase.solve(problem, Rodas5P(); abstol = 1e-10, reltol = 1e-10)
reference = SciMLBase.solve(original, Rodas5P(); abstol = 1e-10, reltol = 1e-10)
solution.u[end], maximum(abs, solution.u[end] - reference.u[end])
```

The trace runs `rober!` once, so the cost of the conversion does not grow with the time
span, and a later change of a parameter value needs no new trace, because the values
enter the generated code as arguments. A value that fixes the structure of the
computation, such as a size or the number of steps of a loop, cannot be a parameter for
the same reason: it is read once during the trace, and a change of it needs a new trace.

## Names of the states and of the parameters

The generated problem has a symbolic system, so that a solution and a problem can be
indexed by name. Without names in the original problem, the states are `u_0`, `u_1` and
`u_2`, and the parameters `p_0`, `p_1` and `p_2`, counted from 0 as CasADi names the
elements of a vector symbol. The keywords `states` and `parameters` give names of our own,
one per element in the order of `vec(u0)` and of the parameters, which then serve
everywhere that SciMLBase.jl takes a name:

```@example traced_problems
named = CasADiCodegen.traced_problem("rober", original; states = [:y₁, :y₂, :y₃], parameters = [:k₁, :k₂, :k₃])
solution = SciMLBase.solve(named, Rodas5P(); abstol = 1e-10, reltol = 1e-10)
solution[:y₂][end], SciMLBase.remake(named; p = [:k₁ => 0.08]).p
```

Without these keywords, the names come from the symbolic system of the original
function, such as a `SciMLBase.SymbolCache` with one name per element, and the names of
the parameters also from the keys of a named tuple, such as `(k₁ = 0.04, k₂ = 3e7, k₃ = 1e4)`.

## A model with loops, views and a sparse Jacobian

A discretized partial differential equation is where the analytic Jacobian matters most,
since each equation depends on a few states only and the Jacobian is large and sparse.
As an example, we take the Gray-Scott reaction-diffusion equations in one dimension, two
species ``U`` and ``V`` on ``n`` points with reflecting ends, written in the style of much
plain Julia code: the state is an ``n \times 2`` matrix with one species per column, the
function takes views of its columns, and a loop over the points adds the diffusion before
`@.` broadcasts the reaction:

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

The loop has a fixed number of steps and is unrolled into the graph during the trace.
The branches that choose the neighbors `l` and `r` compare integer indices, not traced
values, so each is taken once at the trace and leaves no condition in the graph. A branch
on a state, a parameter or the time throws a `TypeError` and is written with `ifelse`
instead (see [Choices that depend on the input](@ref)). With `sparse = true`, which needs
SparseArrays loaded, the `ODEFunction` also holds a `jac_prototype`: a `SparseMatrixCSC`
with the sparsity pattern that CasADi derives from the graph, into whose stored values the
generated Jacobian writes. Each equation depends on its own point, on
its two neighbors and on the other species at the same point, so that a row has at most
four nonzeros:

```@example traced_problems
problem = CasADiCodegen.traced_problem("grayscott", original; sparse = true)
size(problem.f.jac_prototype), nnz(problem.f.jac_prototype)
```

FBDF, a variable-order implicit method, factorizes this sparse matrix of the ``2n = 512``
states rather than a dense one. The generated code takes the state in the order of
`vec(u0)`, here all of ``U`` and then all of ``V``, but it reads and writes a matrix as
well, so that the traced problem keeps the shape of the original state and the solution
is a matrix with one species per column at each saved time:

```@example traced_problems
solution = SciMLBase.solve(problem, FBDF(); abstol = 1e-8, reltol = 1e-8, save_everystep = false)
solution.retcode, size(solution.u[end])
```

The graph of a loop over many points holds one instruction per operation of each step,
here a few thousand, and the generated Julia code compiles at its first call, so that a
model of tens of thousands of instructions takes several seconds to compile. The keywords
of [`ODEFunction`](@ref SciMLBase.ODEFunction(::CasADiCodegen.Function)), which
`traced_problem` passes on, include `chunk_size`, which splits the generated code into
functions of at most that many instructions, `optlevel`, the optimization level of the
compiler, and `cache`, which keeps the compiled code on disk between sessions (see
[Large functions and the cache on disk](@ref)).

## A mass matrix

A differential-algebraic equation written for OrdinaryDiffEq.jl carries its mass matrix in
the `ODEFunction` of the problem, and the traced problem keeps it. As an example, we take
a pendulum of unit mass and length ``L`` in Cartesian coordinates, whose tension ``T`` is
an algebraic state. The constraint ``x^2 + y^2 = L^2``, differentiated twice, gives the
last equation below, which determines ``T``, so that the system has index 1 and the last
row of the mass matrix is zero. Tracing records the equations as they are written, so a
reduction of the index stays the task of the model:

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

The pendulum starts at rest at ``x = L``, where the last equation gives ``T = 0``, so the
initial state is consistent.

## Nonlinear problems

A `NonlinearProblem` traces in the same way, into a function of the state and of the
parameters without the time, and `traced_problem` makes a `NonlinearProblem` of generated
code with the analytic Jacobian of the residual, which Newton's method of NonlinearSolve.jl
then uses. As an example, we take the intersection of a circle of radius ``\sqrt{p_1}``
with the line ``u_1 = u_2``, written out of place, where `traced_problem` calls `f(u, p)`
on the traced state and takes its result:

```@example traced_problems
using NonlinearSolve: NewtonRaphson

circle(u, p) = [u[1]^2 + u[2]^2 - p[1], u[1] - u[2]]
original = SciMLBase.NonlinearProblem(circle, [1.0, 0.5], [2.0])
problem = CasADiCodegen.traced_problem("circle", original)
SciMLBase.solve(problem, NewtonRaphson()).u
```

The generated functions are in place whatever the form of the original function, unless
the initial state is a static vector, which gives functions that return static arrays.
