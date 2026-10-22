# Sparse Jacobians of a discretized PDE

A partial differential equation discretized on a grid gives a large stiff system of
ordinary differential equations in which each state depends on a few neighbors only. An
implicit solver factorizes a matrix built from the Jacobian of such a system at many of its
steps, and when it stores that matrix dense, the factorization takes most of the time of
the solve although almost all of its elements are zero. In this tutorial, we take the right
side of such a system written as an ordinary Julia loop, trace it into a CasADi function,
and give the solver generated code for the right side and its Jacobian together with the
sparsity pattern of the Jacobian, so that the solver factorizes a sparse matrix. The
examples need CasADiCodegen.jl, SciMLBase.jl, OrdinaryDiffEqBDF.jl and the standard
library SparseArrays.

## Write the model in Julia

The two-dimensional Brusselator is a reaction-diffusion system of two species, `u[:, :, 1]`
and `u[:, :, 2]`, on the unit square with periodic boundaries; a source switches on
inside a small disk at the time 1.1. We discretize the Laplacian by central finite
differences on an N by N grid, where `limit` wraps an index across the boundary. The
function is the in-place right side `f(du, u, p, t)` of a SciML problem, written for any
number type:

```@example bruss
const N = 8
const xyd_brusselator = range(0, stop = 1, length = N)
brusselator_f(x, y, t) = (((x - 0.3)^2 + (y - 0.6)^2) <= 0.1^2) * (t >= 1.1) * 5.0
limit(a, N) = a == N + 1 ? 1 : a == 0 ? N : a

function brusselator_2d_loop(du, u, p, t)
    A, B, alpha, dx = p
    alpha = alpha / dx^2
    @inbounds for I in CartesianIndices((N, N))
        i, j = Tuple(I)
        x, y = xyd_brusselator[I[1]], xyd_brusselator[I[2]]
        ip1, im1, jp1, jm1 = limit(i + 1, N), limit(i - 1, N), limit(j + 1, N), limit(j - 1, N)
        du[i, j, 1] = alpha * (u[im1, j, 1] + u[ip1, j, 1] + u[i, jp1, 1] + u[i, jm1, 1] - 4u[i, j, 1]) +
                      B + u[i, j, 1]^2 * u[i, j, 2] - (A + 1) * u[i, j, 1] + brusselator_f(x, y, t)
        du[i, j, 2] = alpha * (u[im1, j, 2] + u[ip1, j, 2] + u[i, jp1, 2] + u[i, jm1, 2] - 4u[i, j, 2]) +
                      A * u[i, j, 1] - u[i, j, 1]^2 * u[i, j, 2]
    end
end
nothing # hide
```

The initial state is a smooth profile of each species, and the parameters are `A`, `B`,
the diffusion coefficient `alpha` and the grid step `dx`. We keep the parameters in a
vector, since the trace below takes a number, an array of numbers or nothing as
parameters:

```@example bruss
import SciMLBase

function init_brusselator_2d(xyd)
    N = length(xyd)
    u = zeros(N, N, 2)
    for I in CartesianIndices((N, N))
        x = xyd[I[1]]
        y = xyd[I[2]]
        u[I, 1] = 22 * (y * (1 - y))^(3 / 2)
        u[I, 2] = 27 * (x * (1 - x))^(3 / 2)
    end
    u
end

u0 = init_brusselator_2d(xyd_brusselator)
p = [3.4, 1.0, 10.0, step(xyd_brusselator)]
original = SciMLBase.ODEProblem(brusselator_2d_loop, u0, (0.0, 11.5), p)
nothing # hide
```

## Trace the loop into a CasADi function

[`CasADiCodegen.traced_function`](@ref) calls the function of the problem on arrays of
traced values, with the shapes of `u0` and `p`, and records each operation as a CasADi
expression (see [Trace a model written in Julia](@ref)). The loop has a fixed number of
steps and is unrolled into the graph. Its branches in `limit` act on integer indices, so
they are taken once during the trace, while `t >= 1.1` compares a traced value and stays
in the graph as a condition on the time. The result is the CasADi function
`rhs(u, p, t)`, whose state is the column vector `vec(u0)` of 2N² elements:

```@example bruss
using CasADiCodegen

rhs = CasADiCodegen.traced_function("brusselator", original)
```

CasADi differentiates `rhs` from its graph. Each equation depends on the state at its
own grid point, on its four neighbors and on the other species at the same point, so that
each row of the Jacobian has six structural nonzeros, the elements that the sparsity
pattern keeps. We call `rhs` on symbols, named `q` for the parameters since `p` holds
their values, and ask CasADi for the pattern of the Jacobian of the result by the state.
The functions `jacobian_sparsity`, `nnz` and `size1` are those of the CasADi front end, in
the module `CasADiCodegen.C`. For N = 8 the pattern has 768 nonzeros in its 2N² = 128
rows:

```@example bruss
u, q, t = SX.sym("u", 2N^2), SX.sym("p", 4), SX.sym("t")
pattern = CasADiCodegen.C.jacobian_sparsity(rhs(u, q, t), u)
CasADiCodegen.C.nnz(pattern), CasADiCodegen.C.size1(pattern)
```

## Make the problem with a sparse Jacobian

`ODEFunction(rhs; sparse = true)` holds generated code for the right side and for its
Jacobian, together with the `jac_prototype` of the solver: a `SparseMatrixCSC` with the
pattern above, into whose stored values the generated Jacobian writes. It needs the
standard library SparseArrays loaded:

```@example bruss
using SparseArrays

f = SciMLBase.ODEFunction(rhs; sparse = true)
f.jac_prototype
```

The problem takes the state as the vector `vec(u0)`; the parameters are those of the
original problem. FBDF, a variable-order implicit method, factorizes a sparse matrix
when the function has a sparse `jac_prototype`:

```@example bruss
using OrdinaryDiffEqBDF: FBDF

problem = SciMLBase.ODEProblem(f, vec(u0), (0.0, 11.5), p)
solution = SciMLBase.solve(problem, FBDF(); save_everystep = false)
solution.retcode
```

The final state is a vector; `reshape(solution.u[end], N, N, 2)` gives it back the shape
of the grid. The original problem, solved by the same method with a Jacobian from
automatic differentiation of the loop, reaches the same state within the default
tolerances of OrdinaryDiffEq.jl, `abstol = 1e-6` and `reltol = 1e-3`:

```@example bruss
reference = SciMLBase.solve(original, FBDF(); save_everystep = false)
maximum(abs, vec(reference.u[end]) - solution.u[end])
```

## Differentiate numerically with the sparsity pattern

A solver may also compute the Jacobian itself, by finite differences or by automatic
differentiation, and still factorize a sparse matrix if it gets the pattern. The function
then holds only the generated right side, and we pass the prototype from above through
the option `jac_prototype` of SciMLBase.jl. The solver colors the pattern with
SparseMatrixColorings.jl, which OrdinaryDiffEqBDF.jl loads: it groups the columns that
share no row and evaluates the right side once for each group, here on the dual numbers
of ForwardDiff.jl. These derivatives agree with the generated Jacobian to rounding, so
that the two solutions agree closely:

```@example bruss
f_pattern = SciMLBase.ODEFunction(rhs; jac = false, paramjac = false, jac_prototype = f.jac_prototype)
problem_pattern = SciMLBase.ODEProblem(f_pattern, vec(u0), (0.0, 11.5), p)
solution_pattern = SciMLBase.solve(problem_pattern, FBDF(); save_everystep = false)
maximum(abs, solution_pattern.u[end] - solution.u[end])
```

For a larger grid, only the constant `N` changes, in a new Julia session: the trace, the
pattern and the generated code grow in proportion to the number of states, since each
row keeps its six nonzeros. For the time of the
build and of the first call of the generated code on large models, see
[Large functions and the cache on disk](@ref).
