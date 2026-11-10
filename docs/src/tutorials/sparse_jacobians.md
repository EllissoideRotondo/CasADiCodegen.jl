# Sparse Jacobians of a discretized PDE

A discretized PDE gives a large stiff system whose Jacobian is almost all zeros, so an
implicit solver that stores it dense spends most of its time in the factorization. In
this tutorial, we trace a right side written as a Julia loop into a CasADi function and
give the solver generated code for its sparse Jacobian.

## Write the model in Julia

The two-dimensional Brusselator is a reaction-diffusion system of two species on the
unit square with periodic boundaries, and a source switches on inside a small disk at
`t = 1.1`. We discretize the Laplacian by central differences on an N by N grid:

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

We keep the parameters `A`, `B`, `alpha` and `dx` in a vector, since the trace takes a
number, an array of numbers or nothing as parameters:

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

[`CasADiCodegen.traced_function`](@ref) records the loop as a CasADi graph (see
[Trace a model written in Julia](@ref)). The branches in `limit` act on integer indices
and are resolved during the trace, while `t >= 1.1` stays in the graph. The state of
`rhs(u, p, t)` is `vec(u0)`:

```@example bruss
using CasADiCodegen

rhs = CasADiCodegen.traced_function("brusselator", original)
```

Each equation depends on its own grid point, its four neighbors and the other species,
so each row of the Jacobian has six structural nonzeros. We call `rhs` on symbols and
ask CasADi (module `CasADiCodegen.C`) for the pattern: 768 nonzeros in 128 rows:

```@example bruss
u, q, t = SX.sym("u", 2N^2), SX.sym("p", 4), SX.sym("t")
pattern = CasADiCodegen.C.jacobian_sparsity(rhs(u, q, t), u)
CasADiCodegen.C.nnz(pattern), CasADiCodegen.C.size1(pattern)
```

## Make the problem with a sparse Jacobian

`ODEFunction(rhs; sparse = true)`, which needs SparseArrays loaded, holds the generated
right side and Jacobian together with a `SparseMatrixCSC` `jac_prototype` with this
pattern:

```@example bruss
using SparseArrays

f = SciMLBase.ODEFunction(rhs; sparse = true)
f.jac_prototype
```

The problem takes the state as `vec(u0)`, and FBDF factorizes a sparse matrix:

```@example bruss
using OrdinaryDiffEqBDF: FBDF

problem = SciMLBase.ODEProblem(f, vec(u0), (0.0, 11.5), p)
solution = SciMLBase.solve(problem, FBDF(); save_everystep = false)
solution.retcode
```

The original problem, solved with a Jacobian from automatic differentiation of the
loop, reaches the same state within the default tolerances:

```@example bruss
reference = SciMLBase.solve(original, FBDF(); save_everystep = false)
maximum(abs, vec(reference.u[end]) - solution.u[end])
```

## Differentiate numerically with the sparsity pattern

The solver may also compute the Jacobian itself from the pattern alone. We pass only
the generated right side and the prototype; the solver colors the pattern and
evaluates the right side on dual numbers once per color:

```@example bruss
f_pattern = SciMLBase.ODEFunction(rhs; jac = false, paramjac = false, jac_prototype = f.jac_prototype)
problem_pattern = SciMLBase.ODEProblem(f_pattern, vec(u0), (0.0, 11.5), p)
solution_pattern = SciMLBase.solve(problem_pattern, FBDF(); save_everystep = false)
maximum(abs, solution_pattern.u[end] - solution.u[end])
```

For a larger grid, change `N` in a new session; the generated code grows in proportion
to the number of states. For build times of large models, see
[Large functions and the cache on disk](@ref).
