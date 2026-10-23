# Linearize a model and order its equations

A controller design or a stability analysis needs the matrices of a model linearized
about an operating point, and the solution of large implicit equations starts from the
order in which they can be solved. This page computes both from CasADi functions: the
matrices as generated code, and the order from the sparsity pattern of a Jacobian.

## Linearize a state-space model

[`CasADiCodegen.linearization`](@ref) takes the state-space model `x' = f(x, u, p)`,
`y = h(x, u, p)` as two CasADi functions of the state `x`, the input `u` and the
parameters `p`. It returns generated code for the matrices `A = ∂f/∂x`, `B = ∂f/∂u`,
`C = ∂h/∂x` and `D = ∂h/∂u` at a point. The example linearizes a pendulum driven by a
torque about its lowest point:

```@example linearize
using CasADiCodegen

x, u, p = SX.sym("x", 2), SX.sym("u"), SX.sym("p", 2)        # p = [g / L, 1 / (m L^2)]
f = CasadiFunction("f", [x, u, p], [vertcat(x[2], -p[1] * sin(x[1]) + p[2] * u)])
h = CasadiFunction("h", [x, u, p], [x[1]])                    # the angle is measured
lin, lin! = CasADiCodegen.linearization(f, h)
A, B, C, D = lin([0.0, 0.0], [0.0], [9.81, 2.0])
A
```

`lin` returns each of the four matrices as a `Matrix` (an `SMatrix` when `x` is a static array), also `B`, which has one column for
one input: `linearization` builds the functions with `vector_columns = false`, so that no
matrix becomes a vector. `lin!(out, (x, u, p))` writes the four matrices into the arrays
of the tuple `out`, which are dense by default. With `expression = Val{true}`,
`linearization` returns the code of both functions as one `Expr`, as
[`build_function`](@ref) does. For a large model
with a sparse Jacobian, give `sparse = true`, so that `lin!` writes into `SparseMatrixCSC` arrays with the sparsity patterns of the outputs of
[`CasADiCodegen.linearization_function`](@ref):

```@example linearize
using SparseArrays

lin_sparse, lin_sparse! = CasADiCodegen.linearization(f, h; sparse = true)
lin_casadi = CasADiCodegen.linearization_function(f, h; sparse = true)    # (x, u, p) -> (A, B, C, D)
out = Tuple(SparseMatrixCSC(CasADiCodegen.C.sparsity_out(lin_casadi, i)) for i in 0:3)
lin_sparse!(out, ([0.0, 0.0], [0.0], [9.81, 2.0]))
out[1]
```

## Linearize a model with algebraic states

A model with the algebraic states `z`, `x' = f(x, z, u, p)`, `0 = g(x, z, u, p)` and
`y = h(x, z, u, p)`, is linearized by `linearization(f, g, h)`, with three CasADi functions
of `(x, z, u, p)`. The generated code eliminates the deviation of `z` by a solve with the
Jacobian `∂g/∂z`, which must be invertible, so `lin(x, z, u, p)` returns the matrices of
the same form as above, in the deviations of `x` and `u` alone. The point must satisfy
`g = 0`, but need not be a steady state; the generated code neither solves for `z` nor
checks the condition. In the
example, the algebraic state `z` follows `x[1]` through `z + z^3 = x[1]`, and at `x[1] = 2`,
where `z = 1`, the slope of `z` is `1 / (1 + 3z^2) = 1/4`:

```@example linearize
x, z, u = SX.sym("x", 2), SX.sym("z"), SX.sym("u")
p = SX.sym("p", 0, 1)                                         # no parameter
f = CasadiFunction("f", [x, z, u, p], [vertcat(x[2], -z + u)])
g = CasadiFunction("g", [x, z, u, p], [z + z^3 - x[1]])
h = CasadiFunction("h", [x, z, u, p], [z])
lin_dae, _ = CasADiCodegen.linearization(f, g, h)
A, B, C, D = lin_dae([2.0, 0.0], [1.0], [0.0], Float64[])
A, C
```

[Linear analysis of a feedback loop](@ref) linearizes a circuit with an algebraic current at
several operating points.

## Order equations by their dependencies

[`CasADiCodegen.block_triangular_form`](@ref) permutes the rows and the columns of a
sparsity pattern so that the matrix becomes block lower triangular. For the Jacobian of a
square system of equations, each diagonal block is a set of equations that must
be solved together, and the blocks come in the order in which they can be solved. Here
the equations 2 and 4 hold the unknowns 1 and 3 only, so they form the first block:

```@example linearize
y = SX.sym("y", 4)
equations = vertcat(y[2] - y[4], y[1] + y[3] - 1, y[4] * y[1] - y[3], y[1] - 2y[3])
form = CasADiCodegen.block_triangular_form(CasADiCodegen.C.sparsity(jacobian(equations, y)))
[form.colperm[form.colblock[k]:form.colblock[k+1]-1] for k in 1:form.blocks]
```

Each element of the result holds the unknowns of one block, in the order of solution.
`colperm` is the permutation of the unknowns, `blocks` the number of blocks, and
`colblock[k]` the position in `colperm` at which block `k` starts.
