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

As for any function of [`build_function`](@ref), `lin` returns a matrix of one column,
such as `B` for one input, as a `Vector`. `lin!(out, (x, u, p))` writes the four
matrices into the arrays of the tuple `out`, which are dense by default. For a large model
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

The model must have no algebraic states. The package has no function that eliminates
them, so a model with algebraic equations needs their solutions substituted by hand, as
in [Model physical systems with differential-algebraic equations](@ref).

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
