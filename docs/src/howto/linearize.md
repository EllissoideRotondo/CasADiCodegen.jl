# Linearize a model and order its equations

This page generates code for the matrices of a model linearized about an operating
point, and orders a system of equations by the sparsity pattern of its Jacobian.

## Linearize a state-space model

[`CasADiCodegen.linearization`](@ref) takes `x' = f(x, u, p)` and `y = h(x, u, p)` as
two CasADi functions and returns generated code for `A = ∂f/∂x`, `B = ∂f/∂u`,
`C = ∂h/∂x` and `D = ∂h/∂u`. Linearize a torque-driven pendulum about its lowest point:

```@example linearize
using CasADiCodegen

x, u, p = SX.sym("x", 2), SX.sym("u"), SX.sym("p", 2)        # p = [g / L, 1 / (m L^2)]
f = CasadiFunction("f", [x, u, p], [vcat(x[2], -p[1] * sin(x[1]) + p[2] * u)])
h = CasadiFunction("h", [x, u, p], [x[1]])                    # the angle is measured
lin, lin! = CasADiCodegen.linearization(f, h)
A, B, C, D = lin([0.0, 0.0], [0.0], [9.81, 2.0])
A
```

`lin` returns each matrix as a `Matrix` (an `SMatrix` for a static `x`), even a `B`
with one column, and `lin!(out, (x, u, p))` writes them into the tuple `out`;
`expression = Val{true}` returns the code as one `Expr`, as in [`build_function`](@ref).
For a large sparse model, pass `sparse = true` and give `lin!` `SparseMatrixCSC` arrays
with the patterns of [`CasADiCodegen.linearization_function`](@ref):

```@example linearize
using SparseArrays

lin_sparse, lin_sparse! = CasADiCodegen.linearization(f, h; sparse = true)
lin_casadi = CasADiCodegen.linearization_function(f, h; sparse = true)    # (x, u, p) -> (A, B, C, D)
out = Tuple(SparseMatrixCSC(CasADiCodegen.C.sparsity_out(lin_casadi, i)) for i in 0:3)
lin_sparse!(out, ([0.0, 0.0], [0.0], [9.81, 2.0]))
out[1]
```

## Linearize a model with algebraic states

For `x' = f(x, z, u, p)`, `0 = g(x, z, u, p)` and `y = h(x, z, u, p)`, call
`linearization(f, g, h)`. The code eliminates `z` with `∂g/∂z`, which must be
invertible, so `lin(x, z, u, p)` returns the same four matrices in `x` and `u` alone.
The point must satisfy `g = 0`, which the code neither solves for nor checks. Here
`z + z^3 = x[1]`, so at `x[1] = 2`, where `z = 1`, the slope of `z` is `1/4`:

```@example linearize
x, z, u = SX.sym("x", 2), SX.sym("z"), SX.sym("u")
p = SX.sym("p", 0, 1)                                         # no parameter
f = CasadiFunction("f", [x, z, u, p], [vcat(x[2], -z + u)])
g = CasadiFunction("g", [x, z, u, p], [z + z^3 - x[1]])
h = CasadiFunction("h", [x, z, u, p], [z])
lin_dae, _ = CasADiCodegen.linearization(f, g, h)
A, B, C, D = lin_dae([2.0, 0.0], [1.0], [0.0], Float64[])
A, C
```

[Linear analysis of a feedback loop](@ref) applies this to a circuit.

## Order equations by their dependencies

[`CasADiCodegen.block_triangular_form`](@ref) permutes a sparsity pattern to block
lower triangular form; for the Jacobian of a square system, each diagonal block is a
set of equations to solve together, in order of solution. Equations 2 and 4 hold only
unknowns 1 and 3, so they form the first block:

```@example linearize
y = SX.sym("y", 4)
equations = vcat(y[2] - y[4], y[1] + y[3] - 1, y[4] * y[1] - y[3], y[1] - 2y[3])
form = CasADiCodegen.block_triangular_form(CasADiCodegen.C.sparsity(jacobian(equations, y)))
[form.colperm[form.colblock[k]:form.colblock[k+1]-1] for k in 1:form.blocks]
```

Each element holds the unknowns of one block; `colblock[k]` is the position in
`colperm` at which block `k` starts.
