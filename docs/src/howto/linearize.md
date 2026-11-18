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
x_op = [0.0, 0.0]                                             # the operating point
u_op = [0.0]
p_op = [9.81, 2.0]
A, B, C, D = lin(x_op, u_op, p_op)
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
# the CasADi function (x, u, p) -> (A, B, C, D), which holds the sparsity patterns
lin_casadi = CasADiCodegen.linearization_function(f, h; sparse = true)
out = Tuple(SparseMatrixCSC(casadi.sparsity_out(lin_casadi, i)) for i in 0:3)
lin_sparse!(out, (x_op, u_op, p_op))
out[1]
```

The matrices are those of a `StateSpace` system of ControlSystemsBase.jl. The package has
no block-diagram signals, so a perturbation of a loop is an input of the model, and the
outputs give the transfer functions. For a plant `x' = (k u - x) / τ` with the controller
`c = K x` and the perturbation `w` at the plant input, the outputs `u = c + w`, `-c` and `x`
give the sensitivity functions `S`, `T` and `P S`:

```@example linearize
using ControlSystemsBase: ss, tf

xl, w, pl = SX.sym("xl"), SX.sym("w"), SX.sym("pl", 3)        # pl = [k, τ, K]
c = pl[3] * xl
loop_f = CasadiFunction("loop_f", [xl, w, pl], [(pl[1] * (c + w) - xl) / pl[2]])
loop_h = CasadiFunction("loop_h", [xl, w, pl], [vcat(c + w, -c, xl)])
loop_lin, _ = CasADiCodegen.linearization(loop_f, loop_h)
closed = ss(loop_lin([0.0], [0.0], [1.0, 1.0, -1.0])...)
tf(closed[1, 1])                                               # S
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

## Order equations to solve them block by block

[`CasADiCodegen.block_triangular_form`](@ref) permutes a sparsity pattern to block
lower triangular form; for the Jacobian of a square system, each diagonal block is a
set of equations to solve together, in order of solution. Equations 2 and 4 hold only
unknowns 1 and 3, so they form the first block:

```@example linearize
y = SX.sym("y", 4)
equations = vcat(y[2] - y[4], y[1] + y[3] - 1, y[4] * y[1] - y[3], y[1] - 2y[3])
form = CasADiCodegen.block_triangular_form(casadi.sparsity(jacobian(equations, y)))
[form.colperm[form.colblock[k]:form.colblock[k+1]-1] for k in 1:form.blocks]
```

Each element holds the unknowns of one block; `colblock[k]` is the position in
`colperm` at which block `k` starts.
