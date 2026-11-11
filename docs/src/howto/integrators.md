# Integrate a differential equation

[`CasADiCodegen.integrator`](@ref) makes a CasADi function that integrates an ordinary
differential equation with a SciML solver and that CasADi differentiates to any order
(for a solver name such as `"cvodes"`, see [Replace the solvers of CasADi](@ref)). The
examples use the planar two-body problem:

```@example ode
using CasADiCodegen
using OrdinaryDiffEqVerner: Vern9

x, mu = MX.sym("x", 4), MX.sym("mu")
r3 = (x[1]^2 + x[2]^2)^1.5
ode = vcat(x[3], x[4], -mu * x[1] / r3, -mu * x[2] / r3)
model = Dict("x" => x, "p" => mu, "ode" => ode)
nothing # hide
```

The model dictionary holds these entries:

| Entry | Content | Required |
|---|---|---|
| `"x"` | The state | Yes |
| `"ode"` | The derivative of the state | Yes |
| `"p"` | The parameters; without them, the function is `F(x0)` | No |
| `"t"` | The time, as a symbol that the right side uses | No |
| `"quad"` | The quadratures: expressions that the solver integrates along the solution | No |
| `"zero"` | The event function; free times only (see [Stop at an event](@ref events)) | No |
| `"transition"` | The state after an event, as an expression of the state before it | No |

## Integrate between two fixed times

Give the initial and final times, and generate the final state with its Jacobian `stm`
by the initial state:

```@example ode
F = integrator("F", Vern9(), model, 0.0, 3.0; abstol = 1e-13, reltol = 1e-13)

x0, p = MX.sym("x0", 4), MX.sym("p")
f = CasadiFunction("f", [x0, p], [F(x0, p), jacobian(F(x0, p), x0)])
g, g! = build_function(f; expression = Val{false})
xf, stm = g([1.0, 0.0, 0.0, 1.0], [1.0])
xf
```

The keywords `specialize`, `adjoint`, `mass` and those of events belong to `integrator`,
and the others, such as `abstol`, `reltol` and `maxiters`, go to the solver.

## Choose the form of the times

| Call | Function | Result |
|---|---|---|
| `integrator(name, alg, model, t0, tf)` | `F(x0, p)` | State at `tf` |
| `integrator(name, alg, model, t0, times)` | `F(x0, p)` | Matrix with the state at each time in one column |
| `integrator(name, alg, model)` | `G(x0, p, t0, tf)` | State at `tf`; CasADi also differentiates by the times |

Free times may run backward. With `"quad"`, each function returns the quadratures as a
second result. The derivative by the final time is the right side at the final state:

```@example ode
G = integrator("G", Vern9(), model; abstol = 1e-13, reltol = 1e-13)
t0, tf = MX.sym("t0"), MX.sym("tf")
h = CasadiFunction("h", [x0, p, t0, tf], [jacobian(G(x0, p, t0, tf), tf)])
build_function(h; expression = Val{false})[1]([1.0, 0.0, 0.0, 1.0], [1.0], [0.0], [3.0])
```

## Get a gradient by many parameters

For the gradient of one number by many inputs, pass `adjoint = true`, so that a reverse
derivative such as `gradient` integrates the adjoint equations backward once instead of
one forward sensitivity per input.

```@example ode
A = integrator("A", Vern9(), model, 0.0, 3.0; adjoint = true, abstol = 1e-13, reltol = 1e-13)
cost = CasadiFunction("cost", [x0, p], [gradient(casadi.sumsqr(A(x0, p)), vcat(x0, p))])
build_function(cost; expression = Val{false})[1]([1.0, 0.0, 0.0, 1.0], [1.0])
```

Forward and higher derivatives use forward sensitivities;
[How an integrator is differentiated](@ref) explains both methods. For a gradient by 6
initial states and 36 parameters, measured once:

| Derivative | Time per call |
|---|---|
| Forward sensitivities | 90.7 µs |
| `adjoint = true` | 52.7 µs |

## Shorten the compilation of the solver

The default `FullSpecialize` compiles the solver for each new model, which takes about
3 s at the first call. To compile it once for all models, run `import SciMLBase` and pass
`specialize = SciMLBase.AutoSpecialize` to `integrator`. The first call of `f` above and a
later one took, measured once on Julia 1.13:

| `specialize` | First model | Second model | Later call |
|---|---|---|---|
| `SciMLBase.FullSpecialize` (default) | 3.7 s | 2.9 s | 36.6 µs |
| `SciMLBase.AutoSpecialize` | 2.8 s | 0.5 s | 41.9 µs |

## Stop at an event

An integrator with free times can stop where a function of the state changes its sign;
see [Stop at an event](@ref events).

## Add algebraic states

Pass `mass`, the diagonal of the mass matrix: 0 for an algebraic state, whose row of
`"ode"` holds the residual of its equation, and 1 for the others.

- The algorithm must accept a mass matrix, as `Rodas5P()` does.
- The initial value of an algebraic state is a guess, corrected by Newton's method.
- The times are `t0` and `tf`, fixed or free; a vector of times is not supported.
- Forward derivatives of any order and `adjoint = true` are supported.
- With free times, `"zero"` may depend on the algebraic states; the algebraic rows of
  `"transition"` are a guess, which the integrator corrects after the jump.

```julia
using OrdinaryDiffEqRosenbrock: Rodas5P

y, k = MX.sym("y", 3), MX.sym("k")                       # y[3] is algebraic: y[3] = k y[1]
model = Dict("x" => y, "p" => k, "ode" => vcat(-y[3], y[1], y[3] - k * y[1]))
stiff = integrator("stiff", Rodas5P(), model, 0.0, 1.0; mass = [1.0, 1.0, 0.0], abstol = 1e-11, reltol = 1e-11)
```

## Write the model to a file

`generate_julia` writes an integrator to a module that loads the package of the
algorithm, so the algorithm and its options must be values that a source file can hold:
numbers, symbols and structures of them.

## Integrate many arcs

For the arcs of a multiple shooting method, map the integrator over the columns of the
initial states with `map` of CasADi:

```julia
X = MX.sym("X", 4, 20)
ends = casadi.map(F, 20)(X, casadi.repmat(p, 1, 20))
f = CasadiFunction("shoot", [X, p], [ends, jacobian(ends, X)])
```

Each arc is integrated once, at the cost of a separate integrator.
