# Integrate a differential equation

[`CasADiCodegen.integrator`](@ref) makes a CasADi function that integrates an
ordinary differential equation with a solver of the SciML ecosystem. CasADi
differentiates this function to any order, and the generated code calls the solver
directly, so a model can contain an integration together with its derivatives.

`integrator` is the function of the CasADi front end. The methods that take a SciML
algorithm exist once a solver package is loaded, for example OrdinaryDiffEqVerner.jl.
With a solver name such as `"cvodes"`, the same function makes an integrator of
CasADi itself; see [Replace the solvers of CasADi](@ref).

We use the planar two-body problem in all examples of this page:

```@example ode
using CasADiCodegen
using OrdinaryDiffEqVerner: Vern9

x, mu = MX.sym("x", 4), MX.sym("mu")
r3 = (x[1]^2 + x[2]^2)^1.5
ode = vertcat(x[3], x[4], -mu * x[1] / r3, -mu * x[2] / r3)
model = Dict("x" => x, "p" => mu, "ode" => ode)
nothing # hide
```

The dictionary holds these entries:

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

We integrate from 0 to 3 and generate the final state with its Jacobian by the
initial state:

```@example ode
F = integrator("F", Vern9(), model, 0.0, 3.0; abstol = 1e-13, reltol = 1e-13)

x0, p = MX.sym("x0", 4), MX.sym("p")
f = CasadiFunction("f", [x0, p], [F(x0, p), jacobian(F(x0, p), x0)])
g, g! = build_function(f; expression = Val{false})
xf, stm = g([1.0, 0.0, 0.0, 1.0], [1.0])
xf
```

`stm` is the Jacobian of the final state by the initial state. The keywords
`specialize`, `adjoint` and `mass`, and those of events (see [Stop at an event](@ref events)),
belong to `integrator`. Other keywords, such as `abstol`, `reltol` and `maxiters`, go
to the solver.

## Choose the form of the times

| Call | Function | Result |
|---|---|---|
| `integrator(name, alg, model, t0, tf)` | `F(x0, p)` | State at `tf` |
| `integrator(name, alg, model, t0, times)` | `F(x0, p)` | Matrix with the state at each time in one column |
| `integrator(name, alg, model)` | `G(x0, p, t0, tf)` | State at `tf`; CasADi also differentiates by the times |

In the last form, the times are free: they are inputs of the function, not numbers
fixed when the integrator is made. With free times, the final time can be before
the initial time or equal to it.
With `"quad"`, each function returns the quadratures as a second result.

We differentiate the state by the final time:

```@example ode
G = integrator("G", Vern9(), model; abstol = 1e-13, reltol = 1e-13)
t0, tf = MX.sym("t0"), MX.sym("tf")
h = CasadiFunction("h", [x0, p, t0, tf], [jacobian(G(x0, p, t0, tf), tf)])
build_function(h; expression = Val{false})[1]([1.0, 0.0, 0.0, 1.0], [1.0], [0.0], [3.0])
```

The derivative by the final time is the right side at the final state.

## Get a gradient by many parameters

By default, CasADi gets each derivative from forward sensitivities, which cost one
integration for each initial state and each parameter. For the gradient of one
number by many inputs, we pass `adjoint = true` to `integrator`. A reverse
derivative, such as `gradient`, then integrates the adjoint equations backward,
once.

```@example ode
A = integrator("A", Vern9(), model, 0.0, 3.0; adjoint = true, abstol = 1e-13, reltol = 1e-13)
cost = CasadiFunction("cost", [x0, p], [gradient(CasADiCodegen.C.sumsqr(A(x0, p)), vertcat(x0, p))])
build_function(cost; expression = Val{false})[1]([1.0, 0.0, 0.0, 1.0], [1.0])
```

Forward derivatives and derivatives of higher order still use the sensitivities.
For the gradient of one number by 6 initial states and 36 parameters, measured once:

| Derivative | Time per call |
|---|---|
| Forward sensitivities | 90.7 µs |
| `adjoint = true` | 52.7 µs |

[How an integrator is differentiated](@ref) explains both methods.

## Shorten the compilation of the solver

By default, the solver compiles for each integrator of a new model, which takes about
3 s at the first call of a function with its Jacobian. An integrator of the same
model compiles nothing new. To compile the solver once for all models:

1. Run `import SciMLBase`.
2. Pass `specialize = SciMLBase.AutoSpecialize` to `integrator`.

The solver then calls the right side through a pointer with fixed argument types.
`AutoSpecialize` is the default of SciML; the default of this package is
`FullSpecialize`, which compiles the solver for each right side. The table gives the
first call of the function `f` above, with the final state and its Jacobian, for a first
model and for a second model with another right side, and the time of a later call of
`f`. The times are from 0 to 3 with `abstol = reltol = 1e-13`, measured once on
Julia 1.13:

| `specialize` | First model | Second model | Later call |
|---|---|---|---|
| `SciMLBase.FullSpecialize` (default) | 3.7 s | 2.9 s | 36.6 µs |
| `SciMLBase.AutoSpecialize` | 2.8 s | 0.5 s | 41.9 µs |

## Stop at an event

An integrator with free times can stop where a function of the state changes its
sign, and it can apply a jump of the state there. See [Stop at an event](@ref events).

## Add algebraic states

A model with algebraic states has a diagonal mass matrix. The keyword `mass` takes
its diagonal: 0 for an algebraic state and 1 for the other states. The rows of
`"ode"` with the mass 0 then hold the residuals of the algebraic equations.

- The algorithm must accept a mass matrix, as `Rodas5P()` of
  OrdinaryDiffEqRosenbrock.jl does.
- The initial value of an algebraic state is a guess. The integrator first solves
  the algebraic equations by the Newton method, so the result does not depend on
  that guess.
- Forward derivatives of any order are supported.
- The times are `t0` and `tf`, fixed or free. `adjoint = true` gives adjoint
  equations with the same mass.
- An event function is supported with free times, `integrator(name, alg, dae; mass, ...)`.
  `"zero"` can depend on the algebraic states. `"transition"` returns all rows of `"x"`;
  its algebraic rows are a guess, which the integrator replaces by the solution of the
  algebraic equations after the jump.
- A vector of times is not supported. This limit applies to `integrator` with a SciML
  algorithm; an integrator `idas` of CasADi that generated code replaces has its own
  rules (see [Replace the solvers of CasADi](@ref)).

```julia
using OrdinaryDiffEqRosenbrock: Rodas5P

y, k = MX.sym("y", 3), MX.sym("k")                       # y[3] is algebraic: y[3] = k y[1]
model = Dict("x" => y, "p" => k, "ode" => vertcat(-y[3], y[1], y[3] - k * y[1]))
stiff = integrator("stiff", Rodas5P(), model, 0.0, 1.0; mass = [1.0, 1.0, 0.0], abstol = 1e-11, reltol = 1e-11)
```

## Write the model to a file

`generate_julia` writes a function with an integrator to a module that makes the
solver again. That module loads CasADiCodegen.jl, SciMLBase.jl and the package of
the algorithm. The algorithm and the options of the solver must therefore be values
that a source file can hold: numbers, symbols and structures of them.

## Integrate many arcs

An arc is one integration between two times. `map` of CasADi applies a function to
each column of its arguments, so a map of an integrator gives the arcs of a multiple
shooting method. We map the integrator `F` above over 20 arcs and differentiate all
final states by all initial states:

```julia
X = MX.sym("X", 4, 20)
ends = CasADiCodegen.C.map(F, 20)(X, CasADiCodegen.C.repmat(p, 1, 20))
f = CasadiFunction("shoot", [X, p], [ends, jacobian(ends, X)])
```

Each arc is integrated once. The time per arc is that of a separate function for one
arc: for the final states only, from 0 to 3 with `abstol = reltol = 1e-13`, 24.2 µs per
arc in the map against 24.1 µs, measured once on Julia 1.13.
