# How an integrator is differentiated

This page explains what CasADi and the generated code do for the derivatives of a
function made by [`CasADiCodegen.integrator`](@ref): how forward sensitivities grow
with the order of the derivative, why each arc is integrated once, what a vector of
output times and free times cost, and when the adjoint equations are the better
choice. It is meant for a reader who wants to know where the time of a differentiated
integrator goes.

## Forward sensitivities

CasADi differentiates a function `F` made by `integrator` to any order, because its
Jacobian is a second integrator for the state with its sensitivity matrices, whose
right side CasADi builds symbolically. Each of these integrators takes the initial state and the parameters
of the first one, and integrates the derivatives by those only, so the number of
states grows by the same factor at each order: 4 states and 1 parameter give 24
states for the Jacobian and 144 for the second derivative. The generated code reuses
the cache of the solver in each Julia task, and the table gives one measurement for
the planar two-body problem.

| Outputs | Time per call | Memory per call |
|---|---|---|
| Final state | 22 µs | 0.7 kB |
| Final state and its Jacobians by the initial state and by the parameter | 59 µs | 3.6 kB |
| The above and the second derivative by the initial state | 164 µs | 17 kB |

An arc is one integration between two times. When a generated function needs the
final state and its Jacobian, the call for the state runs the integration with the
sensitivities and keeps its result, which the call for the Jacobian then takes, so
that each arc is integrated once. This holds for any number of arcs in one function,
and a function that needs the state alone integrates the state alone. The state from
the integration with the sensitivities agrees with the state of an integration
without sensitivities to the tolerances of the solver and not to the last digit,
because the sensitivities take part in the step size control, as the partial
derivatives of dual numbers do.

The table below compares one day of an Earth orbit perturbed by the Moon and the
Sun, whose positions come from tables of polynomials that two callbacks evaluate
(see [Call a Julia function from the model](@ref)), with `Vern9` and tolerances
of 1e-12, against the same model in plain Julia, where
ForwardDiff.jl gives the state transition matrix, which is the Jacobian of the final
state by the initial state.

| Outputs | CasADiCodegen.jl | Plain Julia |
|---|---|---|
| Final state | 80 µs | 76 µs |
| Final state and state transition matrix | 143 µs | 172 µs with ForwardDiff.jl |

## A vector of output times

An integrator over a vector of output times, `integrator(name, alg, dae, t0, times)`,
runs the solver once from `t0` and takes the state at each time from a step that ends
there, as `solve` with `saveat` and `tstops` does in SciML, because the tolerances
control the error of a step but not that of the interpolant between two steps. Its
Jacobian is one integration of the state with its sensitivity matrices over the same
times, so that a gradient, a Hessian or a derivative of higher order costs one run of
the solver for each order rather than one per output time. With `adjoint = true`, a
reverse derivative is one backward integration of the adjoint equations: the solver
stops at each output time, where the multipliers take the seed of that time and the
state starts again from its value of the forward run. The derivatives of that reverse
derivative come from the forward sensitivities, so the second order costs the same with
and without `adjoint`.

The table compares the least-squares fit of the Lotka-Volterra model to 101 samples
on the times 0, 0.1, ..., 10, with 4 parameters, `Vern9` and tolerances of 1e-10,
against ForwardDiff.jl through one SciML `solve` with `saveat`. Each value was
measured once on a shared machine; the loss and its derivatives agreed with the
ForwardDiff.jl values to 5e-11 relative.

| Outputs | Forward sensitivities | `adjoint = true` | ForwardDiff.jl |
|---|---|---|---|
| Loss | 59 µs | 58 µs | 65 µs |
| Loss and gradient | 97 µs | 196 µs | 113 µs |
| Loss and Hessian | 383 µs | 394 µs | 302 µs |

With 4 parameters the forward sensitivities give the cheaper gradient, because the
adjoint run must stop at each of the 101 output times; the adjoint equations give the
cheaper gradient when the parameters are many more than the states.

## Free times

With free times, the derivative by the final time is the right side at the final
state. For a model without the time in its right side, the derivative by the initial
time is the opposite of that value, so free times add no state to the integration;
a model with the time in its right side adds one column of sensitivities for the
initial time.

## Adjoint equations

The sensitivity matrices cost one integration of the state for each initial state
and each parameter, which is more work than the gradient of one number by many
parameters needs. With `adjoint = true`, a reverse derivative comes instead from an
integrator of the adjoint equations, which runs from the final time back to the
initial time and whose number of states does not grow with the number of
parameters; its accuracy is that of the backward integration of the state, which
starts again from its value of the forward run at each output time. Forward derivatives
and the derivatives of higher order use the sensitivities.

For a model with algebraic states, the adjoint equations have the mass of the model,
so the multipliers of the algebraic states are algebraic too. A seed is the vector that
a reverse derivative multiplies with the Jacobian. A seed on a final algebraic state
first goes, through the algebraic equations at the final time, to the
other final states, the parameters and the final time, which costs one linear solve.
