# How an integrator is differentiated

This page explains where the time of a differentiated
[`CasADiCodegen.integrator`](@ref) goes, and when the adjoint equations are the better
choice.

## Forward sensitivities

The Jacobian of an integrator is a second integrator, of the state with its
sensitivity matrices by the initial state and the parameters, so CasADi
differentiates to any order and the number of states grows by the same factor at each
order: 4 states and 1 parameter give 24 states for the Jacobian and 144 for the second
derivative. Each Julia task keeps the solver state after its first call and skips the
setup on later calls.

A function that needs the final state and its Jacobian integrates each arc once, with
sensitivities, and takes both from that run; a function that needs the state alone
integrates the state alone. The two states agree to the solver tolerances only,
because the sensitivities take part in the step size control.

## A vector of output times

`integrator(name, alg, dae, t0, times)` runs the solver once from `t0` and ends a step
at each output time, as `solve` with `saveat` and `tstops` does, because the
tolerances control the error of a step but not of the interpolant. Its Jacobian is one
integration with sensitivities over the same times, so each derivative order costs one
solver run rather than one per output time.

## Free times

With free times, the derivative by the final time is the right side at the final
state, and for an autonomous model the derivative by the initial time is its
opposite, so free times add no state; a model whose right side depends on time adds
one column of sensitivities for the initial time.

## Through an event

The event time is the root of the event function on the trajectory, a node between
two arcs that CasADi differentiates by the implicit function theorem; the chain rule
through the arc before, the transition and the arc after then gives the jump of the
state transition matrix at the event, to any order. A switch in the right side, such
as an `if_else`, has no such node and its derivatives lack the jump; see
[Discontinuous models](@ref discontinuities).

## Adjoint equations

With `adjoint = true`, a reverse derivative comes from one backward integration of the
adjoint equations, whose number of states does not grow with the number of
parameters; at each output time it takes the seed of that time and restarts the state
from the forward run, which sets its accuracy. Forward and higher-order derivatives
use the sensitivities, so the second order costs the same with and without `adjoint`.
Forward sensitivities add one column of states per initial state and parameter but
never stop, so they give the cheaper gradient with few parameters and many output
times, as in fitting a model to samples; the adjoint equations win when the parameters
far outnumber the states.

For a model with algebraic states, the adjoint equations have the mass matrix of the
model, so the multipliers of the algebraic states are algebraic too. A seed (the
vector that a reverse derivative multiplies with the Jacobian) on a final algebraic
state passes through the algebraic equations to the other final states, the
parameters and the final time, at the cost of one linear solve.
