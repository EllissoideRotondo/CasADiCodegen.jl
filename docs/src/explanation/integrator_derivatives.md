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
states for the Jacobian and 144 for the second derivative. The cost of a call grows
with the number of states that the solver integrates, so that a second derivative
costs several times a Jacobian. Each Julia task keeps the state of the solver after its
first call and uses it again, so that a later call in the same task integrates without
setting up the solver again.

An arc is one integration between two times. When a generated function needs the
final state and its Jacobian, the call for the state runs the integration with the
sensitivities and keeps its result, which the call for the Jacobian then takes, so
that each arc is integrated once. This holds for any number of arcs in one function,
and a function that needs the state alone integrates the state alone. The state from
the integration with the sensitivities agrees with the state of an integration
without sensitivities to the tolerances of the solver and not to the last digit,
because the sensitivities take part in the step size control, as the partial
derivatives of dual numbers do.

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

For a gradient, the choice between the two depends on the numbers of parameters,
states and output times. The forward sensitivities add one column of states for each
initial state and each parameter, whereas the backward run integrates one vector of
multipliers whatever the number of parameters but must stop at each output time to take
its seed. With few parameters and many output times, as in the fit of a model to a
series of samples, the forward sensitivities give the cheaper gradient; the adjoint
equations give the cheaper gradient when the parameters are many more than the states.

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
