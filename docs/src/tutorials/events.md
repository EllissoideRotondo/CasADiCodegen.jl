# Simulate a bouncing ball

In this tutorial, we simulate a ball that falls and bounces on the floor, first as a
SciML problem whose bounce is a callback, and then as a CasADi function that returns
the times of the contacts together with their derivatives. We then put the ball in a
box with drag, where it bounces on the floor and on two walls, so that the model has
three event functions and a number of contacts that we do not know in advance. The
examples need CasADiCodegen.jl, SciMLBase.jl and OrdinaryDiffEqTsit5.jl.

```@example events
using CasADiCodegen
using OrdinaryDiffEqTsit5: Tsit5
import SciMLBase
```

## Drop a ball on the floor

The state of the ball is its height ``x`` and its velocity ``v``, and gravity is its only
force:

```math
\dot{x} = v, \qquad \dot{v} = -g, \qquad g = 9.8.
```

A CasADi function of the state alone gives the right side, since the model has no
parameters and does not depend on time, and `SciMLBase.ODEFunction` turns
it into the function of an ODE problem with generated code for the right side and its
Jacobian (see [Give a CasADi function to a SciML problem](@ref)). The ball starts at rest
at the height 1:

```@example events
s = SX.sym("s", 2)                                        # s = [x, v]
fall = CasadiFunction("fall", [s], [vertcat(s[2], SX(-9.8))])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(fall), [1.0, 0.0], (0.0, 5.0))
nothing # hide
```

Without an event, the solver would let the ball pass through the floor and fall for
the whole interval.

## Apply the bounce with a callback

A bounce is an event: the solver finds the time at which the height changes its sign,
by a search for the root of ``x`` in the step that crosses it, and there the velocity
changes its sign. The callbacks of SciMLBase.jl express both parts as Julia functions
of the state, so a `ContinuousCallback` with the condition `u[1]` and an affect that
reverses `u[2]` completes the model:

```@example events
bounce = SciMLBase.ContinuousCallback((u, t, integ) -> u[1], integ -> (integ.u[2] = -integ.u[2]))
solution = SciMLBase.solve(problem, Tsit5(); callback = bounce, abstol = 1e-10, reltol = 1e-10)
minimum(solution[1, :])
```

The lowest height is zero within the tolerance `1e-10`, so the ball comes close to the
floor at each contact but does not pass through it. The callback saves the state before
and after each bounce at the same time, which gives a way to read the times of the
contacts from the solution. Since the ball returns to the height 1 with the speed of
its fall, contact ``k`` comes at ``t_k = (2k - 1) \sqrt{2 h / g}`` for the initial
height ``h = 1``, and we compare the six contacts of the interval with these times:

```@example events
contacts = unique(filter(t -> count(==(t), solution.t) == 2, solution.t))
t_exact = (2 .* (1:6) .- 1) .* sqrt(2 / 9.8)
maximum(abs, contacts - t_exact)
```

The callback is ordinary SciML code, and only the right side and its Jacobian come
from CasADi; the contact times, however, are then read from a SciML solution, outside
any graph that CasADi could differentiate.

## Return the contact times from a CasADi function

When the contact times are results of a model, for example the targets of a
calibration, we want them as outputs of a CasADi function, so that CasADi
differentiates them with the rest of the model. [`integrator`](@ref) makes such a
function from a dictionary in which `"zero"` is the event function and
`"transition"` gives the state after an event as an expression of the state before
it (see [Stop at an event](@ref events)). We write the model with `MX` symbols, since the
integrator is a call inside a graph of matrix operations, and ask for six events that do
not end the integration:

```@example events
m = MX.sym("m", 2)
ball = Dict("x" => m, "ode" => vertcat(m[2], MX(-9.8)), "zero" => m[1],
            "transition" => vertcat(m[1], -m[2]))
bounces = integrator("bounces", Tsit5(), ball; events = 6, terminal = false, direction = -1,
                     abstol = 1e-10, reltol = 1e-10)
nothing # hide
```

The keyword `direction = -1` counts only the crossings from positive to negative
heights, and `terminal = false` applies the sixth bounce and goes on to the final time
instead of stopping there. With free times, `bounces` takes the initial state, the
initial time and the final time, and returns the final state and the vector of the six
contact times. We make the initial height a symbol `h`, so that the Jacobian of the
times by `h` is one more output of the same function. Each argument of the generated
function is an array, so the scalars `h`, `t0` and `tf` take vectors of one element:

```@example events
h, t0, tf = MX.sym("h"), MX.sym("t0"), MX.sym("tf")
sf, te = bounces(vertcat(h, MX(0.0)), t0, tf)
contact = CasadiFunction("contact", [h, t0, tf], [sf, te, jacobian(te, h)])
contact_f, contact_f! = build_function(contact; expression = Val{false})
xf, times, dtimes = contact_f([1.0], [0.0], [5.0])
maximum(abs, times - t_exact)
```

CasADi differentiates each contact time by the implicit function theorem, as the root
of the event function on the trajectory, so the derivative has the accuracy of the
integration rather than that of a finite difference. The exact derivative of
``t_k`` by ``h`` is ``(2k - 1) / \sqrt{2 g h}``:

```@example events
maximum(abs, vec(dtimes) - (2 .* (1:6) .- 1) ./ sqrt(2 * 9.8 * 1.0))
```

## Bounce between two walls

The second ball moves in a plane, between a floor at ``x = 0`` and two walls at
``y = \pm 1.5``, and it loses speed to a small drag:

```math
\dot{x} = v_x, \qquad \dot{y} = v_y, \qquad \dot{v}_x = -9.8 - 0.1 v_x, \qquad \dot{v}_y = -0.1 v_y.
```

A contact with the floor reverses ``v_x``, and a contact with either wall reverses
``v_y``. The entry `"zero"` is then a column of three event functions, one for each
surface, and the entry `"transition"` has one column per event function, of which
column ``i`` applies after an event of function ``i``. Each event function is positive
inside the box, so that `direction = -1` finds the contacts of the ball from the
inside:

```@example events
b = MX.sym("b", 4)                                        # b = [x, y, vx, vy]
flip_x = vertcat(b[1], b[2], -b[3], b[4])
flip_y = vertcat(b[1], b[2], b[3], -b[4])
box = Dict("x" => b, "ode" => vertcat(b[3], b[4], -9.8 - 0.1 * b[3], -0.1 * b[4]),
           "zero" => vertcat(b[1], b[2] + 1.5, 1.5 - b[2]),
           "transition" => horzcat(flip_x, flip_y, flip_y))
nothing # hide
```

Unlike the first ball, this one has no formula for the number of contacts before the
final time. With the default `exact = true`, a call throws when fewer than `events`
events come before the final time; with `exact = false`, `events` is an upper bound: the
function passes at most that many events and returns, as its last result, the number
`ne` of events that came. With several event functions, the function also returns the
index `ie` of the event function of each event. We allow 20 events:

```@example events
walls = integrator("walls", Tsit5(), box; events = 20, exact = false, terminal = false,
                   direction = -1, abstol = 1e-10, reltol = 1e-10)
b0 = MX.sym("b0", 4)
inbox = CasadiFunction("inbox", [b0, t0, tf], [walls(b0, t0, tf)...])
w, w! = build_function(inbox; expression = Val{false})
bf, tw, iw, nw = w([1.0, 0.0, 0.0, 2.0], [0.0], [10.0])
n = Int(nw[1])
```

The ball starts at the height 1 in the middle of the box, with the speed 2 toward the
wall at ``y = 1.5``, and it touches a surface 17 times before ``t = 10``. The first
`n` entries of `iw` say which surface each contact was, 1 for the floor and 2 or 3 for
a wall; the remaining entries of `tw` hold the final time, and those of `iw` hold 0:

```@example events
Int.(iw[1:n])'
```

## Follow the ball through the box

The function returns the state at the final time only, while a check that the ball
stays inside the box needs the whole path. Since `ne` adapts to the final time, one
function serves for every final time, and we evaluate it on a grid of 201 times:

```@example events
path = [w([1.0, 0.0, 0.0, 2.0], [0.0], [t])[1] for t in range(0, 10; length = 201)]
minimum(p[1] for p in path), extrema(p[2] for p in path)
```

The height stays positive and the lateral position stays between the walls, as the
events require. Each call integrates from the initial time again, so this grid costs
201 integrations; for a dense path, the callback form of the first ball, with a
`VectorContinuousCallback` over the three surfaces, saves the trajectory in one solve.
