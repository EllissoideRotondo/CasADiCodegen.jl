# [Stop at an event](@id events)

An integration can end where a function of the state changes its sign, for example
at an apsis of an orbit or at the contact of a ball with the ground, and it can
apply a jump of the state there. This page continues
[Integrate a differential equation](@ref) and uses its model `x`, `mu`, `ode` and
its symbols `x0`, `p`, `t0`, `tf`.

## Find one event

With free times, the entry `"zero"` of the dictionary holds an event function of the
state, the parameters and the time. The integration ends at the event: the first
time `te` between the initial time and `tf` at which the event function changes its
sign.

- The function returns the state at the event, the quadratures if the model has
  them, and `te`.
- A zero at the initial time is not an event, so an integration can start on the
  surface.
- A call throws an error when no event comes before `tf`, and also when fewer events
  come than the keyword `events` asks for, unless `exact = false` (see
  [Pass up to a number of events](@ref)).

We compute the time from one apsis of an orbit to the next one, where the radial
speed is zero:

```julia
apsis = integrator("apsis", Vern9(), Dict("x" => x, "p" => mu, "ode" => ode, "zero" => x[1] * x[3] + x[2] * x[4]);
                   abstol = 1e-12, reltol = 1e-12)
xe, te = apsis(x0, p, t0, tf)
f = CasadiFunction("f", [x0, p, t0, tf], [xe, te, jacobian(xe, x0), jacobian(te, x0)])
```

Each call runs two integrations: one that finds the event, and one to the event
time, with the sensitivities when a result needs a derivative. CasADi
differentiates the event time by the implicit function theorem, as the root of the
event function, and the state at the event as the result of an integrator with free
times, both to any order. The integrator also takes dual numbers, for example when
ForwardDiff.jl differentiates the generated code.

Three keywords control the search:

| Keyword | Default | Effect |
|---|---|---|
| `direction` | 0 | The change of sign: 1 from negative to positive, -1 the opposite, 0 both |
| `interp_points` | 10, as in SciML | The number of points of each step, besides its end, at which the solver looks for the change; 0 is faster, but can miss two changes of sign in one step |
| `specialize` | `SciMLBase.FullSpecialize` | With `SciMLBase.AutoSpecialize`, after `import SciMLBase`, the solver compiles once for all event functions |

The default `direction = 0` differs from the integrators of CasADi, which find a
change from positive to negative only.

For the orbit above, measured once:

| Outputs | `interp_points = 10` | `interp_points = 0` |
|---|---|---|
| State at the event and event time | 91 µs | 50 µs |
| The above with their Jacobians by the initial state | 99 µs | 61 µs |
| An arc to a given time with its Jacobian, for comparison | 35 µs | 35 µs |

## Pass several events

One integrator can also pass several events, which is how a model applies an
impulse or a bounce. The keyword `events` gives their number (1 by default), and
`te` is then the vector of their times. The entry `"transition"` of the dictionary
gives the state after an event as an expression of the state before it, the
parameters and the time. By default, `terminal = true` ends the integration at the
last event, and the result is the state there before its transition. With
`terminal = false`, the integration applies the transition there too and continues to
`tf`, so the result is the state at the final time. CasADi differentiates the result
through each event to any order.

We model a ball that falls and bounces three times. Its state is the height and the
speed, and its parameters are the gravity and the part of the speed that a bounce
keeps:

```julia
s, c = MX.sym("s", 2), MX.sym("c", 2)                    # s = [height, speed], c = [gravity, restitution]
ball = Dict("x" => s, "p" => c, "ode" => vertcat(s[2], -c[1]), "zero" => s[1],
            "transition" => vertcat(s[1], -c[2] * s[2]))  # the speed changes its sign and loses a part
bounce = integrator("bounce", Vern9(), ball; events = 3, terminal = false, direction = -1)

s0, c0 = MX.sym("s0", 2), MX.sym("c0", 2)
sf, te = bounce(s0, c0, t0, tf)                           # te has the three times of contact
```

## Several event functions

The entry `"zero"` can be a column of event functions, for example one for each wall
that a body can reach. The event is then the first time at which any of them changes
its sign, and `direction` applies to each. The entry `"transition"` has either one
column, which applies after every event, or one column per event function, of which
column `i` applies after an event of function `i`. The function returns one more
result, `ie`, which holds for each event the index `i`, from 1, of the function that
changed its sign; when two functions change their sign in the same root search, the
first one counts. CasADi differentiates the event time as the root of that function,
so the derivatives are those of the surface that the body reached.

We model a ball in a box with drag: it bounces on the floor `x = 0`, where the speed
`vx` changes its sign, and between two walls `y = ±1.5`, where `vy` does. Each event
function is positive inside the box, so that `direction = -1` finds the contacts:

```julia
s = MX.sym("s", 4)                                         # s = [x, y, vx, vy]
flip_x, flip_y = vertcat(s[1], s[2], -s[3], s[4]), vertcat(s[1], s[2], s[3], -s[4])
box = Dict("x" => s, "ode" => vertcat(s[3], s[4], -9.8 - 0.1 * s[3], -0.1 * s[4]),
           "zero" => vertcat(s[1], s[2] + 1.5, 1.5 - s[2]),
           "transition" => horzcat(flip_x, flip_y, flip_y))    # one column per event function
walls = integrator("walls", Vern9(), box; events = 17, terminal = false, direction = -1)
sf, te, ie = walls(MX.sym("s0", 4), t0, tf)
```

A time event at a known time `tk` is the event function `tk - t`, with the time `"t"`
of the model. A parameter that changes at an event becomes a state whose element of
`"ode"` is zero, and the column of `"transition"` gives its new value. The solver then looks for the root of
`tk - t` in every step, which costs more than a stop at a given time of the SciML
solver (`tstops`) would.

## Pass up to a number of events

With `exact = false`, a call does not throw when fewer events than `events` come
before `tf`: the integration passes at most `events` events and then goes on to `tf`.
The last result, `ne`, is the number of events; with one event function the results
are `sf, te, ne`. A missing event has the time `tf`,
whose derivative by `tf` is 1, and the index 0 in `ie`, and no transition follows it,
so that `events` is an upper bound rather than a count that the user must know in
advance. Each missing event costs a short call of the solver, which returns at once,
and the state after the last event stays the same through them:

```julia
walls = integrator("walls", Vern9(), box; events = 40, exact = false, terminal = false, direction = -1)
sf, te, ie, ne = walls(MX.sym("s0", 4), t0, tf)            # te[ne+1:end] are tf
```

For the box above, from `s = [1, 0, 0, 2]` to `t = 10`, measured once with `Tsit5()`
at the tolerances `1e-10`; the generated code returns the final state and the times of
the 17 contacts:

| Form | Time of a call | Memory |
|---|---|---|
| One event function, the product `x (2.25 - y²)`, `events = 17` | 0.20 ms | 46 kB |
| Three event functions, `events = 17` | 0.25 ms | 54 kB |
| Three event functions, `events = 40, exact = false` | 0.41 ms | 112 kB |

The three forms gave the same contact times to `3e-10` and the same final state to
`3e-9`. The product form needs the number of contacts in advance and a transition that
guesses the wall from the smaller factor, which can choose the wrong wall when two
factors are small together.

An integrator of CasADi with an event function and the option `"transition"` has
its own rules. [Integrators with events](@ref) shows how generated code replaces it.

## Switches in the right side

A switch on the state written in `"ode"`, such as an `if_else`, rather than as an event
function gives the right values but wrong derivatives, and the integrator warns about it
when it is built. [Discontinuous models](@ref discontinuities) lists the operations that
count and gives the switches as events with the keyword `modes`, or smooths them.
