# [Stop at an event](@id events)

An integration can end where a function of the state changes its sign and apply a jump
of the state there. This page uses the model `x`, `mu`, `ode` and the symbols `x0`, `p`,
`t0`, `tf` of [Integrate a differential equation](@ref).

## Find one event

With free times, put an event function of the state, the parameters and the time in the
entry `"zero"`. The integration ends at the first change of sign, at the time `te`, and
returns the state there, the quadratures if any, and `te`. A zero at the initial time is
not an event, and a call throws when fewer events than `events` come before `tf`, unless
`exact = false` (see [Pass up to a number of events](@ref)). Here we stop at the next apsis
of an orbit, where the radial speed is zero:

```julia
apsis = integrator("apsis", Vern9(), Dict("x" => x, "p" => mu, "ode" => ode, "zero" => x[1] * x[3] + x[2] * x[4]);
                   abstol = 1e-12, reltol = 1e-12)
xe, te = apsis(x0, p, t0, tf)
f = CasadiFunction("f", [x0, p, t0, tf], [xe, te, jacobian(xe, x0), jacobian(te, x0)])
```

CasADi differentiates the event time and the state at the event to any order, and the
integrator also takes dual numbers. Three keywords control the search:

| Keyword | Default | Effect |
|---|---|---|
| `direction` | 0 | The change of sign: 1 from negative to positive, -1 the opposite, 0 both |
| `interp_points` | 10, as in SciML | The number of points of each step, besides its end, at which the solver looks for the change; 0 is faster, but can miss two changes of sign in one step |
| `specialize` | `SciMLBase.FullSpecialize` | With `SciMLBase.AutoSpecialize`, after `import SciMLBase`, the solver compiles once for all event functions |

The default `direction = 0` differs from the integrators of CasADi, which find a change
from positive to negative only. For the orbit above, measured once:

| Outputs | `interp_points = 10` | `interp_points = 0` |
|---|---|---|
| State at the event and event time | 91 µs | 50 µs |
| The above with their Jacobians by the initial state | 99 µs | 61 µs |
| An arc to a given time with its Jacobian, for comparison | 35 µs | 35 µs |

## Pass several events

Pass `events`, the number of events (1 by default), and give the state after an event in
the entry `"transition"`; `te` is then the vector of event times. With the default
`terminal = true` the integration ends at the last event, before its transition; with
`terminal = false` it applies the transition and continues to `tf`. A ball that bounces
three times:

```julia
s, c = MX.sym("s", 2), MX.sym("c", 2)                    # s = [height, speed], c = [gravity, restitution]
ball = Dict("x" => s, "p" => c, "ode" => vcat(s[2], -c[1]), "zero" => s[1],
            "transition" => vcat(s[1], -c[2] * s[2]))  # the speed changes its sign and loses a part
bounce = integrator("bounce", Vern9(), ball; events = 3, terminal = false, direction = -1)

s0, c0 = MX.sym("s0", 2), MX.sym("c0", 2)
sf, te = bounce(s0, c0, t0, tf)                           # te has the three times of contact
```

## Several event functions

Make `"zero"` a column of event functions, and `"transition"` one column for all of them
or one column per function. The function then returns one more result, `ie`, the index
from 1 of the function behind each event; when two change their sign in one root search,
the first counts. A ball in a box with drag bounces on the floor `x = 0` and between the
walls `y = ±1.5`:

```julia
s = MX.sym("s", 4)                                         # s = [x, y, vx, vy]
flip_x, flip_y = vcat(s[1], s[2], -s[3], s[4]), vcat(s[1], s[2], s[3], -s[4])
box = Dict("x" => s, "ode" => vcat(s[3], s[4], -9.8 - 0.1 * s[3], -0.1 * s[4]),
           "zero" => vcat(s[1], s[2] + 1.5, 1.5 - s[2]),
           "transition" => hcat(flip_x, flip_y, flip_y))    # one column per event function
walls = integrator("walls", Vern9(), box; events = 17, terminal = false, direction = -1)
sf, te, ie = walls(MX.sym("s0", 4), t0, tf)
```

For an event at a known time `tk`, use the event function `tk - t`, which costs more
than a `tstops` of the SciML solver. To change a parameter at an event, make it a state
with a zero element of `"ode"`.

## Pass up to a number of events

With `exact = false`, `events` is an upper bound: the integration passes at most
`events` events and goes on to `tf`, and the last result `ne` is their number. A missing
event has the time `tf` and the index 0 in `ie`:

```julia
walls = integrator("walls", Vern9(), box; events = 40, exact = false, terminal = false, direction = -1)
sf, te, ie, ne = walls(MX.sym("s0", 4), t0, tf)            # te[ne+1:end] are tf
```

For the box above, from `s = [1, 0, 0, 2]` to `t = 10` with `Tsit5()` at the tolerances
`1e-10`, measured once:

| Form | Time of a call | Memory |
|---|---|---|
| One event function, the product `x (2.25 - y²)`, `events = 17` | 0.20 ms | 46 kB |
| Three event functions, `events = 17` | 0.25 ms | 54 kB |
| Three event functions, `events = 40, exact = false` | 0.41 ms | 112 kB |

The three forms gave the same contact times to `3e-10`; the product form needs the
number of contacts in advance and can pick the wrong wall when two factors are small
together. For an integrator of CasADi with events, see [Integrators with events](@ref).

## Switches in the right side

A switch on the state written in `"ode"`, such as an `if_else`, gives wrong derivatives,
and the integrator warns when it is built; [Discontinuous models](@ref discontinuities)
gives such switches as events with the keyword `modes`.
