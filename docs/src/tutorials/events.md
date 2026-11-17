# Simulate a bouncing ball

In this tutorial, we simulate a bouncing ball as a SciML problem with a callback and as a
CasADi function that returns the contact times and their derivatives, then put the ball in
a box where the number of contacts is not known in advance.

```@example events
using CasADiCodegen
using OrdinaryDiffEqTsit5: Tsit5
import SciMLBase
```

## Drop a ball on the floor

The state of the ball is its height ``x`` and its velocity ``v``:

```math
\dot{x} = v, \qquad \dot{v} = -g, \qquad g = 9.8.
```

A CasADi function of the state alone gives the right side (see
[Use the generated code with SciML packages](@ref)), and the ball starts at rest at the
height 1:

```@example events
s = SX.sym("s", 2)                                        # s = [x, v]
fall = CasadiFunction("fall", [s], [vcat(s[2], SX(-9.8))])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(fall), [1.0, 0.0], (0.0, 5.0))
nothing # hide
```

## Apply the bounce with a callback

A bounce is a `ContinuousCallback` at a root of the height that reverses the velocity:

```@example events
bounce = SciMLBase.ContinuousCallback((u, t, integ) -> u[1], integ -> (integ.u[2] = -integ.u[2]))
solution = SciMLBase.solve(problem, Tsit5(); callback = bounce, abstol = 1e-10, reltol = 1e-10)
minimum(solution[1, :])
```

The lowest height is zero within the tolerance. The callback saves the state twice at
each contact, and we compare these contact times with the exact
``t_k = (2k - 1) \sqrt{2 h / g}`` for ``h = 1``:

```@example events
contacts = unique(filter(t -> count(==(t), solution.t) == 2, solution.t))
t_exact = (2 .* (1:6) .- 1) .* sqrt(2 / 9.8)
maximum(abs, contacts - t_exact)
```

These contact times come from a SciML solution, outside any graph that CasADi could
differentiate.

## Return the contact times from a CasADi function

To differentiate the contact times, we make them outputs of a CasADi function.
[`integrator`](@ref) takes the event function as `"zero"` and the state
after an event as `"transition"` (see [Stop at an event](@ref events)). The integrator is called
inside an `MX` graph, and we ask for six events that do not end the integration:

```@example events
m = MX.sym("m", 2)
ball = Dict("x" => m, "ode" => vcat(m[2], MX(-9.8)), "zero" => m[1],
            "transition" => vcat(m[1], -m[2]))
bounces = integrator("bounces", Tsit5(), ball; events = 6, terminal = false, direction = -1,
                     abstol = 1e-10, reltol = 1e-10)
nothing # hide
```

With `direction = -1`, only crossings from positive to negative heights count, and
`terminal = false` continues to the final time after the sixth event. `bounces` takes the
initial state and the two times and returns the final state and the contact times; we add
their Jacobian by the initial height `h` as an output, and pass scalars as vectors of one
element:

```@example events
h, t0, tf = MX.sym("h"), MX.sym("t0"), MX.sym("tf")
sf, te = bounces(vcat(h, MX(0.0)), t0, tf)
contact = CasadiFunction("contact", [h, t0, tf], [sf, te, jacobian(te, h)])
contact_f, contact_f! = build_function(contact; expression = Val{false})
xf, times, dtimes = contact_f([1.0], [0.0], [5.0])
maximum(abs, times - t_exact)
```

CasADi differentiates each contact time by the implicit function theorem, so the
derivative has the accuracy of the integration. The exact derivative is
``(2k - 1) / \sqrt{2 g h}``:

```@example events
maximum(abs, vec(dtimes) - (2 .* (1:6) .- 1) ./ sqrt(2 * 9.8 * 1.0))
```
