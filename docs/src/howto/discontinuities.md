# [Discontinuous models](@id discontinuities)

A right side that switches on the state, such as a thrust that is on where a switching
function is positive and off where it is negative, gives an integrator whose values are
right but whose derivatives can be wrong. CasADi differentiates `if_else`, `sign`, a
comparison or `abs` branch by branch, and the variational equations then follow the
branch that is active along the trajectory; they miss the effect of the switching time,
which moves with the initial state and the parameters. This page shows how the package
finds such switches, and the two routes that give the right derivatives: the switches as
modes of the integrator, which are events, and a smooth model whose width falls toward
zero. [Shoot a bang-off-bang control](@ref) applies both routes to one problem.

## The jump of the state transition matrix at the switching time

Let the right side change from ``f^-`` to ``f^+`` where the switching function ``S(y, t)``
changes its sign, at the time ``t_s`` with ``S(y(t_s), t_s) = 0``, and let the state be
continuous there. A change ``\delta y^-`` of the state just before the switch moves the
switching time by ``\delta t_s = -\frac{1}{\dot S} \frac{\partial S}{\partial y} \delta y^-``,
with the rate ``\dot S = \frac{\partial S}{\partial y} f^- + \frac{\partial S}{\partial t}``,
and over that time the state moves with ``f^-`` on one side and ``f^+`` on the other. The
state transition matrix (STM) is therefore multiplied at the switch by

```math
\Psi = I + \left(\dot y^+ - \dot y^-\right) \frac{1}{\dot S} \frac{\partial S}{\partial y},
\qquad \dot y^\pm = f^\pm(y(t_s), t_s),
```

which is the jump of the STM at the switching time. This first-order update goes back to
the stability analysis of periodic solutions of systems with discontinuous right sides
(M. A. Aizerman and F. R. Gantmacher, Quarterly Journal of Mechanics and Applied
Mathematics 11(4):385–398, 1958) and is the base of the trajectory sensitivities of
hybrid systems (I. A. Hiskens and M. A. Pai, IEEE Transactions on Circuits and Systems I
47(2):204–220, 2000). The update is singular where ``\dot S = 0``, at a grazing contact,
where the trajectory touches the surface ``S = 0`` without crossing it.

An integrator with an event function, `"zero"` and `"transition"` (see
[Stop at an event](@ref events)), contains this update without a formula: CasADi
differentiates the event time as the root of the event function, by the implicit
function theorem, and the chain rule through the arcs and the event gives ``\Psi``, and
with it the derivatives of higher order. A switch written with `if_else` in `"ode"` has no
event time in the graph, so the derivatives take ``\Psi = I`` at the switch, and the term
that they omit is of the order of the derivative itself. For a control written as `sign`
of a costate, each branch is a constant, and the Jacobian of the final state by the
costates is zero.

## Find the switches of a model

Each integrator scans its model for switches when it is built: [`integrator`](@ref) scans
`"ode"` and `"quad"`, and an integrator of CasADi also scans `"alg"`. A switch counts when
its operand depends on the state, or on the time and a parameter together, because its
surface then moves with the initial state or with the parameters. A switch on the
parameters alone, or on the time alone, such as `t < 2`, does not move along a
trajectory, and its derivatives are right. These operations switch:

| Operation                                 | Code                                         | Kind                      |
|:----------------------------------------- |:-------------------------------------------- |:------------------------- |
| `a < b`, `a > b`                          | `:OP_LT`                                     | jump                      |
| `a <= b`, `a >= b`                        | `:OP_LE`                                     | jump                      |
| `sign`                                    | `:OP_SIGN`                                   | jump                      |
| `floor`, `ceil`                           | `:OP_FLOOR`, `:OP_CEIL`                      | jump                      |
| `fmod`, `remainder`                       | `:OP_FMOD`, `:OP_REMAINDER`                  | jump                      |
| `copysign(a, b)`                          | `:OP_COPYSIGN`                               | jump in `b`, kink in `a`  |
| the index of `conditional`                | `:Switch`                                    | jump                      |
| `abs`                                     | `:OP_FABS`                                   | kink                      |
| `fmin`, `fmax`, also as `min`, `max`      | `:OP_FMIN`, `:OP_FMAX`                       | kink                      |
| MX `norm_1`, `norm_inf`, `mmin`, `mmax`   | `:OP_NORM1`, `:OP_NORMINF`, `:OP_MMIN`, `:OP_MMAX` | kink                |
| a `linear` interpolant, CasADi's `low`    | `:LinearInterpolant`, `:OP_LOW`              | kink                      |

A jump of the value spoils the first derivatives. A kink of the slope leaves the right
side continuous, so ``\dot y^+ = \dot y^-`` and the STM is right, but it spoils the
derivatives of order 2 and higher. The equality tests `==` and `!=` switch on a set of
measure zero, such as the guard of `if_else(r == 0, 0, x / r)`, and do not count. The
logical operations `!`, `&` and `|` combine comparisons that count already, and an
`if_else` or a `conditional` whose condition is a comparison counts once, at that
comparison. The code is CasADi's name of the operation; CasADi stores `a > b` as `b < a`,
so it counts as `:OP_LT`. An MX function is expanded to scalar code before the scan when
CasADi can expand it, so the codes of matrix operations, such as `:OP_NORM1`, appear only
when it cannot.

The keyword `nonsmooth` selects the effect: `:warn` by default, `:error` for an
`ArgumentError` when the integrator is built, and `:ignore` for no scan. The default warns
rather than throws, because a model that is only simulated is right.

The method of `integrator` that takes the name of a plugin of CasADi as a `String`, such
as `integrator("F", "cvodes", dae, t0, tf, opts)`, makes an integrator of CasADi, which
misses the jump in its own sensitivities too. It is scanned in the same way when it is
built, and its option `"nonsmooth" => "warn" | "error" | "ignore"` in `opts`, `"warn"` by
default, selects the effect; the package removes the option before CasADi reads `opts`.
When `build_function` replaces such an integrator by generated code (see
[Replace the solvers of CasADi](@ref)), it scans the model again, with the keyword
`nonsmooth` of [`CasADiCodegen.Solvers`](@ref), also `:warn` by default. The two settings
are independent, since each acts at its own step, so a model that is known to be right
needs both set to ignore its switches.

[`CasADiCodegen.nonsmooth_operations`](@ref) counts the switches in any CasADi function,
without an integrator, for the inputs that the user chooses. Here a double integrator has
a thrust that switches on the position, scaled by `min(p, 2)`:

```@example disc
using CasADiCodegen
using OrdinaryDiffEqVerner: Vern9
using SimpleNonlinearSolve: SimpleNewtonRaphson
import SciMLBase
const C = CasADiCodegen.C

x, p = SX.sym("x", 2), SX.sym("p")
ode = vertcat(x[2], C.if_else(x[1] < 1, SX(1.0), SX(-1.0)) * min(p, 2))
rhs = CasadiFunction("rhs", [x, p], [ode], ["x", "p"], ["ode"])
CasADiCodegen.nonsmooth_operations(rhs; inputs = "x"), CasADiCodegen.nonsmooth_operations(rhs; inputs = "p")
```

The scan cannot see into the body of a Julia callback or of an external function, so a
switch there gives no warning. Two kinds of switch warn without need: an `if_else` whose
branches agree at the surface, which the graph cannot show, and the guard of a removable
singularity, such as a series near zero beside the closed form away from it.

A finite difference of the generated code is the quick test of a model, because the
adaptive step size of the solver resolves the switch. Here the thrust is +1 below
``x = 1`` and -1 above, written with `if_else`:

```@example disc
written = Dict("x" => x, "ode" => vertcat(x[2], C.if_else(x[1] < 1, SX(1.0), SX(-1.0))))
F = integrator("F", Vern9(), written, 0.0, 2.0; nonsmooth = :ignore, abstol = 1e-12, reltol = 1e-12)

function fd_error(F)
    x0 = MX.sym("x0", 2)
    xf = F(x0)
    g, _ = build_function(CasadiFunction("g", [x0], [xf, jacobian(xf, x0)]); expression = Val{false})
    z, h = [0.0, 0.0], 1e-6
    fd = reduce(hcat, (vec(g(z + h * e)[1]) - vec(g(z - h * e)[1])) / 2h for e in ([1.0, 0.0], [0.0, 1.0]))
    maximum(abs, g(z)[2] - fd)
end
fd_error(F)
```

The Jacobian of the generated code differs from the finite difference by a quantity of
order 1, which is the missing jump.

## Give the switches as modes

The keyword `modes = m => S` of `integrator` gives the switches as events. `m` is a column
of fresh symbols of the type of the state, the modes, which `"ode"` and `"quad"` use in
place of the sign of the switching functions `S`, a column of the same size: mode ``j`` is
+1 where ``S_j > 0`` and -1 where ``S_j < 0``. The integrator flips mode ``j`` at each
change of sign of ``S_j``, and the derivatives get the jump of the STM at the switching
time to any order. The keyword `switches`, which `modes` requires, is an upper bound on the
number of switches between the initial and the final time; `integrator` throws
`ArgumentError` for `modes` without `switches`, and for `switches` or `switch_times`
without `modes`:

```@example disc
m = SX.sym("m")
switched = Dict("x" => x, "ode" => vertcat(x[2], m))
G = integrator("G", Vern9(), switched, 0.0, 2.0; modes = m => 1 - x[1], switches = 2, abstol = 1e-12, reltol = 1e-12)
fd_error(G)
```

The function has the inputs and the results of the model without modes, so it replaces
the smoothed model or the one with `if_else` in the code that calls it; only
`switch_times = true`, below, adds results. The model without modes is the one above with
`m` in place of the sign of ``1 - x_1``, and an `if_else` on the mode itself does not warn.

The initial mode ``m_j`` is the sign of ``S_j`` at the initial time. On the surface, where
``|S_j| \le \sqrt{\epsilon} / (1 - \sqrt{\epsilon}) \approx 1.5 \times 10^{-8}`` with
``\epsilon`` = `eps()`, the initial mode is the one under which the flow leaves the
surface: the value ``m_j = \pm 1`` for which the rate ``\dot S_j``, taken with the right
side of mode ``m_j``, has the sign of ``m_j``, or the opposite sign for an integration
backward in time. A start at rest on a surface therefore works. When the flow leaves on
both sides, or on neither, a call throws an error that names the case.

The bound `switches` is not the exact number of switches, unlike the keyword `events` of
the event functions in `"zero"`: a call passes up to `switches` switches and throws, with
the advice to raise the bound, when more come. With the times `t0, tf` and with free times,
`switch_times = true` adds three results: the switch times `ts`, with the time `tf` for a
missing switch, the index `is` of the switching function of each when there are several,
and their number `ns`; CasADi differentiates the switch times too. A model may have event
functions of its own in `"zero"` beside the modes; `te`, `ie`, `ne`, `events`, `terminal`
and `exact` then concern those events only. With a vector of output times, the integrator
carries the modes from one output time to the next, and the bound applies to each interval.

The integrator finds a switch by its event search: it looks for a change of sign of the
switching functions at 10 points of each step of the solver, or at `interp_points` points
with free times, and at the end of the step, and locates the first change by root finding. Each call checks two conditions that the
result needs, and throws when one fails:

- At each switch, the flow on the far side of the surface must leave it, so that the
  product of the rates ``\dot S`` before and after the flip is positive. Otherwise the
  flow slides on the surface, a sliding arc, and the error says so.
- At the final time and at each output time, each mode must agree with the sign of its
  switching function. This check catches a sliding arc that the first one misses, a
  crossing at an output time, and two switching functions that change their sign in one
  step of the event search, of which only the first flips its mode.

A model with a sliding arc, where the state stays on the surface over an interval, has no
solution as a sequence of modes, and the smoothed route below is the one to take. A
grazing contact gives no event and a right result; near it, however, the switching times
depend on the initial state like a square root, so the derivative is discontinuous where
the number of switches changes.

## Smooth the switches

The second route replaces each switch by a smooth function of width ``\varepsilon``, a
parameter of the model, so that CasADi differentiates the smoothed flow exactly. The
helpers [`CasADiCodegen.smooth_step`](@ref), [`CasADiCodegen.smooth_sign`](@ref),
[`CasADiCodegen.smooth_abs`](@ref), [`CasADiCodegen.smooth_max`](@ref) and
[`CasADiCodegen.smooth_min`](@ref) give the smooth forms of a step, `sign`, `abs`, `max`
and `min` of a switching value, on numbers, SX and MX. The package does not export them,
so they are called with the prefix `CasADiCodegen.`; `smooth_sign(s, ε)` is
``\tanh(s/\varepsilon)``, or ``s/\sqrt{s^2 + \varepsilon^2}`` with `kind = :algebraic`.

```@example disc
ε = SX.sym("width")
model = Dict("x" => x, "p" => ε, "ode" => vertcat(x[2], CasADiCodegen.smooth_sign(1 - x[1], ε)))
nothing # hide
```

[`CasADiCodegen.smoothed`](@ref) applies the same forms to a model that is already
written with switches, such as comparisons, `if_else`, `sign`, `abs`, `fmin` and `fmax`;
its docstring lists the operations that it rewrites and those that it keeps. It takes an
expression, for a model dictionary, or a whole function, which gets the width as one more
input, and shared subexpressions stay shared. The keyword `inputs` restricts the rewrite to
the switches on the chosen inputs, such as the state, and keeps a switch on the parameters:

```@example disc
smooth_ode = CasADiCodegen.smoothed(written["ode"], ε; inputs = x)
nothing # hide
```

For a transversal crossing, at which ``S`` changes its sign with a nonzero rate
``\dot S``, the state and its derivatives converge to those of the switched model as
``\varepsilon^2`` with `:tanh`, because the odd symmetry of ``\tanh`` cancels the error of
the first order; the tails of the algebraic kind decay as ``\varepsilon^2/s^2``, and the
error falls about as ``\varepsilon^2 \log(1/\varepsilon)``. A grazing contact spends a time
of order ``\sqrt{\varepsilon}`` in the layer and converges as ``\sqrt{\varepsilon}``, and on
a sliding arc the smoothed state stays within the layer, at a distance of order
``\varepsilon`` from the surface, and converges as ``\varepsilon``. No rate is claimed here
for the second derivatives of a smoothed model.

The smooth form of `if_else(c, a, b)` weights both branches by the smooth condition,
within about 19 widths of the surface with `:tanh`, so both branches must be finite there: a
branch that is undefined on the far side of the surface, such as `sqrt` of a negative
number, or a guard of a removable singularity, does not smooth well. One width serves all
the switches of a model, so the switching functions should be scaled to comparable
magnitudes.

On a sliding arc the smoothed model is stiff: the switching function stays within the
layer, where the Jacobian of the right side grows as ``1/\varepsilon``, and an explicit
solver takes a number of steps that grows as ``1/\varepsilon`` too. An implicit solver,
such as `Rodas5P()` of OrdinaryDiffEqRosenbrock.jl, keeps the number of steps nearly
constant as the width falls. A transversal crossing spends only a time
``\varepsilon / |\dot S|`` in the layer and does not need an implicit solver.

## Follow the width toward zero

A Newton solve of a smoothed model converges from a smaller region of guesses as the width
falls, since the residual tends to that of the discontinuous model, whose Jacobian is
singular where an iterate has lost a switch.
[`CasADiCodegen.natural_continuation`](@ref), a method of the extension for SciMLBase.jl,
solves a `NonlinearProblem` for a sequence of values of one parameter, each solve starting
from the root of the one before, and tries intermediate values when a solve fails. Here the
smoothed model above is shot from the position 0 to the position 1.2 at the time 2, with
the initial velocity as the unknown:

```@example disc
H = integrator("H", Vern9(), model, 0.0, 2.0; abstol = 1e-10, reltol = 1e-10)
v0, width = MX.sym("v0"), MX.sym("width")
residual = CasadiFunction("residual", [v0, width], [H(vertcat(0, v0), width)[1] - 1.2])
problem = SciMLBase.NonlinearProblem(SciMLBase.NonlinearFunction(residual), [0.0], [1.0])
path = CasADiCodegen.natural_continuation(problem, SimpleNewtonRaphson(), [1.0, 0.1, 0.01, 1e-3];
                                          parameter = :width)
[(value, solution.retcode, only(solution.u)) for (value, solution) in path]
```

`NonlinearFunction` makes the second input of `residual` the parameters `problem.p`, and
each scalar symbol in it a parameter named after the symbol, here `:width`. `parameter` is
such a name, as a `Symbol`, or the place of the parameter in `problem.p`, here 1. For a
path with turning points in the parameter, the extension for BifurcationKit.jl gives
pseudo-arclength continuation (see [Bifurcation diagrams](@ref)).

## Choose a route

Modes give the derivatives of the discontinuous model itself, with no error from a width,
and their cost is that of an integration with events; they need a guess with the right
sequence of switches and an upper bound on their number, and they fail on a sliding arc. A
smoothed model accepts any switch that [`CasADiCodegen.smoothed`](@ref) can rewrite,
including sliding arcs, and with continuation it converges from rougher guesses; its
results carry an error of order ``\varepsilon^2`` at a transversal crossing and more
elsewhere, and a sliding arc needs an implicit solver. The two combine well: the root of
the smoothed model at a moderate width is a guess for the switched model, as
[Shoot a bang-off-bang control](@ref) shows. In both routes, a finite difference of the
generated code, as above, confirms that the derivatives are right.
