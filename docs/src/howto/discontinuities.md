# [Discontinuous models](@id discontinuities)

A right side that switches on the state, through `if_else`, `sign`, a comparison or `abs`,
gives an integrator with right values but wrong derivatives, since CasADi differentiates
each branch and misses the motion of the switching time. This page finds such switches and
fixes the derivatives with modes, which are events, or with a smooth model of small width;
[Shoot a bang-off-bang control](@ref) applies both.

## The jump of the state transition matrix at the switching time

Let the right side change from ``f^-`` to ``f^+`` where the switching function ``S(y, t)``
changes its sign, with the rate ``\dot S = \frac{\partial S}{\partial y} f^- + \frac{\partial S}{\partial t}``.
The state transition matrix (STM) is then multiplied at the switch by

```math
\Psi = I + \left(\dot y^+ - \dot y^-\right) \frac{1}{\dot S} \frac{\partial S}{\partial y},
\qquad \dot y^\pm = f^\pm(y(t_s), t_s),
```

which is singular at a grazing contact, where ``\dot S = 0``. An integrator with an event
function (see [Stop at an event](@ref events)) contains this jump to any order, while a
switch written with `if_else` in `"ode"` has no event time, so its derivatives take
``\Psi = I``.

## Find the switches of a model

[`integrator`](@ref) scans `"ode"` and `"quad"` for switches when it is built, and an
integrator of CasADi also scans `"alg"`. A switch counts when its operand depends on the
state, or on the time and a parameter together; a switch on the parameters alone or on the
time alone, such as `t < 2`, does not move along a trajectory and is not reported. These
operations switch:

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

A jump spoils the first derivatives, and a kink those of order 2 and higher. The tests
`==` and `!=` do not count, and the codes of matrix operations appear only when CasADi
cannot expand an MX function to scalar code.

The keyword `nonsmooth` selects the effect: `:warn` by default, `:error` for an
`ArgumentError` when the integrator is built, and `:ignore` for no scan. An integrator of
CasADi, `integrator("F", "cvodes", dae, t0, tf, opts)`, takes no `nonsmooth`; when
generated code replaces it (see [Replace the solvers of CasADi](@ref)), the keyword
`nonsmooth` of [`CasADiCodegen.Solvers`](@ref) applies.

[`CasADiCodegen.nonsmooth_operations`](@ref) counts the switches of any CasADi function
in the inputs that you choose. Here the thrust of a double integrator switches on the
position and is scaled by `min(p, 2)`:

```@example disc
using CasADiCodegen
using OrdinaryDiffEqVerner: Vern9
using SimpleNonlinearSolve: SimpleNewtonRaphson
import SciMLBase

x, p = SX.sym("x", 2), SX.sym("p")
ode = vcat(x[2], casadi.if_else(x[1] < 1, SX(1.0), SX(-1.0)) * min(p, 2))
rhs = CasadiFunction("rhs", [x, p], [ode], ["x", "p"], ["ode"])
CasADiCodegen.nonsmooth_operations(rhs; inputs = "x"), CasADiCodegen.nonsmooth_operations(rhs; inputs = "p")
```

The scan cannot see into a Julia callback or an external function, and it warns without
need for an `if_else` whose branches agree at the surface or for the guard of a removable
singularity.

A finite difference of the generated code tests a model, since the adaptive step resolves
the switch. With the thrust +1 below ``x = 1`` and -1 above, the Jacobian misses the jump by
a quantity of order 1:

```@example disc
written = Dict("x" => x, "ode" => vcat(x[2], casadi.if_else(x[1] < 1, SX(1.0), SX(-1.0))))
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

## Give the switches as modes

Pass `modes = m => S` to `integrator`, where `m` is a column of fresh symbols that
`"ode"` and `"quad"` use in place of the sign of the switching functions `S`; the
integrator flips mode ``j`` at each change of sign of ``S_j``. Pass also `switches`, an upper bound on the number of switches; `integrator`
throws `ArgumentError` for `modes` without `switches`, and for `switches` or
`switch_times` without `modes`. The finite difference now agrees with the Jacobian:

```@example disc
m = SX.sym("m")
switched = Dict("x" => x, "ode" => vcat(x[2], m))
G = integrator("G", Vern9(), switched, 0.0, 2.0; modes = m => 1 - x[1], switches = 2, abstol = 1e-12, reltol = 1e-12)
fd_error(G)
```

The function has the inputs and results of the model with `if_else`, which it replaces
in the calling code. The rules are:

- The initial mode is the sign of ``S_j``; within ``1.5 \times 10^{-8}`` of the surface it
  is the mode under which the flow leaves the surface, and a call throws when the flow
  leaves on both sides or on neither.
- A call passes up to `switches` switches and throws, with the advice to raise the bound,
  when more come. With a vector of output times the bound applies to each interval.
- With fixed or free times, `switch_times = true` adds the differentiable switch times
  `ts` (`tf` for a missing switch), the index `is` of the switching function of each, and
  their number `ns`.
- Event functions of the model in `"zero"` may stand beside the modes; `te`, `ie`, `ne`,
  `events`, `terminal` and `exact` concern those events only.

The event search checks the sign at 10 points of each solver step, or at
`interp_points` points with free times. A call throws for a sliding arc, where the flow
does not leave the surface, and for two switching functions that change their sign in one
step of the search; a sliding arc needs the smoothed route below. Near a grazing contact
the derivative is discontinuous where the number of switches changes.

## Smooth the switches

Replace each switch by a smooth function of a width ``\varepsilon``, a parameter of the
model. [`CasADiCodegen.smooth_step`](@ref), [`CasADiCodegen.smooth_sign`](@ref),
[`CasADiCodegen.smooth_abs`](@ref), [`CasADiCodegen.smooth_max`](@ref) and
[`CasADiCodegen.smooth_min`](@ref) give the smooth forms on numbers, SX and MX; they are
not exported. `smooth_sign(s, ε)` is ``\tanh(s/\varepsilon)``, or
``s/\sqrt{s^2 + \varepsilon^2}`` with `kind = :algebraic`.

```@example disc
ε = SX.sym("width")
model = Dict("x" => x, "p" => ε, "ode" => vcat(x[2], CasADiCodegen.smooth_sign(1 - x[1], ε)))
nothing # hide
```

[`CasADiCodegen.smoothed`](@ref) rewrites an expression or a whole function that is
already written with switches; its docstring lists the operations it rewrites. The keyword
`inputs` restricts the rewrite to the switches on the chosen inputs and keeps a switch on
the parameters:

```@example disc
smooth_ode = CasADiCodegen.smoothed(written["ode"], ε; inputs = x)
nothing # hide
```

At a transversal crossing the state and its derivatives converge to those of the switched
model as ``\varepsilon^2`` with `:tanh`; at a grazing contact they converge as
``\sqrt{\varepsilon}``, and on a sliding arc as ``\varepsilon``. The limits are:

- The smooth `if_else(c, a, b)` weights both branches within about 19 widths of the
  surface, so a branch that is undefined on the far side, such as `sqrt` of a negative
  number, does not smooth well.
- One width serves all switches of a model, so scale the switching functions to comparable
  magnitudes.
- On a sliding arc the smoothed model is stiff, with a Jacobian that grows as
  ``1/\varepsilon``; use an implicit solver such as `Rodas5P()` of
  OrdinaryDiffEqRosenbrock.jl.

## Follow the width toward zero

A Newton solve of a smoothed model converges from fewer guesses as the width falls, so
lower the width step by step. [`CasADiCodegen.natural_continuation`](@ref), from the
extension for SciMLBase.jl, solves a `NonlinearProblem` for a sequence of values of one
parameter and tries intermediate values when a solve fails. Shoot the smoothed model from
the position 0 to 1.2 at the time 2, with the initial velocity as the unknown:

```@example disc
H = integrator("H", Vern9(), model, 0.0, 2.0; abstol = 1e-10, reltol = 1e-10)
v0, width = MX.sym("v0"), MX.sym("width")
residual = CasadiFunction("residual", [v0, width], [H(vcat(0, v0), width)[1] - 1.2])
problem = SciMLBase.NonlinearProblem(SciMLBase.NonlinearFunction(residual), [0.0], [1.0])
path = CasADiCodegen.natural_continuation(problem, SimpleNewtonRaphson(), [1.0, 0.1, 0.01, 1e-3];
                                          parameter = :width)
[(value, solution.retcode, only(solution.u)) for (value, solution) in path]
```

`parameter` is the name of a scalar symbol of `problem.p`, here `:width`, or its place in
`problem.p`. For a path with turning points, use the extension for BifurcationKit.jl (see
[Bifurcation diagrams](@ref)).

## Choose a route

Modes give the exact derivatives of the discontinuous model, but they need a guess with
the right sequence of switches and fail on a sliding arc; a smoothed model accepts both,
at an error of order ``\varepsilon^2`` or more. Use the root of the smoothed model at a
moderate width as the guess for the switched model, as
[Shoot a bang-off-bang control](@ref) shows.
