# [Get correct derivatives from a model with switches](@id discontinuities)

A right side that switches on the state gives an integrator with right values but wrong
derivatives, since CasADi differentiates each branch and misses that the switching time
moves. This page finds such switches and fixes the derivatives in one of two ways, as in
[Find a fuel-optimal control that switches on and off](@ref): the integrator locates each
switch as an event
(`modes`), or a smooth function of a small width replaces the switch. Exact switches give
the exact derivatives but fail on a sliding arc, where the state stays on the switching
surface; a smoothed model accepts both, at an error of order ``\varepsilon^2`` or more.

## Find the switches of a model

[`integrator`](@ref) scans `"ode"` and `"quad"` for switches when it is built, and a
CasADi integrator also scans `"alg"`. A switch counts when its operand depends on the
state, or on the time and a parameter together; a switch such as `t < 2` does not move
along a trajectory. A switch is a jump, which spoils the first derivatives, or a kink,
which spoils those of order 2 and higher.

!!! details "Operations that count as switches"
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

    The tests `==` and `!=` do not count, and the codes of matrix operations appear only
    when CasADi cannot expand an MX function to scalar code.

The keyword `nonsmooth` selects a warning, an error or no scan. A CasADi integrator
takes no `nonsmooth`; when generated code replaces it (see
[Run CasADi solver objects with SciML solvers](@ref)), the keyword `nonsmooth` of
[`CasADiCodegen.Solvers`](@ref) applies.

[`CasADiCodegen.nonsmooth_operations`](@ref) counts the switches of any CasADi function
in the chosen inputs, here a thrust that switches on the position, scaled by `min(p, 2)`:

```@example disc
using CasADiCodegen
using OrdinaryDiffEqVerner: Vern9
using SimpleNonlinearSolve: SimpleNewtonRaphson
import SciMLBase

x, p = SX.sym("x", 2), SX.sym("p")
thrust = casadi.if_else(x[1] < 1, SX(1.0), SX(-1.0))
ode = vcat(x[2], thrust * min(p, 2))
rhs = CasadiFunction("rhs", [x, p], [ode], ["x", "p"], ["ode"])
in_x = CasADiCodegen.nonsmooth_operations(rhs; inputs = "x")
in_p = CasADiCodegen.nonsmooth_operations(rhs; inputs = "p")
in_x, in_p
```

The scan cannot see into a Julia callback or an external function, and it warns without
need for an `if_else` whose branches agree at the surface or for the guard of a removable
singularity.

A finite difference of the generated code tests a model, since the adaptive step resolves
the switch. Here the Jacobian misses the jump by a quantity of order 1:

```@example disc
written = Dict("x" => x, "ode" => vcat(x[2], thrust))
F = integrator("F", Vern9(), written, 0.0, 2.0;
               nonsmooth = :ignore, abstol = 1e-12, reltol = 1e-12)

function fd_error(F)
    x0 = MX.sym("x0", 2)
    xf = F(x0)
    with_jacobian = CasadiFunction("g", [x0], [xf, jacobian(xf, x0)])
    g, _ = build_function(with_jacobian; expression = Val{false})
    final(y) = vec(g(y)[1])
    z, h = [0.0, 0.0], 1e-6
    directions = ([1.0, 0.0], [0.0, 1.0])
    fd = reduce(hcat, (final(z + h * e) - final(z - h * e)) / 2h for e in directions)
    maximum(abs, g(z)[2] - fd)
end
fd_error(F)
```

## Locate the switches exactly with `modes`

Pass `modes = m => S` to `integrator`, where `m` is a column of fresh symbols that
`"ode"` and `"quad"` use in place of the sign of the switching functions `S`; the
integrator flips mode ``j`` at each change of sign of ``S_j``. Pass also `switches`, an
upper bound on the number of switches. The finite difference now agrees with the
Jacobian:

```@example disc
m = SX.sym("m")
switched = Dict("x" => x, "ode" => vcat(x[2], m))
G = integrator("G", Vern9(), switched, 0.0, 2.0; modes = m => 1 - x[1], switches = 2,
               abstol = 1e-12, reltol = 1e-12)
fd_error(G)
```

The function has the inputs and results of the model with `if_else`, which it replaces
in the calling code. The rules are:

- The initial mode is the sign of ``S_j``; within ``1.5 \times 10^{-8}`` of the surface it
  is the mode under which the flow leaves the surface, and a call throws when the flow
  leaves on both sides or on neither.
- A call throws when more than `switches` switches come; with a vector of output times
  the bound applies to each interval.
- With fixed or free times, `switch_times = true` adds the differentiable switch times
  `ts` (`tf` for a missing switch), the index `is` of the switching function of each, and
  their number `ns`.
- Event functions in `"zero"` may stand beside the modes; `te`, `ie`, `ne`, `events`,
  `terminal` and `exact` concern those events only.

!!! details "When `modes` fails: sliding, close switches and grazing"
    The event search checks the sign at 10 points of each solver step; with free times
    the keyword `interp_points` sets this number, and with fixed times it stays 10. A call
    throws for a sliding arc, where the flow does not leave the surface, and for two
    switching functions that change their sign in one step of the search. Near a grazing
    contact the derivative is discontinuous where the number of switches changes.

## Replace the switches with smooth functions

Replace each switch by a smooth function of a width ``\varepsilon``, a parameter of the
model. [`CasADiCodegen.smooth_step`](@ref), [`CasADiCodegen.smooth_sign`](@ref),
[`CasADiCodegen.smooth_abs`](@ref), [`CasADiCodegen.smooth_max`](@ref) and
[`CasADiCodegen.smooth_min`](@ref) give the smooth forms on numbers, SX and MX:

```@example disc
ε = SX.sym("width")
smooth_thrust = CasADiCodegen.smooth_sign(1 - x[1], ε)
model = Dict("x" => x, "p" => ε, "ode" => vcat(x[2], smooth_thrust))
nothing # hide
```

[`CasADiCodegen.smoothed`](@ref) rewrites an expression or a function already written
with switches; `inputs` restricts the rewrite to the switches on the chosen inputs:

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

## Sharpen the smoothing step by step

A Newton solve of a smoothed model converges from fewer guesses as the width falls, so
lower the width step by step. [`CasADiCodegen.natural_continuation`](@ref) solves a
`NonlinearProblem` for a sequence of values of one parameter, each from the root of the
one before, and tries intermediate values when a solve fails. Here the root is the
initial speed `v0` that takes the smoothed model from the position 0 to 1.2 at the time 2:

```@example disc
H = integrator("H", Vern9(), model, 0.0, 2.0; abstol = 1e-10, reltol = 1e-10)
v0, width = MX.sym("v0"), MX.sym("width")
residual = CasadiFunction("residual", [v0, width], [H(vcat(0, v0), width)[1] - 1.2])
shoot = SciMLBase.NonlinearFunction(residual)
problem = SciMLBase.NonlinearProblem(shoot, [0.0], [1.0])
widths = [1.0, 0.1, 0.01, 1e-3]
path = CasADiCodegen.natural_continuation(problem, SimpleNewtonRaphson(), widths;
                                          parameter = :width)
[(value, solution.retcode, only(solution.u)) for (value, solution) in path]
```

For a path with turning points, use the extension for BifurcationKit.jl (see
[Use the generated code with SciML packages](@ref)).
