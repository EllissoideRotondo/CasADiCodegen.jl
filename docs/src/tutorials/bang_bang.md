# Shoot a bang-off-bang control

In this tutorial, we solve a bang-off-bang optimal control problem by indirect shooting,
once with the switches as modes of the integrator and once with a smoothed control.

```@example bang
using CasADiCodegen
using OrdinaryDiffEqVerner: Vern9
using SimpleNonlinearSolve: SimpleNewtonRaphson
import SciMLBase
const C = CasADiCodegen.C
nothing # hide
```

## The problem and its solution

A double integrator ``\ddot{x} = u`` with ``|u| \le 1`` moves from rest at ``x = 0``
to rest at ``x = 1`` in the time ``T = 3`` with the cost ``\int_0^T |u| \, dt``. The
Hamiltonian is

```math
H = |u| + \lambda_x v + \lambda_v u,
```

so ``\dot\lambda_v = -\lambda_x`` with ``\lambda_x`` constant, and the control is
``-\operatorname{sign}(\lambda_v)`` where ``|\lambda_v| > 1`` and zero elsewhere. It
switches at ``\tau = (3 - \sqrt{5})/2`` and ``T - \tau``, where ``\lambda_v = \mp 1``,
which gives the two unknowns of the shooting:

```math
\lambda_x = -\frac{2}{\sqrt{5}}, \qquad \lambda_v(0) = -1 - \frac{2\tau}{\sqrt{5}}.
```

```@example bang
τ = (3 - sqrt(5)) / 2
exact = [-2 / sqrt(5), -1 - 2τ / sqrt(5)]                  # [λx, λv(0)]
nothing # hide
```

## The control written with `if_else`

The shooting integrates ``y = (x, v, \lambda_v)`` with the parameter ``\lambda_x``, and
its residual is the final position minus 1 and the final velocity. The direct form of the
control is a nested `if_else` on ``\lambda_v``:

```@example bang
y, a = SX.sym("y", 3), SX.sym("a")                         # y = [x, v, λv], a = λx
λv = y[3]
u = C.if_else(λv > 1, SX(-1.0), C.if_else(λv < -1, SX(1.0), SX(0.0)))
ode = vcat(y[2], u, -a)
rhs = CasadiFunction("rhs", [y, a], [ode], ["y", "a"], ["ode"])
CasADiCodegen.nonsmooth_operations(rhs; inputs = ["y"])
```

[`CasADiCodegen.nonsmooth_operations`](@ref) finds two comparisons on the state, and
[`integrator`](@ref) warns about them. CasADi differentiates each constant branch of
`if_else` separately, so the Jacobian of the residual is zero, as we show with the warning
silenced by `nonsmooth = :ignore`:

```@example bang
branchwise = integrator("branchwise", Vern9(), Dict("x" => y, "p" => a, "ode" => ode), 0.0, 3.0;
                        nonsmooth = :ignore, abstol = 1e-12, reltol = 1e-12)
z = MX.sym("z", 2)                                         # z = [λx, λv(0)]
yb = branchwise(vcat(0, 0, z[2]), z[1])
residual = vcat(yb[1] - 1, yb[2])
CasadiFunction("check", [z], [jacobian(residual, z)])(exact)
```

The two routes below add the missing jump of the state transition matrix at the switching
times.

## Route 1: the switches as modes

The keyword `modes = m => S` of `integrator` replaces the sign of each switching function
by a mode, +1 or -1, which an event flips at each change of sign; CasADi differentiates the
event times, so the derivatives include the jump. With the switching functions
``\lambda_v \mp 1``, the control is ``u = -(m_1 + m_2)/2``, and `switches` bounds the
number of switches:

```@example bang
m = SX.sym("m", 2)
switched = integrator("switched", Vern9(), Dict("x" => y, "p" => a, "ode" => vcat(y[2], -(m[1] + m[2]) / 2, -a)), 0.0, 3.0;
                      modes = m => vcat(λv - 1, λv + 1), switches = 2, abstol = 1e-12, reltol = 1e-12)
ym = switched(vcat(0, 0, z[2]), z[1])
shoot = CasadiFunction("shoot", [z], [vcat(ym[1] - 1, ym[2])], ["z"], ["r"])
nothing # hide
```

Without `switch_times`, `switched` has the inputs and result of the model without modes.
Newton's method needs a guess with the right sequence of switches:

```@example bang
problem = SciMLBase.NonlinearProblem(SciMLBase.NonlinearFunction(shoot), [-1.0, -1.5])
solution = SciMLBase.solve(problem, SimpleNewtonRaphson(); abstol = 1e-10)
solution.retcode, maximum(abs, solution.u - exact)
```

From the guess `[-2.0, -3.0]`, Newton's method meets a singular Jacobian:

```@example bang
try
    SciMLBase.solve(SciMLBase.remake(problem; u0 = [-2.0, -3.0]), SimpleNewtonRaphson(); abstol = 1e-10)
catch err
    typeof(err)
end
```

## Route 2: a smoothed control and continuation in its width

[`CasADiCodegen.smoothed`](@ref) rewrites each switch into a smooth form of width
``\varepsilon``; a comparison ``s > 0`` becomes ``(1 + \tanh(s/\varepsilon))/2``. The
width is a parameter after ``\lambda_x``, and `inputs = y` rewrites only the switches on
the state:

```@example bang
w, W = SX.sym("width"), MX.sym("width")
smooth = integrator("smooth", Vern9(), Dict("x" => y, "p" => vcat(a, w), "ode" => CasADiCodegen.smoothed(ode, w; inputs = y)), 0.0, 3.0;
                    abstol = 1e-12, reltol = 1e-12)
ys = smooth(vcat(0, 0, z[2]), vcat(z[1], W))
shoot_smooth = CasadiFunction("shoot_smooth", [z, W], [vcat(ys[1] - 1, ys[2])], ["z", "width"], ["r"])
nothing # hide
```

The root of the smoothed model tends to the exact one as the width falls, but far from
the root a small width makes the Jacobian nearly singular; from `[-2.0, -3.0]`, Newton's
method converges at the width 1 but not at 0.1:

```@example bang
smooth_problem = SciMLBase.NonlinearProblem(SciMLBase.NonlinearFunction(shoot_smooth), [-2.0, -3.0], [1.0])
function direct(width)
    try
        SciMLBase.solve(SciMLBase.remake(smooth_problem; p = [width]), SimpleNewtonRaphson(); abstol = 1e-10).retcode
    catch err
        typeof(err)
    end
end
direct(1.0), direct(0.1)
```

[`CasADiCodegen.natural_continuation`](@ref) solves for a decreasing sequence of widths,
each from the root of the one before, and finds the parameter by its name `:width`:

```@example bang
path = CasADiCodegen.natural_continuation(smooth_problem, SimpleNewtonRaphson(), [1.0, 0.1, 0.01, 1e-3, 1e-4];
                                          parameter = :width, abstol = 1e-10)
[(width, s.retcode, maximum(abs, s.u - exact)) for (width, s) in path]
```

The error falls as the square of the width, since both switching functions cross zero
at a nonzero rate and the odd symmetry of ``\tanh`` cancels the first-order error.

## The two routes together

The smoothed root at a moderate width is a good guess for the switched model:

```@example bang
refined = SciMLBase.solve(SciMLBase.remake(problem; u0 = Dict(path)[0.1].u), SimpleNewtonRaphson(); abstol = 1e-10)
refined.retcode, maximum(abs, refined.u - exact)
```

[Discontinuous models](@ref discontinuities) explains the choice between the routes,
including a sliding arc, where only the smoothed route applies.
