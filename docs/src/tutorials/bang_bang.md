# Find a fuel-optimal control that switches on and off

In this tutorial, we solve an optimal control problem whose control switches between full
thrust, no thrust and full reverse thrust (a bang-off-bang control). Indirect shooting
finds the unknown initial values of the multipliers ``\lambda``, once with exact switch
times found by the integrator and once with a smoothed control.


```@example bang
using CasADiCodegen
using OrdinaryDiffEqVerner: Vern9
using SimpleNonlinearSolve: SimpleNewtonRaphson
import SciMLBase
nothing # hide
```

## Move a mass with least fuel: problem and exact answer

A double integrator ``\ddot{x} = u`` with ``|u| \le 1`` moves from rest at ``x = 0``
to rest at ``x = 1`` in the time ``T = 3`` with the cost ``\int_0^T |u| \, dt``. The
control switches at ``\tau = (3 - \sqrt{5})/2`` and ``T - \tau``, which gives the two
unknowns of the shooting:

```math
\lambda_x = -\frac{2}{\sqrt{5}}, \qquad \lambda_v(0) = -1 - \frac{2\tau}{\sqrt{5}}.
```

!!! details "How the Hamiltonian gives the switching times"
    The Hamiltonian is

    ```math
    H = |u| + \lambda_x v + \lambda_v u,
    ```

    so ``\dot\lambda_v = -\lambda_x`` with ``\lambda_x`` constant, and the control is
    ``-\operatorname{sign}(\lambda_v)`` where ``|\lambda_v| > 1`` and zero elsewhere. It
    switches where ``\lambda_v = \mp 1``, at ``\tau`` and ``T - \tau``.

```@example bang
τ = (3 - sqrt(5)) / 2
exact = [-2 / sqrt(5), -1 - 2τ / sqrt(5)]                  # [λx, λv(0)]
nothing # hide
```

## Why an `if_else` control gives zero derivatives

The shooting integrates ``y = (x, v, \lambda_v)`` with the parameter ``\lambda_x``, and
its residual is the final position minus 1 and the final velocity. The direct form of the
control is a nested `if_else` on ``\lambda_v``:

```@example bang
y, a = SX.sym("y", 3), SX.sym("a")                         # y = [x, v, λv], a = λx
λv = y[3]
u = casadi.if_else(λv > 1, SX(-1.0), casadi.if_else(λv < -1, SX(1.0), SX(0.0)))
ode = vcat(y[2], u, -a)
rhs = CasadiFunction("rhs", [y, a], [ode], ["y", "a"], ["ode"])
CasADiCodegen.nonsmooth_operations(rhs; inputs = ["y"])
```

[`CasADiCodegen.nonsmooth_operations`](@ref) finds two comparisons on the state, and
[`integrator`](@ref) warns about them. CasADi differentiates each constant branch
separately, so the Jacobian of the residual is zero; `nonsmooth = :ignore` silences the
warning:

```@example bang
branch_model = Dict("x" => y, "p" => a, "ode" => ode)
branchwise = integrator("branchwise", Vern9(), branch_model, 0.0, 3.0;
                        nonsmooth = :ignore, abstol = 1e-12, reltol = 1e-12)
z = MX.sym("z", 2)                                         # z = [λx, λv(0)]
yb = branchwise(vcat(0, 0, z[2]), z[1])
residual = vcat(yb[1] - 1, yb[2])
CasadiFunction("check", [z], [jacobian(residual, z)])(exact)
```

The next two sections add the missing jump of the state transition matrix (the derivative
of the state by its initial value) at the switching times.

## Solve with exact switch times found by the integrator

With the keyword `modes = m => S` of `integrator` (see
[Locate the switches exactly with `modes`](@ref)) and the switching functions
``\lambda_v \mp 1``, the
control is ``u = -(m_1 + m_2)/2``, and `switches` bounds the number of switches:

```@example bang
m = SX.sym("m", 2)
switched_ode = vcat(y[2], -(m[1] + m[2]) / 2, -a)
switched_model = Dict("x" => y, "p" => a, "ode" => switched_ode)
switching = vcat(λv - 1, λv + 1)
switched = integrator("switched", Vern9(), switched_model, 0.0, 3.0; switches = 2,
                      modes = m => switching, abstol = 1e-12, reltol = 1e-12)
ym = switched(vcat(0, 0, z[2]), z[1])
shoot = CasadiFunction("shoot", [z], [vcat(ym[1] - 1, ym[2])], ["z"], ["r"])
nothing # hide
```

Newton's method needs a guess with the right sequence of switches:

```@example bang
near_guess = [-1.0, -1.5]
problem = SciMLBase.NonlinearProblem(SciMLBase.NonlinearFunction(shoot), near_guess)
solution = SciMLBase.solve(problem, SimpleNewtonRaphson(); abstol = 1e-10)
solution.retcode, maximum(abs, solution.u - exact)
```

From the guess `[-2.0, -3.0]`, it meets a singular Jacobian.

## Solve with a smoothed control, then sharpen it

[`CasADiCodegen.smoothed`](@ref) rewrites each switch into a smooth form (see
[Replace the switches with smooth functions](@ref)). The width is a parameter after
``\lambda_x``, and
`inputs = y` rewrites only the switches on the state:

```@example bang
w, W = SX.sym("width"), MX.sym("width")
smooth_ode = CasADiCodegen.smoothed(ode, w; inputs = y)
smooth_model = Dict("x" => y, "p" => vcat(a, w), "ode" => smooth_ode)
smooth = integrator("smooth", Vern9(), smooth_model, 0.0, 3.0;
                    abstol = 1e-12, reltol = 1e-12)
ys = smooth(vcat(0, 0, z[2]), vcat(z[1], W))
shoot_smooth = CasadiFunction("shoot_smooth", [z, W], [vcat(ys[1] - 1, ys[2])],
                              ["z", "width"], ["r"])
nothing # hide
```

The smoothed root tends to the exact one as the width falls, but far from the root a
small width makes the Jacobian nearly singular; from `[-2.0, -3.0]`, Newton's method
converges at the width 1 but not at 0.1:

```@example bang
far_guess = [-2.0, -3.0]
smooth_function = SciMLBase.NonlinearFunction(shoot_smooth)
smooth_problem = SciMLBase.NonlinearProblem(smooth_function, far_guess, [1.0])   # width 1
function direct(width)
    try
        at_width = SciMLBase.remake(smooth_problem; p = [width])
        SciMLBase.solve(at_width, SimpleNewtonRaphson(); abstol = 1e-10).retcode
    catch err
        typeof(err)
    end
end
direct(1.0), direct(0.1)
```

[`CasADiCodegen.natural_continuation`](@ref) follows the width toward zero (see
[Sharpen the smoothing step by step](@ref)):

```@example bang
widths = [1.0, 0.1, 0.01, 1e-3, 1e-4]
path = CasADiCodegen.natural_continuation(smooth_problem, SimpleNewtonRaphson(), widths;
                                          parameter = :width, abstol = 1e-10)
[(width, s.retcode, maximum(abs, s.u - exact)) for (width, s) in path]
```

The error falls as the square of the width, as at any transversal crossing (see
[Replace the switches with smooth functions](@ref)).

## Use the smoothed solution as the starting guess

The smoothed root at a moderate width is a good guess for the switched model:

```@example bang
smoothed_root = Dict(path)[0.1].u                          # the root at the width 0.1
from_smoothed = SciMLBase.remake(problem; u0 = smoothed_root)
refined = SciMLBase.solve(from_smoothed, SimpleNewtonRaphson(); abstol = 1e-10)
refined.retcode, maximum(abs, refined.u - exact)
```

[Get correct derivatives from a model with switches](@ref discontinuities) explains the
choice between the two methods.
