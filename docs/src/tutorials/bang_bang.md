# Shoot a bang-off-bang control

In this tutorial, we solve an optimal control problem whose control switches between
full thrust, no thrust and full reverse thrust, by indirect shooting on the initial
costates. The right side of the shooting integration then switches on the state, so the
derivatives that Newton's method needs are wrong when the switch is written with
`if_else`. We solve the problem twice, once with the switches given as modes of the
integrator and once with a smoothed model followed down to a small width, and we check
both against the closed form. The examples need CasADiCodegen.jl, SciMLBase.jl,
OrdinaryDiffEqVerner.jl and SimpleNonlinearSolve.jl.

```@example bang
using CasADiCodegen
using OrdinaryDiffEqVerner: Vern9
using SimpleNonlinearSolve: SimpleNewtonRaphson
import SciMLBase
const C = CasADiCodegen.C
nothing # hide
```

## The problem and its solution

A double integrator ``\ddot{x} = u`` with a bounded control ``|u| \le 1`` moves from rest
at ``x = 0`` to rest at ``x = 1`` in the time ``T = 3``, and the cost is the effort
``\int_0^T |u| \, dt``. With the costates ``\lambda_x`` and ``\lambda_v`` of the position
and the velocity, the Hamiltonian is

```math
H = |u| + \lambda_x v + \lambda_v u,
```

so ``\lambda_x`` is constant and ``\dot\lambda_v = -\lambda_x``. The control that minimizes
``H`` is ``u = -\operatorname{sign}(\lambda_v)`` where ``|\lambda_v| > 1`` and ``u = 0``
where ``|\lambda_v| < 1``. Since ``\lambda_v`` is linear in time, the control is full
thrust up to a time ``\tau``, then no thrust, then full reverse thrust from ``T - \tau``,
and the distance ``3\tau - \tau^2 = 1`` gives ``\tau = (3 - \sqrt{5})/2``. The costate
``\lambda_v`` equals -1 at ``\tau`` and +1 at ``T - \tau``, which fixes the two unknowns of
the shooting:

```math
\lambda_x = -\frac{2}{\sqrt{5}}, \qquad \lambda_v(0) = -1 - \frac{2\tau}{\sqrt{5}}.
```

```@example bang
τ = (3 - sqrt(5)) / 2
exact = [-2 / sqrt(5), -1 - 2τ / sqrt(5)]                  # [λx, λv(0)]
nothing # hide
```

## The control written with `if_else`

The shooting integrates the state ``y = (x, v, \lambda_v)`` with the parameter
``\lambda_x`` from ``y(0) = (0, 0, \lambda_v(0))``, and its residual is the final position
minus 1 and the final velocity. The direct form of the control is a nested `if_else` on
``\lambda_v``, an element of the state:

```@example bang
y, a = SX.sym("y", 3), SX.sym("a")                         # y = [x, v, λv], a = λx
λv = y[3]
u = C.if_else(λv > 1, SX(-1.0), C.if_else(λv < -1, SX(1.0), SX(0.0)))
ode = vertcat(y[2], u, -a)
rhs = CasadiFunction("rhs", [y, a], [ode], ["y", "a"], ["ode"])
CasADiCodegen.nonsmooth_operations(rhs; inputs = ["y"])
```

[`CasADiCodegen.nonsmooth_operations`](@ref) finds two comparisons on the state, and
[`integrator`](@ref) reports the same two surfaces in a warning when it is built. The values
of this model are right, but CasADi differentiates each branch of `if_else` by itself, and
each branch is a constant, so the Jacobian of the residual by the two costates is zero,
whereas its true value depends on the switching times that the costates move. We build
the integrator with `nonsmooth = :ignore`, which silences the warning, to show the zero
Jacobian at the exact root:

```@example bang
branchwise = integrator("branchwise", Vern9(), Dict("x" => y, "p" => a, "ode" => ode), 0.0, 3.0;
                        nonsmooth = :ignore, abstol = 1e-12, reltol = 1e-12)
z = MX.sym("z", 2)                                         # z = [λx, λv(0)]
yb = branchwise(vertcat(0, 0, z[2]), z[1])
residual = vertcat(yb[1] - 1, yb[2])
CasadiFunction("check", [z], [jacobian(residual, z)])(exact)
```

Newton's method cannot move from any guess with this Jacobian. The two routes below give
the derivatives the term that the branch-wise derivative misses, the jump of the state
transition matrix (STM) at the switching time.

## Route 1: the switches as modes

The keyword `modes = m => S` of `integrator` replaces the sign of each switching function
by a symbol, the mode, which is +1 where its switching function is positive and -1 where
it is negative. The integrator turns each change of sign into an event that flips the
mode, and CasADi differentiates the event times by the implicit function theorem, so the
derivatives include the jump. Here the switching functions are ``\lambda_v - 1`` and
``\lambda_v + 1``, and the control is ``u = -(m_1 + m_2)/2``, which is +1, 0 and -1 in the
three regions. The keyword `switches` is an upper bound on the number of switches, 2 for
this control:

```@example bang
m = SX.sym("m", 2)
switched = integrator("switched", Vern9(), Dict("x" => y, "p" => a, "ode" => vertcat(y[2], -(m[1] + m[2]) / 2, -a)), 0.0, 3.0;
                      modes = m => vertcat(λv - 1, λv + 1), switches = 2, abstol = 1e-12, reltol = 1e-12)
ym = switched(vertcat(0, 0, z[2]), z[1])
shoot = CasadiFunction("shoot", [z], [vertcat(ym[1] - 1, ym[2])], ["z"], ["r"])
nothing # hide
```

The function `switched` has the inputs and the result of the model without modes, the
initial state and the parameter, since `switch_times` is not set, so the residual is
written as before.
`SciMLBase.NonlinearFunction` makes generated code of the residual and its Jacobian, and
Newton's method starts from a guess that has the right sequence of switches:

```@example bang
problem = SciMLBase.NonlinearProblem(SciMLBase.NonlinearFunction(shoot), [-1.0, -1.5])
solution = SciMLBase.solve(problem, SimpleNewtonRaphson(); abstol = 1e-10)
solution.retcode, maximum(abs, solution.u - exact)
```

The root agrees with the closed form to the tolerances of the integration. The guess
matters here: from the guess `[-2.0, -3.0]`, Newton's method meets a singular Jacobian.

```@example bang
try
    SciMLBase.solve(SciMLBase.remake(problem; u0 = [-2.0, -3.0]), SimpleNewtonRaphson(); abstol = 1e-10)
catch err
    typeof(err)
end
```

## Route 2: a smoothed control and continuation in its width

[`CasADiCodegen.smoothed`](@ref) rewrites each switch of an expression into a smooth form
of a given width ``\varepsilon``; a comparison ``s > 0`` becomes
``(1 + \tanh(s/\varepsilon))/2``. The width is a parameter of the model, after
``\lambda_x``. The keyword `inputs = y` rewrites the switches on the state only:

```@example bang
w, W = SX.sym("width"), MX.sym("width")
smooth = integrator("smooth", Vern9(), Dict("x" => y, "p" => vertcat(a, w), "ode" => CasADiCodegen.smoothed(ode, w; inputs = y)), 0.0, 3.0;
                    abstol = 1e-12, reltol = 1e-12)
ys = smooth(vertcat(0, 0, z[2]), vertcat(z[1], W))
shoot_smooth = CasadiFunction("shoot_smooth", [z, W], [vertcat(ys[1] - 1, ys[2])], ["z", "width"], ["r"])
nothing # hide
```

The smoothed model has an exact Jacobian for each width, and its root tends to that of the
switched model as the width falls. With a small width, however, the control is nearly
constant in the costates wherever ``|\lambda_v|`` stays away from 1, so the Jacobian at an
iterate far from the root can be nearly singular. From the guess `[-2.0, -3.0]`, Newton's
method converges at the width 1 and meets a singular Jacobian at the width 0.1:

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

[`CasADiCodegen.natural_continuation`](@ref) solves the problem for a decreasing sequence
of widths, each solve starting from the root of the one before, and finds the parameter by
the name of its symbol, `:width`:

```@example bang
path = CasADiCodegen.natural_continuation(smooth_problem, SimpleNewtonRaphson(), [1.0, 0.1, 0.01, 1e-3, 1e-4];
                                          parameter = :width, abstol = 1e-10)
[(width, s.retcode, maximum(abs, s.u - exact)) for (width, s) in path]
```

The error of the root falls by a factor of 100 for each factor of 10 in the width, as the
square of the width. Both switching functions cross zero with the nonzero rate
``\dot\lambda_v = -\lambda_x``, and the odd symmetry of ``\tanh`` cancels the error of the
first order in the width.

## The two routes together

The routes combine: the smoothed problem converges from a rough guess, and its root at a
moderate width has the switching structure of the exact solution, so it is a good guess
for the switched model, whose root has no error from the width:

```@example bang
refined = SciMLBase.solve(SciMLBase.remake(problem; u0 = Dict(path)[0.1].u), SimpleNewtonRaphson(); abstol = 1e-10)
refined.retcode, maximum(abs, refined.u - exact)
```

The width 0.1 was enough here. [Discontinuous models](@ref discontinuities) explains the choice between
the routes in general, and in particular the case of a sliding arc, where the switching
function stays at zero over an interval and only the smoothed route applies.
