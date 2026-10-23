# Interpolated data in a model

A model often contains a quantity that is known only through data: a measured forcing,
a table of a material property, or the output of another program sampled on a grid.
An interpolation turns such data into a function, which the equations then call like
any other function, and the data usually change more often than the equations, from one
experiment or one scenario to the next. In this tutorial, we write interpolated data in
a model with the interpolants of CasADi and solve it with OrdinaryDiffEq.jl. We then
replace the data without building the model again, since the values of the interpolant,
and even its grid, can be parameters of the problem, so that the generated code stays
the same and only the numbers that it receives change. Finally, we turn to smooth
interpolation and to the interpolations of DataInterpolations.jl, which the model calls
through [`CasADiCodegen.callback`](@ref), a CasADi function that runs Julia code.

The examples need CasADiCodegen.jl, SciMLBase.jl, OrdinaryDiffEqTsit5.jl,
DataInterpolations.jl and StaticArrays.jl. `CasADiCodegen.C` is the module of the
Julia bindings of CasADi, which holds the CasADi functions that this package does not
export, such as `interpolant`.

```@example callable
using CasADiCodegen, StaticArrays
import SciMLBase, DataInterpolations
using OrdinaryDiffEqTsit5: Tsit5
C = CasADiCodegen.C
nothing # hide
```

## A table as a function of time

As an example, we take the equation ``x' = d(t)`` with ``x(0) = 0``, where ``d`` is the
linear interpolation of the values ``t_k^2`` on the grid ``t_k = 0, 0.1, \ldots, 10``.
The solution is the integral of the interpolation, which we can compute by hand to check
the solver: on ``(0, 1)``, the trapezoidal rule with the step ``0.1`` gives
``x(1) = 0.335`` exactly. `C.interpolant(name, "linear", grids, values)` makes the
interpolant, a CasADi function of the point at which it interpolates; `grids` is a
vector with one increasing grid per dimension. An interpolant can only be called on
symbols of the matrix (MX) type, so the symbols of the model are MX symbols. The
right side `f(u, p, t)` of an `ODEFunction` takes the time as its third argument; since
the model has no parameters yet, the second argument is empty, `MX(0, 1)`, and the
problem needs no parameters:

```@example callable
grid = collect(0.0:0.1:10.0)
table = C.interpolant("table", "linear", [grid], grid .^ 2)
x, t = MX.sym("x"), MX.sym("t")
rhs = CasadiFunction("rhs", [x, MX(0, 1), t], [table(t)])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(rhs), [0.0], (0.0, 1.0))
SciMLBase.solve(problem, Tsit5()).u[end], SciMLBase.solve(problem, Tsit5(); tstops = grid).u[end]
```

The generated code holds the grid and the values as constants, and it computes the
interpolation without calling CasADi. The right side has a kink at each point of the
grid, where its derivative jumps, and an adaptive solver that steps over a kink
estimates its error from a function that is not smooth there. With the points of the
grid as `tstops`, every step stays inside one interval of the grid, where the right side
is linear, and the solution (the second value above) is exact up to rounding, while
without them the error of the first value is of the order of the tolerance of the
solver. Outside its grid, the linear interpolant continues the line of its first or its
last interval.

## Data as parameters

Data that change from one solve to the next belong in the parameters, so that a new set
of data needs no new code. When `C.interpolant` receives the number of values per point
of the grid, `1` here, in place of the values, the interpolant takes the values as its
second argument, after the point. With the option `"inline" => true`, CasADi builds the
interpolant from elementary operations, which it can differentiate with respect to the
values, and the values of the linear interpolation stay the same. Without the option,
CasADi gives zero as the derivative by the values, so that the Jacobian by the
parameters and the sensitivities of the solution to the data are zero, with no error.
The values become the parameter argument of the right side:

```@example callable
table = C.interpolant("table", "linear", [grid], 1, Dict("inline" => true))
data = MX.sym("data", length(grid))
rhs = CasadiFunction("rhs", [x, data, t], [table(t, data)])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(rhs), [0.0], (0.0, 1.0), grid .^ 2)
SciMLBase.solve(problem, Tsit5(); tstops = grid).u[end]
```

A parameter symbol of several elements, such as `data`, is a parameter of the problem
under its own name, besides its elements `data_0`, `data_1`, and so on. `remake` with
the name thus replaces all the values at once, and the new problem keeps the function,
with its generated code, of the old one. With the values ``t_k^3``, the exact solution
is ``x(1) = 0.2525``:

```@example callable
cubic = SciMLBase.remake(problem; p = [:data => grid .^ 3])
cubic.f === problem.f, SciMLBase.solve(cubic, Tsit5(); tstops = grid).u[end]
```

An assignment such as `problem.ps[:data] = grid .^ 3` writes new values into the
parameters of a problem in place, which avoids the copy that `remake` makes; we keep
`problem` as it is here. The Jacobian by the parameters, which
the `ODEFunction` holds as `paramjac`, is the derivative of the interpolation by its
values: at ``t = 0.55``, the two values on either side, at ``0.5`` and ``0.6``, each
have the weight ``1/2``:

```@example callable
J = zeros(1, length(grid))
problem.f.paramjac(J, [0.0], problem.p, 0.55)
findall(!iszero, J), J[1, 6:7]
```

## A grid as a parameter

The grid, too, can be an argument of a linear interpolant: with the number of points
per dimension in place of the grid, the interpolant takes the point, the grid and the
values, in this order. Both become parameters, which the right side joins into one
argument with `vertcat`; each keeps its name. New data may then come on another
increasing grid, provided that it has the same number of points, here the values of
``\sin \pi t`` at 101 points of ``(0, 1)``, whose integral over ``(0, 1)`` is
``2/\pi = 0.63662``. The `tstops` follow the grid that the parameters hold:

```@example callable
table = C.interpolant("table", "linear", [length(grid)], 1, Dict("inline" => true))
points = MX.sym("points", length(grid))
rhs = CasadiFunction("rhs", [x, vertcat(points, data), t], [table(t, points, data)])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(rhs), [0.0], (0.0, 1.0), [grid; grid .^ 2])
finer = collect(range(0.0, 1.0; length = length(grid)))
sine = SciMLBase.remake(problem; p = [:points => finer, :data => sin.(pi .* finer)])
SciMLBase.solve(sine, Tsit5(); tstops = finer).u[end]
```

The result, 0.63657, differs from ``2/\pi`` by the error of the trapezoidal rule on
this grid. CasADi accepts a grid as an argument for the linear interpolant only, not for the
B-spline below.

## Smooth data

The interpolant `"bspline"` passes a cubic B-spline through the values, which has
continuous first and second derivatives, so that the solver meets no kink and needs no
`tstops`. The cubic B-spline through the values ``t_k^2`` is the polynomial ``t^2``
itself, so that the solution is ``x(1) = 1/3`` up to the tolerance of the
solver:

```@example callable
smooth = C.interpolant("smooth", "bspline", [grid], grid .^ 2)
rhs = CasadiFunction("rhs", [x, MX(0, 1), t], [smooth(t)])
SciMLBase.solve(SciMLBase.ODEProblem(SciMLBase.ODEFunction(rhs), [0.0], (0.0, 1.0)), Tsit5()).u[end]
```

A B-spline can also take its values as an argument, as `C.interpolant("smooth",
"bspline", [grid], 1, Dict("inline" => true))`, and its generated code then fits the
coefficients of the spline to the values at each call, which solves a linear system
with one row per point of the grid and costs far more than the interpolation itself
(see the table at the end). As for the linear interpolant, the option `inline` gives the
derivative by the values. Unlike there, it also changes the values outside the grid:
with `inline`, the B-spline continues the polynomials of its first and last intervals,
whereas without it, the B-spline, also one with fixed values such as `smooth`, is zero
there.

## A Julia interpolation in the model

DataInterpolations.jl has many more methods than CasADi, such as natural cubic splines
(`CubicSpline`), Akima and monotone interpolation, and a model can call any of them
through `callback` (see [Call a Julia function from the model](@ref)). The two tuples
give the size of its one argument, the time, and of its one result, the value, which
the Julia function receives and returns as vectors; the keyword `jacobian` gives its derivative,
which `DataInterpolations.derivative` computes, and static arrays keep the call free of
allocations:

```@example callable
spline = DataInterpolations.CubicSpline(grid .^ 2, grid)
interpolation = callback(s -> SVector(spline(s[1])), "spline", (1,), (1,);
                         jacobian = s -> SMatrix{1,1}(DataInterpolations.derivative(spline, s[1])))
rhs = CasadiFunction("rhs", [x, MX(0, 1), t], [interpolation(t)])
SciMLBase.solve(SciMLBase.ODEProblem(SciMLBase.ODEFunction(rhs), [0.0], (0.0, 1.0)), Tsit5()).u[end]
```

The natural spline differs from the B-spline near the ends of the grid, where its second
derivative is zero, so the result is not exactly ``1/3``. The interpolation object is
part of the Julia function, not of the parameters, so new data need a new callback and a
new `ODEFunction`; with the default `specialize = SciMLBase.FullSpecialize`, the solver is
then compiled again (see [Compilation for each new model](@ref)), and each callback stays
in memory for the rest of the session (see [How the code reaches the function](@ref)).
For speed the two routes are close: the generated code of a linear interpolant reads the
grid and the values where they are, in its constants or in the parameter vector, so that
a call costs a search in the grid whatever the number of points, as the callback, which
receives only the time and searches the table of the Julia object. The B-spline with
values as parameters is the exception, since it fits its coefficients at each call. One
call of the right side took:

| right side | 101 points | 10001 points |
|---|---|---|
| linear interpolant, values as parameters | 23 ns | 28 ns |
| B-spline, values as parameters | 10 µs | |
| callback of DataInterpolations.jl | 18 ns | 21 ns |
