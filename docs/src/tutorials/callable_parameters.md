# Interpolated data in a model

In this tutorial, we put interpolated data in a model with `casadi.interpolant`, and
replace the data and the grid through the
parameters without generating code again. We end with smooth interpolation and with
DataInterpolations.jl, called through
[`CasADiCodegen.callback`](@ref CasADiCodegen.callback(::Any, ::AbstractString, ::Tuple, ::Tuple)).

```@example callable
using CasADiCodegen, StaticArrays
import SciMLBase, DataInterpolations
using OrdinaryDiffEqTsit5: Tsit5
nothing # hide
```

## A table as a function of time

We take ``x' = d(t)``, ``x(0) = 0``, where ``d`` interpolates ``t_k^2`` linearly on
``t_k = 0, 0.1, \ldots, 10``; the trapezoidal rule gives the exact ``x(1) = 0.335``.
`casadi.interpolant(name, "linear", grids, values)` takes one increasing grid per dimension. An
interpolant can only be called on MX symbols, and the empty `MX(0, 1)` stands for the
missing parameters:

```@example callable
grid = collect(0.0:0.1:10.0)
table = casadi.interpolant("table", "linear", [grid], grid .^ 2)
x, t = MX.sym("x"), MX.sym("t")
rhs = CasadiFunction("rhs", [x, MX(0, 1), t], [table(t)])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(rhs), [0.0], (0.0, 1.0))
SciMLBase.solve(problem, Tsit5()).u[end], SciMLBase.solve(problem, Tsit5(); tstops = grid).u[end]
```

The generated code holds the grid and the values as constants. The right side has a kink at each grid point, so with the grid as `tstops` the solution
(the second value) is exact up to rounding, while without them the error is of the order
of the tolerance. Outside its grid, the linear interpolant continues its first or last
interval.

## Data as parameters

With the number of values per grid point, `1`, in place of the values, the interpolant
takes the values as a second argument, which becomes a parameter. The option
`"inline" => true` makes the interpolant differentiable by the values; without it,
CasADi gives zero as that derivative, so the parameter Jacobian and the sensitivities to
the data are zero with no error:

```@example callable
table = casadi.interpolant("table", "linear", [grid], 1, Dict("inline" => true))
data = MX.sym("data", length(grid))
rhs = CasadiFunction("rhs", [x, data, t], [table(t, data)])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(rhs), [0.0], (0.0, 1.0), grid .^ 2)
SciMLBase.solve(problem, Tsit5(); tstops = grid).u[end]
```

The parameter `data` has its own name besides its elements `data_0`, `data_1`, …, so
`remake` replaces all values at once and keeps the generated code. With the values
``t_k^3``, the exact solution is ``x(1) = 0.2525``:

```@example callable
cubic = SciMLBase.remake(problem; p = [:data => grid .^ 3])
cubic.f === problem.f, SciMLBase.solve(cubic, Tsit5(); tstops = grid).u[end]
```

`problem.ps[:data] = grid .^ 3` writes the values in place instead of copying. The
`paramjac` of the function is the derivative by the values: at ``t = 0.55``, the values at
``0.5`` and ``0.6`` each have the weight ``1/2``:

```@example callable
J = zeros(1, length(grid))
problem.f.paramjac(J, [0.0], problem.p, 0.55)
findall(!iszero, J), J[1, 6:7]
```

## A grid as a parameter

With the number of points in place of the grid, a linear interpolant takes the point,
the grid and the values. New data may then come on another increasing grid of the same
number of points, here ``\sin \pi t`` on ``(0, 1)``, whose integral is
``2/\pi = 0.63662``:

```@example callable
table = casadi.interpolant("table", "linear", [length(grid)], 1, Dict("inline" => true))
points = MX.sym("points", length(grid))
rhs = CasadiFunction("rhs", [x, vcat(points, data), t], [table(t, points, data)])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(rhs), [0.0], (0.0, 1.0), [grid; grid .^ 2])
finer = collect(range(0.0, 1.0; length = length(grid)))
sine = SciMLBase.remake(problem; p = [:points => finer, :data => sin.(pi .* finer)])
SciMLBase.solve(sine, Tsit5(); tstops = finer).u[end]
```

The difference from ``2/\pi`` is the error of the trapezoidal rule. CasADi takes a grid
as an argument for the linear interpolant only, not for the B-spline.

## Smooth data

The interpolant `"bspline"` is a cubic B-spline with continuous second derivatives, so
the solver needs no `tstops`. Through ``t_k^2`` it is ``t^2`` itself, and
``x(1) = 1/3`` up to the tolerance:

```@example callable
smooth = casadi.interpolant("smooth", "bspline", [grid], grid .^ 2)
rhs = CasadiFunction("rhs", [x, MX(0, 1), t], [smooth(t)])
SciMLBase.solve(SciMLBase.ODEProblem(SciMLBase.ODEFunction(rhs), [0.0], (0.0, 1.0)), Tsit5()).u[end]
```

A B-spline can also take its values as an argument, with
`casadi.interpolant("smooth", "bspline", [grid], 1, Dict("inline" => true))`; its generated code
then fits the spline coefficients at each call, which costs far more than the
interpolation (see the table below). Outside the grid, a B-spline with `inline` continues
its end polynomials, and one without `inline`, such as `smooth`, is zero.

## A Julia interpolation in the model

A model can call any interpolation of DataInterpolations.jl through `callback` (see
[Call a Julia function from the model](@ref)). The tuples give the sizes of the argument
and the result, `jacobian` gives the derivative, and static arrays avoid allocations:

```@example callable
spline = DataInterpolations.CubicSpline(grid .^ 2, grid)
interpolation = callback(s -> SVector(spline(s[1])), "spline", (1,), (1,);
                         jacobian = s -> SMatrix{1,1}(DataInterpolations.derivative(spline, s[1])))
rhs = CasadiFunction("rhs", [x, MX(0, 1), t], [interpolation(t)])
SciMLBase.solve(SciMLBase.ODEProblem(SciMLBase.ODEFunction(rhs), [0.0], (0.0, 1.0)), Tsit5()).u[end]
```

The interpolation object is part of the Julia function, so new data need a new
callback and a new `ODEFunction`, which the solver compiles again (see
[Compilation for each new model](@ref)); each callback stays in memory for the rest of the
session (see [How the code reaches the function](@ref)). A call of the linear interpolant
or of the callback costs a search in the grid whatever its size, while the B-spline with
values as parameters fits its coefficients at each call. One call of the right side took:

| right side | 101 points | 10001 points |
|---|---|---|
| linear interpolant, values as parameters | 23 ns | 28 ns |
| B-spline, values as parameters | 10 µs | |
| callback of DataInterpolations.jl | 18 ns | 21 ns |
