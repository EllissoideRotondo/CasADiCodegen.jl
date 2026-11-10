# Minimize the Rosenbrock function

In this tutorial, we minimize the Rosenbrock function with Ipopt through Optimization.jl,
first inside a box and then with a nonlinear constraint, using generated code for the
objective, the constraints and their derivatives.

## Write the objective

The Rosenbrock function

```math
f(x, y) = (a - x)^2 + b\,(y - x^2)^2
```

has its minimum ``f = 0`` at ``x = a``, ``y = a^2``. The objective of a SciML optimization
problem takes the unknowns and the parameters as two vectors, so we make one symbol `z`
for ``x`` and ``y`` and join ``a`` and ``b`` in `p`:

```@example optimization
using CasADiCodegen
import SciMLBase

z = SX.sym("z", 2)
a, b = SX.sym("a"), SX.sym("b")
p = vcat(a, b)
x, y = z[1], z[2]
rosenbrock = CasadiFunction("rosenbrock", [z, p], [(a - x)^2 + b * (y - x^2)^2])
```

## Solve inside a box

`SciMLBase.OptimizationFunction` generates code for the objective, its gradient and its
Hessian, which CasADi derives (see [Give a CasADi function to a SciML problem](@ref)), so
the problem takes no `adtype`. We set ``a = 1``, ``b = 100`` and the box
``-2 \le x \le 2``, ``-1 \le y \le 3``:

```@example optimization
f = SciMLBase.OptimizationFunction(rosenbrock)
problem = SciMLBase.OptimizationProblem(f, [1.0, 2.0], [1.0, 100.0]; lb = [-2.0, -1.0], ub = [2.0, 3.0])
nothing # hide
```

We solve with Ipopt and silence its output:

```@example optimization
using OptimizationIpopt: IpoptOptimizer

ipopt = IpoptOptimizer(additional_options = Dict{String,Any}("print_level" => 0, "sb" => "yes"))
solution = SciMLBase.solve(problem, ipopt)
solution.u
```

The solution is ``(1, 1)``, where the objective is zero to the tolerance of Ipopt:

```@example optimization
solution.objective
```

## Change the parameters

The parameters are an argument of the generated code, so `remake` changes them without
generating code again; with ``a = 1/2`` the minimum moves to ``(1/2, 1/4)``:

```@example optimization
solution = SciMLBase.solve(SciMLBase.remake(problem; p = [0.5, 100.0]), ipopt)
solution.u
```

## Add a constraint

We restrict the point to the unit disk ``x^2 + y^2 \le 1``, which excludes ``(1, 1)``.
The constraints form a second CasADi function with the arguments of the objective, and the
problem bounds them with `lcons` and `ucons`:

```@example optimization
disk = CasadiFunction("disk", [z, p], [x^2 + y^2])
nothing # hide
```

With `sparse = true`, the function holds the Hessian of the Lagrange function
``\sigma f + \mu^T g`` as one generated function, `lag_h`, and the Jacobian and the
Hessians become sparse matrices whose prototypes carry the pattern that CasADi finds.
The sparse form needs SparseArrays to be loaded:

```@example optimization
using SparseArrays

f = SciMLBase.OptimizationFunction(rosenbrock, disk; sparse = true)
f.lag_hess_prototype
```

We start inside the disk; `lcons = [-Inf]` makes the constraint one-sided:

```@example optimization
problem = SciMLBase.OptimizationProblem(f, [0.14, 0.14], [1.0, 100.0];
                                        lb = [-2.0, -1.0], ub = [2.0, 3.0], lcons = [-Inf], ucons = [1.0])
solution = SciMLBase.solve(problem, ipopt)
solution.u
```

The minimum lies on the circle, where the constraint is active:

```@example optimization
g = zeros(1)
f.cons(g, solution.u, problem.p)
g
```

## Write the problem with the Opti stack

The `Opti` stack of CasADi keeps the variables, the parameter values, the initial point
and the constraints with their bounds in one object, and
`SciMLBase.OptimizationProblem(opti)` makes the SciML problem from it. Its functions are
those of the Python API, and `set_value` is `C.set_value`. A chained comparison such as
`-2 <= w[1] <= 2` is not a constraint, because Julia evaluates it with `&&`; write
`bounded(lb, w, ub)` instead:

```@example optimization
C = CasADiCodegen.C
opti = Opti()
w = variable(opti, 2)
pa, pb = parameter(opti), parameter(opti)
minimize(opti, (pa - w[1])^2 + pb * (w[2] - w[1]^2)^2)
subject_to(opti, w[1]^2 + w[2]^2 <= 1)
subject_to(opti, bounded([-2.0, -1.0], w, [2.0, 3.0]))
set_initial(opti, w, DM([0.14, 0.14]))
C.set_value(opti, pa, DM(1.0))
C.set_value(opti, pb, DM(100.0))

problem = SciMLBase.OptimizationProblem(opti; sparse = true)
problem.lcons, problem.ucons
```

`opti` keeps a bound on a variable as a constraint, so the problem has three constraints,
in the order of the calls of `subject_to`, and no `lb` and `ub`. Ipopt reaches the same
minimum:

```@example optimization
solution = SciMLBase.solve(problem, ipopt)
solution.u
```

`C.value` evaluates an expression of `opti` at the solution:

```@example optimization
Float64(C.value(opti, w[1]^2 + w[2]^2, [C.x(opti) == DM(solution.u)]))
```
