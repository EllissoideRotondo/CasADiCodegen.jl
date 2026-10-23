# Minimize the Rosenbrock function

In this tutorial, we minimize the Rosenbrock function of two variables, first inside a box
and then with a nonlinear inequality constraint as well, with generated code for the
objective, the constraints and their derivatives, and we solve each problem with Ipopt
through Optimization.jl. We write the model once as CasADi functions, turn them into a
SciML optimization problem, and change the values of the parameters without generating
the code again; at the end, we write the constrained problem with the `Opti` stack of
CasADi, which keeps the bounds and the constraints together with the model. The examples
need CasADiCodegen.jl, SciMLBase.jl, OptimizationIpopt.jl and the standard library SparseArrays.

## Write the objective

The Rosenbrock function

```math
f(x, y) = (a - x)^2 + b\,(y - x^2)^2
```

has its minimum ``f = 0`` at ``x = a``, ``y = a^2``, at the bottom of a narrow curved
valley, which makes it a common test of an optimization method. The objective of a SciML
optimization problem takes the unknowns as a column vector and, optionally, the
parameters as a second argument, so we make one symbol `z` of two elements for ``x`` and
``y``, and join the parameters ``a`` and ``b`` in the vector `p`:

```@example optimization
using CasADiCodegen
import SciMLBase

z = SX.sym("z", 2)
a, b = SX.sym("a"), SX.sym("b")
p = vertcat(a, b)
x, y = z[1], z[2]
rosenbrock = CasadiFunction("rosenbrock", [z, p], [(a - x)^2 + b * (y - x^2)^2])
```

## Solve inside a box

`SciMLBase.OptimizationFunction` generates Julia code for the objective, its gradient
and its Hessian, which CasADi computes from the graph of the expression (see
[Give a CasADi function to a SciML problem](@ref)). The function therefore needs no
automatic differentiation in Julia, and the problem takes no `adtype`. The problem
holds the initial point, the values of the parameters, here ``a = 1`` and ``b = 100``,
and the box ``-2 \le x \le 2``, ``-1 \le y \le 3`` as the bounds `lb` and `ub`:

```@example optimization
f = SciMLBase.OptimizationFunction(rosenbrock)
problem = SciMLBase.OptimizationProblem(f, [1.0, 2.0], [1.0, 100.0]; lb = [-2.0, -1.0], ub = [2.0, 3.0])
nothing # hide
```

Ipopt, an interior-point method for nonlinear problems, uses the gradient and the
Hessian of the function, and we silence its output with the option `print_level` and its
banner with `sb`:

```@example optimization
using OptimizationIpopt: IpoptOptimizer

ipopt = IpoptOptimizer(additional_options = Dict{String,Any}("print_level" => 0, "sb" => "yes"))
solution = SciMLBase.solve(problem, ipopt)
solution.u
```

The solution is the point ``(1, 1)``, where the objective is zero to the tolerance of
Ipopt:

```@example optimization
solution.objective
```

## Change the parameters

The parameters are an argument of the generated code, not numbers fixed in it, so a
problem with other values of ``a`` and ``b`` reuses the same function. `remake` replaces
the parameters, and with ``a = 1/2`` the minimum moves to ``(1/2, 1/4)``:

```@example optimization
solution = SciMLBase.solve(SciMLBase.remake(problem; p = [0.5, 100.0]), ipopt)
solution.u
```

## Add a constraint

We now restrict the point to the unit disk, ``x^2 + y^2 \le 1``, which excludes the
minimum ``(1, 1)``. The constraints form a second CasADi function, with the arguments of
the objective and a column vector as its result, and the problem gives each of them a
lower bound in `lcons` and an upper bound in `ucons`. An inequality has an infinite
bound on one side, and an equality has the same value on both sides.

```@example optimization
disk = CasadiFunction("disk", [z, p], [x^2 + y^2])
nothing # hide
```

Ipopt asks for the Hessian of the Lagrange function ``\sigma f + \mu^T g``, in which
``g`` are the constraints, ``\sigma`` is the factor of the objective and ``\mu`` the
vector of the Lagrange multipliers, which Ipopt sets at each iteration. With `sparse = true`, the function holds that Hessian as one generated
function, `lag_h`, which writes the stored values of its upper triangle, in place of one
Hessian per constraint; the Jacobian of the constraints and the Hessian of the objective
become sparse matrices as well. Each of them has a prototype, a `SparseMatrixCSC` with the
sparsity pattern that CasADi finds, which the solver reads before the first iteration.
The sparse form needs the standard library SparseArrays to be loaded:

```@example optimization
using SparseArrays

f = SciMLBase.OptimizationFunction(rosenbrock, disk; sparse = true)
f.lag_hess_prototype
```

For two unknowns the pattern holds the whole upper triangle; for a large problem, in
which each constraint couples a few unknowns, it holds a small part of it, and the cost
of a call follows the number of stored values. The initial point ``(0.14, 0.14)`` lies
inside the disk:

```@example optimization
problem = SciMLBase.OptimizationProblem(f, [0.14, 0.14], [1.0, 100.0];
                                        lb = [-2.0, -1.0], ub = [2.0, 3.0], lcons = [-Inf], ucons = [1.0])
solution = SciMLBase.solve(problem, ipopt)
solution.u
```

The minimum lies on the circle, so the constraint is active there, and the generated
constraint function evaluates to 1 at the solution within the tolerance of Ipopt:

```@example optimization
g = zeros(1)
f.cons(g, solution.u, problem.p)
g
```

## Write the problem with the Opti stack

With the functions above, the order of the unknowns, the parameters and the constraints
is the order of the vectors, and the bounds are numbers written apart from the model.
The `Opti` stack of CasADi keeps the variables, the
parameters with their values, the initial point and each constraint with its bounds in
one object, and `SciMLBase.OptimizationProblem(opti)` makes the SciML problem from it.
Its functions, such as `variable` and `subject_to`, are those of the Python API of
CasADi, in the module `CasADiCodegen.C` that holds the whole CasADi front end. A
comparison such as `w[1]^2 + w[2]^2 <= 1` gives an expression of CasADi that `subject_to`
takes as a constraint. A chain such as `-2 <= w[1] <= 2` cannot be one, because Julia
evaluates it as `-2 <= w[1] && w[1] <= 2`, and `&&` takes a `Bool` only;
[`bounded`](@ref CasADiCodegen.bounded)`(lb, w, ub)`, the `opti.bounded` of Python, gives
the constraint with its two bounds:

```@example optimization
C = CasADiCodegen.C
opti = C.Opti()
w = C.variable(opti, 2)
pa, pb = C.parameter(opti), C.parameter(opti)
C.minimize(opti, (pa - w[1])^2 + pb * (w[2] - w[1]^2)^2)
C.subject_to(opti, w[1]^2 + w[2]^2 <= 1)
C.subject_to(opti, bounded([-2.0, -1.0], w, [2.0, 3.0]))
C.set_initial(opti, w, DM([0.14, 0.14]))
C.set_value(opti, pa, DM(1.0))
C.set_value(opti, pb, DM(100.0))

problem = SciMLBase.OptimizationProblem(opti; sparse = true)
problem.lcons, problem.ucons
```

`opti` keeps a bound on a variable as a constraint, so the problem has three constraints
and no `lb` and `ub`: the disk, then the two bounded variables, in the order of the calls
of `subject_to`. Ipopt reaches the same minimum:

```@example optimization
solution = SciMLBase.solve(problem, ipopt)
solution.u
```

`C.value` evaluates an expression of `opti` at the values that its third argument gives
the unknowns `C.x(opti)`, here the solution, and returns a `DM`, which `Float64` converts
to a number:

```@example optimization
Float64(C.value(opti, w[1]^2 + w[2]^2, [C.x(opti) == DM(solution.u)]))
```
