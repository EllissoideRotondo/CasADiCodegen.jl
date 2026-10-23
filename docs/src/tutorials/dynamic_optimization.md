# Solve optimal control problems

In this tutorial, we solve three dynamic optimization problems by direct collocation:
the climb of a rocket whose thrust is the control, the double integrator that reaches
the origin in the shortest time, and the estimation of two parameters of a
Lotka-Volterra model from data. In each of them, we write the differential equation
and the conditions of the problem as CasADi expressions, replace the trajectory by its
values on a grid, which turns the problem into a nonlinear program, and give that
program to Ipopt through Optimization.jl with generated code for all its derivatives.
The examples need CasADiCodegen.jl, SciMLBase.jl, the standard library SparseArrays,
which the sparse matrices need, OptimizationIpopt.jl and, for the data of the last problem, OrdinaryDiffEqTsit5.jl.

## Collocation on one interval

Direct collocation divides the time interval into ``N`` intervals of length ``\Delta t``
and approximates the state on each of them by a polynomial of degree 3. The polynomial
passes through the state ``x_k`` at the start of the interval and through three
collocation states ``x_{k,1}, x_{k,2}, x_{k,3}`` at the Radau points
``0 < \tau_1 < \tau_2 < \tau_3 = 1`` of the interval, and the nonlinear program requires
its slope at each Radau point to equal the right side ``f`` of the differential
equation. Its value at the end of the interval is then the state ``x_{k+1}`` at the
start of the next interval. This is the three-stage Radau IIA method, of order 5, written
in the states rather than in the stage derivatives.

CasADi gives the Radau points and the coefficients of the polynomial. Of the three
matrices that `collocation_coeff` returns, the first gives the slopes at the Radau points
and the second the value at the end of the interval, both as linear combinations of the
four values that the polynomial passes through:

```@example dynopt
using CasADiCodegen
using SparseArrays
using OptimizationIpopt: IpoptOptimizer
import SciMLBase

C = CasADiCodegen.C                              # the module of the CasADi front end
τ = C.collocation_points(3, "radau")
slope, final, _ = Matrix.(C.collocation_coeff(τ))
τ
```

The function below appends to the vector `g` the equations of one interval, as
expressions that the nonlinear program sets to zero. It takes the right side as a Julia
function `f(x, u)` of a state and a control, which returns a CasADi expression, so that
the three problems can share it:

```@example dynopt
function collocate!(g, f, x, xc, u, xnext, Δt)
    z = [x, xc...]                               # the values that the polynomial passes through
    for j in 1:3
        push!(g, Δt * f(xc[j], u) - sum(slope[r, j] * z[r] for r in 1:4))
    end
    push!(g, sum(final[r] * z[r] for r in 1:4) - xnext)
    return g
end
nothing # hide
```

The control `u` is constant on each interval. A nonlinear program in the vector `w` of
all unknowns then consists of an objective `J`, the equations `g`, the bounds of `w` and
a start point `w0`. Each problem below ends with the same steps: two CasADi functions of
`w`, a SciML function with generated code for the gradient, the Jacobian of the
constraints and the Hessian of the Lagrange function, and an `OptimizationProblem` in
which all constraints are equations. With `sparse = true`, the Jacobian of the
constraints and the Hessian of the Lagrange function are sparse matrices, and the Hessian
of the Lagrange function is one generated function instead of one per constraint (see
[Give a CasADi function to a SciML problem](@ref)). The option `sb` of Ipopt hides its
banner, and `verbose = false` in each call of `solve` below sets its `print_level` to 0,
which hides its iteration log:

```@example dynopt
function collocation_problem(w, J, g, w0; lb, ub)
    G = vertcat(g...)
    objective = CasadiFunction("J", [w], [J])
    constraints = CasadiFunction("g", [w], [G])
    f = SciMLBase.OptimizationFunction(objective, constraints; sparse = true)
    n = size(G, 1)
    return SciMLBase.OptimizationProblem(f, w0; lb, ub, lcons = zeros(n), ucons = zeros(n))
end

ipopt = IpoptOptimizer(additional_options = Dict{String,Any}("sb" => "yes"))
nothing # hide
```

## Maximize the height of a rocket

A rocket of height ``h``, velocity ``v`` and mass ``m`` climbs against gravity and a
drag that decreases with height. The thrust ``T`` of its engine, bounded by
``0 \le T \le T_m``, burns fuel, and the mass cannot fall below the mass ``m_c`` of the
empty rocket:

```math
\dot{h} = v, \qquad
\dot{v} = \frac{T - D_c v^2 e^{-h_c (h - h_0)/h_0}}{m} - g_0 \frac{h_0}{h}, \qquad
\dot{m} = -\frac{T}{c}.
```

All quantities are scaled so that the initial height, the initial mass and the
gravity at the ground are 1. The right side is a Julia function of the state and the
thrust:

```@example dynopt
h₀, m₀, g₀ = 1.0, 1.0, 1.0
h_c, c, D_c = 500.0, 0.5 * sqrt(g₀ * h₀), 0.5 * 620 * m₀ / g₀
Tₘ, m_c = 3.5 * g₀ * m₀, 0.6

function rocket(x, T)
    h, v, m = x[1], x[2], x[3]
    drag = D_c * v^2 * exp(-h_c * (h - h₀) / h₀)
    return vertcat(v, (T - drag) / m - g₀ * h₀ / h, -T / c)
end
nothing # hide
```

We look for the thrust that maximizes the height at the final time ``t_e = 0.2``, with
all the fuel spent at that time. On 50 intervals, the unknowns are the states at the 51
nodes, the three collocation states of each interval and the thrust on each interval.
Each of them is a symbol of its own, and `w` stacks them in this order:

```@example dynopt
N, te = 50, 0.2
X = [SX.sym("x_$k", 3) for k in 0:N]                           # states at the nodes
Xc = [[SX.sym("xc_$(k)_$j", 3) for j in 1:3] for k in 1:N]     # collocation states
T = [SX.sym("T_$k") for k in 1:N]                              # thrust on each interval

g = SX[]
for k in 1:N
    collocate!(g, rocket, X[k], Xc[k], T[k], X[k+1], te / N)
end
push!(g, X[end][3] - m_c)                                      # empty at the final time
w = vertcat(X..., reduce(vcat, Xc)..., T...)
size(w)
```

The bounds follow the same order. Equal lower and upper bounds fix the initial state,
the mass at the nodes stays between ``m_c`` and 1, and the collocation states have no
bounds. The start point holds the initial state at every node and a thrust of zero. The
objective is the height at the final time with a minus sign, since Ipopt minimizes:

```@example dynopt
x0 = [h₀, 0.0, m₀]
lb = [x0; repeat([-Inf, -Inf, m_c], N); fill(-Inf, 9N); zeros(N)]
ub = [x0; repeat([Inf, Inf, 1.0], N); fill(Inf, 9N); fill(Tₘ, N)]
w0 = [repeat(x0, 4N + 1); zeros(N)]

problem = collocation_problem(w, -X[end][1], g, w0; lb, ub)
solution = SciMLBase.solve(problem, ipopt; verbose = false)
-solution.objective
```

The maximum height is about 1.013, and the first solve compiles the generated code. The
states at the nodes are the first ``3(N + 1)`` elements of the solution and the thrust
the last ``N``; the state at the final time holds the maximum height and the mass of the
empty rocket:

```@example dynopt
states = reshape(solution.u[1:3(N + 1)], 3, N + 1)
thrust = solution.u[end-N+1:end]
states[:, end]
```

The rocket burns at full thrust at first, then at a lower thrust that limits the losses
to drag while the air is dense, and coasts with the engine off once the fuel is spent:

```@example dynopt
round.(thrust; digits = 2)'
```

## Reach a point in the shortest time

A block on a line, of position ``x`` and velocity ``v``, starts at rest at ``x = 1``, and
its acceleration ``u`` is bounded by ``|u| \le 1``. We look for the acceleration that
brings it to rest at the origin in the shortest time ``t_f``. Since the final time is an
unknown, we write the problem on the normalized time ``s = t/t_f \in [0, 1]``, on which
the right side is multiplied by ``t_f``:

```math
\frac{dx}{ds} = t_f v, \qquad \frac{dv}{ds} = t_f u.
```

The symbol `tf` is an unknown of the nonlinear program, the last element of `w`, and
the objective. The right side refers to it, and the collocation equations do not change:

```@example dynopt
N = 20
tf = SX.sym("tf")
block(x, u) = tf * vertcat(x[2], u)

X = [SX.sym("x_$k", 2) for k in 0:N]
Xc = [[SX.sym("xc_$(k)_$j", 2) for j in 1:3] for k in 1:N]
U = [SX.sym("u_$k") for k in 1:N]

g = SX[]
for k in 1:N
    collocate!(g, block, X[k], Xc[k], U[k], X[k+1], 1 / N)
end
push!(g, X[end])                                               # at rest at the origin
w = vertcat(X..., reduce(vcat, Xc)..., U..., tf)

x0 = [1.0, 0.0]
lb = [x0; fill(-Inf, 2N + 6N); fill(-1.0, N); 0.0]
ub = [x0; fill(Inf, 2N + 6N); fill(1.0, N); Inf]
w0 = [repeat(x0, 4N + 1); zeros(N); 1.0]

problem = collocation_problem(w, tf, g, w0; lb, ub)
solution = SciMLBase.solve(problem, ipopt; verbose = false)
solution.objective
```

The shortest time is 2: the block accelerates at the bound towards the origin for half
of the time and brakes for the other half. The switch falls on a node of the grid, and
on each interval the collocation polynomial is exact for this motion, so the nonlinear
program finds this time to the tolerance of Ipopt. The acceleration on each interval is:

```@example dynopt
solution.u[end-N:end-1]'
```

## Estimate parameters from data

The Lotka-Volterra model describes a prey ``x`` and a predator ``y``:

```math
\dot{x} = \alpha x - \beta x y, \qquad \dot{y} = -\gamma y + \delta x y.
```

We keep ``\beta = 1`` and ``\gamma = 3`` fixed and estimate ``\alpha`` and ``\delta``
from 51 samples of the prey. The samples come from a solution with ``\alpha = 2.5``
and ``\delta = 1.8``, which a CasADi function of the state and the two parameters gives
to OrdinaryDiffEq.jl:

```@example dynopt
using OrdinaryDiffEqTsit5: Tsit5

β, γ = 1.0, 3.0
lotka_volterra(x, p) = vertcat(p[1] * x[1] - β * x[1] * x[2], -γ * x[2] + p[2] * x[1] * x[2])

x, p = SX.sym("x", 2), SX.sym("p", 2)
rhs = CasadiFunction("lotka_volterra", [x, p], [lotka_volterra(x, p)])
ode = SciMLBase.ODEProblem(SciMLBase.ODEFunction(rhs), [4.0, 2.0], (0.0, 1.0), [2.5, 1.8])
times = range(0.0, 1.0; length = 51)
data = first.(SciMLBase.solve(ode, Tsit5(); saveat = times).u)
data[1:5]
```

The same Julia function gives the right side of the collocation equations, now with
the parameters as the last two unknowns of the nonlinear program. With one interval
between two samples, the samples fall on the nodes, and the objective is the sum of the
squared differences between the prey at the nodes and the data. The problem has no
control, so `collocate!` gets `nothing` for it, and all states are free except the
initial ones:

```@example dynopt
N = length(times) - 1
q = SX.sym("q", 2)                                             # α and δ
X = [SX.sym("x_$k", 2) for k in 0:N]
Xc = [[SX.sym("xc_$(k)_$j", 2) for j in 1:3] for k in 1:N]

g = SX[]
for k in 1:N
    collocate!(g, (x, _) -> lotka_volterra(x, q), X[k], Xc[k], nothing, X[k+1], 1 / N)
end
J = sum((X[k][1] - data[k])^2 for k in 1:N+1)
w = vertcat(X..., reduce(vcat, Xc)..., q)

x0 = [4.0, 2.0]
lb = [x0; fill(-Inf, 8N + 2)]
ub = [x0; fill(Inf, 8N + 2)]
w0 = [repeat(x0, 4N + 1); 1.5; 1.0]

problem = collocation_problem(w, J, g, w0; lb, ub)
solution = SciMLBase.solve(problem, ipopt; verbose = false)
solution.u[end-1:end]
```

The estimates differ from 2.5 and 1.8 in the fifth decimal. The data come from
Tsit5 at its default tolerances and the model from collocation on 50 intervals, so that
the two trajectories, and with them the best parameters, differ slightly even without
noise in the data.
