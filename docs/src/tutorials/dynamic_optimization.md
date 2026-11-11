# Solve optimal control problems

In this tutorial, we solve three dynamic optimization problems by direct collocation: the
climb of a rocket, a minimum-time double integrator, and a parameter estimation for a
Lotka-Volterra model. Each becomes a nonlinear program that Ipopt solves through
Optimization.jl with generated code for all its derivatives.

## Collocation on one interval

Direct collocation divides the time interval into ``N`` intervals of length ``\Delta t``
and approximates the state on each by a cubic polynomial through the state ``x_k`` and
three collocation states at the Radau points ``0 < \tau_1 < \tau_2 < \tau_3 = 1``; the
slope of the polynomial must equal the right side ``f`` at each Radau point, and its value
at the end of the interval is ``x_{k+1}``. This is the three-stage Radau IIA method.
`collocation_coeff` gives the matrices of the slopes and of the final value:

```@example dynopt
using CasADiCodegen
using SparseArrays
using OptimizationIpopt: IpoptOptimizer
import SciMLBase

τ = casadi.collocation_points(3, "radau")
slope, final, _ = Matrix.(casadi.collocation_coeff(τ))
τ
```

`collocate!` appends to `g` the equations of one interval, for a right side `f(x, u)`
that returns a CasADi expression:

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

The control is constant on each interval. `collocation_problem` turns the unknowns `w`,
the objective `J`, the equations `g`, the bounds and a start point into a sparse
`OptimizationProblem` (see [Give a CasADi function to a SciML problem](@ref)):

```@example dynopt
function collocation_problem(w, J, g, w0; lb, ub)
    G = vcat(g...)
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

A rocket of height ``h``, velocity ``v`` and mass ``m`` climbs against gravity and drag
with thrust ``0 \le T \le T_m``, and its mass cannot fall below the empty mass ``m_c``:

```math
\dot{h} = v, \qquad
\dot{v} = \frac{T - D_c v^2 e^{-h_c (h - h_0)/h_0}}{m} - g_0 \frac{h_0}{h}, \qquad
\dot{m} = -\frac{T}{c}.
```

The quantities are scaled so that the initial height, mass and gravity are 1:

```@example dynopt
h₀, m₀, g₀ = 1.0, 1.0, 1.0
h_c, c, D_c = 500.0, 0.5 * sqrt(g₀ * h₀), 0.5 * 620 * m₀ / g₀
Tₘ, m_c = 3.5 * g₀ * m₀, 0.6

function rocket(x, T)
    h, v, m = x[1], x[2], x[3]
    drag = D_c * v^2 * exp(-h_c * (h - h₀) / h₀)
    return vcat(v, (T - drag) / m - g₀ * h₀ / h, -T / c)
end
nothing # hide
```

We maximize the height at ``t_e = 0.2`` with all the fuel spent. On 50 intervals, `w`
stacks the states at the nodes, the collocation states and the thrust:

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
w = vcat(X..., reduce(vcat, Xc)..., T...)
size(w)
```

Equal bounds fix the initial state, and the objective is the final height with a minus
sign, since Ipopt minimizes:

```@example dynopt
x0 = [h₀, 0.0, m₀]
lb = [x0; repeat([-Inf, -Inf, m_c], N); fill(-Inf, 9N); zeros(N)]
ub = [x0; repeat([Inf, Inf, 1.0], N); fill(Inf, 9N); fill(Tₘ, N)]
w0 = [repeat(x0, 4N + 1); zeros(N)]

problem = collocation_problem(w, -X[end][1], g, w0; lb, ub)
solution = SciMLBase.solve(problem, ipopt; verbose = false)
-solution.objective
```

The final state holds the maximum height and the empty mass:

```@example dynopt
states = reshape(solution.u[1:3(N + 1)], 3, N + 1)
thrust = solution.u[end-N+1:end]
states[:, end]
```

The rocket burns at full thrust, then at a lower thrust while the air is dense, and coasts
once the fuel is spent:

```@example dynopt
round.(thrust; digits = 2)'
```

## Reach a point in the shortest time

A block starts at rest at ``x = 1`` with acceleration ``|u| \le 1``, and we bring it to rest
at the origin in the shortest time ``t_f``. Since ``t_f`` is unknown, we write the problem
on the normalized time ``s = t/t_f \in [0, 1]``:

```math
\frac{dx}{ds} = t_f v, \qquad \frac{dv}{ds} = t_f u.
```

The symbol `tf` is the last unknown in `w` and the objective:

```@example dynopt
N = 20
tf = SX.sym("tf")
block(x, u) = tf * vcat(x[2], u)

X = [SX.sym("x_$k", 2) for k in 0:N]
Xc = [[SX.sym("xc_$(k)_$j", 2) for j in 1:3] for k in 1:N]
U = [SX.sym("u_$k") for k in 1:N]

g = SX[]
for k in 1:N
    collocate!(g, block, X[k], Xc[k], U[k], X[k+1], 1 / N)
end
push!(g, X[end])                                               # at rest at the origin
w = vcat(X..., reduce(vcat, Xc)..., U..., tf)

x0 = [1.0, 0.0]
lb = [x0; fill(-Inf, 2N + 6N); fill(-1.0, N); 0.0]
ub = [x0; fill(Inf, 2N + 6N); fill(1.0, N); Inf]
w0 = [repeat(x0, 4N + 1); zeros(N); 1.0]

problem = collocation_problem(w, tf, g, w0; lb, ub)
solution = SciMLBase.solve(problem, ipopt; verbose = false)
solution.objective
```

The shortest time is 2, which the program finds to the tolerance of Ipopt because the
switch falls on a node and the polynomial is exact for this motion. The block accelerates
for half of the time and brakes for the other half:

```@example dynopt
solution.u[end-N:end-1]'
```

## Estimate parameters from data

The Lotka-Volterra model describes a prey ``x`` and a predator ``y``:

```math
\dot{x} = \alpha x - \beta x y, \qquad \dot{y} = -\gamma y + \delta x y.
```

We fix ``\beta = 1`` and ``\gamma = 3`` and estimate ``\alpha`` and ``\delta`` from 51
samples of the prey, simulated with ``\alpha = 2.5`` and ``\delta = 1.8``:

```@example dynopt
using OrdinaryDiffEqTsit5: Tsit5

β, γ = 1.0, 3.0
lotka_volterra(x, p) = vcat(p[1] * x[1] - β * x[1] * x[2], -γ * x[2] + p[2] * x[1] * x[2])

x, p = SX.sym("x", 2), SX.sym("p", 2)
rhs = CasadiFunction("lotka_volterra", [x, p], [lotka_volterra(x, p)])
ode = SciMLBase.ODEProblem(SciMLBase.ODEFunction(rhs), [4.0, 2.0], (0.0, 1.0), [2.5, 1.8])
times = range(0.0, 1.0; length = 51)
data = first.(SciMLBase.solve(ode, Tsit5(); saveat = times).u)
data[1:5]
```

The parameters become the last two unknowns, the samples fall on the nodes, and the
objective is the sum of squared errors of the prey; the problem has no control, so
`collocate!` gets `nothing`:

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
w = vcat(X..., reduce(vcat, Xc)..., q)

x0 = [4.0, 2.0]
lb = [x0; fill(-Inf, 8N + 2)]
ub = [x0; fill(Inf, 8N + 2)]
w0 = [repeat(x0, 4N + 1); 1.5; 1.0]

problem = collocation_problem(w, J, g, w0; lb, ub)
solution = SciMLBase.solve(problem, ipopt; verbose = false)
solution.u[end-1:end]
```

The estimates differ from 2.5 and 1.8 in the fifth decimal, because the data come from
Tsit5 at its default tolerances and the model from collocation on 50 intervals.
