# Models and helpers of the tests that integrate differential equations.
import ForwardDiff
import SciMLBase
using LinearAlgebra: exp
using OrdinaryDiffEqVerner: Vern9

const ODE_X0, ODE_MU = [1.0, 0.0, 0.1, 1.1], [1.0]

# Tight tolerances, and one compilation of the solver for all the integrators of the tests.
const TIGHT = (; abstol = 1e-13, reltol = 1e-13, specialize = SciMLBase.AutoSpecialize)

"Planar two-body problem; the parameter is the gravitational parameter."
function two_body()
    x, p = SX.sym("x", 4), SX.sym("mu", 1)
    r3 = (x[1]^2 + x[2]^2)^1.5
    Dict("x" => x, "p" => p, "ode" => vertcat([x[3], x[4], -p[1] * x[1] / r3, -p[1] * x[2] / r3]))
end

two_body_integrator(; options...) = integrator("F", Vern9(), two_body(), 0.0, 3.0; TIGHT..., options...)
two_body_cvodes() = integrator("R", "cvodes", two_body(), 0.0, 3.0, Dict("abstol" => 1e-12, "reltol" => 1e-12))

"Run the generated code of `f`; one vector of nonzeros per output."
function ode_emitted(f, arg; T = Float64, kwargs...)
    res = [zeros(T, C.nnz_out(f, i - 1)) for i in 1:C.n_out(f)]
    build_function(f; nonzeros = true, expression = Val{false}, kwargs...)(res, arg)
    res
end

"The final state and its derivatives to the given order, as a CasADi function."
function with_derivatives(F, order)
    x0, p = MX.sym("x0", 4), MX.sym("p", 1)
    xf = F(x0, p)
    outputs = [xf]
    order >= 1 && append!(outputs, [jacobian(xf, x0), jacobian(xf, p)])
    order >= 2 && push!(outputs, jacobian(jacobian(xf, x0), x0))
    CasADiCodegen.Function("d", [x0, p], outputs)
end

cvodes_call(R) = (x0, p) -> R(Dict("x0" => x0, "p" => p))["xf"]
closeto(a, b, tolerance) = all(maximum(abs, u - v) <= tolerance * max(1, maximum(abs, v)) for (u, v) in zip(a, b))

"""
A bouncing ball, exactly: the height and the speed at the output times `times`, and the integral of the
square of the speed to each of them. The gravity is the control of each interval, and an impact multiplies the speed
by `-e`. The result is generic in the number type, so that ForwardDiff.jl gives the derivatives through the impacts.
"""
function ball_reference(x0, e, gravity, times)
    T = promote_type(eltype(x0), typeof(e), eltype(gravity))
    h, v, sum, t = T(x0[1]), T(x0[2]), zero(T), zero(T)
    finals, sums = T[], T[]
    # The integral of the square of the speed over a flight of the length `s` from the speed `v`.
    flight(v, g, s) = (v^3 - (v - g * s)^3) / (3g)
    for (k, stop) in enumerate(times)
        g = gravity[k]
        while true
            # Loop invariant: `h`, `v` and `sum` are the values at the time `t`, which is before `stop` or at it.
            s = (v + sqrt(v^2 + 2g * h)) / g          # the time to the next impact
            t + s < stop || break
            sum += flight(v, g, s)
            h, v, t = zero(T), -e * (v - g * s), t + s
        end
        s = stop - t
        sum += flight(v, g, s)
        h, v, t = h + v * s - g * s^2 / 2, v - g * s, T(stop)
        append!(finals, (h, v))
        push!(sums, sum)
    end
    finals, sums
end

"""
The state at the time `t` of the decay x' = -r x from `x0` at the time 0, which doubles
when r x falls to 0.55: the first event at x = 0.55 / r, then one event each time x halves.
"""
function refilled(t, x0, rate)
    onset = log(rate * x0 / 0.55) / rate
    t < onset && return x0 * exp(-rate * t)
    events = 1 + floor((t - onset) * rate / log(2))
    2 * 0.55 / rate * exp(-rate * (t - onset) + (events - 1) * log(2))
end
