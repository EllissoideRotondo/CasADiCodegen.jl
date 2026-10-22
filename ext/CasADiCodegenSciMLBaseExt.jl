module CasADiCodegenSciMLBaseExt

import CasADiCodegen
import SciMLBase
import StaticArrays
import LinearAlgebra
using CasADiCodegen.Codegen: @requires, @ensures

"""
    Solver

The Julia function `(x0, q) -> final state` that integrates `rhs` from `t0` to `tf` with
the SciML algorithm `alg`. With `free`, the run is from zero to `q[end] - q[end-1]`.

Each task keeps one solver state per number type and uses it again at each call, so later
calls allocate little memory. The result is the state vector of that solver, which the
next call in the same task overwrites.
"""
struct Solver{F,A,O}
    name::String
    rhs::F
    alg::A
    t0::Float64
    tf::Float64
    free::Bool
    specialize::DataType
    options::O
end

span(s::Solver, q, ::Type{T}) where {T} = s.free ? (zero(T), T(q[end]) - T(q[end-1])) : (T(s.t0), T(s.tf))

# The solver state of this task for the number type `T`. The first call makes
# it from the actual inputs, so that the solver finds its first step there.
# A solver keeps the direction of time of its first run, so the runs backward
# with free times have a state of their own.
function state(s::Solver, ::Type{T}, x0, q) where {T}
    t0, tf = s.free ? span(s, q, T) : (s.t0, s.tf)
    get!(task_local_storage(), s.free && tf < t0 ? (s, T, :backward) : (s, T)) do
        tf == t0 && (tf = t0 + one(tf))          # an arc of no length makes no first step
        u0, q0 = copyto!(Vector{T}(undef, length(x0)), x0), copyto!(Vector{T}(undef, length(q)), q)
        make_state(s, CasADiCodegen.Codegen.consistent!(u0, s.rhs, q0, t0), q0, t0, tf)
    end
end

make_state(s, u0::Vector, q0::Vector, t0, tf) = make_state(s, s.specialize, u0, q0, t0, tf)

# The form of specialization is an argument, so that Julia knows the type of the state from the types alone.
function make_state(s::Solver, ::Type{S}, u0::Vector, q0::Vector, t0, tf) where {S}
    problem = SciMLBase.ODEProblem{true,S}(problem_function(s.rhs, S), u0, (t0, tf), q0)
    SciMLBase.init(problem, s.alg; save_everystep = false, save_start = false, save_end = false, s.options...)
end

"The right side for a SciML problem: itself, or a SciML function with the mass matrix of a model with algebraic states."
problem_function(rhs, ::Type) = rhs
problem_function(rhs::CasADiCodegen.Codegen.RightSide{F,Vector{Float64}}, ::Type{S}) where {F,S} =
    SciMLBase.ODEFunction{true,S}(rhs; mass_matrix = LinearAlgebra.Diagonal(rhs.mass))

# The inputs go to the vectors of the state, so that the run is one method for each solver,
# whatever arrays the generated code gives, and the cache on disk can hold its machine code.
function integrate!(state, s::Solver, x0, q)
    copyto!(state.sol.prob.u0, x0)
    copyto!(state.p, q)
    CasADiCodegen.Codegen.consistent!(state.sol.prob.u0, s.rhs, state.p, span(s, state.p, typeof(state.t))[1])
    run!(state, s)
end

function run!(state, s::Solver)
    x0 = state.sol.prob.u0
    t0, tf = span(s, state.p, typeof(state.t))
    t0 == tf && return copyto!(state.u, x0)
    SciMLBase.reinit!(state, x0; t0, tf)
    SciMLBase.solve!(state)
    code = state.sol.retcode
    SciMLBase.successful_retcode(code) || error("integrator \"$(s.name)\": the solver stopped with the return code $code")
    state.u
end

(s::Solver)(x0, q) = integrate!(state(s, promote_type(eltype(x0), eltype(q)), x0, q), s, x0, q)

"""
    integrator(name, alg, dae, t0, tf; specialize, adjoint = false, mass, options...) -> CasADiCodegen.Function
    integrator(name, alg, dae, t0, times::AbstractVector; specialize, adjoint = false, options...) -> CasADiCodegen.Function
    integrator(name, alg, dae; specialize, adjoint = false, mass, direction = 0, interp_points = 10,
               events = 1, terminal = true, options...) -> CasADiCodegen.Function

Make a CasADi function that integrates an ordinary differential equation with the SciML
algorithm `alg`, for example `Vern9()` of OrdinaryDiffEqVerner.jl. CasADi differentiates
the function to any order, and `build_function` emits code that calls the solver directly.

The times select the form of the function. `p` is absent when the model has no parameters.

| Call                                    | Function           | Result                                        |
|-----------------------------------------|--------------------|-----------------------------------------------|
| `integrator(name, alg, dae, t0, tf)`    | `F(x0, p)`         | the state at `tf`                             |
| `integrator(name, alg, dae, t0, times)` | `F(x0, p)`         | a matrix; column `k` is the state at `times[k]` |
| `integrator(name, alg, dae)`            | `G(x0, p, t0, tf)` | the state at `tf`; CasADi also differentiates by the two times |

With free times, `tf` can be before `t0` or equal to it. The times of the vector `times`
can be in any order. A model with quadratures returns them as a second result.

# Arguments
- `name`: the name of the CasADi function, a valid identifier.
- `dae`: a dictionary as for the CasADi integrators. It holds the state `"x"` and its
  derivative `"ode"`, both dense column vectors. Optional entries are the parameters `"p"`,
  the time `"t"` and the integrands `"quad"`. With free times, the entry `"zero"` is an
  event function, and `"transition"` is the state after an event.
- `t0`, `tf`, `times`: finite times. `t0` and `tf` must be different.

# Keywords
- `specialize = SciMLBase.FullSpecialize`: the specialization of the SciML problem. With
  `SciMLBase.AutoSpecialize` the solver is compiled once for all integrators, and the runs
  are slower.
- `adjoint = false`: with `true`, a reverse derivative comes from an integrator of the
  adjoint equations, which runs from the final time back to the initial time. Its cost
  does not grow with the number of parameters.
- `mass`: the diagonal of the mass matrix of a model with algebraic states: 0 for an
  algebraic state and 1 for the others. The element of `"ode"` for an algebraic state is
  the residual of an algebraic equation, and its initial value is a guess. `alg` must
  accept a mass matrix, as `Rodas5P()` of OrdinaryDiffEqRosenbrock.jl does. A vector of
  times is not supported with it.
- `direction = 0`: the change of sign of `"zero"` that is an event: any with 0, from
  negative to positive with 1, and from positive to negative with -1.
- `interp_points = 10`: the number of points of each step at which the solver looks for
  the change of sign, besides the end of the step.
- `events = 1`: the number of events that the integration passes.
- `terminal = true`: the integration ends at the last event. With `false` it applies the
  transition there too and goes on to `tf`.
- `options...`: options of the solver, for example `abstol`, `reltol` and `maxiters`.

With an event function, the function returns the state at the event, the quadratures if
the model has them, and the event time `te` as its last result; `te` is a vector with
`events` more than 1. A zero at the initial time is not an event. A call throws when no
event comes before `tf`.

See [Integrate a differential equation](@ref) and [Stop at an event](@ref events).

# Examples
```julia
using CasADiCodegen
using OrdinaryDiffEqVerner: Vern9

x, mu = MX.sym("x", 4), MX.sym("mu")
r3 = (x[1]^2 + x[2]^2)^1.5
model = Dict("x" => x, "p" => mu, "ode" => vertcat(x[3], x[4], -mu * x[1] / r3, -mu * x[2] / r3))
F = integrator("F", Vern9(), model, 0.0, 3.0; abstol = 1e-13, reltol = 1e-13)

x0, p = MX.sym("x0", 4), MX.sym("p")
f = CasadiFunction("f", [x0, p], [F(x0, p), jacobian(F(x0, p), x0)])
```
"""
function CasADiCodegen.integrator(name::AbstractString, alg::SciMLBase.AbstractODEAlgorithm, dae, t0::Real, tf::Real;
                           specialize::Type{<:SciMLBase.AbstractSpecialization} = SciMLBase.FullSpecialize,
                           adjoint::Bool = false, mass::AbstractVector{<:Real} = Float64[], options...)
    # The adjoint equations run backward, on the solver with free times.
    make_solver(rhs, n, m) = adjoint ? Solver(String(name), rhs, alg, NaN, NaN, true, specialize, values(options)) :
                                       Solver(String(name), rhs, alg, Float64(t0), Float64(tf), false, specialize, values(options))
    CasADiCodegen.Codegen.ode_integrator(name, dae, t0, tf, make_solver; adjoint, mass = Vector{Float64}(mass))
end

function CasADiCodegen.integrator(name::AbstractString, alg::SciMLBase.AbstractODEAlgorithm, dae, t0::Real, times::AbstractVector{<:Real};
                           specialize::Type{<:SciMLBase.AbstractSpecialization} = SciMLBase.FullSpecialize,
                           adjoint::Bool = false, mass::AbstractVector{<:Real} = Float64[], options...)
    @requires isempty(mass) "integrator \"$name\": mass is not supported with a vector of times"
    make_solver(rhs, n, m) = Solver(String(name), rhs, alg, NaN, NaN, true, specialize, values(options))
    CasADiCodegen.Codegen.ode_grid(name, dae, t0, times, make_solver; adjoint)
end

function CasADiCodegen.integrator(name::AbstractString, alg::SciMLBase.AbstractODEAlgorithm, dae;
                           specialize::Type{<:SciMLBase.AbstractSpecialization} = SciMLBase.FullSpecialize,
                           direction::Integer = 0, interp_points::Integer = 10, adjoint::Bool = false,
                           events::Integer = 1, terminal::Bool = true, mass::AbstractVector{<:Real} = Float64[], options...)
    @requires direction in (-1, 0, 1) "integrator \"$name\": the direction of the event must be -1, 0 or 1; got $direction"
    @requires interp_points >= 0 "integrator \"$name\": interp_points must not be negative; got $interp_points"
    make_solver(rhs, n, m) = Solver(String(name), rhs, alg, NaN, NaN, true, specialize, values(options))
    @requires haskey(dae, "zero") || (events == 1 && terminal) "integrator \"$name\": events and terminal need an event function, \"zero\""
    haskey(dae, "zero") || return CasADiCodegen.Codegen.ode_integrator(name, dae, make_solver; adjoint, mass = Vector{Float64}(mass))
    # Without full specialization the solver does not know the type of the event function,
    # so that it compiles once for all integrators with an event.
    crossing(zero) = specialize === SciMLBase.FullSpecialize ? Crossing(zero) : Crossing{Any}(zero)
    make_event(rhs, zero, n, m) = EventSolver(String(name), rhs, crossing(zero), alg, Int(direction), Int(interp_points), specialize, values(options))
    CasADiCodegen.Codegen.ode_event(name, dae, make_solver, make_event; adjoint, events, terminal, mass = Vector{Float64}(mass))
end

"The event function of an integrator as the condition `(u, t, integrator)` of a SciML callback."
struct Crossing{Z}
    zero::Z
end
(c::Crossing)(u, t, integrator) = (T = promote_type(eltype(u), typeof(t)); number(c.zero, T, (u, integrator.p, (t,)))::T)

"""
    EventSolver

The Julia function `(tf, x0, p, t0) -> te`, or `(tf, x0, t0) -> te`, that integrates `rhs`
from `t0` toward `tf` with the SciML algorithm `alg` and returns the time of the first
event, where `zero` changes its sign. A call throws when no event comes before `tf`.

`direction` selects the change of sign: any with 0, from negative to positive with 1, and
from positive to negative with -1. The solver looks for the change at `interp_points`
points of each step and at its end. Each task keeps one solver state per number type and
per direction of time, as `Solver` does.
"""
struct EventSolver{F,Z,A,O}
    name::String
    rhs::F
    zero::Z
    alg::A
    direction::Int
    interp_points::Int
    specialize::DataType
    options::O
end

function (s::EventSolver)(limit, x0, rest...)
    q = length(rest) == 2 ? rest[1] : CasADiCodegen.Codegen.NO_PARAMETERS
    T = float(promote_type(eltype(limit), eltype(x0), eltype(q), eltype(rest[end])))
    t0, tf = T(rest[end][1]), T(limit[1])
    state = get!(task_local_storage(), tf < t0 ? (s, T, :backward) : (s, T)) do
        u0, q0 = copyto!(Vector{T}(undef, length(x0)), x0), copyto!(Vector{T}(undef, length(q)), q)
        make_state(s, CasADiCodegen.Codegen.consistent!(u0, s.rhs, q0, t0), q0, t0, tf == t0 ? t0 + one(T) : tf)
    end
    if tf != t0
        copyto!(state.sol.prob.u0, x0)
        copyto!(state.p, q)
        CasADiCodegen.Codegen.consistent!(state.sol.prob.u0, s.rhs, state.p, t0)
        run!(state, s, t0, tf) && return StaticArrays.SVector{1,T}(event_time(state, s, T))
    end
    error("integrator \"$(s.name)\": no event between the initial time $t0 and the final time $tf")
end

"""
    event_time(state, s::EventSolver, T)

Return the time of the event that ended the run of `state`; `state.u` and `state.t` are
then the state and the time of the event. For `Float64` the result is `state.t`.

The root search of the solver does not give exact derivatives for a number type that
carries them, such as a dual number. For such a type, one Newton step on the event function
along the trajectory makes the derivatives those of the implicit function. The value does
not change, because the event function is zero there. The slope of the event function
comes from a finite difference, which is sufficient because it multiplies a residual that
is almost zero.
"""
event_time(state, s::EventSolver, ::Type{Float64}) = state.t
function event_time(state, s::EventSolver, ::Type{T}) where {T}
    u, t = state.u, state.t
    flow = similar(u)
    s.rhs(flow, u, state.p, t)
    h = T(sqrt(eps(Float64)))
    residual = s.zero(u, t, state)
    # The rows of the algebraic states hold residuals, so those states follow from their equations at the later time.
    moved = CasADiCodegen.Codegen.consistent!(u .+ h .* flow, s.rhs, state.p, t + h)
    slope = (s.zero(moved, t + h, state) - residual) / h
    t - residual / slope
end

"Integrate from `t0` toward `tf`; true when an event ended the run."
function run!(state, s::EventSolver, t0, tf)
    SciMLBase.reinit!(state, state.sol.prob.u0; t0, tf)
    SciMLBase.solve!(state)
    state.sol.retcode == SciMLBase.ReturnCode.Terminated
end

function make_state(s::EventSolver, ::Type{S}, u0::Vector, q0::Vector, t0, tf) where {S}
    stop = SciMLBase.terminate!
    callback = SciMLBase.ContinuousCallback(s.zero, s.direction >= 0 ? stop : nothing, s.direction <= 0 ? stop : nothing;
                                            save_positions = (false, false), s.interp_points)
    problem = SciMLBase.ODEProblem{true,S}(problem_function(s.rhs, S), u0, (t0, tf), q0)
    SciMLBase.init(problem, s.alg; callback, save_everystep = false, save_start = false, save_end = false, s.options...)
end

"""
    CasADiCodegen.Codegen.warm(s::Union{Solver,EventSolver})

Compile the solver for `Float64`, with no run: the code that makes its state, and the
integration of that state. A package of the cache on disk calls it when Julia compiles the
package, so that the machine code is stored.
"""
function CasADiCodegen.Codegen.warm(s::Union{Solver,EventSolver})
    types = (typeof(s), Type{s.specialize}, Vector{Float64}, Vector{Float64}, Float64, Float64)
    precompile(make_state, types)
    made = Base.return_types(make_state, types)
    length(made) == 1 && isconcretetype(made[1]) &&
        precompile(run!, s isa Solver ? (made[1], typeof(s)) : (made[1], typeof(s), Float64, Float64))
    nothing
end

"""
    event_callback(name, rhs, zero, jump, events, count, limit)

Return the SciML callback of the event functions `zero!((e,), (u, p, (t,)))` of an
integrator of CasADi (see `CasADiCodegen.Codegen.event_functions`): when `e[i]` changes its sign
from positive to negative, the state becomes that of
`jump!((after,), (before, p, (t,), (i - 1,), z))`, where `z` holds the algebraic states
after the event. The algebraic states then satisfy the algebraic equations of `rhs` again
(see `CasADiCodegen.Codegen.consistent!`). With algebraic states, `jump!` runs twice: first
with those before the event as `z`, then with those that satisfy the algebraic equations
after it, which the derivatives of the states need.

Each event increments `count[]`, and the event after `limit` of them throws an error.
"""
function event_callback(name::String, rhs, zero, jump, events::Int, count::Base.RefValue{Int}, limit::Int)
    @requires pkgversion(SciMLBase) >= v"3" "integrator \"$name\": an event function needs SciMLBase.jl 3 or later; got $(pkgversion(SciMLBase))"
    before, algebraic = Float64[], rhs.mass === nothing ? Int[] : findall(iszero, rhs.mass)
    z = zeros(length(algebraic))
    condition(out, u, t, integrator) = (zero((out,), (u, integrator.p, (t,))); nothing)
    # `crossings[i]` is -1 when value `i` changes its sign from positive to negative, and +1 for the opposite change.
    function event!(integrator, crossings)
        any(<(0), crossings) || return SciMLBase.derivative_discontinuity!(integrator, false)
        for i in eachindex(crossings)
            # Loop invariant: the state of `integrator` follows the events before `i`.
            crossings[i] < 0 || continue
            count[] += 1
            count[] > limit && error("integrator \"$name\": more than $limit events, the last one at the time $(integrator.t)")
            copyto!(resize!(before, length(integrator.u)), integrator.u)
            z .= view(before, algebraic)
            jump((integrator.u,), (before, integrator.p, (integrator.t,), (Float64(i - 1),), z))
            isempty(algebraic) && continue
            CasADiCodegen.Codegen.consistent!(integrator.u, rhs, integrator.p, integrator.t)
            z .= view(integrator.u, algebraic)
            jump((integrator.u,), (before, integrator.p, (integrator.t,), (Float64(i - 1),), z))
            integrator.u[algebraic] .= z
        end
        CasADiCodegen.Codegen.consistent!(integrator.u, rhs, integrator.p, integrator.t)
        nothing
    end
    SciMLBase.VectorContinuousCallback(condition, event!, events; save_positions = (false, false))
end

"""
    EventIntegration

The Julia function `(x0, p, u) -> path` of an integrator of CasADi with an event function
(see `CasADiCodegen.Codegen.solver_events`). It integrates `rhs` from `t0` to each output time of
`times` with the SciML algorithm `alg`, with the callback of `event_callback`. `path` has
the state at each output time, one after the other.

- `u` has the controls of each interval, which follow the parameters `p` in the
  parameters of the solver; they change at each output time.
- `sizes` are the number of states, of parameters, of controls of one interval, of event
  functions, and the most events of one call.

Each task keeps one solver state and uses it again at each call, as `Solver` does; the
result is memory of that state. The numbers must be `Float64`: the derivatives come from
the forward directions of CasADi, not from a number type.
"""
struct EventIntegration{F,Z,J,A,O}
    name::String
    rhs::F
    zero::Z
    jump::J
    alg::A
    t0::Float64
    times::Vector{Float64}
    sizes::NTuple{5,Int}
    specialize::DataType
    options::O
end

function CasADiCodegen.Codegen.event_integrator(name::AbstractString, alg, rhs, zero, jump, problem, options)
    specialize = get(options, :specialize, SciMLBase.FullSpecialize)
    others = Base.structdiff(options, NamedTuple{(:specialize,)})
    sizes = (problem.n, problem.np, problem.nu, problem.events, problem.max_events)
    EventIntegration(String(name), rhs, zero, jump, alg, Float64(problem.t0), Vector{Float64}(problem.times), sizes, specialize, others)
end

function (s::EventIntegration)(x0, p, u)
    for v in (x0, p, u)
        eltype(v) === Float64 || error("integrator \"$(s.name)\" with an event function: the numbers must be Float64, not $(eltype(v)); use the derivatives of CasADi")
    end
    n, np, nu = s.sizes
    state = get!(task_local_storage(), s) do
        q0 = Vector{Float64}([p; u[1:nu]])
        make_state(s, s.specialize, CasADiCodegen.Codegen.consistent!(Vector{Float64}(x0), s.rhs, q0, s.t0), q0)
    end
    integrate!(state, s, x0, p, u)
end

function make_state(s::EventIntegration, ::Type{S}, u0::Vector{Float64}, q0::Vector{Float64}) where {S}
    count = Ref(0)
    callback = event_callback(s.name, s.rhs, s.zero, s.jump, s.sizes[4], count, s.sizes[5])
    tf = s.times[end] == s.t0 ? s.t0 + 1 : s.times[end]       # an integration of no length makes no first step
    problem = SciMLBase.ODEProblem{true,S}(problem_function(s.rhs, S), u0, (s.t0, tf), q0)
    integrator = SciMLBase.init(problem, s.alg; callback, save_everystep = false, save_start = false, save_end = false, s.options...)
    (; integrator, count, path = zeros(s.sizes[1] * length(s.times)))
end

function integrate!(state, s::EventIntegration, x0, p, u)
    integrator, path = state.integrator, state.path
    n, np, nu = s.sizes
    q, u0 = integrator.p, integrator.sol.prob.u0
    copyto!(q, 1, p, 1, np)
    copyto!(q, np + 1, u, 1, nu)
    CasADiCodegen.Codegen.consistent!(copyto!(u0, x0), s.rhs, q, s.t0)
    state.count[] = 0
    SciMLBase.reinit!(integrator, u0; t0 = s.t0, tf = max(s.times[end], s.t0 + 1))
    for stop in unique(s.times)
        s.t0 < stop < s.times[end] && SciMLBase.add_tstop!(integrator, stop)
    end
    for (k, stop) in enumerate(s.times)
        # Loop invariant: `path` holds the states at the output times before `k`, and the controls are those of interval `k`.
        while integrator.t < stop
            SciMLBase.step!(integrator)
            code = integrator.sol.retcode
            code in (SciMLBase.ReturnCode.Default, SciMLBase.ReturnCode.Success) ||
                error("integrator \"$(s.name)\": the solver stopped with the return code $code")
        end
        copyto!(path, (k - 1) * n + 1, stop == s.t0 ? u0 : integrator.u, 1, n)
        if k < length(s.times) && nu > 0
            copyto!(q, np + 1, u, k * nu + 1, nu)
            SciMLBase.derivative_discontinuity!(integrator, true)
        end
    end
    path
end

"""
    RootSolver{N}

The Julia function `(z0, p1, p2, ...) -> root` that solves the equations of `residual` for
`N` unknowns with the SciML algorithm `alg`. The problem is on static arrays, so a small
system allocates little memory.

The algorithm differentiates the residual itself: with the Jacobian as a second function,
SimpleNonlinearSolve.jl was measured 40 times slower on one equation.
"""
struct RootSolver{N,G,A,O}
    name::String
    residual::G
    alg::A
    options::O
end

function (s::RootSolver{N})(z0, parameters...) where {N}
    problem = SciMLBase.NonlinearProblem{false}((u, q) -> s.residual(u, q...), StaticArrays.SVector{N}(z0), parameters)
    solution = SciMLBase.solve(problem, s.alg; s.options...)
    code = solution.retcode
    SciMLBase.successful_retcode(code) || error("rootfinder \"$(s.name)\": the solver stopped with the return code $code")
    solution.u
end

"""
    rootfinder(name, alg, f; options...) -> CasADiCodegen.Function

Make a CasADi function that solves `f(z, p1, p2, ...) = 0` for `z` with the SciML algorithm
`alg`, for example `SimpleNewtonRaphson()` of SimpleNonlinearSolve.jl. The function takes a
first guess of `z` and the other arguments of `f`, and returns the root.

`f` is a CasADi function with dense arguments. Its first argument `z` is a column vector
that is not empty, and its only result, the residual, has the size of `z`. `name` is a
valid identifier. The `options` go to the solver, for example `abstol` and `maxiters`.

CasADi differentiates the root to any order by the implicit function theorem.
`build_function` emits code that calls the solver directly. See [Solve an equation](@ref).

# Examples
```julia
using CasADiCodegen
using SimpleNonlinearSolve: SimpleNewtonRaphson

E, M, e = SX.sym("E"), SX.sym("M"), SX.sym("e")
kepler = CasadiFunction("kepler", [E, M, e], [E - e * sin(E) - M])
anomaly = rootfinder("anomaly", SimpleNewtonRaphson(), kepler; abstol = 1e-14)

m, ec = MX.sym("m"), MX.sym("ec")
f = CasadiFunction("f", [m, ec], [anomaly(m, m, ec)])    # the mean anomaly is the first guess
```
"""
function CasADiCodegen.rootfinder(name::AbstractString, alg::SciMLBase.AbstractNonlinearAlgorithm, f::CasADiCodegen.Function; options...)
    make_solver(residual, slope, n) =
        RootSolver{n,typeof(residual),typeof(alg),typeof(values(options))}(String(name), residual, alg, values(options))
    CasADiCodegen.Codegen.root_function(name, f, make_solver)
end

# The solvers in a source file: each is made again from its parts, with the code of its
# functions, the algorithm and the options. The file gets the extension from CasADiCodegen.jl.
const G = CasADiCodegen.Codegen
G.writable(::Union{Solver,EventSolver,EventIntegration,RootSolver,Crossing}) = true
extension(writer) = (G.module_path(writer, SciMLBase); :(Base.get_extension($(G.module_path(writer, CasADiCodegen)), :CasADiCodegenSciMLBaseExt)))
parts(writer, values...) = map(value -> G.value_source(writer, value), values)

G.object_source(writer::G.SourceWriter, s::Solver) = :($(extension(writer)).Solver(
    $(parts(writer, s.name, s.rhs, s.alg, s.t0, s.tf, s.free, s.specialize, s.options)...)))
G.object_source(writer::G.SourceWriter, s::EventSolver) = :($(extension(writer)).EventSolver(
    $(parts(writer, s.name, s.rhs, s.zero, s.alg, s.direction, s.interp_points, s.specialize, s.options)...)))
G.object_source(writer::G.SourceWriter, s::EventIntegration) = :($(extension(writer)).EventIntegration(
    $(parts(writer, s.name, s.rhs, s.zero, s.jump, s.alg, s.t0, s.times, s.sizes, s.specialize, s.options)...)))
G.object_source(writer::G.SourceWriter, c::Crossing{Z}) where {Z} =
    Z === Any ? :($(extension(writer)).Crossing{Any}($(G.source(writer, c.zero)))) : :($(extension(writer)).Crossing($(G.source(writer, c.zero))))
G.object_source(writer::G.SourceWriter, s::RootSolver{N}) where {N} = :(
    let residual = $(G.source(writer, s.residual)), alg = $(G.value_source(writer, s.alg)), options = $(G.value_source(writer, s.options))
        $(extension(writer)).RootSolver{$N,typeof(residual),typeof(alg),typeof(options)}($(s.name), residual, alg, options)
    end)

# CasADi functions as the functions of SciML problems. Each wrapper calls the
# in-place generated function with the arguments that the CasADi function has.

const C = CasADiCodegen.C

"A generated in-place function `g!` as a SciML function of `N` of the arguments (u, p, t)."
struct Call{N,G}
    g!::G
end
(c::Call{1})(out, u, p, t) = (c.g!((out,), (u,)); nothing)
(c::Call{2})(out, u, p, t) = (c.g!((out,), (u, p)); nothing)
(c::Call{3})(out, u, p, t) = (c.g!((out,), (u, p, t)); nothing)
(c::Call{1})(out, u, p) = (c.g!((out,), (u,)); nothing)
(c::Call{2})(out, u, p) = (c.g!((out,), (u, p)); nothing)

"The value of a generated function with one number as its result, for the arguments (x, p)."
struct Scalar{N,G}
    g!::G
end
(s::Scalar{1})(x, p) = number(s.g!, float(eltype(x)), (x,))
(s::Scalar{2})(x, p) = number(s.g!, float(promote_type(eltype(x), eltype(p))), (x, p))

"One place in memory as a vector, so that the result of a generated function can stay on the stack."
struct Slot{T} <: AbstractVector{T}
    pointer::Ptr{T}
end
Base.size(::Slot) = (1,)
Base.getindex(s::Slot, i::Int) = unsafe_load(s.pointer)
Base.setindex!(s::Slot, value, i::Int) = unsafe_store!(s.pointer, value)

"""
    number(g!, T, arguments) -> T

Return the one number that the in-place function `g!` gives for `arguments`, with no memory
from the heap. `T` must be a bits type, and `g!` must have one result with one element.
"""
function number(g!::G, ::Type{T}, arguments) where {G,T}
    @requires isbitstype(T) "the numbers of the arguments must have a bits type; got $T"
    cell = Ref{T}()
    GC.@preserve cell g!((Slot{T}(Base.unsafe_convert(Ptr{T}, cell)),), arguments)
    cell[]
end

"The Hessians of the constraints, one matrix per constraint, for the arguments (x, p)."
struct Hessians{N,G}
    g!::G
end
(h::Hessians{1})(H, x, p) = (h.g!(H, (x,)); nothing)
(h::Hessians{2})(H, x, p) = (h.g!(H, (x, p)); nothing)

symbols(f::CasADiCodegen.Function) = [CasADiCodegen.MX.sym(C.name_in(f, i), C.sparsity_in(f, i)) for i in 0:C.n_in(f)-1]
# CasADi does not accept the function names "jac" and "hess", thus each name gets a prefix.
inplace(name, arguments, value) = build_function(CasADiCodegen.Function("sciml_" * name, arguments, [value]); expression = Val{false})[2]
build_function(f; options...) = CasADiCodegen.build_function(f; options...)
shape(sp) = (C.size1(sp), C.size2(sp))

"The arguments of `f` as symbols and its result on them, when it maps a column vector to one of the same size."
function vector_field(kind, f::CasADiCodegen.Function, most::Int)
    @requires 1 <= C.n_in(f) <= most && C.n_out(f) == 1 "$kind: the function \"$(C.name(f))\" must have 1 to $most arguments and one result"
    n = C.size1(C.sparsity_in(f, 0))
    @requires shape(C.sparsity_in(f, 0)) == (n, 1) && shape(C.sparsity_out(f, 0)) == (n, 1) "$kind: the first argument and the result of \"$(C.name(f))\" must be column vectors of one size"
    arguments = symbols(f)
    arguments, CasADiCodegen.densify(f(arguments...))
end

"""
    SciMLBase.ODEFunction(f::CasADiCodegen.Function; jac = true, paramjac = true, options...)

Make the in-place SciML function of the ordinary differential equation whose right side is
the CasADi function `f(u)`, `f(u, p)` or `f(u, p, t)`. It holds generated code for the
right side and for its Jacobians by the state (`jac`) and by the parameters (`paramjac`).

`f` must have one result, a column vector of the size of its first argument. Set `jac` or
`paramjac` to `false` to leave that Jacobian out. The other `options` go to
`SciMLBase.ODEFunction`, for example `jac_prototype` with a sparse matrix that has the
sparsity pattern of the Jacobian. See [Give a CasADi function to a SciML problem](@ref).

# Examples
```julia
using CasADiCodegen
import SciMLBase

u, k, t = SX.sym("u", 2), SX.sym("k"), SX.sym("t")
rhs = CasadiFunction("rhs", [u, k, t], [vertcat(u[2], -k * u[1])])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(rhs), [1.0, 0.0], (0.0, pi), [4.0])
```
"""
function SciMLBase.ODEFunction(f::CasADiCodegen.Function; jac::Bool = true, paramjac::Bool = true, options...)
    arguments, value = vector_field("ODEFunction", f, 3)
    n = length(arguments)
    wrap(name, expression) = Call{n,Any}(inplace(name, arguments, expression)) |> concrete
    SciMLBase.ODEFunction{true,SciMLBase.FullSpecialize}(wrap("rhs", value);
        jac = jac ? wrap("jac", CasADiCodegen.jacobian(value, arguments[1])) : nothing,
        paramjac = paramjac && n >= 2 ? wrap("paramjac", CasADiCodegen.jacobian(value, arguments[2])) : nothing, options...)
end
concrete(c::Call{N}) where {N} = Call{N,typeof(c.g!)}(c.g!)

"""
    SciMLBase.NonlinearFunction(f::CasADiCodegen.Function; jac = true, options...)

Make the in-place SciML function of the equations `f(u) = 0` or `f(u, p) = 0`, where `f` is
a CasADi function. It holds generated code for the residual and for its Jacobian (`jac`).

`f` must have one result, a column vector of the size of its first argument. Set
`jac = false` to leave the Jacobian out. The other `options` go to
`SciMLBase.NonlinearFunction`.

# Examples
```julia
residual = CasADiCodegen.Function("r", [z, q], [equations])
problem = SciMLBase.NonlinearProblem(SciMLBase.NonlinearFunction(residual), z0, q0)
```
"""
function SciMLBase.NonlinearFunction(f::CasADiCodegen.Function; jac::Bool = true, options...)
    arguments, value = vector_field("NonlinearFunction", f, 2)
    n = length(arguments)
    wrap(name, expression) = concrete(Call{n,Any}(inplace(name, arguments, expression)))
    SciMLBase.NonlinearFunction{true,SciMLBase.FullSpecialize}(wrap("residual", value);
        jac = jac ? wrap("jac", CasADiCodegen.jacobian(value, arguments[1])) : nothing, options...)
end

"""
    SciMLBase.OptimizationFunction(objective::CasADiCodegen.Function, constraints = nothing; options...)

Make the SciML function of an optimization problem from the CasADi functions `objective(x)`
or `objective(x, p)` and, if the problem has them, `constraints` with the same arguments.

The result holds generated code for the objective, its gradient (`grad`) and its Hessian
(`hess`). With constraints it also holds `cons`, their Jacobian `cons_j`, and
`cons_h(H, x, p)`, where `H` has one matrix per constraint. The matrices are dense. The
other `options` go to `SciMLBase.OptimizationFunction`.

`x` must be a column vector, and the objective one number. The constraints must be a column
vector. The result has no Hessian of the Lagrange function (`lag_h`): the solvers of
Optimization.jl make it from `hess` and `cons_h`.

# Examples
```julia
x = SX.sym("x", 2)
objective = CasADiCodegen.Function("cost", [x], [(x[1] - 1)^2 + x[2]^2])
constraints = CasADiCodegen.Function("cons", [x], [x[1] + x[2]])
f = SciMLBase.OptimizationFunction(objective, constraints)
problem = SciMLBase.OptimizationProblem(f, [0.0, 0.0]; lcons = [1.0], ucons = [1.0])
```
"""
function SciMLBase.OptimizationFunction(objective::CasADiCodegen.Function, constraints::Union{Nothing,CasADiCodegen.Function} = nothing; options...)
    n = C.n_in(objective)
    @requires 1 <= n <= 2 && C.n_out(objective) == 1 && shape(C.sparsity_out(objective, 0)) == (1, 1) "OptimizationFunction: the objective must have 1 or 2 arguments and one number as its result"
    @requires C.size2(C.sparsity_in(objective, 0)) == 1 "OptimizationFunction: the unknown must be a column vector"
    arguments = symbols(objective)
    x = arguments[1]
    cost = CasADiCodegen.densify(objective(arguments...))
    call(name, expression) = concrete(Call{n,Any}(inplace(name, arguments, expression)))
    given = (; grad = call("grad", CasADiCodegen.gradient(cost, x)), hess = call("hess", CasADiCodegen.hessian(cost, x)))
    if constraints !== nothing
        @requires C.n_in(constraints) == n && C.n_out(constraints) == 1 && C.size2(C.sparsity_out(constraints, 0)) == 1 &&
                  all(shape(C.sparsity_in(constraints, i)) == shape(C.sparsity_in(objective, i)) for i in 0:n-1) "OptimizationFunction: the constraints must be a column vector and have the arguments of the objective"
        value = CasADiCodegen.densify(constraints(arguments...))
        hessians = CasADiCodegen.Function("sciml_cons_h", arguments, [CasADiCodegen.densify(CasADiCodegen.hessian(value[j], x)) for j in 1:size(value, 1)])
        cons_h = build_function(hessians; expression = Val{false})[2]
        given = (; given..., cons = call("cons", value), cons_j = call("cons_j", CasADiCodegen.jacobian(value, x)), cons_h = Hessians{n,typeof(cons_h)}(cons_h))
    end
    scalar = Scalar{n,Any}(inplace("cost", arguments, cost))
    SciMLBase.OptimizationFunction{true}(Scalar{n,typeof(scalar.g!)}(scalar.g!), SciMLBase.NoAD(); given..., options...)
end

# The problems of the solver objects of CasADi as SciML problems, which the
# user solves with a solver of the SciML ecosystem.

function check_count(label, what, values, n)
    @requires length(values) == n "$label: $what must have $(G.numbers(n)); got $(length(values))"
end

"""
    SciMLBase.ODEProblem(I::CasADiCodegen.Function, x0, p = Float64[]; z0 = Float64[], u = Float64[], options...)

Make the SciML problem of the integrator `I` of CasADi (`cvodes`, `idas`, `rk` or
`collocation`), with generated code for its right side and for the Jacobians of it.

- The state has the differential states from `x0`, then the algebraic states from `z0`,
  then the quadratures, which start at zero. `z0` must satisfy the algebraic equations at
  the initial time: the default initialization of the SciML solvers checks it and stops.
- The parameters are `p` and then the controls of the current interval. `u` has one block
  of controls per output time of `I`, as for the integrators of CasADi: block `k` holds
  from output time `k - 1`, or from the initial time, to output time `k`. With more than
  one block the problem has a callback that changes the controls at each output time.
- The time span is from the initial time of `I` to its last output time. The output times
  are the option `saveat`, so the solution has one state per output time of `I`.
- The tolerances of `cvodes` and of `idas` are the options `abstol` and `reltol`.
- With algebraic states the problem has a mass matrix, so it needs a solver for such
  problems, for example `Rodas5P()` of OrdinaryDiffEqRosenbrock.jl.

The other `options` go to the problem, and replace these ones.

An event function gives a callback: when one of its values changes its sign from positive
to negative, the state becomes that of the option `transition` of `I`. The vectors must
have the sizes of `I`. A `callback` among the options runs after the one that changes the
controls and after that of the events. The
callback writes the controls into the parameters of the problem, so do not solve one
problem with several intervals from two tasks at the same time. See [Replace the solvers of CasADi](@ref).

# Examples
```julia
problem = SciMLBase.ODEProblem(I, [1.0, 0.0, 0.0, 1.0], [1.0])      # I is a `cvodes` integrator
solution = SciMLBase.solve(problem, Vern9(); abstol = 1e-13, reltol = 1e-13)
```
"""
function SciMLBase.ODEProblem(I::CasADiCodegen.Function, x0::AbstractVector{<:Real}, p::AbstractVector{<:Real} = Float64[];
                              z0::AbstractVector{<:Real} = Float64[], u::AbstractVector{<:Real} = Float64[], options...)
    model = G.integrator_model(I)
    label = "integrator \"$(model.name)\""
    intervals = length(model.times)
    check_count(label, "the initial state", x0, model.nx)
    check_count(label, "the algebraic states", z0, model.nz)
    check_count(label, "the parameters", p, model.np)
    check_count(label, "the controls", u, model.nu * intervals)
    mass = model.nz == 0 ? (;) : (; mass_matrix = LinearAlgebra.Diagonal([ones(model.nx); zeros(model.nz); ones(model.nq)]))
    f = SciMLBase.ODEFunction(G.right_side(model); mass...)
    controls = reshape(float(collect(u)), model.nu, intervals)
    callback = get(options, :callback, nothing)
    if model.events > 0
        transition = CasADiCodegen.C.has_function(I, "transition") ? CasADiCodegen.C.get_function(I, "transition") : nothing
        events = G.event_functions(model, transition)
        emitted(g) = G.Emitted(G.build_function(g; expression = Val{false}, nonzeros = true), g, :nonzeros)
        rhs = model.nz == 0 ? G.RightSide(nothing) : G.emitted_right_side(model)
        found = event_callback(model.name, rhs, emitted(events.zero), emitted(events.jump), model.events, Ref(0), typemax(Int))
        callback = callback === nothing ? found : SciMLBase.CallbackSet(found, callback)
    end
    changes = model.nu > 0 && intervals > 1 ? (; control_changes(controls, model.times, model.np, callback)...) :
              callback === nothing ? (;) : (; callback)
    SciMLBase.ODEProblem(f, float([x0; z0; zeros(model.nq)]), (model.t0, model.times[end]), float([p; controls[:, 1]]);
                         saveat = model.times, model.tolerances..., options..., changes...)
end

"""
    control_changes(controls, times, np, callback) -> NamedTuple

Return the options `callback` and `tstops` of a problem whose controls are constant
between the output times `times`. Column `k` of `controls` holds until `times[k]`. The
controls of the current interval are the parameters after the first `np`. The callback
writes them at the start of a solve and at each output time, and the solver stops at each
output time, so that no step crosses a jump of the right side. `callback` is a callback of
the user or `nothing`; it runs after the change of the controls.
"""
function control_changes(controls::Matrix{Float64}, times::Vector{Float64}, np::Int, callback)
    slots = np+1:np+size(controls, 1)
    set!(integrator, k) = (integrator.p[slots] .= view(controls, :, k); nothing)
    start!(_, _, _, integrator) = set!(integrator, 1)
    # The last output time is the end of the integration, where no interval follows.
    inner = view(times, 1:length(times)-1)
    at_change(_, t, _) = t in inner
    next!(integrator) = set!(integrator, findfirst(==(integrator.t), inner) + 1)
    change = SciMLBase.DiscreteCallback(at_change, next!; initialize = start!, save_positions = (false, false))
    (; callback = callback === nothing ? change : SciMLBase.CallbackSet(change, callback), tstops = times)
end

"""
    SciMLBase.NonlinearProblem(R::CasADiCodegen.Function, z0, others...; options...)

Make the SciML problem of the rootfinder `R` of CasADi (`newton`, `fast_newton` or
`kinsol`), with generated code for the residual and for its Jacobian.

`z0` is the first guess of the unknown. `others` are the other arguments of `R`, in their
order, as vectors of their nonzeros; the parameters of the problem are those vectors,
joined. The vectors must have the sizes of `R`. The `options` go to the problem.

# Examples
```julia
problem = SciMLBase.NonlinearProblem(R, [1.0], [1.0, 0.3])          # R(z0, p) is a `kinsol` rootfinder
solution = SciMLBase.solve(problem, SimpleNewtonRaphson())
```
"""
function SciMLBase.NonlinearProblem(R::CasADiCodegen.Function, z0::AbstractVector{<:Real}, others::AbstractVector{<:Real}...; options...)
    model = G.rootfinder_model(R)
    label = "rootfinder \"$(model.name)\""
    check_count(label, "the unknown", z0, model.n)
    sizes = G.other_sizes(model)
    @requires collect(map(length, others)) == sizes "$label: the other arguments must have the sizes ($(join(sizes, ", "))); got ($(join(map(length, others), ", ")))"
    f = SciMLBase.NonlinearFunction(G.residual_function(model))
    SciMLBase.NonlinearProblem(f, float(collect(z0)), float(reduce(vcat, others; init = Float64[])); options...)
end

"""
    SciMLBase.OptimizationProblem(S::CasADiCodegen.Function, x0, p = Float64[]; lbx, ubx, lbg, ubg, options...)

Make the SciML problem of the solver `S` that `nlpsol` or `conic` of CasADi made, with
generated code for the objective, the constraints and their derivatives (see
`OptimizationFunction`).

For a solver of `conic`, the parameters `p` are the nonzeros of `h`, then `g`, then the
nonzeros of `a`, and the constraints `g` of the bounds are the rows of `a`.

The bounds have the names and the default values of CasADi: no bound on the unknowns, and
none on the constraints. The other `options` go to the problem. The unknowns must be a
column vector, and the vectors must have the sizes of `S`.

# Examples
```julia
problem = SciMLBase.OptimizationProblem(S, [0.0, 0.0]; lbg = [1.0], ubg = [1.0])
solution = SciMLBase.solve(problem, Ipopt.Optimizer())              # with OptimizationMOI.jl
```
"""
function SciMLBase.OptimizationProblem(S::CasADiCodegen.Function, x0::AbstractVector{<:Real}, p::AbstractVector{<:Real} = Float64[];
                                       lbx = nothing, ubx = nothing, lbg = nothing, ubg = nothing, options...)
    model = G.optimization_model(S)
    label = "solver \"$(model.name)\""
    check_count(label, "the unknowns", x0, model.nx)
    check_count(label, "the parameters", p, model.np)
    for (what, bound, n) in (("the lower bounds of the unknowns", lbx, model.nx), ("the upper bounds of the unknowns", ubx, model.nx),
                             ("the lower bounds of the constraints", lbg, model.ng), ("the upper bounds of the constraints", ubg, model.ng))
        bound === nothing || check_count(label, what, bound, n)
    end
    constrained = model.ng > 0
    f = SciMLBase.OptimizationFunction(model.objective, constrained ? model.constraints : nothing)
    limit(bound, value) = !constrained ? nothing : bound === nothing ? fill(value, model.ng) : float(collect(bound))
    # A SciML problem takes both bounds of the unknowns or none.
    bounded = lbx !== nothing || ubx !== nothing
    side(bound, value) = !bounded ? nothing : bound === nothing ? fill(value, model.nx) : float(collect(bound))
    SciMLBase.OptimizationProblem(f, float(collect(x0)), float(collect(p)); lb = side(lbx, -Inf), ub = side(ubx, Inf),
                                  lcons = limit(lbg, -Inf), ucons = limit(ubg, Inf), options...)
end

# A solver of CasADi for optimization problems in the graph of a function,
# replaced in generated code by a solver of the SciML ecosystem.

"How near a bound a value must be for the bound to count as active, relative to the bound when that is more than 1."
const ACTIVE_TOLERANCE = 1e-6
"The least size of the multiplier of an active bound, relative to the gradient of the objective when that is more than 1."
const MULTIPLIER_TOLERANCE = 1e-8

"""
    OptimizationSolver

The Julia function `(x0, p, lbx, ubx, lbg, ubg) -> (x, lam_x, lam_g)` that solves the
optimization problem of `functions` (see `CasADiCodegen.Codegen.optimization_functions`) with the
SciML algorithm `alg`, for `sizes[1]` unknowns, `sizes[2]` parameters and `sizes[3]`
constraints. The multipliers have the signs of CasADi (see `multipliers`). A problem with
no constraint gives no `lam_g`.
"""
struct OptimizationSolver{F,A,O}
    name::String
    functions::F
    alg::A
    options::O
    sizes::NTuple{3,Int}
end

G.optimizer(name::AbstractString, alg, functions, sizes, options) = OptimizationSolver(String(name), functions, alg, options, sizes)

"The functions of the solver `s` as a SciML function with the arguments (x, p)."
function problem_function(s::OptimizationSolver)
    wrap(W, g) = W{2,typeof(g)}(g)
    F = s.functions
    given = s.sizes[3] == 0 ? (;) : (; cons = wrap(Call, F.cons), cons_j = wrap(Call, F.cons_j), cons_h = wrap(Hessians, F.cons_h))
    SciMLBase.OptimizationFunction{true}(wrap(Scalar, F.cost), SciMLBase.NoAD(); grad = wrap(Call, F.grad), hess = wrap(Call, F.hess), given...)
end

"""
    active_side(value, lower, upper) -> -1, 0, 1 or nothing

Return the bound at which `value` is: -1 for the lower one, 1 for the upper one, 0 when the
two bounds are one number, and `nothing` when the value is at none.
"""
function active_side(value, lower, upper)
    near(bound) = isfinite(bound) && abs(value - bound) <= ACTIVE_TOLERANCE * max(1.0, abs(bound))
    lower == upper && return 0
    near(lower) ? -1 : near(upper) ? 1 : nothing
end

"""
    multipliers(gradient, J, x, lbx, ubx, g, lbg, ubg) -> (lam_x, lam_g)

Return the multipliers of the bounds of the unknowns and of the constraints at the solution
`x` of an optimization problem. The objective has the `gradient`, and the constraints have
the values `g` and the Jacobian `J`. The multipliers have the signs of CasADi:
`gradient + J' lam_g + lam_x = 0`, and a multiplier is positive at an upper bound and
negative at a lower bound.

The multiplier of a bound that is not active is exactly zero, because CasADi reads the
active bounds from the signs of the multipliers when it differentiates the solution. The
others are the solution of the equation above by least squares. A bound whose multiplier
then has the wrong sign, or is zero to rounding, is not active, and the equation is solved
again without it.
"""
function multipliers(gradient, J, x, lbx, ubx, g, lbg, ubg)
    nx, ng = length(x), length(g)
    sides = Union{Nothing,Int}[[active_side(x[i], lbx[i], ubx[i]) for i in 1:nx]; [active_side(g[j], lbg[j], ubg[j]) for j in 1:ng]]
    active = findall(!isnothing, sides)
    least = MULTIPLIER_TOLERANCE * max(1.0, LinearAlgebra.norm(gradient, Inf))
    column(k) = k <= nx ? Float64.(1:nx .== k) : J[k-nx, :]
    solved(set) = isempty(set) ? Float64[] : LinearAlgebra.qr(reduce(hcat, map(column, set)), LinearAlgebra.ColumnNorm()) \ -gradient
    values = solved(active)
    kept = [k for (k, value) in zip(active, values) if sides[k] == 0 || (sign(value) == sides[k] && abs(value) > least)]
    length(kept) == length(active) || (values = solved(kept))
    lam = zeros(nx + ng)
    lam[kept] = values
    @ensures all(iszero(lam[k]) for k in 1:nx+ng if sides[k] === nothing) "the multiplier of a bound that is not active must be zero"
    lam[1:nx], lam[nx+1:end]
end

function (s::OptimizationSolver)(x0, p, lbx, ubx, lbg, ubg)
    nx, np, ng = s.sizes
    for a in (x0, p, lbx, ubx, lbg, ubg)
        @requires eltype(a) <: Union{AbstractFloat,Integer} "solver \"$(s.name)\" takes floating-point numbers only; got $(eltype(a))"
    end
    f = problem_function(s)
    parameters = Vector{Float64}(p)
    limits = ng == 0 ? (;) : (; lcons = Vector{Float64}(lbg), ucons = Vector{Float64}(ubg))
    problem = SciMLBase.OptimizationProblem(f, Vector{Float64}(x0), parameters; lb = Vector{Float64}(lbx), ub = Vector{Float64}(ubx), limits...)
    solution = SciMLBase.solve(problem, s.alg; s.options...)
    code = solution.retcode
    SciMLBase.successful_retcode(code) || error("solver \"$(s.name)\": the solver stopped with the return code $code")
    x = Vector{Float64}(solution.u)
    gradient, g, J = zeros(nx), zeros(ng), zeros(ng, nx)
    f.grad(gradient, x, parameters)
    ng == 0 || (f.cons(g, x, parameters); f.cons_j(J, x, parameters))
    lam_x, lam_g = multipliers(gradient, J, x, lbx, ubx, g, lbg, ubg)
    ng == 0 ? (x, lam_x) : (x, lam_x, lam_g)
end

G.writable(::OptimizationSolver) = true
G.object_source(writer::G.SourceWriter, s::OptimizationSolver) = :($(extension(writer)).OptimizationSolver(
    $(parts(writer, s.name, s.functions, s.alg, s.options, s.sizes)...)))

end # module
