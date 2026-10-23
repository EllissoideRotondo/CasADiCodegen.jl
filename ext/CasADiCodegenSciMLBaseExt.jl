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
    GridSolver

The solver of an integrator over a grid of times (see `CasADiCodegen.Codegen.ode_grid`): it
integrates `rhs` from `t0` through `times` with the SciML algorithm `alg`, in one run.

- `(x0, q) -> path`: the states at the `times`, where the solver stops (`tstops`),
  the first state at all the times, then the second one, and so on.
- `(x0, q, jump!) -> u`: the solver stops at each of the `times` (`tstops`), calls
  `jump!(u, k)` there, and goes on from the changed state; `u` is the state at the last time.

Each task keeps one solver state per number type and form, as `Solver` does, and the
result is memory of that state.
"""
struct GridSolver{F,A,O}
    name::String
    rhs::F
    alg::A
    t0::Float64
    times::Vector{Float64}
    specialize::DataType
    options::O
end

function grid_state(s::GridSolver, ::Type{T}, x0, q, jumps::Bool) where {T}
    get!(task_local_storage(), (s, T, jumps)) do
        u0, q0 = copyto!(Vector{T}(undef, length(x0)), x0), copyto!(Vector{T}(undef, length(q)), q)
        make_grid_state(s, s.specialize, u0, q0, jumps)
    end
end

function make_grid_state(s::GridSolver, ::Type{S}, u0::Vector{T}, q0::Vector, jumps::Bool) where {S,T}
    problem = SciMLBase.ODEProblem{true,S}(problem_function(s.rhs, S), u0, (T(s.t0), T(s.times[end])), q0)
    integrator = jumps ?
        SciMLBase.init(problem, s.alg; tstops = s.times, save_everystep = false, save_start = false, save_end = false, s.options...) :
        # The solver stops at each time: the error of the interpolant between two steps is not
        # under the control of the tolerances, and an adjoint gradient multiplies it.
        SciMLBase.init(problem, s.alg; saveat = s.times, tstops = s.times, save_everystep = false, save_start = false, s.options...)
    (; integrator, path = Vector{T}(undef, length(u0) * length(s.times)))
end

function restart!(integrator, x0, q)
    u0 = integrator.sol.prob.u0
    copyto!(u0, x0)
    copyto!(integrator.p, q)
    SciMLBase.reinit!(integrator, u0)
    integrator
end

function (s::GridSolver)(x0, q)
    state = grid_state(s, promote_type(eltype(x0), eltype(q)), x0, q, false)
    sample!(state.integrator, state.path, s, x0, q)
end

function sample!(integrator, path, s::GridSolver, x0, q)
    SciMLBase.solve!(restart!(integrator, x0, q))
    code = integrator.sol.retcode
    SciMLBase.successful_retcode(code) || error("integrator \"$(s.name)\": the solver stopped with the return code $code")
    saved, n, N = integrator.sol.u, length(x0), length(s.times)
    @ensures length(saved) == N "integrator \"$(s.name)\": the solver must save the state at each of the $N times; do not give saveat or save_start"
    for k in 1:N, i in 1:n
        path[(i-1)*N+k] = saved[k][i]
    end
    path
end

function (s::GridSolver)(x0, q, jump!)
    state = grid_state(s, promote_type(eltype(x0), eltype(q)), x0, q, true)
    jumped!(state.integrator, s, x0, q, jump!)
end

function jumped!(integrator, s::GridSolver, x0, q, jump!)
    restart!(integrator, x0, q)
    direction = sign(s.times[end] - s.t0)
    for (k, stop) in enumerate(s.times)
        # Loop invariant: the state of `integrator` follows the calls of `jump!` at the times before `k`.
        while direction * (stop - integrator.t) > 0
            SciMLBase.step!(integrator)
            code = integrator.sol.retcode
            code in (SciMLBase.ReturnCode.Default, SciMLBase.ReturnCode.Success) ||
                error("integrator \"$(s.name)\": the solver stopped with the return code $code")
        end
        jump!(integrator.u, k)
        SciMLBase.derivative_discontinuity!(integrator, true)
    end
    integrator.u
end

"""
    integrator(name, alg, dae, t0, tf; specialize, adjoint = false, mass, options...) -> CasADiCodegen.Function
    integrator(name, alg, dae, t0, times::AbstractVector; specialize, adjoint = false, options...) -> CasADiCodegen.Function
    integrator(name, alg, dae; specialize, adjoint = false, mass, direction = 0, interp_points = 10,
               events = 1, terminal = true, exact = true, options...) -> CasADiCodegen.Function

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
can be in any order and on both sides of `t0`. One run of the solver, which stops at
each time, gives the states at all the times after `t0`, and one more those before it;
`options` must not hold a saving option, such as `saveat` or `tstops`. A model with
quadratures returns them as a second result.

# Arguments
- `name`: the name of the CasADi function, a valid identifier.
- `dae`: a dictionary as for the CasADi integrators. It holds the state `"x"` and its
  derivative `"ode"`, both dense column vectors. Optional entries are the parameters `"p"`,
  the time `"t"` and the integrands `"quad"`. With free times, the entry `"zero"` is a
  column of event functions, and `"transition"` is the state after an event: one column,
  or one column per event function.
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
- `direction = 0`: the change of sign of an element of `"zero"` that is an event: any with 0, from
  negative to positive with 1, and from positive to negative with -1.
- `interp_points = 10`: the number of points of each step at which the solver looks for
  the change of sign, besides the end of the step.
- `events = 1`: the number of events that the integration passes.
- `terminal = true`: the integration ends at the last event. With `false` it applies the
  transition there too and goes on to `tf`.
- `exact = true`: a call throws when fewer than `events` events come before `tf`. With
  `false` it passes at most `events` events and returns their count.
- `options...`: options of the solver, for example `abstol`, `reltol` and `maxiters`.

With an event function, the function returns the state at the event, the quadratures if
the model has them, and the event time `te`; `te` is a vector with `events` more than 1.
With several event functions, the result `ie` follows: for each event, the index from 1 of
the event function that changed its sign. With `exact = false`, the last result `ne` is the
number of events, and a missing event has the time `tf` and the index 0. A zero at the
initial time is not an event.

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
    @requires !any(in(keys(options)), (:saveat, :tstops, :save_start, :save_end, :save_everystep, :save_idxs)) "integrator \"$name\": the times give saveat and tstops; do not pass saving options"
    make_solver(rhs, n, m, start, stops) = GridSolver(String(name), rhs, alg, start, stops, specialize, values(options))
    CasADiCodegen.Codegen.ode_grid(name, dae, t0, times, make_solver; adjoint)
end

function CasADiCodegen.integrator(name::AbstractString, alg::SciMLBase.AbstractODEAlgorithm, dae;
                           specialize::Type{<:SciMLBase.AbstractSpecialization} = SciMLBase.FullSpecialize,
                           direction::Integer = 0, interp_points::Integer = 10, adjoint::Bool = false,
                           events::Integer = 1, terminal::Bool = true, exact::Bool = true, mass::AbstractVector{<:Real} = Float64[], options...)
    @requires direction in (-1, 0, 1) "integrator \"$name\": the direction of the event must be -1, 0 or 1; got $direction"
    @requires interp_points >= 0 "integrator \"$name\": interp_points must not be negative; got $interp_points"
    make_solver(rhs, n, m) = Solver(String(name), rhs, alg, NaN, NaN, true, specialize, values(options))
    @requires haskey(dae, "zero") || (events == 1 && terminal && exact) "integrator \"$name\": events, terminal and exact need an event function, \"zero\""
    haskey(dae, "zero") || return CasADiCodegen.Codegen.ode_integrator(name, dae, make_solver; adjoint, mass = Vector{Float64}(mass))
    # Without full specialization the solver does not know the type of the event function,
    # so that it compiles once for all integrators with an event.
    crossing(zero) = specialize === SciMLBase.FullSpecialize ? Crossing(zero) : Crossing{Any}(zero)
    make_event(rhs, zero, n, m, k) = EventSolver(String(name), rhs, crossing(zero), alg, Int(direction), Int(interp_points), specialize, values(options), k, exact)
    CasADiCodegen.Codegen.ode_event(name, dae, make_solver, make_event; adjoint, events, terminal, exact, mass = Vector{Float64}(mass))
end

"""
    Crossing(zero)

The event function `zero((out,), (u, q, (t,)))` of an integrator as the condition of a SciML
callback: `(u, t, integrator)` for one element, and `(out, u, t, integrator)` for several.
"""
struct Crossing{Z}
    zero::Z
end
(c::Crossing)(u, t, integrator) = (T = promote_type(eltype(u), typeof(t)); number(c.zero, T, (u, integrator.p, (t,)))::T)
(c::Crossing)(out, u, t, integrator) = (c.zero((out,), (u, integrator.p, (t,))); nothing)

"""
    EventSolver

The Julia function `(z0, x0, p, t0, tf) -> (te, i)`, or `(z0, x0, t0, tf) -> (te, i)`, that
integrates `rhs` from `t0` toward `tf` with the SciML algorithm `alg` and returns the time of
the first event, where an element of the event function `zero` changes its sign, and the
place `i` of that element, from 1. The guess `z0` has no use. With `exact`, a call throws
when no event comes before `tf`; otherwise it returns `(tf, 0)`.

`direction` selects the change of sign: any with 0, from negative to positive with 1, and
from positive to negative with -1. The solver looks for the change at `interp_points`
points of each step and at its end. `width` is the number of elements of `zero`. When two
elements change their sign in one root search, the first one counts. Each task keeps one
solver state per number type and per direction of time, as `Solver` does.
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
    width::Int
    exact::Bool
end

function (s::EventSolver)(guess, x0, rest...)
    q = length(rest) == 3 ? rest[1] : CasADiCodegen.Codegen.NO_PARAMETERS
    T = float(promote_type(eltype(guess), eltype(x0), eltype(q), eltype(rest[end-1]), eltype(rest[end])))
    t0, tf = T(rest[end-1][1]), T(rest[end][1])
    if tf != t0
        state = get!(task_local_storage(), tf < t0 ? (s, T, :backward) : (s, T)) do
            u0, q0 = copyto!(Vector{T}(undef, length(x0)), x0), copyto!(Vector{T}(undef, length(q)), q)
            make_state(s, CasADiCodegen.Codegen.consistent!(u0, s.rhs, q0, t0), q0, t0, tf)
        end
        integrator = state.integrator
        copyto!(integrator.sol.prob.u0, x0)
        copyto!(integrator.p, q)
        CasADiCodegen.Codegen.consistent!(integrator.sol.prob.u0, s.rhs, integrator.p, t0)
        run!(state, s, t0, tf) && return StaticArrays.SVector{2,T}(event_time(state, s, T), T(state.place[]))
    end
    s.exact && error("integrator \"$(s.name)\": no event between the initial time $t0 and the final time $tf")
    StaticArrays.SVector{2,T}(tf, zero(T))
end

"""
    event_time(state, s::EventSolver, T)

Return the time of the event that ended the run of `state`; `state.integrator.u` and
`state.integrator.t` are then the state and the time of the event, and `state.place[]` the
element of the event function that changed its sign. For `Float64` the result is the time
of the solver.

The root search of the solver does not give exact derivatives for a number type that
carries them, such as a dual number. For such a type, one Newton step on that element of
the event function along the trajectory makes the derivatives those of the implicit
function. The value does not change, because the element is zero there. Its slope comes
from a finite difference, which is sufficient because it multiplies a residual that is
almost zero.
"""
event_time(state, s::EventSolver, ::Type{Float64}) = state.integrator.t
function event_time(state, s::EventSolver, ::Type{T}) where {T}
    integrator = state.integrator
    u, t = integrator.u, integrator.t
    flow = similar(u)
    s.rhs(flow, u, integrator.p, t)
    h = T(sqrt(eps(Float64)))
    residual = element(s, state, u, t)
    # The rows of the algebraic states hold residuals, so those states follow from their equations at the later time.
    moved = CasADiCodegen.Codegen.consistent!(u .+ h .* flow, s.rhs, integrator.p, t + h)
    slope = (element(s, state, moved, t + h) - residual) / h
    t - residual / slope
end

"The element of the event function that ended the run of `state`, at the state `u` and the time `t`."
element(s::EventSolver, state, u, t) = s.width == 1 ? s.zero(u, t, state.integrator) :
    (values = similar(u, promote_type(eltype(u), typeof(t)), s.width); s.zero(values, u, t, state.integrator); values[state.place[]])

"Integrate from `t0` toward `tf`; true when an event ended the run."
function run!(state, s::EventSolver, t0, tf)
    integrator = state.integrator
    SciMLBase.reinit!(integrator, integrator.sol.prob.u0; t0, tf)
    SciMLBase.solve!(integrator)
    integrator.sol.retcode == SciMLBase.ReturnCode.Terminated
end

"""
    Stop(place, direction)

The affect of the callback of an `EventSolver`: it ends the run at the first element whose
change of sign has the `direction` (0 for any), and keeps the place of that element in
`place[]`. With one element it is the affect of a `ContinuousCallback`, which calls it for
the right direction only.
"""
struct Stop
    place::Base.RefValue{Int}
    direction::Int
end
(a::Stop)(integrator) = (a.place[] = 1; SciMLBase.terminate!(integrator))
function (a::Stop)(integrator, crossings)
    # `crossings[i]` is -1 when element `i` changes its sign from positive to negative, +1 for the opposite change, and 0 otherwise.
    i = findfirst(c -> c != 0 && (a.direction == 0 || c == a.direction), crossings)
    # A change in the other direction leaves the state and its derivative as they are.
    i === nothing && return SciMLBase.derivative_discontinuity!(integrator, false)
    a.place[] = i
    SciMLBase.terminate!(integrator)
end

function make_state(s::EventSolver, ::Type{S}, u0::Vector, q0::Vector, t0, tf) where {S}
    stop = Stop(Ref(0), s.direction)
    callback = s.width == 1 ?
        SciMLBase.ContinuousCallback(s.zero, s.direction >= 0 ? stop : nothing, s.direction <= 0 ? stop : nothing;
                                     save_positions = (false, false), s.interp_points) :
        SciMLBase.VectorContinuousCallback(s.zero, stop, s.width; save_positions = (false, false), s.interp_points)
    problem = SciMLBase.ODEProblem{true,S}(problem_function(s.rhs, S), u0, (t0, tf), q0)
    integrator = SciMLBase.init(problem, s.alg; callback, save_everystep = false, save_start = false, save_end = false, s.options...)
    (; integrator, place = stop.place)
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

function CasADiCodegen.Codegen.warm(s::GridSolver)
    precompile(make_grid_state, (typeof(s), Type{s.specialize}, Vector{Float64}, Vector{Float64}, Bool))
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

"The largest number of unknowns for which the SciML problem of a rootfinder is on static arrays."
const STATIC_ROOT_SIZE = 16

"""
    RootSolver{N}

The Julia function `(z0, p1, p2, ...) -> root` that solves the equations of `residual` for
`N` unknowns with the SciML algorithm `alg`. Up to `STATIC_ROOT_SIZE` unknowns the problem
is on static arrays, so a small system allocates little memory, and the algorithm
differentiates the residual itself: with the Jacobian as a second function,
SimpleNonlinearSolve.jl was measured 40 times slower on one equation. A larger problem is
on vectors, with the Jacobian `slope` of the generated code.
"""
struct RootSolver{N,G,J,A,O}
    name::String
    residual::G
    slope::J
    alg::A
    options::O
end

function (s::RootSolver{N})(z0, parameters...) where {N}
    problem = if N <= STATIC_ROOT_SIZE
        SciMLBase.NonlinearProblem{false}((u, q) -> s.residual(u, q...), StaticArrays.SVector{N}(z0), parameters)
    else
        f = SciMLBase.NonlinearFunction{false}((u, q) -> s.residual(u, q...); jac = (u, q) -> s.slope(u, q...))
        SciMLBase.NonlinearProblem{false}(f, collect(z0), parameters)
    end
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

Up to 16 unknowns the solver works on static arrays and differentiates the residual
itself. With more unknowns it works on vectors and gets the dense Jacobian of the
generated code.

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
        RootSolver{n,typeof(residual),typeof(slope),typeof(alg),typeof(values(options))}(String(name), residual, slope, alg, values(options))
    CasADiCodegen.Codegen.root_function(name, f, make_solver)
end

# The solvers in a source file: each is made again from its parts, with the code of its
# functions, the algorithm and the options. The file gets the extension from CasADiCodegen.jl.
const G = CasADiCodegen.Codegen
G.writable(::Union{Solver,GridSolver,EventSolver,EventIntegration,RootSolver,Crossing}) = true
extension(writer) = (G.module_path(writer, SciMLBase); :(Base.get_extension($(G.module_path(writer, CasADiCodegen)), :CasADiCodegenSciMLBaseExt)))
parts(writer, values...) = map(value -> G.value_source(writer, value), values)

G.object_source(writer::G.SourceWriter, s::Solver) = :($(extension(writer)).Solver(
    $(parts(writer, s.name, s.rhs, s.alg, s.t0, s.tf, s.free, s.specialize, s.options)...)))
G.object_source(writer::G.SourceWriter, s::GridSolver) = :($(extension(writer)).GridSolver(
    $(parts(writer, s.name, s.rhs, s.alg, s.t0, s.times, s.specialize, s.options)...)))
G.object_source(writer::G.SourceWriter, s::EventSolver) = :($(extension(writer)).EventSolver(
    $(parts(writer, s.name, s.rhs, s.zero, s.alg, s.direction, s.interp_points, s.specialize, s.options, s.width, s.exact)...)))
G.object_source(writer::G.SourceWriter, s::EventIntegration) = :($(extension(writer)).EventIntegration(
    $(parts(writer, s.name, s.rhs, s.zero, s.jump, s.alg, s.t0, s.times, s.sizes, s.specialize, s.options)...)))
G.object_source(writer::G.SourceWriter, c::Crossing{Z}) where {Z} =
    Z === Any ? :($(extension(writer)).Crossing{Any}($(G.source(writer, c.zero)))) : :($(extension(writer)).Crossing($(G.source(writer, c.zero))))
G.object_source(writer::G.SourceWriter, s::RootSolver{N}) where {N} = :(
    let residual = $(G.source(writer, s.residual)), slope = $(G.source(writer, s.slope)), alg = $(G.value_source(writer, s.alg)),
        options = $(G.value_source(writer, s.options))
        $(extension(writer)).RootSolver{$N,typeof(residual),typeof(slope),typeof(alg),typeof(options)}($(s.name), residual, slope, alg, options)
    end)

# CasADi functions as the functions of SciML problems. Each wrapper calls the
# in-place generated function with the arguments that the CasADi function has.

const C = CasADiCodegen.C

"A generated in-place function `g!` as a SciML function of `N` of the arguments (u, p, t)."
struct Call{N,G}
    g!::G
end
# The calls are inline, so that the tuples of arrays stay in registers; the code of a small
# function comes inline with them (see `Codegen.Inlined`).
(c::Call{1})(out, u, p, t) = (@inline c.g!((out,), (u,)); nothing)
(c::Call{2})(out, u, p, t) = (@inline c.g!((out,), (u, given(p))); nothing)
(c::Call{3})(out, u, p, t) = (@inline c.g!((out,), (u, given(p), t)); nothing)
(c::Call{1})(out, u, p) = (@inline c.g!((out,), (u,)); nothing)
(c::Call{2})(out, u, p) = (@inline c.g!((out,), (u, given(p))); nothing)

"""
    given(p)

The parameters `p` as the generated code takes them: a problem without parameters
(`SciMLBase.NullParameters`) gives an empty vector, which a parameter argument without
elements accepts.
"""
given(p) = p
given(::SciMLBase.NullParameters) = StaticArrays.SVector{0,Float64}()

"""
    Allocating{K}(call, shape)

The in-place function `call(out, u, p, t)` (`K = 3`) or `call(out, u, p)` (`K = 2`) of a
SciML function, with an out-of-place method `(u, p, t)` or `(u, p)` that returns a new
result. `shape` gives the result: `Val(dims)` for a dense array of the dimensions `dims`,
`Each{m}(Val(dims))` for a vector of `m` such arrays, a sparse matrix whose pattern the
result copies, or a vector of sparse matrices. The element type is that of the arguments,
at least `Float64`. A dense shape keeps the wrapper of a function without fields free of
fields too, so that a call through dynamic dispatch allocates no memory.
"""
struct Allocating{K,F,S}
    call::F
    shape::S
end
Allocating{K}(call, shape) where {K} = Allocating{K,typeof(call),typeof(shape)}(call, shape)
(a::Allocating{3})(out, u, p, t) = a.call(out, u, p, t)
(a::Allocating{2})(out, u, p) = a.call(out, u, p)
function (a::Allocating{3})(u, p, t)
    out = new_result(a.shape, float(promote_type(eltype(u), number_type(p), typeof(t))))
    a.call(out, u, p, t)
    out
end
function (a::Allocating{2})(u, p)
    out = new_result(a.shape, float(promote_type(eltype(u), number_type(p))))
    a.call(out, u, p)
    out
end

"`M` dense arrays of the dimensions `D`, as the result of an `Allocating` function."
struct Each{M,D} end
Each{M}(::Val{D}) where {M,D} = Each{M,D}()
number_type(p::AbstractArray) = eltype(p)
number_type(p::Number) = typeof(p)
number_type(p) = Bool
new_result(::Val{D}, ::Type{T}) where {D,T} = zeros(T, D)
new_result(prototype::AbstractArray{<:Number}, ::Type{T}) where {T} = fill!(similar(prototype, T), zero(T))
new_result(prototypes::Vector, ::Type{T}) where {T} = [new_result(prototype, T) for prototype in prototypes]
new_result(::Each{M,D}, ::Type{T}) where {M,D,T} = [zeros(T, D) for _ in 1:M]

"""
    Static{N,S}(g)

A generated out-of-place function `g` with static arrays as a SciML function of `N` of the
arguments (u, p, t). `S` is `Nothing` for a vector, or `Tuple{R,C}` for an R x C matrix,
which the result of `g` has whatever the number of its columns.
"""
struct Static{N,S,G}
    g::G
end
(s::Static{1})(u, p, t) = shaped(s, s.g(u))
(s::Static{2})(u, p, t) = shaped(s, s.g(u, given(p)))
(s::Static{3})(u, p, t) = shaped(s, s.g(u, given(p), t))
(s::Static{1})(u, p) = shaped(s, s.g(u))
(s::Static{2})(u, p) = shaped(s, s.g(u, given(p)))
shaped(::Static{N,Nothing}, a) where {N} = a
shaped(::Static{N,Tuple{R,C}}, a) where {N,R,C} = StaticArrays.SMatrix{R,C}(Tuple(a))

"The value of a generated function with one number as its result, for the arguments (x, p)."
struct Scalar{N,G} <: CasADiCodegen.Codegen.GeneratedObjective
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
(h::Hessians{1})(H, x, p) = (@inline h.g!(H, (x,)); nothing)
(h::Hessians{2})(H, x, p) = (@inline h.g!(H, (x, p)); nothing)

"""
    LagrangeHessian{N}(g)

The Hessian of the Lagrange function `σ f(x, p) + μ' g(x, p)` by `x` as the `lag_h(h, x, σ, μ, p)`
of a SciML function, for an objective of `N` arguments. `g` is the generated function on
nonzeros of the arguments (x, σ, μ) or (x, p, σ, μ), whose result is the upper triangle of
the Hessian; `h` receives its stored values.
"""
struct LagrangeHessian{N,G} <: CasADiCodegen.Codegen.GeneratedLagrangeHessian
    g::G
end
(l::LagrangeHessian{1})(h, x, σ, μ, p) = (l.g((h,), (x, σ, μ)); nothing)
(l::LagrangeHessian{2})(h, x, σ, μ, p) = (l.g((h,), (x, p, σ, μ)); nothing)

"""
    Opaque(f)

The in-place function `f(out, u, p, t)` behind a field of abstract type, so that the type
of a SciML function that holds it is the same for each model. A call costs a dynamic
dispatch.
"""
struct Opaque
    f::Any
end
(o::Opaque)(out, u, p, t) = (o.f(out, u, p, t); nothing)
(o::Opaque)(u, p, t) = o.f(u, p, t)
"The in-place function `f(out, u, p)`, with its out-of-place method `f(u, p)`, behind a field of abstract type, as `Opaque` is for (u, p, t)."
struct OpaqueNonlinear
    f::Any
end
(o::OpaqueNonlinear)(out, u, p) = (o.f(out, u, p); nothing)
(o::OpaqueNonlinear)(u, p) = o.f(u, p)

# The numbers of arguments of the methods of each wrapper. SciMLBase finds them by reflection
# otherwise, at each construction of a SciML function and so at each `remake` of a problem.
SciMLBase.numargs(::Union{Call{1},Call{2}}) = [4, 3]
SciMLBase.numargs(::Call{3}) = [4]
SciMLBase.numargs(::Union{Static{1},Static{2}}) = [3, 2]
SciMLBase.numargs(::Static{3}) = [3]
SciMLBase.numargs(::Scalar) = [2]
SciMLBase.numargs(::Hessians) = [3]
SciMLBase.numargs(::LagrangeHessian) = [5]
SciMLBase.numargs(::Opaque) = [4, 3]
SciMLBase.numargs(::OpaqueNonlinear) = [3, 2]
SciMLBase.numargs(::Allocating{3}) = [4, 3]
SciMLBase.numargs(::Allocating{2}) = [3, 2]

# SX symbols for an SX function, whose call on them gives its expressions again; MX symbols
# would make a call node that the code generator expands back to SX, at a cost in time and memory.
function symbols(f::CasADiCodegen.Function)
    T = C.is_a(f, "SXFunction") ? CasADiCodegen.SX : CasADiCodegen.MX
    [T.sym(C.name_in(f, i), C.sparsity_in(f, i)) for i in 0:C.n_in(f)-1]
end
"The keywords of the code generation that the SciML constructors pass on."
codegen_options(; chunk_size = nothing, split_above = nothing, optlevel = nothing, cache = false) =
    (; chunk_size, split_above, optlevel, cache)

# CasADi does not accept the function names "jac" and "hess", thus each name gets a prefix.
# The in-place function comes alone, without the out-of-place forms, unless it comes from the cache.
function inplace(name, arguments, values::Vector; cache = false, options...)
    f = CasADiCodegen.Function("sciml_" * name, arguments, values)
    cache === false ? G.build_inplace(f; options...) : G.build_function(f; expression = Val{false}, cache, options...)[2]
end
inplace(name, arguments, value; options...) = inplace(name, arguments, [value]; options...)

"""
    outofplace(name, arguments, value; matrix, options...) -> Static

Return the out-of-place function of `value` that returns static arrays, as a SciML function
of the `arguments`. With `matrix` it returns an `SMatrix` also for one column.
"""
function outofplace(name, arguments, value; matrix::Bool, options...)
    elements = prod(shape(C.sparsity(value)))
    @requires elements <= G.STATIC_LIMIT "$name: inplace = false returns static arrays of at most $(G.STATIC_LIMIT) elements, and this one has $elements; use inplace = true"
    g = G.build_function(CasADiCodegen.Function("sciml_" * name, arguments, [value]); expression = Val{false}, force_SA = true, options...)[1]
    Static{length(arguments),matrix ? Tuple{shape(C.sparsity(value))...} : Nothing,typeof(g)}(g)
end

"The function `name` of `value` for a SciML function: a `Call` in place, or else a `Static`."
sciml_function(name, arguments, value, inplace::Bool; matrix::Bool = false, codegen...) =
    inplace ? concrete(Call{length(arguments),Any}(CasADiCodegenSciMLBaseExt.inplace(name, arguments, value; codegen...))) :
              outofplace(name, arguments, value; matrix, codegen...)
shape(sp) = (C.size1(sp), C.size2(sp))

"""
    sparse_prototype(kind, J) -> SparseMatrixCSC

Return a sparse matrix of stored zeros with the sparsity pattern of the expression `J`, as
the `jac_prototype` of a SciML function. It needs SparseArrays.jl loaded.
"""
function sparse_prototype(kind, J)
    sparse_extension = Base.get_extension(CasADiCodegen, :CasADiCodegenSparseArraysExt)
    @requires sparse_extension !== nothing "$kind: sparse = true needs SparseArrays.jl; load it with `using SparseArrays`"
    sparse_extension.SparseArrays.SparseMatrixCSC(C.sparsity(J))
end

function check_sparse(kind, sparse::Bool, jac::Bool, inplace::Bool, options)
    @requires !sparse || (jac && inplace && !haskey(options, :jac_prototype)) "$kind: sparse = true needs jac = true and inplace = true, and makes jac_prototype itself"
end

"""
    vector_field(kind, f, most; square = true) -> (arguments, value, observed)

Return the arguments of `f` as symbols, its first result on them, which is a column vector
of the size of the first argument (of any size without `square`), and its other results on
them, which are column vectors.
"""
function vector_field(kind, f::CasADiCodegen.Function, most::Int; square::Bool = true)
    @requires 1 <= C.n_in(f) <= most && C.n_out(f) >= 1 "$kind: the function \"$(C.name(f))\" must have 1 to $most arguments and at least one result"
    n = C.size1(C.sparsity_in(f, 0))
    m = square ? n : C.size1(C.sparsity_out(f, 0))
    @requires shape(C.sparsity_in(f, 0)) == (n, 1) && shape(C.sparsity_out(f, 0)) == (m, 1) "$kind: the first argument and the first result of \"$(C.name(f))\" must be column vectors$(square ? " of one size" : "")"
    @requires all(C.size2(C.sparsity_out(f, o)) == 1 for o in 1:C.n_out(f)-1) "$kind: the other results of \"$(C.name(f))\" must be column vectors"
    arguments = symbols(f)
    values = C.n_out(f) == 1 ? [f(arguments...)] : collect(f(arguments...))
    arguments, CasADiCodegen.densify(values[1]), map(CasADiCodegen.densify, values[2:end])
end

# Names of the variables of a solution, from the names of the symbols of CasADi.

const SII = SciMLBase.SymbolicIndexingInterface

"""
    element_names(f, i) -> Vector{Symbol} or nothing

Return the names of the elements of argument `i` (from 0) of `f`: the names of its scalar
symbols, and for an MX symbol of several elements, its name and the index from 0, as CasADi
names the elements of an SX symbol. An MX argument may join column symbols with `vertcat`.
Return `nothing` when the argument has other symbols or is not dense.
"""
function element_names(f::CasADiCodegen.Function, i::Int)
    C.is_dense(C.sparsity_in(f, i)) || return nothing
    if C.is_a(f, "SXFunction")
        e = C.sx_in(f, i)
        n = C.numel(e)
        all(k -> C.is_symbolic(e[k]), 1:n) || return nothing
        return [Symbol(C.name(e[k])) for k in 1:n]
    end
    e = C.mx_in(f, i)
    n = C.numel(e)
    if C.is_symbolic(e)
        return n == 1 ? [Symbol(C.name(e))] : [Symbol(C.name(e), "_", k - 1) for k in 1:n]
    end
    C.is_valid_input(e) || return nothing
    parts = C.primitives(e)
    all(part -> C.size2(part) == 1 && C.is_dense(C.sparsity(part)), parts) && sum(C.numel, parts; init = 0) == n || return nothing
    names = [C.numel(part) == 1 ? [Symbol(C.name(part))] : [Symbol(C.name(part), "_", k - 1) for k in 1:C.numel(part)] for part in parts]
    reduce(vcat, names; init = Symbol[])
end

"""
    symbol_blocks(names) -> Vector{Pair{Symbol,UnitRange{Int}}}

Return `x => i:j` for each run `names[i:j] == [x_0, x_1, ...]` of more than one name, the
names of the elements of one symbol `x` of several elements.
"""
function symbol_blocks(names::AbstractVector)
    blocks = Pair{Symbol,UnitRange{Int}}[]
    i = 1
    while i <= length(names)
        head = match(r"^(.+)_0$", String(names[i]))
        stop = i
        while head !== nothing && stop < length(names) && names[stop+1] == Symbol(head[1], "_", stop + 1 - i)
            stop += 1
        end
        stop > i && push!(blocks, Symbol(head[1]) => i:stop)
        @assert stop >= i
        i = stop + 1
    end
    blocks
end

"The name of an observed variable as a property of `Names`, through which SciMLBase looks up a `Symbol`."
struct ObservedName
    name::Symbol
end

"""
    Names

The symbolic system of a SciML function made from a CasADi function: a
`SciMLBase.SymbolCache` with the names of the states, the parameters and the time, and the
functions `(u, p, t) -> value`, or `(u, p) -> value` without time, of the observed
variables by name. A property of it is an `ObservedName`, as SciMLBase expects of the
system of an observed variable. `blocks` holds the indices of each parameter symbol of
several elements by its name; these names are parameters but not `parameter_symbols`, so
that `remake` writes each element once.
"""
struct Names{S}
    cache::S
    observed::Dict{Symbol,Any}
    blocks::Dict{Symbol,UnitRange{Int}}
end
Base.getproperty(n::Names, name::Symbol) = ObservedName(name)
Base.propertynames(n::Names) = collect(keys(getfield(n, :observed)))

SII.symbolic_container(n::Names) = getfield(n, :cache)
SII.is_time_dependent(n::Names) = SII.is_time_dependent(getfield(n, :cache))
SII.constant_structure(::Names) = true
SII.all_symbols(n::Names) = [SII.all_symbols(getfield(n, :cache)); collect(keys(getfield(n, :observed)))]
SII.is_parameter(n::Names, name::Symbol) = haskey(getfield(n, :blocks), name) || SII.is_parameter(getfield(n, :cache), name)
SII.is_parameter(n::Names, other) = SII.is_parameter(getfield(n, :cache), other)
SII.parameter_index(n::Names, name::Symbol) = get(() -> SII.parameter_index(getfield(n, :cache), name), getfield(n, :blocks), name)
SII.parameter_index(n::Names, other) = SII.parameter_index(getfield(n, :cache), other)
SII.is_observed(n::Names, name::Symbol) = haskey(getfield(n, :observed), name)
SII.is_observed(::Names, ::ObservedName) = true
SII.is_observed(n::Names, names::Union{AbstractArray,Tuple}) = any(name -> SII.is_observed(n, name), names)
SII.is_observed(n::Names, other) = SII.is_observed(getfield(n, :cache), other)
SII.observed(n::Names, name::Symbol) = getfield(n, :observed)[name]
SII.observed(n::Names, name::ObservedName) = getfield(n, :observed)[name.name]
SII.observed(n::Names, other) = SII.observed(getfield(n, :cache), other)
function SII.observed(n::Names, names::Union{AbstractArray,Tuple})
    getters = map(name -> getter(n, name), names)
    (arguments...) -> map(g -> g(arguments...), getters)
end
function SII.get_all_timeseries_indexes(n::Names, name::Union{Symbol,ObservedName})
    observed = name isa ObservedName || SII.is_observed(n, name)
    !observed ? SII.get_all_timeseries_indexes(getfield(n, :cache), name) :
    SII.is_time_dependent(n) ? Set([SII.ContinuousTimeseries()]) : Set()
end

"The function `(u, p, t...) -> value` of the state, the parameter, the time or the observed variable `name` of `n`."
function getter(n::Names, name)
    SII.is_observed(n, name) && !(name isa Union{AbstractArray,Tuple}) && return SII.observed(n, name)
    cache = getfield(n, :cache)
    SII.is_variable(cache, name) && return (index = SII.variable_index(cache, name); (u, p, t...) -> u[index])
    SII.is_parameter(n, name) && return (index = SII.parameter_index(n, name); (u, p, t...) -> p[index])
    SII.is_independent_variable(cache, name) && return (u, p, t) -> t
    SII.observed(cache, name)
end

"""
    observed_functions(f, arguments, values, codegen) -> Dict{Symbol,Any}

Return the observed variables of the other results `values` of `f` on the symbols
`arguments`: result `o` with the name `y` gives `y`, its value, and for more than one
element also `y_0`, `y_1`, ..., each element.
"""
function observed_functions(f::CasADiCodegen.Function, arguments, values, codegen)
    observed = Dict{Symbol,Any}()
    for (o, value) in enumerate(values)
        name = C.name_out(f, o)
        g = G.build_function(CasADiCodegen.Function("sciml_observed_" * name, arguments, [value]); expression = Val{false}, codegen...)[1]
        call = length(arguments) == 1 ? (u, p, t...) -> g(u) : length(arguments) == 2 ? (u, p, t...) -> g(u, given(p)) : (u, p, t) -> g(u, given(p), t)
        n = C.size1(C.sparsity(value))
        observed[Symbol(name)] = n == 1 ? (arguments...) -> call(arguments...)[1] : (arguments...) -> collect(vec(call(arguments...)))
        for k in 1:(n == 1 ? 0 : n)
            observed[Symbol(name, "_", k - 1)] = (arguments...) -> call(arguments...)[k]
        end
    end
    observed
end

"""
    named_system(f, arguments, values, time, codegen; places = (0, 1, 2)) -> NamedTuple

Return the option `sys` of the SciML function of `f`, with the names of the elements of
its arguments (see `element_names`) and the observed variables of its other results
`values`, or no option when an argument has no names or a name occurs twice. Throw an
`ArgumentError` in that case when `f` has further results, since an observed variable
needs the names. `places` are the arguments of `f`, from 0, that hold the state, the
parameters and the time; the arguments after the state are optional.
"""
function named_system(f::CasADiCodegen.Function, arguments, values, time::Bool, codegen; places::NTuple{3,Int} = (0, 1, 2))
    states = element_names(f, places[1])
    parameters = C.n_in(f) > places[2] ? element_names(f, places[2]) : Symbol[]
    clock = !time ? Symbol[] : C.n_in(f) > places[3] ? element_names(f, places[3]) : [:t]
    outputs = [Symbol(C.name_out(f, o)) for o in 1:length(values)]
    elements = [Symbol(C.name_out(f, o), "_", k - 1) for (o, v) in enumerate(values) if C.size1(C.sparsity(v)) > 1 for k in 1:C.size1(C.sparsity(v))]
    named = !any(isnothing, (states, parameters, clock)) && (!time || length(clock) == 1) &&
            allunique([states; parameters; clock; outputs; elements])
    @requires named || isempty(values) "the further results of \"$(C.name(f))\" are observed variables, which need distinct names for the symbols of its arguments and for its results"
    named || return (;)
    # A parameter symbol of several elements is also one parameter, the vector of its elements.
    taken = Set([states; parameters; clock; outputs; elements])
    blocks = Dict{Symbol,UnitRange{Int}}(filter(block -> first(block) ∉ taken, symbol_blocks(parameters)))
    cache = time ? SciMLBase.SymbolCache(states, parameters, only(clock)) : SciMLBase.SymbolCache(states, parameters)
    isempty(values) && isempty(blocks) && return (; sys = cache)
    (; sys = Names(cache, isempty(values) ? Dict{Symbol,Any}() : observed_functions(f, arguments, values, codegen), blocks))
end

"""
    SciMLBase.ODEFunction(f::CasADiCodegen.Function; jac = true, paramjac = true, sparse = false,
                          inplace = true, specialize = SciMLBase.FullSpecialize, chunk_size = nothing,
                          split_above = nothing, optlevel = nothing, cache = false, initialization = nothing,
                          solved_parameters = Symbol[], options...)

Make the SciML function of the ordinary differential equation whose right side is the
CasADi function `f(u)`, `f(u, p)` or `f(u, p, t)`. It holds generated code for the right
side and for its Jacobians by the state (`jac`) and by the parameters (`paramjac`).

The first result of `f` is the right side, a column vector of the size of its first
argument. Each further result, a column vector, is an observed variable. Set `jac` or
`paramjac` to `false` to leave that Jacobian out.

The function has the names of the symbols of `f` as its `sys`, so that `solution[:x]`,
`remake(problem; p = [:k => 2.0])` and `problem.ps[:k]` work:

- The states, the parameters and the time have the names of the scalar symbols of the
  arguments of `f`. The elements of one symbol of several elements, such as
  `MX.sym("x", 2)`, are `x_0` and `x_1`, as CasADi names the elements of `SX.sym("x", 2)`.
  A parameter symbol of several elements is also one parameter by its own name, so that
  `remake(problem; p = [:x => [1.0, 2.0]])` replaces all its elements. Without a third
  argument the time is `t`.

A function of the time without parameters takes an empty parameter argument, such as
`SX(0, 1)`; its problem then needs no parameters.
- An observed variable has the name of its result (see `C.name_out`). A result of more
  than one element `y` also gives `y_0`, `y_1`, ..., one per element.

A function whose arguments are not made of named symbols, or whose names occur twice,
gets no names, and then a further result throws an `ArgumentError`. A `sys` among the
`options` replaces the names and the observed variables. With `sparse = true` the function has a
`jac_prototype`: a `SparseMatrixCSC` with the sparsity pattern of the Jacobian, into whose
stored values `jac` writes. It needs SparseArrays.jl loaded and `jac = true`. Besides the
in-place `jac(J, u, p, t)`, the Jacobians have the out-of-place method `jac(u, p, t)`, which
returns a new matrix: dense, or with `sparse = true` a sparse matrix of the pattern of
`jac_prototype`.

With `inplace = false` the functions are out of place and return static arrays: an
`SVector` for the right side and an `SMatrix` for each Jacobian, also of one row or one
column. Use it with an `SVector` state. Each of these arrays holds at most 1024
elements, so `jac = true` allows at most 32 states; a larger function throws an
`ArgumentError`. `specialize` is the specialization of the SciML
function: `SciMLBase.AutoSpecialize` lets OrdinaryDiffEq.jl wrap the in-place functions of
a `Vector{Float64}` state, so that the solver code compiled for one model serves the next
one; an explicit solver wraps them only with `jac = false`. `chunk_size`, `split_above`,
`optlevel` and `cache` go to [`build_function`](@ref CasADiCodegen.build_function).

`initialization` is a CasADi function with the arguments of `f` whose first result, a
column vector, holds conditions `c(u, p, t) = 0` on the initial values. With it the
function has an `initialization_data`, from which the default `initializealg` of
OrdinaryDiffEq.jl solves the algebraic equations of `f` and the conditions together before
the first step. The algebraic equations are the rows of zeros of the option `mass_matrix`.
The unknowns are the state, whose values in the problem are a guess, and the parameters in
`solved_parameters`, given by name or by place. The initialization problem is a
`NonlinearProblem` when it has as many equations as unknowns, and a
`NonlinearLeastSquaresProblem` otherwise; see
[`initialization_problem`](@ref CasADiCodegen.initialization_problem). A condition on a
derivative calls `f` on the arguments of `initialization`. With
`specialize = SciMLBase.AutoSpecialize`, functions of one `f` with other conditions have one
type, so that the solvers compiled for the first serve the others.

The default nonlinear solver of OrdinaryDiffEq.jl needs OrdinaryDiffEqNonlinearSolve.jl
loaded, and starts with Broyden's method, which does not use the generated Jacobian and
can leave the guess for another solution. A method that uses it is given as
`initializealg = SciMLBase.OverrideInit(; nlsolve = TrustRegion())`, to `solve` or to the
problem.

The other `options` go to `SciMLBase.ODEFunction`. See
[Give a CasADi function to a SciML problem](@ref).

# Examples
```julia
using CasADiCodegen
import SciMLBase

x, v, k, t = SX.sym("x"), SX.sym("v"), SX.sym("k"), SX.sym("t")
rhs = CasadiFunction("rhs", [vertcat(x, v), k, t], [vertcat(v, -k * x), (v^2 + k * x^2) / 2],
                     ["u", "k", "t"], ["du", "energy"])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(rhs), [1.0, 0.0], (0.0, pi), [4.0])
# After `solution = solve(problem, Tsit5())`: solution[:x], solution[:energy]
```
"""
function SciMLBase.ODEFunction(f::CasADiCodegen.Function; jac::Bool = true, paramjac::Bool = true, sparse::Bool = false,
                               inplace::Bool = true, specialize::Type{<:SciMLBase.AbstractSpecialization} = SciMLBase.FullSpecialize,
                               chunk_size = nothing, split_above = nothing, optlevel = nothing, cache = false,
                               initialization::Union{Nothing,CasADiCodegen.Function} = nothing,
                               solved_parameters::AbstractVector = Symbol[], options...)
    check_sparse("ODEFunction", sparse, jac, inplace, options)
    @requires initialization !== nothing || isempty(solved_parameters) "ODEFunction: solved_parameters needs initialization"
    @requires initialization === nothing || !haskey(options, :initialization_data) "ODEFunction: give initialization or initialization_data, not both"
    codegen = codegen_options(; chunk_size, split_above, optlevel, cache)
    _, rhs, keywords = drift("ODEFunction", f, codegen; jac, paramjac, sparse, inplace, specialize, user_sys = haskey(options, :sys))
    start = initialization === nothing ? (;) :
            (; initialization_data = initialization_data(f, initialization, solved_parameters, get(options, :mass_matrix, LinearAlgebra.I), inplace, specialize, codegen))
    SciMLBase.ODEFunction{inplace,specialize}(rhs; start..., keywords..., options...)
end
concrete(c::Call{N}) where {N} = Call{N,typeof(c.g!)}(c.g!)

"""
    drift(kind, f, codegen; jac, paramjac, sparse, inplace, specialize, user_sys) -> (arguments, rhs, keywords)

Return what the SciML function of a differential equation with the right side `f` takes
from it: the symbols of the arguments of `f`, the generated right side, and the keywords
`jac`, `paramjac`, `jac_prototype` (with `sparse`) and, unless `user_sys`, `sys`.
"""
function drift(kind, f::CasADiCodegen.Function, codegen; jac::Bool, paramjac::Bool, sparse::Bool, inplace::Bool, specialize, user_sys::Bool)
    arguments, value, observed = vector_field(kind, f, 3)
    wrap(name, expression; matrix = true) = sciml_function(name, arguments, expression, inplace; matrix, codegen...)
    # In place, a Jacobian also has an out-of-place method, which returns a new matrix of the shape of `result`.
    matrix_function(name, M, result) = inplace ? Allocating{3}(wrap(name, M), result) : wrap(name, M)
    J = jac ? CasADiCodegen.jacobian(value, arguments[1]) : nothing
    prototype = sparse ? (; jac_prototype = sparse_prototype(kind, J)) : (;)
    names = user_sys ? (;) : named_system(f, arguments, observed, true, codegen)
    # OrdinaryDiffEq.jl wraps the right side and the Jacobian of a function that is not
    # FullSpecialize, but not the Jacobian by the parameters, whose type would stay in that of the solver.
    pJ = paramjac && length(arguments) >= 2 ? (P = CasADiCodegen.jacobian(value, arguments[2]); matrix_function("paramjac", P, Val(shape(C.sparsity(P))))) : nothing
    keywords = (; jac = jac ? matrix_function("jac", J, sparse ? prototype.jac_prototype : Val(shape(C.sparsity(J)))) : nothing,
                paramjac = pJ !== nothing && inplace && specialize !== SciMLBase.FullSpecialize ? Opaque(pJ) : pJ,
                prototype..., names...)
    arguments, wrap("rhs", value; matrix = false), keywords
end

"""
    SciMLBase.SDEFunction(f::CasADiCodegen.Function, g::CasADiCodegen.Function; jac = true, paramjac = true,
                          sparse = false, inplace = true, specialize = SciMLBase.FullSpecialize,
                          chunk_size = nothing, split_above = nothing, optlevel = nothing, cache = false, options...)

Make the SciML function of the stochastic differential equation `du = f(u, p, t) dt + g(u, p, t) dW`
from the CasADi functions `f` of the drift and `g` of the noise. It holds generated code for
the drift, the noise and the Jacobians of the drift.

`f` is as for [`ODEFunction`](@ref SciMLBase.ODEFunction(::CasADiCodegen.Function)), and the
keywords have the meaning that they have there; its further results are observed variables.
`g` has the arguments of `f` and one result with a row per state:

- a column vector gives diagonal noise, with one Wiener process per state, or scalar noise,
  with one Wiener process for all, when the problem has a scalar Wiener process as its
  option `noise`, such as `WienerProcess(0.0, 0.0, 0.0)` of DiffEqNoiseProcess.jl, whose
  third value starts the auxiliary process that methods of high order such as `SRIW1` need;
- a matrix gives non-diagonal noise, with one Wiener process per column. The problem needs
  the option `noise_rate_prototype`, a matrix of the size of `g`, which
  [`SDEProblem`](@ref SciMLBase.SDEProblem(::CasADiCodegen.Function, ::CasADiCodegen.Function, ::Any, ::Any))
  makes.

The noise `g(out, u, p, t)` writes a dense matrix or a `SparseMatrixCSC` of any pattern, which
then takes the sparsity pattern of `g`. Its out-of-place method `g(u, p, t)` returns a new vector or matrix. With
`inplace = false` the noise returns an `SVector` or an `SMatrix`.

# Examples
```julia
using CasADiCodegen
import SciMLBase

x, mu, sigma, t = SX.sym("x"), SX.sym("mu"), SX.sym("sigma"), SX.sym("t")
drift = CasadiFunction("drift", [x, vertcat(mu, sigma), t], [mu * x])
noise = CasadiFunction("noise", [x, vertcat(mu, sigma), t], [sigma * x])
f = SciMLBase.SDEFunction(drift, noise)          # geometric Brownian motion
```
"""
function SciMLBase.SDEFunction(f::CasADiCodegen.Function, g::CasADiCodegen.Function; jac::Bool = true, paramjac::Bool = true,
                               sparse::Bool = false, inplace::Bool = true,
                               specialize::Type{<:SciMLBase.AbstractSpecialization} = SciMLBase.FullSpecialize,
                               chunk_size = nothing, split_above = nothing, optlevel = nothing, cache = false, options...)
    check_sparse("SDEFunction", sparse, jac, inplace, options)
    codegen = codegen_options(; chunk_size, split_above, optlevel, cache)
    arguments, rhs, keywords = drift("SDEFunction", f, codegen; jac, paramjac, sparse, inplace, specialize, user_sys = haskey(options, :sys))
    SciMLBase.SDEFunction{inplace,specialize}(rhs, noise_function(f, g, arguments, inplace, codegen); keywords..., options...)
end

"""
    SciMLBase.SDEProblem(f::CasADiCodegen.Function, g::CasADiCodegen.Function, u0, tspan, p = SciMLBase.NullParameters();
                         jac = true, paramjac = true, sparse = false, inplace = true,
                         specialize = SciMLBase.FullSpecialize, chunk_size = nothing, split_above = nothing,
                         optlevel = nothing, cache = false, options...)

Make the SciML problem of the stochastic differential equation with the drift `f` and the
noise `g` from the initial state `u0` over `tspan`. The function of the problem is
[`SDEFunction(f, g)`](@ref SciMLBase.SDEFunction(::CasADiCodegen.Function, ::CasADiCodegen.Function))
with the keywords before `options`, which go to `SciMLBase.SDEProblem`.

For a noise matrix, the problem has a `noise_rate_prototype` of zeros of the size of `g`:
a `Matrix`, with `sparse = true` a `SparseMatrixCSC` of the sparsity pattern of `g`, and with
`inplace = false` an `SMatrix`. An option `noise_rate_prototype` replaces it.

# Examples
```julia
using CasADiCodegen
import SciMLBase
using StochasticDiffEqLowOrder: LambaEM

x = SX.sym("x", 2)
drift = CasadiFunction("drift", [x], [-x])
noise = CasadiFunction("noise", [x], [[0.1 * x[1] 0.0 0.2; 0.0 0.3 * x[2] 0.2]])   # three Wiener processes
problem = SciMLBase.SDEProblem(drift, noise, [1.0, 1.0], (0.0, 1.0))
solution = SciMLBase.solve(problem, LambaEM(); seed = 1)
```
"""
function SciMLBase.SDEProblem(f::CasADiCodegen.Function, g::CasADiCodegen.Function, u0, tspan, p = SciMLBase.NullParameters();
                              jac::Bool = true, paramjac::Bool = true, sparse::Bool = false, inplace::Bool = true,
                              specialize::Type{<:SciMLBase.AbstractSpecialization} = SciMLBase.FullSpecialize,
                              chunk_size = nothing, split_above = nothing, optlevel = nothing, cache = false, options...)
    sde = SciMLBase.SDEFunction(f, g; jac, paramjac, sparse, inplace, specialize, chunk_size, split_above, optlevel, cache)
    SciMLBase.SDEProblem(sde, float.(u0), tspan, p; noise_rate_prototype = noise_prototype(g, inplace, sparse), options...)
end

"""
    noise_shape(f, g) -> (rows, columns)

Return the size of the result of the noise `g` of the drift `f`, after a check that `g`
has the arguments of `f`, one result, and a row per state.
"""
function noise_shape(f::CasADiCodegen.Function, g::CasADiCodegen.Function)
    @requires C.n_in(g) == C.n_in(f) && all(shape(C.sparsity_in(g, i)) == shape(C.sparsity_in(f, i)) for i in 0:C.n_in(f)-1) &&
              C.n_out(g) == 1 "SDEFunction: the noise \"$(C.name(g))\" must have the arguments of \"$(C.name(f))\" and one result"
    n = C.size1(C.sparsity_in(f, 0))
    @requires C.size1(C.sparsity_out(g, 0)) == n "SDEFunction: the noise \"$(C.name(g))\" must have $n rows, one per state; got $(C.size1(C.sparsity_out(g, 0)))"
    shape(C.sparsity_out(g, 0))
end

"""
    noise_prototype(g, inplace, sparse) -> AbstractMatrix or nothing

Return the `noise_rate_prototype` of a problem with the noise `g`: `nothing` for a column,
and for a matrix, zeros of its size: a `Matrix`, a `SparseMatrixCSC` of its pattern with
`sparse`, or an `SMatrix` out of place.
"""
function noise_prototype(g::CasADiCodegen.Function, inplace::Bool, sparse::Bool)
    n, m = shape(C.sparsity_out(g, 0))
    m == 1 && return nothing
    !inplace ? zero(StaticArrays.SMatrix{n,m,Float64}) : sparse ? sparse_prototype("SDEProblem", g(symbols(g)...)) : zeros(n, m)
end

"The generated noise of `g` on the symbols `arguments` of the drift `f`, as the `g` of a SciML function."
function noise_function(f::CasADiCodegen.Function, g::CasADiCodegen.Function, arguments, inplace::Bool, codegen)
    n, m = noise_shape(f, g)
    value = g(arguments...)
    m == 1 && (value = CasADiCodegen.densify(value))
    call = sciml_function("noise", arguments, value, inplace; matrix = m > 1, codegen...)
    inplace || return call
    m == 1 && return Allocating{3}(call, Val((n,)))
    _, _, colptr, rowval = G.sparsity_tuple(C.sparsity_out(g, 0))
    Allocating{3}(Patterned(call, colptr, rowval), Val((n, m)))
end

"""
    Patterned(call, colptr, rowval)

The in-place noise `call(out, u, p, t)` of a matrix whose sparsity pattern has the column
pointers `colptr` and the rows `rowval`, from 1. It first gives a sparse `out` that pattern:
a solver makes the matrices of its caches with `zero` of the prototype, which has no stored
entries.
"""
struct Patterned{F}
    call::F
    colptr::Vector{Int}
    rowval::Vector{Int}
end
function (n::Patterned)(out, u, p, t)
    # `is_stored` is true for a sparse matrix only, whose methods the extension for SparseArrays.jl defines.
    G.Helpers.is_stored(out) && Base.get_extension(CasADiCodegen, :CasADiCodegenSparseArraysExt).take_pattern!(out, n.colptr, n.rowval)
    n.call(out, u, p, t)
end

"""
    SciMLBase.DiscreteFunction(f::CasADiCodegen.Function; inplace = true, specialize = SciMLBase.FullSpecialize,
                               chunk_size = nothing, split_above = nothing, optlevel = nothing, cache = false, options...)

Make the SciML function of the map `u⁺ = f(u, p, t)` of a discrete-time system from the
CasADi function `f(u)`, `f(u, p)` or `f(u, p, t)`. It holds generated code for the map.

The first result of `f` is the next state, a column vector of the size of its first
argument. The solver `FunctionMap` of OrdinaryDiffEqFunctionMap.jl calls the map with the
time of the next state, `t + dt`. Each further result, a column vector, is an observed
variable, which a solution computes from each saved state at the time of that state. The states, the parameters, the time and the observed
variables have names as for [`ODEFunction`](@ref SciMLBase.ODEFunction(::CasADiCodegen.Function)),
and the keywords have the meaning that they have there. With `inplace = false` the map
returns an `SVector`. The other `options` go to `SciMLBase.DiscreteFunction`.

# Examples
```julia
using CasADiCodegen
import SciMLBase

using OrdinaryDiffEqFunctionMap: FunctionMap

x, r = SX.sym("x"), SX.sym("r")
logistic = CasadiFunction("logistic", [x, r], [r * x * (1 - x)])
problem = SciMLBase.DiscreteProblem(SciMLBase.DiscreteFunction(logistic), [0.2], (0.0, 50.0), [3.2])
solution = SciMLBase.solve(problem, FunctionMap())
solution[:x]
```
"""
function SciMLBase.DiscreteFunction(f::CasADiCodegen.Function; inplace::Bool = true,
                                    specialize::Type{<:SciMLBase.AbstractSpecialization} = SciMLBase.FullSpecialize,
                                    chunk_size = nothing, split_above = nothing, optlevel = nothing, cache = false, options...)
    codegen = codegen_options(; chunk_size, split_above, optlevel, cache)
    arguments, value, observed = vector_field("DiscreteFunction", f, 3)
    names = haskey(options, :sys) ? (;) : named_system(f, arguments, observed, true, codegen)
    SciMLBase.DiscreteFunction{inplace,specialize}(sciml_function("map", arguments, value, inplace; codegen...); names..., options...)
end

"""
    SciMLBase.DiscreteProblem(f::CasADiCodegen.Function, u0, tspan, p = SciMLBase.NullParameters();
                              inplace = true, specialize = SciMLBase.FullSpecialize, chunk_size = nothing,
                              split_above = nothing, optlevel = nothing, cache = false, options...)

Make the SciML problem of the map `u⁺ = f(u, p, t)` from the initial state `u0` over `tspan`.
The function of the problem is
[`DiscreteFunction(f)`](@ref SciMLBase.DiscreteFunction(::CasADiCodegen.Function)) with the
keywords before `options`, which go to `SciMLBase.DiscreteProblem`. `FunctionMap` steps with
`dt = 1` unless `solve` receives another `dt`.
"""
function SciMLBase.DiscreteProblem(f::CasADiCodegen.Function, u0, tspan, p = SciMLBase.NullParameters();
                                   inplace::Bool = true, specialize::Type{<:SciMLBase.AbstractSpecialization} = SciMLBase.FullSpecialize,
                                   chunk_size = nothing, split_above = nothing, optlevel = nothing, cache = false, options...)
    map = SciMLBase.DiscreteFunction(f; inplace, specialize, chunk_size, split_above, optlevel, cache)
    SciMLBase.DiscreteProblem(map, float.(u0), tspan, p; options...)
end

"A generated in-place function `g!` of `N` of the arguments (u⁺, u, p, t), as the residual `(out, u⁺, u, p, t)` of an implicit map."
struct ImplicitCall{N,G}
    g!::G
end
(c::ImplicitCall{2})(out, v, u, p, t) = (@inline c.g!((out,), (v, u)); nothing)
(c::ImplicitCall{3})(out, v, u, p, t) = (@inline c.g!((out,), (v, u, given(p))); nothing)
(c::ImplicitCall{4})(out, v, u, p, t) = (@inline c.g!((out,), (v, u, given(p), t)); nothing)
"A generated out-of-place function `g` of `N` of the arguments (u⁺, u, p, t), which returns a static vector."
struct ImplicitStatic{N,G}
    g::G
end
(s::ImplicitStatic{2})(v, u, p, t) = s.g(v, u)
(s::ImplicitStatic{3})(v, u, p, t) = s.g(v, u, given(p))
(s::ImplicitStatic{4})(v, u, p, t) = s.g(v, u, given(p), t)
SciMLBase.numargs(::ImplicitCall) = [5]
SciMLBase.numargs(::ImplicitStatic) = [4]

"""
    SciMLBase.ImplicitDiscreteFunction(f::CasADiCodegen.Function; inplace = true,
                                       specialize = SciMLBase.FullSpecialize, chunk_size = nothing,
                                       split_above = nothing, optlevel = nothing, cache = false, options...)

Make the SciML function of the implicit map `0 = f(u⁺, u, p, t)` of a discrete-time system
from the CasADi function `f(u⁺, u)`, `f(u⁺, u, p)` or `f(u⁺, u, p, t)`. It holds generated
code for the residual.

The first two arguments are the next and the current state, column vectors of one size.
The one result, a column vector, is the residual. The solver `IDSolve` of
ImplicitDiscreteSolve.jl solves `f(u⁺, u, p, t + dt) = 0` for `u⁺` at each step; its
nonlinear solver differentiates the residual with dual numbers, since a
`SciMLBase.ImplicitDiscreteFunction` holds no Jacobian. A residual of more elements than
the state gives the function a `resid_prototype` of zeros of its size, and the step is
then a least-squares problem.

By default `IDSolve` takes the initial state `u0` of the problem as the state before the
first time: it solves `f(u, u0, p, t0) = 0` and starts from `u`. With
`initializealg = SciMLBase.NoInit()` it starts from `u0`, whose algebraic elements must
then satisfy their equations. `IDSolve` adapts its steps to the convergence of the
nonlinear solver unless it receives `adaptive = false` and a `dt`.

The states have the names of the elements of `u⁺`, and the parameters and the time those
of `p` and `t`, as for [`ODEFunction`](@ref SciMLBase.ODEFunction(::CasADiCodegen.Function)),
whose keywords have the same meaning here. With `inplace = false` the residual returns an
`SVector`; `IDSolve` does not take that form, since it writes the state in place. The
other `options` go to `SciMLBase.ImplicitDiscreteFunction`.
"""
function SciMLBase.ImplicitDiscreteFunction(f::CasADiCodegen.Function; inplace::Bool = true,
                                            specialize::Type{<:SciMLBase.AbstractSpecialization} = SciMLBase.FullSpecialize,
                                            chunk_size = nothing, split_above = nothing, optlevel = nothing, cache = false, options...)
    @requires 2 <= C.n_in(f) <= 4 && C.n_out(f) == 1 "ImplicitDiscreteFunction: the function \"$(C.name(f))\" must have 2 to 4 arguments and one result"
    n = C.size1(C.sparsity_in(f, 0))
    @requires shape(C.sparsity_in(f, 0)) == (n, 1) && shape(C.sparsity_in(f, 1)) == (n, 1) && C.size2(C.sparsity_out(f, 0)) == 1 "ImplicitDiscreteFunction: the next and the current state of \"$(C.name(f))\" must be column vectors of the same size, and its result a column vector"
    codegen = codegen_options(; chunk_size, split_above, optlevel, cache)
    arguments = symbols(f)
    value = CasADiCodegen.densify(f(arguments...))
    m = C.size1(C.sparsity(value))
    names = haskey(options, :sys) ? (;) : named_system(f, arguments, [], true, codegen; places = (0, 2, 3))
    prototype = m == n || haskey(options, :resid_prototype) ? (;) :
                (; resid_prototype = inplace ? zeros(m) : zero(StaticArrays.SVector{m,Float64}))
    residual = if inplace
        g! = CasADiCodegenSciMLBaseExt.inplace("residual", arguments, value; codegen...)
        ImplicitCall{length(arguments),typeof(g!)}(g!)
    else
        g = outofplace("residual", arguments, value; matrix = false, codegen...).g
        ImplicitStatic{length(arguments),typeof(g)}(g)
    end
    SciMLBase.ImplicitDiscreteFunction{inplace,specialize}(residual; prototype..., names..., options...)
end

"""
    SciMLBase.ImplicitDiscreteProblem(f::CasADiCodegen.Function, u0, tspan, p = SciMLBase.NullParameters();
                                      inplace = true, specialize = SciMLBase.FullSpecialize, chunk_size = nothing,
                                      split_above = nothing, optlevel = nothing, cache = false, options...)

Make the SciML problem of the implicit map `0 = f(u⁺, u, p, t)` from the initial state `u0`
over `tspan`. The function of the problem is
[`ImplicitDiscreteFunction(f)`](@ref SciMLBase.ImplicitDiscreteFunction(::CasADiCodegen.Function))
with the keywords before `options`, which go to `SciMLBase.ImplicitDiscreteProblem`.
"""
function SciMLBase.ImplicitDiscreteProblem(f::CasADiCodegen.Function, u0, tspan::Tuple, p = SciMLBase.NullParameters();
                                           inplace::Bool = true, specialize::Type{<:SciMLBase.AbstractSpecialization} = SciMLBase.FullSpecialize,
                                           chunk_size = nothing, split_above = nothing, optlevel = nothing, cache = false, options...)
    residual = SciMLBase.ImplicitDiscreteFunction(f; inplace, specialize, chunk_size, split_above, optlevel, cache)
    SciMLBase.ImplicitDiscreteProblem(residual, float.(u0), tspan, p; options...)
end

"""
    algebraic_rows(mass) -> Vector{Int}

Return the rows of zeros of the mass matrix `mass`, which are the algebraic equations.
"""
algebraic_rows(mass::LinearAlgebra.UniformScaling) = Int[]
algebraic_rows(mass::AbstractMatrix) = [i for i in axes(mass, 1) if all(iszero, view(mass, i, :))]

"""
    parameter_places(f, solved) -> Vector{Int}

Return the places, from 1, of the parameters `solved` of `f`, each given by its place or by
its name (see `element_names`).
"""
function parameter_places(f::CasADiCodegen.Function, solved::AbstractVector)
    np = C.n_in(f) >= 2 ? C.nnz_in(f, 1) : 0
    names = np > 0 ? element_names(f, 1) : Symbol[]
    place(i::Integer) = (@requires 1 <= i <= np "ODEFunction: $i is not a parameter of \"$(C.name(f))\", which has $np"; Int(i))
    function place(name::Symbol)
        found = names === nothing ? nothing : findfirst(==(name), names)
        @requires found !== nothing "ODEFunction: $name is not a parameter of \"$(C.name(f))\""
        found
    end
    places = map(place, solved)
    @requires allunique(places) "ODEFunction: a parameter occurs twice in solved_parameters"
    collect(Int, places)
end

"""
    Initialization(n, solved, fixed, time, static)

The maps between a SciML problem and its initialization problem. The problem has a state
of `n` elements, a static array when `static`. The initialization problem has the unknowns
`[u; p[solved]]` and the parameters `[p[fixed]; t]`, or `p[fixed]` without `time`, in
vectors of `Float64`. Called as `update_initializeprob!`, it writes into them the current
state, parameters and time of a problem or an integrator. It writes in place because
`remake` of a SciML problem rebuilds the initialization data with an in-place update;
`StateMap` and `ParameterMap` take the solution back.
"""
struct Initialization
    n::Int
    solved::Vector{Int}
    fixed::Vector{Int}
    time::Bool
    static::Bool
end
parameter_vector(p::AbstractArray) = vec(p)
parameter_vector(p) = Float64[]
function (s::Initialization)(problem, provider)
    u, p = SII.state_values(provider), parameter_vector(SII.parameter_values(provider))
    w, q = problem.u0, problem.p
    T = promote_type(eltype(u), eltype(p))
    @requires T <: Float64 || Float64 <: T "initialization: the state and the parameters have the number type $T, and the initialization problem holds Float64"
    @requires length(u) == s.n && length(w) == s.n + length(s.solved) "initialization: the state has $(length(u)) elements, and the initialization problem was made for $(s.n)"
    copyto!(w, 1, vec(u), 1, s.n)
    copyto!(w, s.n + 1, p[s.solved], 1, length(s.solved))
    copyto!(q, 1, p[s.fixed], 1, length(s.fixed))
    s.time && (q[end] = SII.current_time(provider))
    nothing
end
"The state of the solution of the initialization problem of `Initialization`."
struct StateMap
    s::Initialization
end
(m::StateMap)(solution) = m.s.static ? StaticArrays.SVector{m.s.n}(solution.u[1:m.s.n]) : solution.u[1:m.s.n]
"The parameters of a problem with the solved parameters of the solution of its initialization problem."
struct ParameterMap
    s::Initialization
end
function (m::ParameterMap)(provider, solution)
    p = SII.parameter_values(provider)
    q = Vector{promote_type(eltype(p), eltype(solution.u))}(vec(p))
    q[m.s.solved] .= solution.u[m.s.n+1:end]
    p isa StaticArrays.StaticArray ? StaticArrays.SVector{length(q)}(q) : q
end

"""
    CasADiCodegen.initialization_problem(problem::SciMLBase.ODEProblem) -> SciMLBase.NonlinearProblem or SciMLBase.NonlinearLeastSquaresProblem

Return the initialization problem of `problem`, whose function comes from
[`ODEFunction`](@ref SciMLBase.ODEFunction(::CasADiCodegen.Function)) with `initialization`,
at the initial state, parameters and time of `problem`.

Its unknowns are the state and then the solved parameters, which start from their values in
`problem`; its parameters are the other parameters and then the initial time. Its residual
holds the algebraic equations and then the conditions. The solver of the ODE solves this
problem before its first step; solve it alone to see whether the conditions can be met, and
read its residual at the solution.

# Examples
```julia
initial = CasADiCodegen.initialization_problem(problem)
solution = SciMLBase.solve(initial, SimpleNewtonRaphson())
solution.resid
```
"""
function CasADiCodegen.initialization_problem(problem::SciMLBase.ODEProblem)
    data = problem.f.initialization_data
    @requires data isa SciMLBase.OverrideInitData && data.update_initializeprob! isa Initialization "initialization_problem: the function of the problem has no initialization of CasADiCodegen; give initialization to ODEFunction"
    stored = data.initializeprob
    data.update_initializeprob!(stored, problem)
    # A copy, since the solver of the ODE writes into the stored problem.
    stored isa SciMLBase.NonlinearLeastSquaresProblem ? SciMLBase.NonlinearLeastSquaresProblem(stored.f, copy(stored.u0), copy(stored.p)) :
                                                         SciMLBase.NonlinearProblem(stored.f, copy(stored.u0), copy(stored.p))
end

"""
    initialization_data(f, conditions, solved, mass, inplace, specialize, codegen) -> SciMLBase.OverrideInitData

Return the initialization of the SciML function of `f` with the mass matrix `mass`: the
problem of the algebraic equations of `f` and the conditions `conditions(u, p, t) = 0`,
with the unknowns `u` and the parameters `solved`. It is a `NonlinearProblem` when it has
as many equations as unknowns, and a `NonlinearLeastSquaresProblem` otherwise.
"""
function initialization_data(f::CasADiCodegen.Function, conditions::CasADiCodegen.Function, solved::AbstractVector, mass, inplace::Bool, specialize, codegen)
    k = C.n_in(f)
    @requires C.n_in(conditions) == k && all(shape(C.sparsity_in(conditions, i)) == shape(C.sparsity_in(f, i)) for i in 0:k-1) &&
              C.n_out(conditions) >= 1 && C.size2(C.sparsity_out(conditions, 0)) == 1 "ODEFunction: the conditions \"$(C.name(conditions))\" must have the arguments of \"$(C.name(f))\" and a column vector as their first result"
    places = parameter_places(f, solved)
    n, np = C.nnz_in(f, 0), k >= 2 ? C.nnz_in(f, 1) : 0
    fixed = setdiff(1:np, places)
    T = C.is_a(f, "SXFunction") && C.is_a(conditions, "SXFunction") ? CasADiCodegen.SX : CasADiCodegen.MX
    # Scalar symbols with the names of the state and of the parameters, so that the initialization problem has them.
    states, parameters = element_names(f, 0), np > 0 ? element_names(f, 1) : Symbol[]
    named = states !== nothing && parameters !== nothing
    scalar(name, i) = T.sym(named ? string(name) : "w_$i")
    u = [scalar(named ? states[i] : nothing, i) for i in 1:n]
    p = Vector{T}(undef, np)
    for (j, i) in enumerate(places)
        p[i] = scalar(named ? parameters[i] : nothing, n + j)
    end
    for i in fixed
        p[i] = T.sym(named ? string(parameters[i]) : "q_$i")
    end
    t = T.sym("t")
    arguments = [CasADiCodegen.vertcat(u...), CasADiCodegen.vertcat(p...), t][1:k]
    call(g) = (r = g(arguments...); CasADiCodegen.densify(C.n_out(g) == 1 ? r : r[1]))
    column(parts) = isempty(parts) ? T(0, 1) : CasADiCodegen.vertcat(parts...)
    value, extra = call(f), call(conditions)
    equations = column([[value[i] for i in algebraic_rows(mass)]; [extra[i] for i in 1:C.size1(C.sparsity(extra))]])
    unknowns = column([u; p[places]])
    known = column(k >= 3 ? [p[fixed]; t] : p[fixed])
    residual = CasADiCodegen.Function("initialization", [unknowns, known], [equations])
    # The problem is in place also for an out-of-place ODE, since its update writes into its vectors.
    g = SciMLBase.NonlinearFunction(residual; specialize, codegen...)
    # Without full specialization, the type of the function, and so of the ODEFunction, does
    # not depend on the conditions, so that the solvers compiled for one set of conditions serve the next.
    if specialize !== SciMLBase.FullSpecialize
        g = SciMLBase.NonlinearFunction{true,specialize}(OpaqueNonlinear(g.f); jac = OpaqueNonlinear(g.jac), g.resid_prototype, g.sys)
    end
    m, w = C.size1(C.sparsity(equations)), zeros(n + length(places))
    problem = m == length(w) ? SciMLBase.NonlinearProblem(g, w, zeros(C.numel(known))) :
              SciMLBase.NonlinearLeastSquaresProblem(g, w, zeros(C.numel(known)))
    s = Initialization(n, places, fixed, k >= 3, !inplace)
    SciMLBase.OverrideInitData(problem, s, StateMap(s), isempty(places) ? nothing : ParameterMap(s))
end

"""
    SciMLBase.NonlinearFunction(f::CasADiCodegen.Function; jac = true, sparse = false, inplace = true,
                                specialize = SciMLBase.FullSpecialize, chunk_size = nothing,
                                split_above = nothing, optlevel = nothing, cache = false, options...)

Make the SciML function of the equations `f(u) = 0` or `f(u, p) = 0`, where `f` is
a CasADi function. It holds generated code for the residual and for its Jacobian (`jac`).

The first result of `f` is the residual, a column vector; each further result is an
observed variable. A residual of another size than the first argument of `f` gives the
function a `resid_prototype` of zeros of that size, for a
`SciMLBase.NonlinearLeastSquaresProblem`. Set `jac = false` to leave the Jacobian out. The names, the keywords `sparse`, `inplace`, `specialize` and those of the
code have the meaning that they have for `ODEFunction`, without the time; in place, the
Jacobian also has the out-of-place method `jac(u, p)`. The other `options` go to
`SciMLBase.NonlinearFunction`.

# Examples
```julia
residual = CasADiCodegen.Function("r", [z, q], [equations])
problem = SciMLBase.NonlinearProblem(SciMLBase.NonlinearFunction(residual), z0, q0)
```
"""
function SciMLBase.NonlinearFunction(f::CasADiCodegen.Function; jac::Bool = true, sparse::Bool = false, inplace::Bool = true,
                                     specialize::Type{<:SciMLBase.AbstractSpecialization} = SciMLBase.FullSpecialize,
                                     chunk_size = nothing, split_above = nothing, optlevel = nothing, cache = false, options...)
    check_sparse("NonlinearFunction", sparse, jac, inplace, options)
    codegen = codegen_options(; chunk_size, split_above, optlevel, cache)
    arguments, value, observed = vector_field("NonlinearFunction", f, 2; square = false)
    wrap(name, expression; matrix = true) = sciml_function(name, arguments, expression, inplace; matrix, codegen...)
    J = jac ? CasADiCodegen.jacobian(value, arguments[1]) : nothing
    m = C.size1(C.sparsity(value))
    residual = m == C.size1(arguments[1]) || haskey(options, :resid_prototype) ? (;) :
               (; resid_prototype = inplace ? zeros(m) : zeros(StaticArrays.SVector{m,Float64}))
    prototype = sparse ? (; jac_prototype = sparse_prototype("NonlinearFunction", J), residual...) : residual
    names = haskey(options, :sys) ? (;) : named_system(f, arguments, observed, false, codegen)
    # In place, the Jacobian also has an out-of-place method, which returns a new matrix.
    jacobian = !jac ? nothing : inplace ? Allocating{2}(wrap("jac", J), sparse ? prototype.jac_prototype : Val(shape(C.sparsity(J)))) : wrap("jac", J)
    SciMLBase.NonlinearFunction{inplace,specialize}(wrap("residual", value; matrix = false);
        jac = jacobian, prototype..., names..., options...)
end

"""
    SciMLBase.OptimizationFunction(objective::CasADiCodegen.Function, constraints = nothing; hess = true,
                                   sparse = false, lag_h = sparse && hess, chunk_size = nothing,
                                   split_above = nothing, optlevel = nothing, cache = false, options...)

Make the SciML function of an optimization problem from the CasADi functions `objective(x)`
or `objective(x, p)` and, if the problem has them, `constraints` with the same arguments.

The result holds generated code for the objective, its gradient (`grad`) and its Hessian
(`hess`). With constraints it also holds `cons`, their Jacobian `cons_j`, and, unless
`lag_h`, `cons_h(H, x, p)`, where `H` has one matrix per constraint. `hess = false` leaves out the
derivatives of the second order, for a solver that uses first derivatives only; their code
can take long to build.

Each of `grad`, `hess`, `cons`, `cons_j` and `cons_h` is in place, as `hess(H, x, p)`, and
also has the out-of-place method `hess(x, p)`, which returns a new array. Optimization.jl
calls these functions and differentiates nothing, also for a solver that needs the
second derivatives.

The matrices are dense unless `sparse = true`. Then `hess`, `cons_j` and `cons_h` write the
stored values of a `SparseMatrixCSC` with the sparsity pattern of CasADi, which the result
holds as `hess_prototype`, `cons_jac_prototype` and `cons_hess_prototype`; `options` must
not hold a prototype then. It needs SparseArrays.jl loaded.

`lag_h = true` needs `sparse = true` and `hess = true`. It adds the Hessian of the Lagrange
function `σ f + μ' g`, as `lag_h(h, x, σ, μ, p)`, in place of `cons_h`: `h` receives the
stored values of its upper triangle, whose pattern is `lag_hess_prototype`. Ipopt, through
OptimizationIpopt.jl, then makes one call per iteration in place of one Hessian per
constraint.

`chunk_size`, `split_above`, `optlevel` and `cache` go to
[`build_function`](@ref CasADiCodegen.build_function). The other `options` go to
`SciMLBase.OptimizationFunction`. `x` must be a column vector, and the objective one number.
The constraints must be a column vector.

# Examples
```julia
using SparseArrays

x = SX.sym("x", 2)
objective = CasADiCodegen.Function("cost", [x], [(x[1] - 1)^2 + x[2]^2])
constraints = CasADiCodegen.Function("cons", [x], [x[1] + x[2]])
f = SciMLBase.OptimizationFunction(objective, constraints; sparse = true)
problem = SciMLBase.OptimizationProblem(f, [0.0, 0.0]; lcons = [1.0], ucons = [1.0])
```
"""
function SciMLBase.OptimizationFunction(objective::CasADiCodegen.Function, constraints::Union{Nothing,CasADiCodegen.Function} = nothing;
                                        hess::Bool = true, sparse::Bool = false, lag_h::Bool = sparse && hess,
                                        chunk_size = nothing, split_above = nothing, optlevel = nothing, cache = false, options...)
    @requires !lag_h || (sparse && hess) "OptimizationFunction: lag_h = true needs sparse = true and hess = true"
    @requires !sparse || !any(haskey(options, k) for k in (:hess_prototype, :cons_jac_prototype, :cons_hess_prototype, :lag_hess_prototype)) "OptimizationFunction: sparse = true makes the prototypes itself"
    codegen = codegen_options(; chunk_size, split_above, optlevel, cache)
    n = C.n_in(objective)
    @requires 1 <= n <= 2 && C.n_out(objective) == 1 && shape(C.sparsity_out(objective, 0)) == (1, 1) "OptimizationFunction: the objective must have 1 or 2 arguments and one number as its result"
    @requires C.size2(C.sparsity_in(objective, 0)) == 1 "OptimizationFunction: the unknown must be a column vector"
    arguments = symbols(objective)
    x = arguments[1]
    cost = CasADiCodegen.densify(objective(arguments...))
    # Each function is in place, with an out-of-place method that returns a new array: dense, or a copy of the sparse prototype.
    result(M) = sparse ? sparse_prototype("OptimizationFunction", M) : Val(shape(C.sparsity(M)))
    call(name, expression, new) = Allocating{2}(sciml_function(name, arguments, expression, true; codegen...), new)
    # A prototype for each sparse matrix; the dense form has none.
    prototype(name, M) = sparse ? (; name => sparse_prototype("OptimizationFunction", M)) : (;)
    H = hess ? CasADiCodegen.hessian(cost, x) : nothing
    given = (; grad = call("grad", CasADiCodegen.gradient(cost, x), Val((C.size1(x),))), hess = hess ? call("hess", H, result(H)) : nothing,
             (hess ? prototype(:hess_prototype, H) : (;))...)
    value = nothing
    if constraints !== nothing
        @requires C.n_in(constraints) == n && C.n_out(constraints) == 1 && C.size2(C.sparsity_out(constraints, 0)) == 1 &&
                  all(shape(C.sparsity_in(constraints, i)) == shape(C.sparsity_in(objective, i)) for i in 0:n-1) "OptimizationFunction: the constraints must be a column vector and have the arguments of the objective"
        value = CasADiCodegen.densify(constraints(arguments...))
        J = CasADiCodegen.jacobian(value, x)
        given = (; given..., cons = call("cons", value, Val((C.size1(value),))), cons_j = call("cons_j", J, result(J)), prototype(:cons_jac_prototype, J)...)
        if hess && !lag_h
            # The dense form has m·n² entries: with `sparse` only the stored values are written.
            each = [CasADiCodegen.hessian(value[j], x) for j in 1:size(value, 1)]
            cons_h = inplace("cons_h", arguments, sparse ? each : map(CasADiCodegen.densify, each); codegen...)
            given = (; given..., cons_h = Allocating{2}(Hessians{n,typeof(cons_h)}(cons_h), sparse ? map(result, each) : Each{length(each)}(result(each[1]))),
                     (sparse ? (; cons_hess_prototype = [sparse_prototype("OptimizationFunction", Hj) for Hj in each]) : (;))...)
        end
    end
    if lag_h
        T = typeof(x)
        σ, μ = T.sym("sigma"), T.sym("mu", value === nothing ? 0 : size(value, 1))
        upper = C.triu(CasADiCodegen.hessian(value === nothing ? σ * cost : σ * cost + C.dot(μ, value), x))
        lagrange = G.build_function(CasADiCodegen.Function("sciml_lag_h", [arguments..., σ, μ], [upper]);
                                    expression = Val{false}, nonzeros = true, codegen...)
        given = (; given..., lag_h = LagrangeHessian{n,typeof(lagrange)}(lagrange), prototype(:lag_hess_prototype, upper)...)
    end
    scalar = Scalar{n,Any}(inplace("cost", arguments, cost; codegen...))
    SciMLBase.OptimizationFunction{true}(Scalar{n,typeof(scalar.g!)}(scalar.g!), SciMLBase.NoAD(); given..., options...)
end

# The problems of the solver objects of CasADi as SciML problems, which the
# user solves with a solver of the SciML ecosystem.

function check_count(label, what, values, n)
    @requires length(values) == n "$label: $what must have $(G.numbers(n)); got $(length(values))"
end

"""
    SciMLBase.ODEProblem(I::CasADiCodegen.Function, x0, p = Float64[]; z0 = Float64[], u = Float64[],
                         specialize = SciMLBase.FullSpecialize, options...)

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
- `specialize` is the specialization of the function of the problem. With
  `SciMLBase.AutoSpecialize`, the problems of several integrators with the same kind of
  mass matrix and callback share one compiled solver.

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
                              z0::AbstractVector{<:Real} = Float64[], u::AbstractVector{<:Real} = Float64[],
                              specialize::Type{<:SciMLBase.AbstractSpecialization} = SciMLBase.FullSpecialize, options...)
    model = G.integrator_model(I)
    label = "integrator \"$(model.name)\""
    intervals = length(model.times)
    check_count(label, "the initial state", x0, model.nx)
    check_count(label, "the algebraic states", z0, model.nz)
    check_count(label, "the parameters", p, model.np)
    check_count(label, "the controls", u, model.nu * intervals)
    mass = model.nz == 0 ? (;) : (; mass_matrix = LinearAlgebra.Diagonal([ones(model.nx); zeros(model.nz); ones(model.nq)]))
    f = SciMLBase.ODEFunction(G.right_side(model); specialize, mass...)
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
    SciMLBase.ODEProblem(dae::CasADiCodegen.C.DaeBuilder, tspan = nothing; jac = true, paramjac = true,
                         sparse = false, specialize = SciMLBase.FullSpecialize, chunk_size = nothing,
                         split_above = nothing, optlevel = nothing, cache = false, options...)

Make the SciML problem of the model `dae` of CasADi, with generated code for its right side
and for the Jacobians of it, and with the start values of its variables.

- The state has the differential states `x`, then the algebraic states `z` of `dae`. The
  initial state is their start values. The algebraic start values must satisfy the
  algebraic equations, as for `ODEProblem(I; z0)` of an integrator.
- The parameters are `p`, then the inputs `u`, from their start values. The inputs are
  constant over the solve.
- The time span is `tspan`, or the start and stop times of `dae` when `tspan` is
  `nothing`.
- With algebraic states the problem has a mass matrix, so it needs a solver for such
  problems, for example `Rodas5P()` of OrdinaryDiffEqRosenbrock.jl.
- When each variable is a scalar, the problem has the names of the variables, so that
  `solution[:x]` gives the values of the variable `x`.

The dependent variables `w` and the constants `c` of `dae` are substituted into the
equations; CasADi takes the value of a constant from `set(dae, name, value)`, not from its
start value. A model with quadratures `q` or with `when` equations is not supported. The keywords from `jac` to `cache` go to
[`ODEFunction`](@ref SciMLBase.ODEFunction(::CasADiCodegen.Function)); the other `options`
go to the problem.

# Examples
```julia
dae = CasADiCodegen.C.DaeBuilder("decay")
x, k = CasADiCodegen.C.add(dae, "x", "local", "continuous"), CasADiCodegen.C.add(dae, "k", "parameter", "tunable")
CasADiCodegen.C.eq(dae, CasADiCodegen.C.der(dae, x), -k * x)
CasADiCodegen.C.set_start(dae, "x", 1.0)
CasADiCodegen.C.set_start(dae, "k", 0.5)
solution = SciMLBase.solve(SciMLBase.ODEProblem(dae, (0.0, 2.0)), Vern9())
solution[:x]
```
"""
function SciMLBase.ODEProblem(dae::CasADiCodegen.C.DaeBuilder, tspan::Union{Nothing,Tuple{Real,Real}} = nothing;
                              jac::Bool = true, paramjac::Bool = true, sparse::Bool = false,
                              specialize::Type{<:SciMLBase.AbstractSpecialization} = SciMLBase.FullSpecialize,
                              chunk_size = nothing, split_above = nothing, optlevel = nothing, cache = false, options...)
    name = C.name(dae)
    label = "DaeBuilder \"$name\""
    @requires isempty(C.q(dae)) "$label: a model with quadratures is not supported"
    @requires C.nzero(dae) == 0 "$label: a model with when equations is not supported"
    span = tspan === nothing ? (C.start_time(dae), C.stop_time(dae)) : tspan
    @requires all(isfinite, span) "$label: give the time span, or set the start and stop times of the DaeBuilder"
    g = C.create(dae, "$(name)_dae", ["x", "z", "p", "u", "t"], ["ode", "alg"])
    nx, nz, np, nu = (C.nnz_in(g, i) for i in 0:3)
    state, parameters, t = CasADiCodegen.MX.sym("u", nx + nz), CasADiCodegen.MX.sym("p", np + nu), CasADiCodegen.MX.sym("t")
    # A model that does not use the time has a time of no element.
    time = C.nnz_in(g, 4) == 0 ? CasADiCodegen.MX(C.sparsity_in(g, 4)) : t
    ode, alg = g(G.part(state, 0, nx), G.part(state, nx, nz), G.part(parameters, 0, np), G.part(parameters, np, nu), time)
    rhs = CasADiCodegen.Function(name, [state, parameters, t], [CasADiCodegen.vertcat(ode, alg)])
    differential, algebraic, constant, inputs = C.x(dae), C.z(dae), C.p(dae), C.u(dae)
    u0, p0 = C.start(dae, [differential; algebraic]), C.start(dae, [constant; inputs])
    @ensures length(u0) == nx + nz && length(p0) == np + nu "$label: the start values do not have the sizes of the model"
    # CasADi replaces a constant by its value, which is NaN until `set(dae, name, value)` gives it.
    @requires all(isfinite, C.nonzeros(rhs(C.DM(u0), C.DM(p0), C.DM(float(span[1]))))) "$label: the right side is not finite at the start values; give each constant its value with set(dae, name, value)"
    scalar = length(u0) == length(differential) + length(algebraic) && length(p0) == length(constant) + length(inputs)
    names = scalar ? (; sys = SciMLBase.SymbolCache(Symbol.([differential; algebraic]), Symbol.([constant; inputs]), :t)) : (;)
    mass = nz == 0 ? (;) : (; mass_matrix = LinearAlgebra.Diagonal([ones(nx); zeros(nz)]))
    f = SciMLBase.ODEFunction(rhs; jac, paramjac, sparse, specialize, chunk_size, split_above, optlevel, cache, mass..., names...)
    SciMLBase.ODEProblem(f, u0, float.(span), p0; options...)
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
    SciMLBase.OptimizationProblem(S::CasADiCodegen.Function, x0, p = Float64[]; lbx, ubx, lbg, ubg,
                                  hess = true, sparse = false, lag_h = sparse && hess, chunk_size = nothing,
                                  split_above = nothing, optlevel = nothing, cache = false, options...)

Make the SciML problem of the solver `S` that `nlpsol` or `conic` of CasADi made, with
generated code for the objective, the constraints and their derivatives. The keywords from
`hess` to `cache` have the meaning that they have for
[`OptimizationFunction`](@ref SciMLBase.OptimizationFunction(::CasADiCodegen.Function, ::Union{Nothing,CasADiCodegen.Function})).

For a solver of `conic`, the parameters `p` are the nonzeros of `h`, then `g`, then the
nonzeros of `a`, and the constraints `g` of the bounds are the rows of `a`.

The bounds have the names and the default values of CasADi: no bound on the unknowns, and
none on the constraints. The other `options` go to the problem. The unknowns must be a
column vector, and the vectors must have the sizes of `S`.

# Examples
```julia
problem = SciMLBase.OptimizationProblem(S, [0.0, 0.0]; lbg = [1.0], ubg = [1.0], sparse = true)
solution = SciMLBase.solve(problem, OptimizationIpopt.IpoptOptimizer())
```
"""
function SciMLBase.OptimizationProblem(S::CasADiCodegen.Function, x0::AbstractVector{<:Real}, p::AbstractVector{<:Real} = Float64[];
                                       lbx = nothing, ubx = nothing, lbg = nothing, ubg = nothing, hess::Bool = true,
                                       sparse::Bool = false, lag_h::Bool = sparse && hess, chunk_size = nothing,
                                       split_above = nothing, optlevel = nothing, cache = false, options...)
    model = G.optimization_model(S)
    label = "solver \"$(model.name)\""
    check_count(label, "the unknowns", x0, model.nx)
    check_count(label, "the parameters", p, model.np)
    for (what, bound, n) in (("the lower bounds of the unknowns", lbx, model.nx), ("the upper bounds of the unknowns", ubx, model.nx),
                             ("the lower bounds of the constraints", lbg, model.ng), ("the upper bounds of the constraints", ubg, model.ng))
        bound === nothing || check_count(label, what, bound, n)
    end
    constrained = model.ng > 0
    f = SciMLBase.OptimizationFunction(model.objective, constrained ? model.constraints : nothing;
                                       hess, sparse, lag_h, chunk_size, split_above, optlevel, cache)
    limit(bound, value) = !constrained ? nothing : bound === nothing ? fill(value, model.ng) : float(collect(bound))
    # A SciML problem takes both bounds of the unknowns or none.
    bounded = lbx !== nothing || ubx !== nothing
    side(bound, value) = !bounded ? nothing : bound === nothing ? fill(value, model.nx) : float(collect(bound))
    SciMLBase.OptimizationProblem(f, float(collect(x0)), float(collect(p)); lb = side(lbx, -Inf), ub = side(ubx, Inf),
                                  lcons = limit(lbg, -Inf), ucons = limit(ubg, Inf), options...)
end

"""
    SciMLBase.OptimizationProblem(opti::CasADiCodegen.C.Opti; hess = true, sparse = false, lag_h = sparse && hess,
                                  chunk_size = nothing, split_above = nothing, optlevel = nothing, cache = false, options...)

Make the SciML problem of the optimization problem that the `Opti` object of CasADi holds,
with generated code for the objective, the constraints and their derivatives.

- The unknowns are the variables of `opti`, in the order of their declaration, as
  `C.x(opti)` joins them. Their initial values come from `set_initial`, or are zero.
- The parameters are the parameters of `opti` (`C.p(opti)`), with the values from
  `set_value`. Each parameter must have a value.
- The constraints are those of `subject_to`, with the bounds that `opti` gives them.
  `opti` keeps a bound on a variable as a constraint, so the problem has no `lb` and
  `ub`. A bound that depends on a parameter has the value of this call; a later change
  of the parameters in the problem does not change it.

The keywords from `hess` to `cache` have the meaning that they have for
[`OptimizationFunction`](@ref SciMLBase.OptimizationFunction(::CasADiCodegen.Function, ::Union{Nothing,CasADiCodegen.Function})).
The other `options` go to the problem. `C.value(opti, expression, [C.x(opti) == CasADiCodegen.DM(solution.u)])`
evaluates an expression of `opti` at the solution.

# Examples
```julia
C = CasADiCodegen.C
opti = C.Opti()
x, a = C.variable(opti, 2), C.parameter(opti)
C.minimize(opti, (x[1] - a)^2 + x[2]^2)
C.subject_to(opti, x[1] + x[2] >= 1)
C.set_value(opti, a, C.DM(0.25))
problem = SciMLBase.OptimizationProblem(opti; sparse = true)
solution = SciMLBase.solve(problem, OptimizationIpopt.IpoptOptimizer())
```
"""
function SciMLBase.OptimizationProblem(opti::C.Opti; hess::Bool = true, sparse::Bool = false, lag_h::Bool = sparse && hess,
                                       chunk_size = nothing, split_above = nothing, optlevel = nothing, cache = false, options...)
    x, p = C.x(opti), C.p(opti)
    given = C.value_parameters(opti)
    numbers(expression, values) = Vector{Float64}(C.nonzeros(CasADiCodegen.DM(C.value(opti, expression, values))))
    # CasADi refuses the value of a parameter that has none.
    p0 = C.np(opti) == 0 ? Float64[] : try
        numbers(p, given)
    catch err
        err isa C.SwigError || rethrow()
        Float64[NaN]
    end
    @requires all(isfinite, p0) "Opti: give each parameter its value with set_value"
    x0 = numbers(x, C.initial(opti))
    arguments = C.np(opti) == 0 ? [x] : [x, p]
    objective = CasADiCodegen.Function("opti_objective", arguments, [C.f(opti)])
    constrained = C.ng(opti) > 0
    constraints = constrained ? CasADiCodegen.Function("opti_constraints", arguments, [C.g(opti)]) : nothing
    f = SciMLBase.OptimizationFunction(objective, constraints; hess, sparse, lag_h, chunk_size, split_above, optlevel, cache)
    lcons, ucons = constrained ? (numbers(C.lbg(opti), given), numbers(C.ubg(opti), given)) : (nothing, nothing)
    @ensures length(x0) == C.nx(opti) && (lcons === nothing || length(lcons) == length(ucons) == C.ng(opti)) "Opti: the values do not have the sizes of the problem"
    SciMLBase.OptimizationProblem(f, x0, p0; lcons, ucons, options...)
end

"""
    CasADiCodegen.traced_function(name, problem; symbolic = SX) -> CasADiCodegen.Function

Trace the function of the SciML `problem`, an `ODEProblem` or a `NonlinearProblem`, into a
CasADi function, with the shapes of its initial state and of its parameters (see
[`traced_function`](@ref CasADiCodegen.traced_function(::AbstractString, ::Any, ::AbstractArray{<:Real}))).
The function of an `ODEProblem` takes the time; that of a `NonlinearProblem` does not.
Parameters of the type `SciMLBase.NullParameters` count as none.

# Examples
```julia
original = SciMLBase.ODEProblem(brusselator!, u0, (0.0, 11.5), p)
rhs = CasADiCodegen.traced_function("brusselator", original)
f = SciMLBase.ODEFunction(rhs; sparse = true)
problem = SciMLBase.ODEProblem(f, vec(u0), (0.0, 11.5), p)
```
"""
function CasADiCodegen.traced_function(name::AbstractString, problem::Union{SciMLBase.ODEProblem,SciMLBase.NonlinearProblem};
                                       symbolic::Type = CasADiCodegen.SX)
    p = problem.p isa SciMLBase.NullParameters ? nothing : problem.p
    CasADiCodegen.traced_function(name, SciMLBase.unwrapped_f(problem.f.f), problem.u0, p;
                                  inplace = SciMLBase.isinplace(problem), time = problem isa SciMLBase.ODEProblem, symbolic)
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
    # A solver of the first order gets no Hessians (see `needs_hessians`).
    hessians = F.hess !== nothing
    given = s.sizes[3] == 0 ? (;) : (; cons = wrap(Call, F.cons), cons_j = wrap(Call, F.cons_j),
                                      (hessians ? (; cons_h = wrap(Hessians, F.cons_h)) : (;))...)
    SciMLBase.OptimizationFunction{true}(wrap(Scalar, F.cost), SciMLBase.NoAD(); grad = wrap(Call, F.grad),
                                         hess = hessians ? wrap(Call, F.hess) : nothing, given...)
end

G.needs_hessians(alg) = SciMLBase.requireshessian(alg) || SciMLBase.requiresconshess(alg) || SciMLBase.requireslagh(alg)

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
