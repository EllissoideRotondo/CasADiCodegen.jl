# Ordinary differential equations in CasADi graphs. A solver that is a plain
# Julia function integrates the emitted right side; CasADi builds the
# sensitivity equations, so that the Jacobian of an integrator is a call to a
# second integrator of the same kind, which is made only when CasADi asks.

"""
    RightSide

The emitted right side of an ordinary differential equation, as the
function `(du, u, q, t)` that SciML solvers call. It writes the derivative of
the state `u` at the time `t`, with the parameters `q`, to `du`.

- `mass`: `nothing`, or the diagonal of the mass matrix of a model with
  algebraic states. An element is zero for such a state; the element of `du`
  is then the residual of an algebraic equation.
- `slope`: the emitted Jacobian of those residuals by those states (see
  `consistent!`), or `nothing`.
"""
struct RightSide{F,M,J}
    f::F
    mass::M
    slope::J
end
RightSide(f) = RightSide(f, nothing, nothing)
RightSide(f, mass::Vector{Float64}, slope) = isempty(mass) ? RightSide(f) : RightSide{typeof(f),Vector{Float64},typeof(slope)}(f, mass, slope)
(r::RightSide)(du, u, q, t) = (r.f((du,), (u, q, (t,))); nothing)

"""
    consistent!(u, r::RightSide, q, t) -> u

Make the algebraic states of `u` satisfy the algebraic equations of the model
at the time `t`. Newton's method starts from their values in `u`; the other
states do not change. A model with no mass matrix returns `u` unchanged.

With a number type that carries derivatives, such as a dual number, one more
step after convergence makes the derivatives those of the implicit function.
Throws an error when the method does not converge in 50 steps.
"""
consistent!(u, ::RightSide{F,Nothing}, q, t) where {F} = u
function consistent!(u, r::RightSide, q, t)
    rows = findall(iszero, r.mass)
    T = promote_type(eltype(u), eltype(q))
    du, slope = Vector{T}(undef, length(u)), Matrix{T}(undef, length(rows), length(rows))
    converged = false
    for _ in 1:50
        # Loop invariant: `converged` is true when the last step did not change the values of the algebraic states.
        r.f((du,), (u, q, (t,)))
        r.slope((vec(slope),), (u, q, (t,)))
        step = slope \ du[rows]
        u[rows] .-= step
        converged && return u
        converged = all(k -> abs(step[k]) <= 1e-12 * (1 + abs(u[rows[k]])), eachindex(rows))
    end
    error("the algebraic equations of the model have no consistent initial values near the guess")
end

"""
    Integration

The Julia function behind an integrator: `(x0, q) -> xf`, or `x0 -> xf` when
the model has no parameters. The result can be memory that the solver owns, so
the caller copies it before the next call.

- `tail`: constant values that follow `x0` in the initial state of the solver.
  An integration of a state with its sensitivities starts from the state, the
  identity matrix and zeros.
- `width`: the number of output times. With more than one, the result has the
  values of each state at all the times, one state after the other (see
  `ode_grid`).
- `sensitivity`: that integration, once CasADi has asked for the Jacobian. It
  takes the same `x0` and `q`.
"""
struct Integration{S}
    name::String
    solver::S
    tail::Vector{Float64}
    width::Int
    sensitivity::Base.RefValue{Any}
end
Integration(name, solver, tail, width = 1) = Integration(name, solver, tail, width, Ref{Any}(nothing))
function (i::Integration)(x0, q)
    isempty(i.tail) && return i.solver(x0, q)
    T, n = promote_type(eltype(x0), eltype(q)), length(x0)
    initial = get!(() -> Vector{T}(undef, n + length(i.tail)), task_local_storage(), (i, T))::Vector{T}
    copyto!(initial, x0)
    copyto!(initial, n + 1, i.tail, 1, length(i.tail))
    i.solver(initial, q)
end
(i::Integration)(x0) = i(x0, NO_PARAMETERS)

# One integration for the state and its Jacobian. A generated function that
# needs both calls the integration of the state and, later, that of the
# sensitivities with the same arguments, and the second result starts with
# the first. So the first call runs the second integration ahead and keeps its
# result (`Ahead`), and the second call takes that result (`Remembered`).

"The largest number of results that a task keeps for one integration."
const MEMO_LIMIT = 1024

"""
    Memo{T}

Results of the integrations that ran ahead in one task and that no call has
taken yet.

- Records `head:count` are live. Each has the initial state, the parameters
  and the final state. The vectors of the other records are free.
- `latest` holds the arguments and the result of the last integration.
- `epoch` is the call of a generated function in which that integration ran
  (see `CALL_EPOCH`). A call with the same arguments in the same epoch takes
  that result with no new run.
"""
mutable struct Memo{T}
    initial::Vector{Vector{T}}
    parameters::Vector{Vector{T}}
    results::Vector{Vector{T}}
    head::Int
    count::Int
    latest::NTuple{3,Vector{T}}
    epoch::Int
end
Memo{T}() where {T} = Memo{T}(Vector{T}[], Vector{T}[], Vector{T}[], 1, 0, (T[], T[], T[]), 0)

"""
Count of the calls of generated functions that pair integrations; `new_call!`
increments it. A result is reused only inside the call that made it, because a
Julia function in the right side can read data that the user changes
between two calls. The count only grows, so a call in another task can make an
integration run again, but never gives it an old result.
"""
const CALL_EPOCH = Threads.Atomic{Int}(0)
new_call!() = (Threads.atomic_add!(CALL_EPOCH, 1); nothing)

first_statements(sites) = any(site -> site isa Union{Ahead,Remembered}, values(sites)) ? Any[:($new_call!())] : Any[]

"""
    latest!(memo, run, x0, q) -> result

Return the result of `run(x0, q)`. The memo supplies it when the last
integration had the same arguments in this call of the generated function. A
graph that needs the sensitivities at one point more than once, as for an
event, then integrates them once.

Only floating-point numbers are compared, because two dual numbers with equal
values can carry different derivatives. The result is memory of the memo; the
next call overwrites it.
"""
function latest!(memo::Memo{T}, run, x0, q) where {T}
    initial, parameters, result = memo.latest
    T <: AbstractFloat && memo.epoch == CALL_EPOCH[] && initial == x0 && parameters == q && return result
    fresh = run(x0, q)
    memo.epoch = 0                          # no result while the vectors change
    copyto!(resize!(initial, length(x0)), x0)
    copyto!(resize!(parameters, length(q)), q)
    copyto!(resize!(result, length(fresh)), fresh)
    memo.epoch = CALL_EPOCH[]
    result
end

memo(sensitivity, ::Type{T}) where {T} = get!(() -> Memo{T}(), task_local_storage(), (sensitivity, Memo{T}))::Memo{T}

"""
    claim!(memo, n, m, len) -> k

Return the place of the next record, with vectors for an initial state of `n`
elements, `m` parameters and a result of `len` elements. The record counts
only when the caller sets `memo.count = k`.

The memo restarts from its first place when every record was taken, and also
when it is full. A lost record costs only a new integration.
"""
function claim!(memo::Memo{T}, n::Integer, m::Integer, len::Integer) where {T}
    if memo.head > memo.count || memo.count == MEMO_LIMIT
        memo.head, memo.count = 1, 0
    end
    k = memo.count + 1
    if k > length(memo.results)
        push!(memo.initial, Vector{T}(undef, n))
        push!(memo.parameters, Vector{T}(undef, m))
        push!(memo.results, Vector{T}(undef, len))
    end
    @ensures 1 <= k <= MEMO_LIMIT "a record must have a place in the memo"
    k
end

"""
    Ahead

What a generated function calls in place of the integration `state` when it
also calls `sensitivity`. `(x0, q) -> xf` runs `run`, which integrates the
state with its sensitivities, and keeps the result in the memo of
`sensitivity`. It returns the part that is the result of `state`.
"""
struct Ahead{I,G}
    state::I
    sensitivity::Integration
    run::G
end
function (a::Ahead)(x0, q)
    records = memo(a.sensitivity, promote_type(eltype(x0), eltype(q)))
    result = latest!(records, a.run, x0, q) # a run that fails leaves no record
    k = claim!(records, length(x0), length(q), length(result))
    copyto!(records.initial[k], x0)
    copyto!(records.parameters[k], q)
    copyto!(records.results[k], result)
    records.count = k
    # The values of the state at each output time come first, whatever the number of times (see `ode_grid`).
    view(records.results[k], 1:(length(x0)+length(a.state.tail))*a.state.width)
end
(a::Ahead)(x0) = a(x0, NO_PARAMETERS)

"""
    Remembered

What a generated function calls in place of the integration `sensitivity`
when it also calls the integration of the state: the result that `Ahead` kept
for the same initial state and parameters, or else the result of `run`.
"""
struct Remembered{G}
    sensitivity::Integration
    run::G
end
function (r::Remembered)(x0, q)
    records = memo(r.sensitivity, promote_type(eltype(x0), eltype(q)))
    # The calls take the records in the order in which they were kept, so the first comparison succeeds.
    for k in records.head:records.count
        if records.initial[k] == x0 && records.parameters[k] == q
            head = records.head
            for vectors in (records.initial, records.parameters, records.results)
                vectors[k], vectors[head] = vectors[head], vectors[k]
            end
            records.head = head + 1
            return records.results[head]
        end
    end
    latest!(records, r.run, x0, q)
end
(r::Remembered)(x0) = r(x0, NO_PARAMETERS)

"""
    GridAdjoint

The Julia function `(x0, q, Y, Λ) -> [l0; s0]`, or `(x0, Y, Λ) -> l0` when the model has no
parameters, that gives the reverse derivative of an integration over a grid of `width`
output times (see `ode_grid`). `Y` holds the `n` states at each time and `Λ` their seeds,
one state after the other.

`solver` integrates the adjoint equations of the state `x`, its multipliers `l` and the sums
`s` of the `columns` parameters, from the last output time back to the initial time, in
one run. At each output time on the way, the state starts again from its value in `Y` and
the multipliers take the seed of that time. The result has the multipliers of the first
`given` states and the sums at the initial time.
"""
struct GridAdjoint{S}
    solver::S
    n::Int
    given::Int
    columns::Int
    width::Int
end

"""
    Seeds(Y, Λ, n, width)

What the solver of a `GridAdjoint` calls at the output time `k` of its run, `(u, k)`: the
state of `u` takes its value at that time and the multipliers add their seed. The last
time of the run is the initial time, which has neither.
"""
struct Seeds{A,B}
    Y::A
    Λ::B
    n::Int
    width::Int
end
function (seeds::Seeds)(u, k)
    k < seeds.width || return u
    j, n, N = seeds.width - k, seeds.n, seeds.width      # the run visits the output times backward
    for i in 1:n
        u[i] = seeds.Y[(i-1)*N+j]
        u[n+i] += seeds.Λ[(i-1)*N+j]
    end
    u
end

function (a::GridAdjoint)(x0, q, Y, Λ)
    n, N = a.n, a.width
    @requires length(Y) == length(Λ) == n * N "the states and the seeds must have $(n * N) elements"
    T = promote_type(eltype(x0), eltype(q), eltype(Y), eltype(Λ))
    u0 = zeros(T, 2n + a.columns)
    for i in 1:n
        u0[i], u0[n+i] = Y[i*N], Λ[i*N]
    end
    uf = a.solver(u0, q, Seeds(Y, Λ, n, N))
    [uf[n+1:n+a.given]; uf[2n+1:2n+a.columns]]
end
(a::GridAdjoint)(x0, Y, Λ) = a(x0, NO_PARAMETERS, Y, Λ)

"""
    call_sites(functions) -> IdDict

Return what the generated code calls for each Julia function of `functions`,
which are all those of one generated function. An integration whose
sensitivities are also integrated becomes `Ahead`. The integration of those
sensitivities becomes `Remembered`. Every other function maps to itself.
"""
function call_sites(functions)
    present = IdDict{Any,Bool}(f => true for f in functions)
    paired(f) = f isa Integration && f.sensitivity[] !== nothing && haskey(present, f.sensitivity[])
    ahead(f) = paired(f) ? Ahead(f, f.sensitivity[], ahead(f.sensitivity[])) : f
    targets = IdDict{Any,Bool}(f.sensitivity[] => true for f in functions if paired(f))
    IdDict{Any,Any}(f => (haskey(targets, f) ? Remembered(f, ahead(f)) : ahead(f)) for f in functions)
end
Base.show(io::IO, i::Integration) = print(io, "the integrator \"", i.name, "\"")

# The objects of an integration in a source file. The pairs of integrations are made again
# as they are, and each solver from its own parts.
writable(::Union{Integration,Ahead,Remembered,RightSide,ExternalCall,NewtonCall,GridAdjoint}) = true
codegen_path(writer::SourceWriter, name::Symbol) = Expr(:., module_path(writer, @__MODULE__), QuoteNode(name))

"""
    object_source(writer, object) -> expression

Return the expression that makes `object` again in a source file. Each type
with `writable` has a method; the SciML extension has those of its solvers.
"""
object_source(writer::SourceWriter, i::Integration) =
    :($(codegen_path(writer, :Integration))($(i.name), $(source(writer, i.solver)), $(value_source(writer, i.tail)), $(i.width)))
object_source(writer::SourceWriter, a::GridAdjoint) =
    :($(codegen_path(writer, :GridAdjoint))($(source(writer, a.solver)), $(a.n), $(a.given), $(a.columns), $(a.width)))
object_source(writer::SourceWriter, a::Ahead) =
    :($(codegen_path(writer, :Ahead))($(source(writer, a.state)), $(source(writer, a.sensitivity)), $(source(writer, a.run))))
object_source(writer::SourceWriter, r::Remembered) =
    :($(codegen_path(writer, :Remembered))($(source(writer, r.sensitivity)), $(source(writer, r.run))))
object_source(writer::SourceWriter, e::ExternalCall) =
    :($(codegen_path(writer, :ExternalCall))($(e.name), $(e.library), $(value_source(writer, e.nnz_in)), $(value_source(writer, e.nnz_out))))
object_source(writer::SourceWriter, s::NewtonCall) =
    :($(codegen_path(writer, :NewtonCall))($(s.name), $(source(writer, s.step)), $(source(writer, s.residual)), $(s.n), $(value_source(writer, s.rows)),
        $(value_source(writer, s.columns)), $(s.iin), $(s.iout), $(value_source(writer, s.nnz_in)), $(value_source(writer, s.nnz_out)), $(s.max_iter), $(s.abstol), $(s.abstol_step), $(s.line_search), $(s.error_on_fail)))
object_source(writer::SourceWriter, r::RightSide) = r.mass === nothing ? :($(codegen_path(writer, :RightSide))($(source(writer, r.f)))) :
    :($(codegen_path(writer, :RightSide))($(source(writer, r.f)), $(value_source(writer, r.mass)), $(source(writer, r.slope))))

const NO_PARAMETERS = Float64[]
const DAE_KEYS = ("x", "p", "t", "ode", "quad", "zero", "transition")

# The CasADi side of an integrator or of a rootfinder: a Julia function whose
# Jacobian is a CasADi function. `jacobian(name)` makes that function.
# `reverse` is the function of the reverse derivatives for one seed, or `nothing`. CasADi gets
# it from the cache of the callback: the front end cannot pass the request for such a function.
struct SymbolicCallback <: RecordedCallback
    call::JuliaCall
    evaluate::Any
    jacobian::Any
    reverse::Any
end
SymbolicCallback(call, evaluate, jacobian) = SymbolicCallback(call, evaluate, jacobian, nothing)

C.Callback_has_jacobian(cb::SymbolicCallback, args...) = true
C.Callback_get_jacobian(cb::SymbolicCallback, name, names_in, names_out, options) = cb.jacobian(String(name))
C.Callback_has_reverse(cb::SymbolicCallback, nadj) = cb.reverse !== nothing

column(v) = size(v, 2) == 1 && C.is_dense(v)
flat(M) = reshape(M, length(M), 1)

"""
    ode_integrator(name, dae, t0, tf, make_solver; adjoint = false, mass = Float64[], nonsmooth = :warn) -> CasADiCodegen.Function
    ode_integrator(name, dae, make_solver; adjoint = false, mass = Float64[], nonsmooth = :warn) -> CasADiCodegen.Function

Return the CasADi function `(x0, p) -> xf` that integrates the ordinary
differential equation `dae` from `t0` to `tf`. Without `t0` and `tf` the times
are free, and the function is `(x0, p, t0, tf) -> xf`. A model with no
parameters has no `p`.

# Arguments
- `dae`: a dictionary with the state `"x"` and its derivative `"ode"`, and
  optionally the parameters `"p"`, the time `"t"` and the quadratures `"quad"`.
  The function returns the integrals of `"quad"` as a second result, `"qf"`.
- `make_solver(rhs!, n, m)`: gets the right side `rhs!(du, u, q, t)` for
  `n` states and `m` parameters, and returns the Julia function
  `(x0, q) -> final state`. `n` counts the quadratures, which the solver
  integrates as more states.

# Keywords
- `mass`: the diagonal of the mass matrix of a model with algebraic states: 0
  for such a state and 1 for the others. The element of `"ode"` for an
  algebraic state is the residual of an algebraic equation, and its initial
  value is a guess. The solver must accept such a model.
- `adjoint`: CasADi gets its reverse derivatives from one more integrator,
  which integrates the state and its multipliers backward in time. The cost of
  a gradient then does not grow with the number of parameters. The state is
  integrated again backward, so the result is as accurate as that integration.
  `make_solver` must then be that of free times, also when `t0` and `tf` are
  given.
- `nonsmooth`: what a switch on the state in `"ode"` or `"quad"` gives: `:warn`,
  `:error` or `:ignore` (see `check_nonsmooth`). `ode_grid` and `ode_event` take it too.

With free times, the two times follow the model parameters in `q`, so `m`
counts them. The solver integrates from 0 to `q[m] - q[m-1]`, because the
right side adds the initial time itself.

CasADi gets the Jacobian from a second integrator, for the state with its
derivatives by the initial state and by the parameters. That integrator is
made when CasADi first asks for the Jacobian (see `integration`).
"""
function ode_integrator(name::AbstractString, dae, t0::Real, tf::Real, make_solver; adjoint::Bool = false, mass::Vector{Float64} = Float64[],
                        nonsmooth::Symbol = :warn)
    check_dae(name, dae)
    check_model(name, dae, nonsmooth)
    @requires isfinite(t0) && isfinite(tf) && t0 != tf "integrator \"$name\": the times must be finite and different"
    if adjoint
        # The adjoint equations run from the final time to the initial one, which the arc with free times can do.
        arc = ode_integrator(name, dae, make_solver; adjoint, mass, nonsmooth = :ignore)
        x0 = Front.MX.sym("x0", size(dae["x"], 1))
        p = haskey(dae, "p") ? [Front.MX.sym("p", size(dae["p"], 1))] : Front.MX[]
        return Front.Function(String(name), [x0, p...], as_list(arc(x0, p..., Front.MX(Float64(t0)), Front.MX(Float64(tf)))),
                              haskey(dae, "p") ? ["x0", "p"] : ["x0"], haskey(dae, "quad") ? ["xf", "qf"] : ["xf"],
                              Dict("always_inline" => true))
    end
    joined, start = with_quadratures(dae)
    core = integration(name, joined, false, make_solver, start; mass = full_mass(name, dae, mass, start))
    isempty(start) && return core
    x0 = Front.MX.sym("x0", size(dae["x"], 1))
    p = haskey(dae, "p") ? [Front.MX.sym("p", size(dae["p"], 1))] : Front.MX[]
    split_function(name, [x0, p...], core(x0, p...), size(x0, 1), haskey(dae, "p") ? ["x0", "p"] : ["x0"])
end

function ode_integrator(name::AbstractString, dae, make_solver; adjoint::Bool = false, mass::Vector{Float64} = Float64[], nonsmooth::Symbol = :warn)
    check_dae(name, dae)
    check_model(name, dae, nonsmooth)
    joined, start = with_quadratures(dae)
    x, has_p = dae["x"], haskey(dae, "p")
    times = typeof(x).sym("times", 2)
    joined["p"] = has_p ? Front.vertcat([dae["p"], times]) : times
    # The time of the solver starts at zero.
    haskey(dae, "t") && (joined["ode"] = Front.substitute(joined["ode"], dae["t"], dae["t"] + times[1]))
    core = integration(name, joined, true, make_solver, start; adjoint, mass = full_mass(name, dae, mass, start))
    x0, t0, tf = Front.MX.sym("x0", size(x, 1)), Front.MX.sym("t0"), Front.MX.sym("tf")
    p = has_p ? [Front.MX.sym("p", size(dae["p"], 1))] : Front.MX[]
    # Inline, so that the graph of the user calls the integrator itself.
    split_function(name, [x0, p..., t0, tf], core(x0, Front.vertcat([p..., t0, tf])), size(x0, 1),
                   has_p ? ["x0", "p", "t0", "tf"] : ["x0", "t0", "tf"])
end

"""
    ode_event(name, dae, make_solver, make_event; adjoint = false, events = 1, terminal = true, exact = true, mass = Float64[],
              nonsmooth = :warn, switching = 0, switches = 0, guard = nothing) -> CasADiCodegen.Function

Return the CasADi function `(x0, p, t0, tf) -> (xf, te)` that integrates `dae`
from `t0` toward `tf` and stops at the event. The event is the first time `te`
at which an element of the event function `"zero"` of the state, the
parameters and the time changes sign. A model with no parameters has no `p`.

`xf` is the state at `te`. The quadratures to `te` follow it as `"qf"` when the
model has them. A zero of the event function at `t0` is not an event, so an
integration can start on the surface. With `exact`, the results do not depend
on `tf`.

With more than one element of `"zero"`, the result `"ie"` follows `te`: for
each event, the place of the element that changed its sign, from 1. When two
elements change their sign at the same time, the first one counts.

# Arguments
- `make_solver`: that of `ode_integrator` with free times.
- `make_event(rhs!, zero!, n, m, k)`: gets the right side `rhs!(du, u, q, t)`
  and the event function `zero!((out,), (u, q, (t,)))` with `k` elements, for
  `n` states and `m` parameters. It returns the Julia function
  `(z0, x0, p, t0, tf) -> (te, i)`, without `p` when the model has none, where
  `i` is the place of the element that changed its sign. `z0` is a guess that
  the function does not use. When no event comes before `tf`, the function
  throws, or with `exact = false` returns `(tf, 0)`.
- `mass`: that of `ode_integrator`. `rhs!` then carries it, and the algebraic states
  of `x0` are a guess.

# Keywords
- `events`: the number of events that the integration passes. With more than
  one, `te` is the vector of their times.
- `terminal`: when true, the integration ends at the last event, where the
  state is that before its transition. Otherwise it goes on to `tf` after the
  last transition, and later events have no effect.
- `exact`: when false, fewer than `events` events can come before `tf`. The
  result `"ne"` then counts the events, a missing event has the time `tf` and
  the place 0, and the state goes on to `tf` with no transition.
- `switching`, `switches`, `guard`: the switches of a model with modes (see
  `switched_events`). The last `switching` elements of `"zero"` are switches,
  at most `switches` of them before `tf`, and `make_event` then gets
  `switching` as a sixth argument.

The entry `"transition"` of `dae` is the state after an event, as an expression
of the state before it, the parameters and the time. It has one column, or one
column per element of `"zero"`: column `i` then applies after an event of
element `i`. Without it the state does not change.

The event time is the root of the element of the event function that changed
its sign, on the state of the free-time integrator, so CasADi differentiates it
by the implicit function theorem, to any order. A missing event has the
derivatives of `tf`. The places and the count have no derivatives.
"""
function ode_event(name::AbstractString, dae, make_solver, make_event; adjoint::Bool = false, events::Integer = 1, terminal::Bool = true,
                   exact::Bool = true, mass::Vector{Float64} = Float64[], nonsmooth::Symbol = :warn,
                   switching::Integer = 0, switches::Integer = 0, guard = nothing)
    check_dae(name, dae; event = true)
    check_model(name, dae, nonsmooth)
    @requires haskey(dae, "zero") "integrator \"$name\": the entry \"zero\" is necessary"
    @requires events >= 1 "integrator \"$name\": the number of events must be 1 or more; got $events"
    @requires 0 <= switching <= size(dae["zero"], 1) && (switching == 0) == (switches == 0) && switches >= 0 &&
              (guard === nothing || switching > 0) "integrator \"$name\": switching, switches and guard go together"
    arc = ode_integrator(name * "_arc", Dict{String,Any}(key => value for (key, value) in dae if !(key in ("zero", "transition"))), make_solver; adjoint, mass,
                         nonsmooth = :ignore)
    x, has_p, has_quad = dae["x"], haskey(dae, "p"), haskey(dae, "quad")
    symbol = typeof(x).sym
    p = has_p ? dae["p"] : symbol("p", 0)
    t = haskey(dae, "t") ? dae["t"] : symbol("t")
    n, m, k = size(x, 1), size(p, 1), size(dae["zero"], 1)
    surface = Front.Function("surface", [x, p, t], [Front.densify(dae["zero"])])
    model = Front.Function("rhs", [x, p, t], [Front.densify(dae["ode"])])
    emitted(g) = Emitted(build_function(g; expression = Val{false}, nonzeros = true), g, :nonzeros)
    right_side = RightSide(emitted(model), mass, algebraic_slope(Front.densify(dae["ode"]), x, p, t, mass))
    find = switching == 0 ? make_event(right_side, emitted(surface), n, m, k) : make_event(right_side, emitted(surface), n, m, k, switching)

    x0, t0, tf, z = Front.MX.sym("x0", n), Front.MX.sym("t0"), Front.MX.sym("tf"), Front.MX.sym("z", 2)
    q = has_p ? [Front.MX.sym("p", m)] : Front.MX[]
    # The place `i` of the element that changed its sign selects it: one_hot(i) has a 1 in row i, and zeros for i = 0.
    one_hot(i) = Front.vertcat([C.eq(i, Front.MX(Float64(j))) for j in 1:k])
    te, place = z[1], z[2]
    at_time = arc(x0, q..., t0, te)
    value = surface(has_quad ? at_time[1] : at_time, has_p ? q[1] : Front.MX(0, 1), te)
    chosen = k == 1 ? value : Front.dot(one_hot(place), value)
    # The unknowns are the time and the place. The place is the root of i - round(i), which no
    # input moves, so CasADi gives it no derivatives; a missing event is the root of te - tf.
    residual = Front.vertcat([exact && switching == 0 ? chosen : C.if_else(C.eq(place, Front.MX(0.0)), te - tf, chosen), place - C.floor(place + Front.MX(0.5))])
    # Inline, so that the calls of the integrator are at the top level of the graph of the user.
    residual_function = Front.Function(name * "_zero", [z, x0, q..., t0, tf], [residual], Dict("always_inline" => true))
    event_time = implicit_function(name * "_time", residual_function, find)
    jump = haskey(dae, "transition") ? Front.Function("transition", [x, p, t], [Front.densify(dae["transition"])]) : nothing
    state, start, times, places = x0, t0, Front.MX[], Front.MX[]
    sums = Front.MX[]
    function advance(to)
        result = arc(state, q..., start, to)
        has_quad && push!(sums, result[2])
        state, start = has_quad ? result[1] : result, to
    end
    if switching > 0
        parts = (; arc, event_time, jump, one_hot, x0, q, t0, tf, has_p, has_quad, k)
        outputs, names = switched_events(name, parts; events, terminal, exact, switching, switches, guard)
        return Front.Function(String(name), [x0, q..., t0, tf], outputs, has_p ? ["x0", "p", "t0", "tf"] : ["x0", "t0", "tf"], names,
                              Dict("always_inline" => true))
    end
    for _ in 1:events
        # Loop invariant: `state` at the time `start` follows the events of `times`, each with its jump, and `places` holds their places.
        # The guess carries the place of the event at `start`, so that the search does not find it again.
        found = event_time(Front.vertcat([tf, isempty(places) ? Front.MX(0.0) : places[end]]), state, q..., start, tf)
        push!(times, found[1])
        push!(places, found[2])
        advance(times[end])
        if (length(times) < events || !terminal) && jump !== nothing
            after = jump(state, has_p ? q[1] : Front.MX(0, 1), start)
            size(after, 2) > 1 && (after = Front.mtimes(after, one_hot(places[end])))
            state = exact ? after : C.if_else(places[end] > 0, after, state)
        end
    end
    terminal || advance(tf)
    outputs, names = [state], ["xf"]
    has_quad && (push!(outputs, reduce(+, sums)); push!(names, "qf"))
    push!(outputs, Front.vertcat(times)); push!(names, "te")
    k > 1 && (push!(outputs, Front.vertcat(places)); push!(names, "ie"))
    exact || (push!(outputs, reduce(+, [place > 0 for place in places])); push!(names, "ne"))
    Front.Function(String(name), [x0, q..., t0, tf], outputs, has_p ? ["x0", "p", "t0", "tf"] : ["x0", "t0", "tf"], names,
                   Dict("always_inline" => true))
end

"""
    full_mass(name, dae, mass, start) -> Vector{Float64}

Return the diagonal of the mass matrix for the state of `dae` with its
quadratures, whose initial values are `start`: `mass` followed by ones. An
empty `mass` stays empty.
"""
function full_mass(name::AbstractString, dae, mass::Vector{Float64}, start::Vector{Float64})
    isempty(mass) && return mass
    @requires length(mass) == size(dae["x"], 1) && all(in((0.0, 1.0)), mass) "integrator \"$name\": the mass must have one number per state, 0 for an algebraic state and 1 for the others"
    [mass; ones(length(start))]
end

"""
    with_quadratures(dae) -> (dae, start)

The dictionary of an integrator with the quadratures `"quad"` as more states
after the state, and the initial values of those states, which are zero.
"""
function with_quadratures(dae)
    joined = Dict{String,Any}(key => value for (key, value) in dae if key != "quad")
    haskey(dae, "quad") || return joined, Float64[]
    quad = Front.densify(dae["quad"])
    count = size(quad, 1)
    joined["x"] = Front.vertcat([dae["x"], typeof(dae["x"]).sym("quad", count)])
    joined["ode"] = Front.vertcat([dae["ode"], quad])
    joined, zeros(count)
end

"""
    split_function(name, inputs, result, n, names) -> CasADiCodegen.Function

The function of the user for the `result` of an integration: the final state
`"xf"`, which is its first `n` elements, and the quadratures `"qf"` when
elements follow. CasADi inlines it, so that the graph of the user calls the
integrator itself.
"""
function split_function(name::AbstractString, inputs, result, n::Integer, names)
    total = size(result, 1)
    outputs = total == n ? [result] : [result[1:n], result[n+1:total]]
    Front.Function(String(name), inputs, outputs, names, total == n ? ["xf"] : ["xf", "qf"], Dict("always_inline" => true))
end

"""
    ode_grid(name, dae, t0, times, make_solver; adjoint = false, nonsmooth = :warn) -> CasADiCodegen.Function

Return the CasADi function `(x0, p) -> X`, or `x0 -> X`, whose column `k` is
the state at `times[k]` of the integration that starts at `t0`. With
quadratures, a second result `"qf"` has their values at each time. The times can
be in any order and on both sides of `t0`.

One integration from `t0` gives the states at all the distinct times after
`t0`, and one more those before it; a time equal to `t0` gives `x0`.
`make_solver(rhs!, n, m, t0, stops)` gets the right side `rhs!(du, u, q, t)` of
`n` states and `m` parameters, the initial time and the times `stops`, in the
order of the integration, and returns a solver `run` with two methods:

- `run(x0, q)` integrates once and returns the state at each of the `stops`, in
  a vector of `n * length(stops)` elements: the first state at all the times,
  then the second one, and so on.
- `run(x0, q, jump!)` integrates once, calls `jump!(u, k)` with the state `u`
  when it reaches `stops[k]`, and goes on from the state that `jump!` changed. It
  returns the state at the last of the `stops`.

The Jacobian is one integration of the states with their sensitivities over the
same times. With `adjoint`, a reverse derivative is one integration of the
adjoint equations from the last time back to `t0` (see `GridAdjoint`), whose
derivatives come from the forward sensitivities.
"""
function ode_grid(name::AbstractString, dae, t0::Real, times::AbstractVector{<:Real}, make_solver; adjoint::Bool = false, nonsmooth::Symbol = :warn)
    check_dae(name, dae)
    check_model(name, dae, nonsmooth)
    @requires !isempty(times) && isfinite(t0) && all(isfinite, times) "integrator \"$name\": the times must be finite, and at least one is necessary"
    joined, start = with_quadratures(dae)
    has_p, nx = haskey(dae, "p"), size(dae["x"], 1)
    n = nx + length(start)
    x0 = Front.MX.sym("x0", nx)
    p = has_p ? [Front.MX.sym("p", size(dae["p"], 1))] : Front.MX[]
    columns = Vector{Front.MX}(undef, length(times))
    whole = nothing                     # the matrix of all the times, when it has them in their order
    for (suffix, side) in (("", >), ("_before", <))
        stops = sort!(unique(Float64[time for time in times if side(time, t0)]); by = time -> abs(time - t0))
        isempty(stops) && continue
        core = integration(name * suffix, joined, false, make_solver, start; adjoint, grid = (Float64(t0), stops))
        X = C.transpose(reshape(core(x0, p...), length(stops), n))
        stops == times && (whole = X)
        for (k, time) in enumerate(times)
            side(time, t0) && (columns[k] = X[:, searchsortedfirst(stops, time; by = s -> abs(s - t0))])
        end
    end
    initial = isempty(start) ? x0 : Front.vertcat([x0, Front.MX(C.DM(start))])
    for k in eachindex(times)
        times[k] == t0 && (columns[k] = initial)
    end
    M = whole === nothing ? Front.horzcat(columns) : whole
    outputs = isempty(start) ? [M] : [M[1:nx, :], M[nx+1:n, :]]
    Front.Function(String(name), [x0, p...], outputs, has_p ? ["x0", "p"] : ["x0"], isempty(start) ? ["xf"] : ["xf", "qf"])
end

function check_dae(name::AbstractString, dae; event::Bool = false)
    @requires Base.isidentifier(name) "\"$name\" is not a valid function name"
    for key in keys(dae)
        @requires key in DAE_KEYS "integrator \"$name\": the entry \"$key\" is not supported; give \"x\", \"ode\" and, if necessary, \"p\", \"t\", \"quad\", \"zero\" and \"transition\""
    end
    @requires haskey(dae, "x") && haskey(dae, "ode") "integrator \"$name\": the entries \"x\" and \"ode\" are necessary"
    x = dae["x"]
    @requires column(x) && size(x, 1) > 0 "integrator \"$name\": the state must be a dense column vector that is not empty"
    @requires !haskey(dae, "p") || column(dae["p"]) "integrator \"$name\": the parameters must be a dense column vector"
    @requires size(dae["ode"]) == size(x) "integrator \"$name\": \"ode\" must have the size of the state"
    @requires !haskey(dae, "t") || size(dae["t"]) == (1, 1) "integrator \"$name\": the time must be a scalar"
    @requires !haskey(dae, "quad") || size(dae["quad"], 2) == 1 "integrator \"$name\": the quadratures must be a column vector"
    @requires event || !haskey(dae, "zero") "integrator \"$name\": an event function, \"zero\", needs free times: give no times to the integrator"
    @requires !haskey(dae, "zero") || (size(dae["zero"], 2) == 1 && size(dae["zero"], 1) >= 1) "integrator \"$name\": the event function, \"zero\", must be a column vector that is not empty"
    @requires !haskey(dae, "transition") || haskey(dae, "zero") "integrator \"$name\": the state after an event, \"transition\", needs an event function, \"zero\""
    @requires !haskey(dae, "transition") || (size(dae["transition"], 1) == size(x, 1) && size(dae["transition"], 2) in (1, size(dae["zero"], 1))) "integrator \"$name\": the state after an event, \"transition\", must have the size of the state, or one column per event function"
end

"""
    state_rate(ode, x, p, t, mass) -> CasADiCodegen.Function

Return the function `(x, p, t) -> dx/dt` of the model `mass .* x' = ode` on its solution.
A differential state has its right side as rate. An algebraic state `a` follows from the
algebraic equations, so its rate is `-inv(A_aa) (A_ad ode_d + d(ode_a)/dt)`, with the
Jacobian `A` of `ode` by `x`, and `d` the differential states.
"""
function state_rate(ode, x, p, t, mass::Vector{Float64})
    (isempty(mass) || all(!iszero, mass)) && return Front.Function("rate", [x, p, t], [ode])
    n = length(mass)
    select(rows) = Front.MX(C.DM([Float64(i == r) for r in rows, i in 1:n]))
    Pa, Pd = select(findall(iszero, mass)), select(findall(!iszero, mass))
    X, P, T = Front.MX.sym("x", n), Front.MX.sym("p", size(p, 1)), Front.MX.sym("t")
    at(value) = Front.Function("at", [x, p, t], [Front.densify(value)])(X, P, T)
    f, A, by_time = at(ode), at(Front.jacobian(ode, x)), at(Front.jacobian(ode, t))
    differential = Front.mtimes(C.transpose(Pd), Front.mtimes(Pd, f))
    coupling = Front.mtimes(Pa, Front.mtimes(A, differential)) + Front.mtimes(Pa, by_time)
    algebraic = -C.solve(Front.mtimes(Pa, Front.mtimes(A, C.transpose(Pa))), coupling)
    Front.Function("rate", [X, P, T], [differential + Front.mtimes(C.transpose(Pa), algebraic)])
end

"""
    algebraic_slope(ode, x, p, t, mass)

Return the generated code of the Jacobian of the algebraic equations of `ode` by the
algebraic states, the rows and states of `x` with a zero in `mass`, for the right side of a
solver (see `RightSide`); `nothing` when `mass` is empty.
"""
function algebraic_slope(ode, x, p, t, mass::Vector{Float64})
    isempty(mass) && return nothing
    rows = findall(iszero, mass)
    by_all = Front.jacobian(Front.vertcat([ode[r] for r in rows]), x)
    by_state = Front.Function("slope", [x, p, t], [Front.densify(Front.horzcat([by_all[:, r] for r in rows]))])
    Emitted(build_function(by_state; expression = Val{false}, nonzeros = true), by_state, :nonzeros)
end

"""
    integration(name, dae, free, make_solver, tail = Float64[]; adjoint = false, mass = Float64[], grid = nothing) -> CasADiCodegen.Function

Return the CasADi function of `ode_integrator` for a checked `dae`. With
`free`, the last two parameters of `dae` are the initial and the final time,
and the time of `dae` starts at zero. The function takes the first states of
`dae` only; `tail` holds the initial values of the others.

With `grid = (t0, stops)`, the times are fixed and the function returns the
states at each of the `stops`, as the solver of `ode_grid` does; `make_solver`
is then that of `ode_grid`.
"""
function integration(name::AbstractString, dae, free::Bool, make_solver, tail::Vector{Float64} = Float64[];
                     adjoint::Bool = false, mass::Vector{Float64} = Float64[], grid::Union{Nothing,Tuple{Float64,Vector{Float64}}} = nothing)
    @requires free || !adjoint || grid !== nothing "integrator \"$name\": the adjoint equations need an integrator with free times or a grid of times"
    @requires grid === nothing || (!free && isempty(mass)) "integrator \"$name\": a grid of times needs fixed times and no mass matrix"
    x, ode = dae["x"], Front.densify(dae["ode"])
    symbol = typeof(x).sym
    has_p = haskey(dae, "p")
    p = has_p ? dae["p"] : symbol("p", 0)
    t = haskey(dae, "t") ? dae["t"] : symbol("t")
    n, m = size(x, 1), size(p, 1)
    width = grid === nothing ? 1 : length(grid[2])
    given = n - length(tail)                    # the states that the function takes
    own = free ? m - 2 : m                      # the parameters of the model, without the times
    # The parameters whose sensitivities are integrated: those of the model and,
    # when the right side has the time in it, the initial time, which follows them.
    # An element of a matrix (MX) symbol is not a symbol, so the sparsity pattern of the Jacobian answers for it.
    by_start() = p isa MX ? C.nnz(C.sparsity(C.jacobian(ode, p)[:, own+1])) > 0 : C.depends_on(ode, p[own+1])
    columns = free && by_start() ? own + 1 : own

    model = Front.Function("rhs", [x, p, t], [ode])
    rhs = build_function(model; expression = Val{false}, nonzeros = true)
    # A solver that is compiled before the right side would call it with boxed arguments.
    precompile(rhs, (Tuple{Vector{Float64}}, Tuple{Vector{Float64},Vector{Float64},Tuple{Float64}}))
    right_side = RightSide(Emitted(rhs, model, :nonzeros), mass, algebraic_slope(ode, x, p, t, mass))
    run = Integration(String(name), grid === nothing ? make_solver(right_side, n, m) : make_solver(right_side, n, m, grid...), tail, width)

    # CasADi can ask for the Jacobian more than once, because it does not keep a function that
    # nothing uses; the second integrator is made once, so that each call of it is the same one.
    # The values of each state at all the output times are next to each other, so that the sensitivities
    # by one element of x0 or of the parameters are a column of nN elements: the Jacobian of the result.
    nN = n * width
    sensitivity = nothing
    function sensitivities(x0, q)
        if sensitivity === nothing
            sensitivity = sensitivity_integration()
            run.sensitivity[] = JULIA_CALLS[C.__hash__(sensitivity)][1].f
        end
        zf = has_p ? sensitivity(x0, q) : sensitivity(x0)
        reshape(zf[nN+1:nN+nN*given], nN, given), reshape(zf[nN+nN*given+1:nN+nN*given+nN*columns], nN, columns)
    end
    function sensitivity_integration()
        # State z = (x, S, Q) with S = dx/dx0 and Q = dx/dp: S' = A S and Q' = A Q + B.
        S, A = symbol("S", n, given), Front.jacobian(ode, x)
        z, zdot = [x, flat(S)], [ode, flat(Front.mtimes(A, S))]
        longer = [tail; [Float64(i == j) for j in 1:given for i in 1:n]]
        if columns > 0
            Q = symbol("Q", n, columns)
            push!(z, flat(Q))
            push!(zdot, flat(Front.mtimes(A, Q) + Front.jacobian(ode, p)[:, 1:columns]))
            append!(longer, zeros(n * columns))
        end
        augmented = Dict("x" => Front.vertcat(z), "t" => t, "ode" => Front.vertcat(zdot))
        has_p && (augmented["p"] = p)
        # Each column of a sensitivity matrix has the mass of the state.
        longer_mass = isempty(mass) ? mass : repeat(mass, 1 + given + columns)
        integration(name * "_s", augmented, free, make_solver, longer; mass = longer_mass, grid)
    end

    made = Ref{Any}(nothing)
    jacobian = function (jacobian_name)
        made[] === nothing || return Front.Function(jacobian_name, made[]..., Dict("always_inline" => true))
        x0, xf, q = Front.MX.sym("x0", given), Front.MX.sym("xf", nN), Front.MX.sym("p", m)
        Sf, Qf = sensitivities(x0, q)
        parts = own > 0 ? Any[Qf[:, 1:own]] : Any[]
        if free
            rate = state_rate(ode, x, p, t, mass)(xf, q, q[own+2] - q[own+1])
            push!(parts, columns > own ? Qf[:, own+1] - rate : -rate, rate)
        end
        blocks = has_p ? [Sf, Front.horzcat(parts)] : [Sf]
        # Inline, so that the call of the second integrator is at the top level of the graph of the user,
        # where the emitter writes scalar code around it.
        made[] = (has_p ? [x0, q, xf] : [x0, xf], blocks)
        Front.Function(jacobian_name, made[]..., Dict("always_inline" => true))
    end

    # The adjoint equations in the time of the model: the state x, the multipliers l with
    # l' = -A' l, and the sums s with s' = -B' l. From the final time back to the initial one,
    # l goes from a seed to the derivative by the initial state, and s from zero to those by the parameters.
    reverse = nothing
    if adjoint && grid !== nothing
        reverse = grid_reverse(name, x, p, t, ode, has_p, n, given, columns, grid, make_solver, sensitivities)
    elseif adjoint
        l, sums = symbol("l", n), symbol("s", columns)
        rates = [ode, -Front.jtimes(ode, x, l, true)]
        columns > 0 && push!(rates, -Front.jtimes(ode, p, l, true)[1:columns])
        equations = Dict("x" => Front.vertcat([x, l, sums][1:length(rates)]), "p" => p, "t" => t, "ode" => Front.vertcat(rates))
        # The derivatives of that integrator are forward sensitivities, so that the chain of integrators ends.
        backward = integration(name * "_a", equations, true, make_solver, zeros(columns); mass = isempty(mass) ? mass : [mass; mass; ones(columns)])
        x0, q, xf, seed = Front.MX.sym("x0", given), Front.MX.sym("p", m), Front.MX.sym("xf", n), Front.MX.sym("adj_xf", n)
        returned = Front.vertcat([q[1:own], q[own+2], q[own+1]])       # the parameters with the two times exchanged
        at_end(value) = Front.Function("at_end", [x, p, t], [Front.densify(value)])(xf, q, q[own+2] - q[own+1])
        by_final = Front.dot(state_rate(ode, x, p, t, mass)(xf, q, q[own+2] - q[own+1]), seed)
        if isempty(mass)
            zf = backward(Front.vertcat([xf, seed]), returned)
            by_initial = columns > own ? zf[2n+own+1] - by_final : -by_final
            results = [zf[n+1:n+given], Front.vertcat([zf[2n+1:2n+own], by_initial, by_final])]
        else
            # The final algebraic states are functions of the other final states, of the parameters and of the
            # time through the algebraic equations, so a seed on them goes to those: with the rows `a` of the
            # algebraic equations and states and the rows `d` of the others, and c = -inv(A_aa') seed_a, the seed
            # of the states `d` grows by A_ad' c, and the direct derivatives by the parameters are B_a' c.
            select(rows) = Front.MX(C.DM([Float64(i == r) for r in rows, i in 1:n]))
            Pa, Pd = select(findall(iszero, mass)), select(findall(!iszero, mass))
            A = at_end(Front.jacobian(ode, x))
            c = -C.solve(C.transpose(Front.mtimes(Pa, Front.mtimes(A, C.transpose(Pa)))), Front.mtimes(Pa, seed))
            by_states = Front.mtimes(Pd, seed) + Front.mtimes(Pd, Front.mtimes(C.transpose(A), Front.mtimes(C.transpose(Pa), c)))
            pulled(value) = Front.mtimes(C.transpose(at_end(value)), Front.mtimes(C.transpose(Pa), c))
            # The multipliers of the algebraic states start from a guess, as those states do.
            zf = backward(Front.vertcat([xf, Front.mtimes(C.transpose(Pd), by_states)]), returned)
            direct = columns > 0 ? pulled(Front.jacobian(ode, p)[:, 1:columns]) : nothing
            by_initial = columns > own ? zf[2n+own+1] + direct[own+1] - by_final : -by_final
            by_parameters = own > 0 ? [zf[2n+1:2n+own] + direct[1:own]] : []
            # An initial value of an algebraic state is a guess, on which no result depends.
            results = [C.times(Front.MX(C.DM(mass[1:given])), zf[n+1:n+given]), Front.vertcat([by_parameters..., by_initial, by_final])]
        end
        reverse = Front.Function("adj1_" * name, [x0, q, xf, seed], results, Dict("always_inline" => true))
    end

    shapes_in = has_p ? [(given,), (m,)] : [(given,)]
    call = JuliaCall(String(name), run, length(shapes_in), shapes_in, [(nN,)], map(prod, shapes_in), [nN])
    # CasADi asks for the reverse derivatives of one seed at a time, and finds their function in the cache.
    options = adjoint ? (; cache = Dict("adj1_" * name => reverse), max_num_dir = 1) : (;)
    proxy = Front.callback(SymbolicCallback(call, run, jacobian, reverse); name = String(name), options...)
    F = Front.Function(proxy)
    JULIA_CALLS[C.__hash__(F)] = (call, proxy)
    F
end

"""
    grid_reverse(name, x, p, t, ode, has_p, n, given, columns, grid, make_solver, sensitivities) -> CasADiCodegen.Function

Return the reverse derivative `"adj1_" * name` of an integration over the `grid` of
`integration`: `(x0, p, X, seed) -> (seed of x0, seed of p)`, without `p` when the model has
none. One call of a `GridAdjoint` gives it, with one run of the adjoint equations of `ode`.

That call has the Jacobian of `Σₖ Jₖ' seedₖ`, with the Jacobian `Jₖ` of the state at the
time `k` from the forward sensitivities that `sensitivities(x0, q)` returns, so that the
derivatives of every order come from integrations with fixed times.
"""
function grid_reverse(name::AbstractString, x, p, t, ode, has_p::Bool, n::Int, given::Int, columns::Int,
                      grid::Tuple{Float64,Vector{Float64}}, make_solver, sensitivities)
    symbol = typeof(x).sym
    l, sums = symbol("l", n), symbol("s", columns)
    rates = [ode, -Front.jtimes(ode, x, l, true)]
    columns > 0 && push!(rates, -Front.jtimes(ode, p, l, true)[1:columns])
    model = Front.Function("rhs", [Front.vertcat([x, l, sums][1:length(rates)]), p, t], [Front.densify(Front.vertcat(rates))])
    rhs = build_function(model; expression = Val{false}, nonzeros = true)
    t0, stops = grid
    m, width = size(p, 1), length(stops)
    nN, nout = n * width, given + columns
    # From the last output time back to the initial time, through the others.
    solver = make_solver(RightSide(Emitted(rhs, model, :nonzeros)), 2n + columns, m, stops[end], [reverse(stops[1:end-1]); t0])
    run = GridAdjoint(solver, n, given, columns, width)
    symbols() = (Front.MX.sym("x0", given), Front.MX.sym("p", m), Front.MX.sym("xf", nN), Front.MX.sym("adj_xf", nN))
    arguments(x0, q, X, seed) = has_p ? [x0, q, X, seed] : [x0, X, seed]

    made = Ref{Any}(nothing)
    jacobian = function (jacobian_name)
        made[] === nothing || return Front.Function(jacobian_name, made[]..., Dict("always_inline" => true))
        x0, q, X, seed = symbols()
        Sf, Qf = sensitivities(x0, q)
        J = columns > 0 ? Front.horzcat([Sf, Qf]) : Sf
        result = Front.mtimes(C.transpose(J), seed)
        blocks = [Front.jacobian(result, x0)]
        has_p && push!(blocks, Front.jacobian(result, q))
        push!(blocks, Front.MX(nout, nN), C.transpose(J))        # the result does not depend on the states X
        made[] = ([arguments(x0, q, X, seed)..., Front.MX.sym("r", nout)], blocks)
        Front.Function(jacobian_name, made[]..., Dict("always_inline" => true))
    end

    shapes_in = Tuple{Vararg{Int}}[(given,), (m,), (nN,), (nN,)][has_p ? (1:4) : [1, 3, 4]]
    call = JuliaCall(String(name) * "_a", run, length(shapes_in), shapes_in, [(nout,)], map(prod, shapes_in), [nout])
    proxy = Front.callback(SymbolicCallback(call, run, jacobian); name = call.name)
    F = Front.Function(proxy)
    JULIA_CALLS[C.__hash__(F)] = (call, proxy)
    x0, q, X, seed = symbols()
    r = F(arguments(x0, q, X, seed)...)
    results = has_p ? [r[1:given], r[given+1:nout]] : [r[1:given]]
    Front.Function("adj1_" * name, arguments(x0, q, X, seed), results, Dict("always_inline" => true))
end
