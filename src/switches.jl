# A right side that switches between modes. Each mode is a symbol that the model reads in
# place of the sign of a switching function S: +1 where S > 0 and -1 where S < 0. The mode
# is a state with no rate, and a change of sign of S is an event whose transition flips the
# mode. `ode_event` then differentiates each switching time by the implicit function
# theorem, so the derivatives of the integration have the jump of the STM at each switch.

"""
    mode_model(name, dae, modes; gate = false) -> (augmented, k, nu)

Return the model with modes of the integrator `name` as a model with events for
`ode_event`, the number `k` of modes and the number `nu` of event functions of the user.
`modes` is the pair `m => S` of the column of modes and the column of their switching
functions.

The state of `augmented` is `[x; m]`, with `m' = 0`. Its event function is the user's
`"zero"`, then `m .* S`, which changes its sign from positive to negative when the state
leaves the region of its mode. Its `"transition"` has one column per event function: the
user's transition with the modes unchanged, then for each switch the state with that mode
negated.

With `gate`, one more state counts the events of the user that are still to come: each
event of the user lowers it by one, and at zero the event functions of the user no longer
change their sign. Its initial value is the number of events.

Throws `ArgumentError` when the modes are not a dense column of symbols of the type of the
state that the state, the parameters and the time do not hold, when `S` does not have the
size of `m` or depends on `m`, or when the user's `"zero"` or `"transition"` depends on `m`.
"""
function mode_model(name::AbstractString, dae, modes::Pair; gate::Bool = false)
    x = dae["x"]
    T = typeof(x)
    m = modes.first
    S = modes.second isa Real ? T(Float64(modes.second)) : modes.second
    @requires m isa T && column(m) && size(m, 1) >= 1 && C.is_valid_input(m) "integrator \"$name\": the modes must be a dense column of symbols of the type of the state, $T"
    for key in ("x", "p", "t")
        @requires !haskey(dae, key) || !C.depends_on(dae[key], m) "integrator \"$name\": a mode must not be a symbol of \"$key\""
    end
    @requires S isa T && size(S) == size(m) "integrator \"$name\": the switching functions must be a column of $T with one element per mode"
    S = Front.densify(S)
    @requires !C.depends_on(S, m) "integrator \"$name\": a switching function must not depend on the modes; a surface that moves with the mode needs two event functions"
    for key in ("zero", "transition")
        @requires !haskey(dae, key) || !C.depends_on(dae[key], m) "integrator \"$name\": \"$key\" must not depend on the modes"
    end
    n, k = size(x, 1), size(m, 1)
    counter = gate ? T[T.sym("events_left")] : T[]
    augmented = Dict{String,Any}(key => value for (key, value) in dae if !(key in ("x", "ode", "zero", "transition")))
    augmented["x"] = Front.vertcat([x, m, counter...])
    augmented["ode"] = Front.vertcat([Front.densify(dae["ode"]), casadi_matrix(T, zeros(k + length(counter)))])

    user = haskey(dae, "zero") ? Front.densify(dae["zero"]) : nothing
    nu = user === nothing ? 0 : size(user, 1)
    # An event function of the user stays at 1, with no change of sign, when no event is left to count.
    gated = gate ? C.if_else(counter[1] > 0.5, user, casadi_matrix(T, ones(nu))) : user
    surfaces = C.times(m, S)
    augmented["zero"] = nu == 0 ? surfaces : Front.vertcat([gated, surfaces])

    columns = T[]
    if nu > 0
        given = haskey(dae, "transition") ? Front.densify(dae["transition"]) : x
        for i in 1:nu
            after = size(given, 2) == 1 ? given : given[:, i]
            push!(columns, Front.vertcat([after, m, (gate ? [counter[1] - 1] : T[])...]))
        end
    end
    for j in 1:k
        flip = casadi_matrix(T, [i == j ? -1.0 : 1.0 for i in 1:k])
        push!(columns, Front.vertcat([x, C.times(m, flip), counter...]))
    end
    augmented["transition"] = Front.horzcat(columns)
    @ensures size(augmented["x"], 1) == n + k + length(counter) && size(augmented["transition"], 2) == size(augmented["zero"], 1) "integrator \"$name\": the model with modes has the wrong size"
    augmented, k, nu
end

"""
    switched_events(name, parts; events, terminal, exact, switching, switches, guard) -> (outputs, names)

The results of `ode_event` for a model whose last `switching` event functions are switches.
`parts` holds the pieces that `ode_event` built: the arc integrator, the event search, the
transition, and the inputs.

The integration searches `events + switches + 1` times for the next event, with `events`
counted only when the model has event functions of the user. A switch does not count
toward `events`, `exact` or `terminal`, and its transition always applies. After the last
event of the user with `terminal`, the later searches run on the empty interval at its
time. The last search must find no switch, or the result throws an assertion error: more
than `switches` switches came before the final time.

`guard` is `nothing` or a pair of a function `(x, p, t, place) -> value` and a message. At
each switch, the state before the transition throws with the message when the value is
not positive.

The results are the state, the quadratures, then for the events of the user the times
`"te"`, their places `"ie"` (more than one event function) and their count `"ne"` (with
`exact = false`), then for the switches the times `"ts"`, the places `"is"` among the
switches (more than one) and their count `"ns"`. A missing event or switch has the time
`tf` and the place 0; the selection of each by a running count carries no derivative.
"""
function switched_events(name::AbstractString, parts; events::Integer, terminal::Bool, exact::Bool, switching::Integer, switches::Integer, guard)
    (; arc, event_time, jump, one_hot, x0, q, t0, tf, has_p, has_quad, k) = parts
    number(v) = Front.MX(Float64(v))
    nu = k - switching
    wanted = nu > 0 ? Int(events) : 0
    searches = wanted + switches + 1
    p_value = has_p ? q[1] : Front.MX(0, 1)
    state, start, sums = x0, t0, Front.MX[]
    users, switched, done, previous = number(0), number(0), number(0), number(0)
    user_time, user_place, user_hit = fill(number(0), wanted), fill(number(0), wanted), fill(number(0), wanted)
    switch_time, switch_place, switch_hit = fill(number(0), switches), fill(number(0), switches), fill(number(0), switches)
    for search in 1:searches
        # Loop invariant: `state` at the time `start` follows the events found so far, each with its transition;
        # `users` and `switched` count the events of the user and the switches among them.
        horizon = terminal && wanted > 0 ? C.if_else(done, start, tf) : tf
        # The guess carries the place of the event at `start`, so that the search does not find it again.
        found = event_time(Front.vertcat([horizon, previous]), state, q..., start, horizon)
        te, place = found[1], found[2]
        previous = place
        result = arc(state, q..., start, te)
        has_quad && push!(sums, result[2])
        state, start = has_quad ? result[1] : result, te
        is_user, is_switch = C.and(place > 0.5, place < nu + 0.5), place > nu + 0.5
        users, switched = users + is_user, switched + is_switch
        for j in 1:wanted
            mask = C.and(is_user, C.eq(users, number(j)))
            user_time[j], user_place[j], user_hit[j] = user_time[j] + mask * te, user_place[j] + mask * place, user_hit[j] + mask
        end
        for j in 1:switches
            mask = C.and(is_switch, C.eq(switched, number(j)))
            switch_time[j], switch_place[j], switch_hit[j] = switch_time[j] + mask * te, switch_place[j] + mask * (place - nu), switch_hit[j] + mask
        end
        if search == searches
            state = C.attachAssert(state, C.not(is_switch), "integrator \"$name\": more than $switches switches before the final time; raise switches")
            break
        end
        if guard !== nothing
            value = guard[1](state, p_value, te, place)
            state = C.attachAssert(state, C.or(C.not(is_switch), value > 0), guard[2])
        end
        if jump !== nothing
            after = jump(state, p_value, te)
            size(after, 2) > 1 && (after = Front.mtimes(after, one_hot(place)))
            # The last event of the user with `terminal` ends the integration before its transition.
            apply = terminal && wanted > 0 ? C.or(is_switch, C.and(is_user, users < wanted - 0.5)) : place > 0.5
            state = C.if_else(apply, after, state)
        end
        terminal && wanted > 0 && (done = C.or(done, C.and(is_user, C.eq(users, number(wanted)))))
    end
    exact && wanted > 0 &&
        (state = C.attachAssert(state, users > wanted - 0.5, "integrator \"$name\": fewer than $wanted events between the initial and the final time"))
    outputs, names = Front.MX[state], ["xf"]
    has_quad && (push!(outputs, reduce(+, sums)); push!(names, "qf"))
    if wanted > 0
        push!(outputs, Front.vertcat([user_time[j] + (1 - user_hit[j]) * tf for j in 1:wanted])); push!(names, "te")
        nu > 1 && (push!(outputs, Front.vertcat(user_place)); push!(names, "ie"))
        exact || (push!(outputs, users); push!(names, "ne"))
    end
    push!(outputs, Front.vertcat([switch_time[j] + (1 - switch_hit[j]) * tf for j in 1:switches])); push!(names, "ts")
    switching > 1 && (push!(outputs, Front.vertcat(switch_place)); push!(names, "is"))
    push!(outputs, switched); push!(names, "ns")
    @ensures length(outputs) == length(names) "integrator \"$name\": each result must have a name"
    outputs, names
end

"""
    ModeFunctions

The functions of a model with modes that the wrapper of `mode_integrator` evaluates, all of
MX: `surface(x, p, t)`, the switching functions; `rate(X, p, t)`, the rate of each
switching function on the augmented state `X = [x; m; ...]`, `∂S/∂x ẋ + ∂S/∂t`; and the
sizes `n` of the state, `k` of the modes and `extra` of the counter after them.
"""
struct ModeFunctions
    surface::C.Function
    rate::C.Function
    n::Int
    k::Int
    extra::Int
end

function ModeFunctions(dae, augmented, S, mass::Vector{Float64})
    x = dae["x"]
    T = typeof(x)
    p = haskey(dae, "p") ? dae["p"] : T.sym("p", 0)
    t = haskey(dae, "t") ? dae["t"] : T.sym("t")
    surface = Front.Function("surface", [x, p, t], [Front.densify(S)])
    na, n, k = size(augmented["x"], 1), size(x, 1), size(S, 1)
    # The rate of the state from `state_rate`: with a mass matrix, an element of "ode" of an algebraic state is a residual.
    states = state_rate(Front.densify(augmented["ode"]), augmented["x"], p, t, mass)
    X, P, τ = Front.MX.sym("x", na), Front.MX.sym("p", size(p, 1)), Front.MX.sym("t")
    value = surface(X[1:n], P, τ)
    rate = Front.mtimes(Front.jacobian(value, X), states(X, P, τ)) + Front.jacobian(value, τ)
    ModeFunctions(surface, Front.Function("switching_rate", [X, P, τ], [Front.densify(rate)]), n, k, na - n - k)
end

"The augmented state `[x; m; 0...]` with the state `x` and the modes `m`."
augmented_state(f::ModeFunctions, x, m) = Front.vertcat([x, m, (f.extra > 0 ? [Front.MX(f.extra, 1)] : Front.MX[])...])

"""
    initial_mode(f::ModeFunctions, name, x0, p, t0, direction) -> MX

Return the column of initial modes at the state `x0`, the parameters `p` and the time `t0`,
for an integration toward `direction` (+1 forward, -1 backward). Off its surface a mode is
the sign of its switching function. On its surface, where `|S| <= √eps (1 + |S|)`, it is the
mode under which the flow leaves the surface toward the integration, with the other modes
at the sign of their switching functions. The result throws an assertion error when the
flow leaves on both sides, or on neither. A mode carries no derivative.
"""
function initial_mode(f::ModeFunctions, name::AbstractString, x0, p, t0, direction)
    number(v) = Front.MX(Float64(v))
    S0 = f.surface(x0, p, t0)
    off = C.abs(S0) > sqrt(eps()) * (1 + C.abs(S0))
    base = C.sign(S0)
    modes = map(1:f.k) do j
        with(v) = augmented_state(f, x0, Front.vertcat([i == j ? number(v) : base[i] for i in 1:f.k]))
        leaves_up = direction * f.rate(with(1), p, t0)[j] > 0
        leaves_down = direction * f.rate(with(-1), p, t0)[j] < 0
        mode = C.if_else(off[j], base[j], C.if_else(leaves_up, number(1), number(-1)))
        on = "integrator \"$name\": switching function $j is zero at the initial time, and"
        mode = C.attachAssert(mode, C.or(off[j], C.not(C.and(leaves_up, leaves_down))),
                              "$on the flow leaves its surface on both sides, so the initial mode is not defined; start off the surface")
        C.attachAssert(mode, C.or(off[j], C.or(leaves_up, leaves_down)),
                       "$on the flow leaves its surface on neither side: it slides on the surface or touches it; start off the surface, or smooth the switch")
    end
    Front.vertcat(modes)
end

"""
    switch_guard(f::ModeFunctions, nu) -> CasADiCodegen.Function

The guard of the switches for `switched_events`: `(X, p, t, place) -> value`, with the
product of the rates of the switching function that fired before and after the flip of its
mode, and 1 at an event of the user (`place <= nu`). A product that is not positive means
that the flow on the far side points back to the surface: a sliding arc.
"""
function switch_guard(f::ModeFunctions, nu::Integer)
    na = f.n + f.k + f.extra
    X, P, τ, place = Front.MX.sym("x", na), Front.MX.sym("p", C.size1(C.sparsity_in(f.rate, 1))), Front.MX.sym("t"), Front.MX.sym("place")
    before = f.rate(X, P, τ)
    value = place < nu + 0.5
    for j in 1:f.k
        flip = Front.MX(C.DM([i == f.n + j ? -1.0 : 1.0 for i in 1:na]))
        after = f.rate(C.times(X, flip), P, τ)
        value = value + C.eq(place, Front.MX(Float64(nu + j))) * before[j] * after[j]
    end
    Front.Function("switch_guard", [X, P, τ, place], [value])
end

"""
    checked_modes(f::ModeFunctions, name, X, p, t) -> MX

The state of the augmented state `X` at the time `t`, which throws an assertion error when
a mode does not match the sign of its switching function there.
"""
function checked_modes(f::ModeFunctions, name::AbstractString, X, p, t)
    x, m = X[1:f.n], X[f.n+1:f.n+f.k]
    S = f.surface(x, p, t)
    agree = C.mmin(C.times(m, S) + sqrt(eps()) * (1 + C.abs(S))) >= 0
    C.attachAssert(x, agree, "integrator \"$name\": a mode does not match the sign of its switching function at an output time. " *
                             "Two switching functions changed their sign in one step of the event search, as two on one surface do, one changed its sign at an output time, " *
                             "or the flow slides on a surface")
end

"""
    mode_integrator(name, dae, modes, make_solver, make_event; span = nothing, switches, switch_times = false,
                    events = 1, terminal = true, exact = true, adjoint = false, mass = Float64[], nonsmooth = :warn)
        -> CasADiCodegen.Function

Return the integrator of a model whose right side switches between modes, with the inputs
and the results of the same integrator without modes. `modes = m => S` gives the column of
modes `m`, which `"ode"` and `"quad"` read, and their switching functions `S`. A mode is +1
where its switching function is positive and -1 where it is negative.

`span` selects the form, as the times of `integrator` do: `nothing` for free times,
`(t0, tf)` for fixed times, and `(t0, times)` for a vector of output times, through which
the integration is chained with the modes carried from one time to the next. `switches` is
an upper bound on the number of switches before the final time (per interval of a vector
of times). With `switch_times`, the results end with the switch times `"ts"`, their places
`"is"` with more than one mode, and their count `"ns"`; a vector of times does not support
it. `events`, `terminal` and `exact` concern the user's event functions only (free times).

`make_solver` and `make_event` are those of `ode_event`. The initial mode comes from
`initial_mode`; at each switch a guard throws for a sliding arc (`switch_guard`); at each
output time the modes must match the signs of their switching functions
(`checked_modes`).
"""
function mode_integrator(name::AbstractString, dae, modes, make_solver, make_event; span = nothing, switches = nothing, switch_times::Bool = false,
                         events::Integer = 1, terminal::Bool = true, exact::Bool = true, adjoint::Bool = false, mass::Vector{Float64} = Float64[],
                         nonsmooth::Symbol = :warn)
    free, grid = span === nothing, span isa Tuple{Real,AbstractVector}
    check_dae(name, dae; event = free)
    @requires modes isa Pair "integrator \"$name\": modes must be a pair m => S of the modes and their switching functions"
    @requires switches isa Integer && switches >= 1 "integrator \"$name\": modes need switches, an upper bound on the number of switches, 1 or more"
    @requires !(grid && switch_times) "integrator \"$name\": switch_times needs a final time; a vector of times does not return the switch times"
    if span !== nothing
        t0, times = Float64(span[1]), grid ? Float64.(span[2]) : [Float64(span[2])]
        @requires isfinite(t0) && !isempty(times) && all(isfinite, times) && (grid || times[1] != t0) "integrator \"$name\": the times must be finite and different"
    end
    gate = free && haskey(dae, "zero") && !terminal
    augmented, k, nu = mode_model(name, dae, modes; gate)
    check_model(name, dae, nonsmooth; modes = modes.first)
    full = isempty(mass) ? mass : [mass; ones(size(augmented["x"], 1) - size(dae["x"], 1))]
    f = ModeFunctions(dae, augmented, Front.densify(modes.second), full)
    slides = "integrator \"$name\": at a switch the flow on the far side of the surface points back to it, so the flow slides on the surface; " *
             "a model with modes cannot follow a sliding arc: smooth the switch with CasADiCodegen.smooth_sign or CasADiCodegen.smoothed"
    G = ode_event(name, augmented, make_solver, make_event; adjoint, events, terminal, exact, mass = full, nonsmooth = :ignore,
                  switching = k, switches, guard = (switch_guard(f, nu), slides))
    names_out = [C.name_out(G, i) for i in 0:C.n_out(G)-1]

    n, has_p, has_quad = size(dae["x"], 1), haskey(dae, "p"), haskey(dae, "quad")
    x0 = Front.MX.sym("x0", n)
    q = has_p ? [Front.MX.sym("p", size(dae["p"], 1))] : Front.MX[]
    p_value = has_p ? q[1] : Front.MX(0, 1)
    names_in = has_p ? ["x0", "p"] : ["x0"]
    run(X0, from, to) = Dict(zip(names_out, as_list(G(X0, q..., from, to))))
    start(from, direction) = Front.vertcat([x0, initial_mode(f, name, x0, p_value, from, direction), (gate ? [Front.MX(Float64(events))] : Front.MX[])...])

    if !grid
        from, to = free ? (Front.MX.sym("t0"), Front.MX.sym("tf")) : (Front.MX(t0), Front.MX(times[1]))
        direction = free ? C.if_else(from <= to, Front.MX(1.0), Front.MX(-1.0)) : Front.MX(sign(times[1] - t0))
        r = run(start(from, direction), from, to)
        ends = terminal && nu > 0 ? r["te"][Int(events)] : to
        kept = ["xf"; filter(key -> key in ("qf", "te", "ie", "ne") || (switch_times && key in ("ts", "is", "ns")), names_out[2:end])]
        outputs = [key == "xf" ? checked_modes(f, name, r["xf"], p_value, ends) : r[key] for key in kept]
        inputs = free ? [x0, q..., from, to] : [x0, q...]
        return Front.Function(String(name), inputs, outputs, free ? [names_in; "t0"; "tf"] : names_in, kept)
    end

    # A vector of times: the integration is chained from t0 through the times after it, and backward through those before it.
    nq = has_quad ? size(dae["quad"], 1) : 0
    columns = Vector{Front.MX}(undef, length(times))
    for (side, direction) in ((>, 1.0), (<, -1.0))
        stops = sort!(unique(Float64[time for time in times if side(time, t0)]); by = time -> abs(time - t0))
        isempty(stops) && continue
        X, from, total = start(Front.MX(t0), Front.MX(direction)), t0, Front.MX(C.DM(zeros(nq)))
        for stop in stops
            # Loop invariant: `X` is the augmented state at the time `from`, and `total` the quadratures from t0 to it.
            r = run(X, Front.MX(from), Front.MX(stop))
            X, from = r["xf"], stop
            has_quad && (total = total + r["qf"])
            at = Front.vertcat([checked_modes(f, name, X, p_value, Front.MX(stop)), (has_quad ? [total] : Front.MX[])...])
            foreach(i -> times[i] == stop && (columns[i] = at), eachindex(times))
        end
    end
    initial = has_quad ? Front.vertcat([x0, Front.MX(C.DM(zeros(nq)))]) : x0
    foreach(i -> times[i] == t0 && (columns[i] = initial), eachindex(times))
    M = Front.horzcat(columns)
    Front.Function(String(name), [x0, q...], has_quad ? [M[1:n, :], M[n+1:n+nq, :]] : [M], names_in, has_quad ? ["xf", "qf"] : ["xf"])
end
