# A solver object of CasADi (an integrator, a rootfinder, a solver of an
# optimization problem) holds a problem: equations as CasADi functions, and
# data in its serial form. This file reads the problem, so that a solver of
# the SciML ecosystem can take it: as a SciML problem that the user solves
# (see the extension for SciMLBase.jl), or in place of the solver of CasADi in
# generated code.

"""
    Solver(alg; options...)
    Solver(target::CasADiCodegen.Function, alg; options...)

A SciML algorithm `alg`, for example `Vern9()`, with the `options` that go to its solver,
for example `abstol`. Use it as a field of [`Solvers`](@ref) when the algorithm has options.

With `target`, a solver object of CasADi, the result is an override: the solver for that
object and for its derivatives only. Give an override in the field `overrides` of `Solvers`.
"""
struct Solver
    target::Union{Nothing,C.Function}
    alg::Any
    options::NamedTuple
end
Solver(alg; options...) = Solver(nothing, alg, values(options))
Solver(target::C.Function, alg; options...) = Solver(target, alg, values(options))
Base.:(==)(a::Solver, b::Solver) = a.target === b.target && a.alg == b.alg && a.options == b.options

"""
    Solvers(; ode = nothing, dae = nothing, nonlinear = nothing, optimization = nothing,
              overrides = ())

Select the SciML solvers that generated code calls in place of the solver objects of
CasADi in the graph of a function. Pass the result as the keyword `solvers` of
[`build_function`](@ref) or [`generate_julia`](@ref).

Each field is an algorithm, or a [`Solver`](@ref) when the algorithm has options.

| Field          | Replaces                                                             |
|----------------|----------------------------------------------------------------------|
| `ode`          | the integrator `cvodes`, and `idas` without algebraic states         |
| `dae`          | the integrator `idas` with algebraic states                          |
| `nonlinear`    | the rootfinders `kinsol` and `fast_newton`                           |
| `optimization` | the solvers of `nlpsol` and of `conic`                               |
| `overrides`    | one object each: a collection of `Solver(target, alg; options...)`   |

An override has priority over the other fields. Only an override replaces an integrator
`rk` or `collocation` or a rootfinder `newton`, for which the package has Julia code of the
method of CasADi. An integrator with an event function is replaced by the same fields, and
`rk` and `collocation` then need an override.

The tolerances of an integrator of CasADi are the options `abstol` and `reltol` of its
solver, unless the solver has them. A solver of the first four fields must have no
`target`, and each override must have one. See [Replace the solvers of CasADi](@ref).

# Examples
```julia
using CasADiCodegen
using OrdinaryDiffEqVerner: Vern9

solvers = CasADiCodegen.Solvers(ode = CasADiCodegen.Solver(Vern9(); abstol = 1e-13, reltol = 1e-13))
g, g! = build_function(f; expression = Val{false}, expand = false, solvers)   # f calls `cvodes`
```
"""
struct Solvers
    ode::Union{Nothing,Solver}
    dae::Union{Nothing,Solver}
    nonlinear::Union{Nothing,Solver}
    optimization::Union{Nothing,Solver}
    overrides::Vector{Solver}
end
function Solvers(; ode = nothing, dae = nothing, nonlinear = nothing, optimization = nothing, overrides = ())
    function general(kind, solver)
        solver isa Solver || return solver === nothing ? nothing : Solver(solver)
        @requires solver.target === nothing "the solver for \"$kind\" has a function of CasADi; give it in overrides"
        solver
    end
    list = collect(Solver, overrides)
    @requires all(s -> s.target !== nothing, list) "an override needs the function of CasADi that it replaces: Solver(function, algorithm)"
    Solvers(general("ode", ode), general("dae", dae), general("nonlinear", nonlinear), general("optimization", optimization), list)
end

"The solvers of the function that is read now (see `algorithm`); `nothing` when the user gave none."
const SOLVERS = Base.ScopedValues.ScopedValue{Union{Nothing,Solvers}}(nothing)

"The name of the solver object of CasADi from which CasADi made the function `name` for derivatives."
function base_name(name::AbstractString)
    while (found = match(r"^(?:fsens|asens)\d+_(.+)$", name)) !== nothing
        # Loop invariant: `name` is the name of the function without the prefixes that were removed.
        name = found[1]
    end
    String(name)
end

"""
    selected_solver(callee, kind) -> Solver or nothing

Return the solver that the user selected for the solver object `callee` of CasADi: the
override for the object of that name and class, and else the solver of the field `kind` of
`Solvers`. `kind` is `:none` for an object that only an override can replace.
"""
function selected_solver(callee::C.Function, kind::Symbol)
    solvers = SOLVERS[]
    solvers === nothing && return nothing
    name = base_name(C.name(callee))
    for solver in solvers.overrides
        C.name(solver.target) == name && C.class_name(solver.target) == C.class_name(callee) && return solver
    end
    kind === :none ? nothing : getfield(solvers, kind)
end

"The classes of the integrators of CasADi; those with a variable step have tolerances."
const INTEGRATOR_CLASSES = ("CvodesInterface", "IdasInterface", "RungeKutta", "Collocation")
const VARIABLE_STEP_CLASSES = ("CvodesInterface", "IdasInterface")

const ROOTFINDER_CLASSES = ("Newton", "FastNewton", "KinsolInterface")

"The names that CasADi gives to the solvers of the classes that need a solver of the SciML ecosystem."
const PLUGIN_NAMES = Dict("CvodesInterface" => "cvodes", "IdasInterface" => "idas", "KinsolInterface" => "kinsol", "FastNewton" => "fast_newton")

"""
    refuse(callee)

Throw an `ArgumentError` with the reason for which generated code cannot hold the function
`callee` of CasADi, and with what the user can do. Such a function has no instructions and
no Julia code in this package, and generated code never calls CasADi. The manual page
"Matrix code and functions without instructions" lists the cases.
"""
function refuse(callee::C.Function)
    name, class = C.name(callee), C.class_name(callee)
    needs(kind, field) = "is $kind of CasADi, which generated code replaces by a solver of the SciML ecosystem; give one with the option solvers = CasADiCodegen.Solvers($field = ...)"
    reason = if class == "CallbackInternal"
        "is a callback of a subtype of CasadiCallback, whose Julia function this package does not know; make it with CasADiCodegen.callback(f, name, sizes_in, sizes_out)"
    elseif class in FIXED_STEP_CLASSES && C.nnz_out(C.get_function(callee, "dae"), 3) > 0
        "is an integrator of CasADi with a fixed step and an event function, which generated code replaces by a solver of the SciML ecosystem; give one with the option solvers = CasADiCodegen.Solvers(overrides = [CasADiCodegen.Solver(I, alg)]), where I is the integrator"
    elseif class in VARIABLE_STEP_CLASSES
        needs("the integrator \"$(PLUGIN_NAMES[class])\"", serial_value(serial_bytes(callee), "Integrator::nz") > 0 ? "dae" : "ode")
    elseif class in ROOTFINDER_CLASSES
        needs("the rootfinder \"$(PLUGIN_NAMES[class])\"", "nonlinear")
    elseif C.is_a(callee, "Nlpsol")
        needs("a solver of optimization problems", "optimization")
    elseif C.is_a(callee, "Conic")
        needs("a solver of quadratic problems", "optimization")
    else
        "has the class $class of CasADi, for which this package has no Julia code"
    end
    throw(ArgumentError("function \"$name\" $reason"))
end

numbers(n::Integer) = n == 1 ? "1 number" : "$n numbers"

"The elements `from + 1` to `from + n` of the column vector `v`, with the sparsity pattern `pattern`."
function piece(v, from::Int, pattern)
    n = C.nnz(pattern)
    n == 0 && return MX(pattern)
    values = v[from+1:from+n]::MX
    C.is_dense(pattern) && C.size2(pattern) == 1 ? values : MX(pattern, values)
end

column_of(value) = C.vec(C.densify(value))

"Whether `f`, or a function of its matrix graph, is an integrator of CasADi with an event function."
function has_events(f::C.Function)
    C.class_name(f) in INTEGRATOR_CLASSES && return C.nnz_out(C.get_function(f, "dae"), 3) > 0
    C.is_a(f, "MXFunction") || return false
    any(k -> C.instruction_id(f, k) == C.OP_CALL && has_events(C.which_function(C.instruction_MX(f, k))), 0:C.n_instructions(f)-1)
end

"""
    event_jacobian(ex::MX, arg::MX) -> Union{MX,Nothing}

The Jacobian of `ex` by `arg` from one forward direction per element of `arg`, dense, when
the graph of `ex` holds an integrator of CasADi with an event function; `nothing` otherwise.
`jacobian` of CasADi 3.8.1 gives such an integrator the sparsity pattern of the model without
events, so the Jacobian that it builds on that pattern is wrong. The method `jacobian(ex, arg)`
of the front end returns this Jacobian when it is not `nothing`.
"""
function event_jacobian(ex::C.MX, arg::C.MX)
    has_events(Front.Function("graph", C.symvar(ex), [ex])) || return nothing
    n = C.numel(arg)
    C.jtimes(C.vec(ex), C.vec(arg), MX(C.DM([Float64(i == j) for i in 1:n, j in 1:n])))
end

"""
    check_jacobian(f::CasADiCodegen.Function)

Throw an `ArgumentError` when `f` holds an integrator of CasADi with an event function, whose
Jacobian function CasADi 3.8.1 builds on a wrong sparsity pattern (see `event_jacobian`). The
method `jacobian(f)` of the front end calls it.
"""
check_jacobian(f::C.Function) =
    @requires !has_events(f) "the Jacobian function of \"$(C.name(f))\" would be wrong: it holds an integrator of CasADi with an event function, to which CasADi 3.8.1 gives the sparsity pattern of the model without events; take the derivatives with jtimes, one direction per input, or with jacobian of the expressions"

"""
    integrator_model(I::CasADiCodegen.Function) -> NamedTuple

Return the problem of the integrator `I` of CasADi:

- `dae`: its function `dae(t, x, z, p, u) -> (ode, alg, quad, zero)`;
- `t0`, `times`: the initial time and the output times;
- `nx`, `nz`, `nq`, `np`, `nu`, `events`: the numbers of differential states, algebraic
  states, quadratures, parameters, controls and event functions;
- `nfwd`, `nadj`: the numbers of forward and of adjoint directions;
- `tolerances`: those of an integrator with a variable step.

The sizes are those of `dae`: an integrator for sensitivities has the equations of the
sensitivities in it.
"""
function integrator_model(I::C.Function)
    name = String(C.name(I))::String
    @requires C.class_name(I) in INTEGRATOR_CLASSES "function \"$name\" is not an integrator of CasADi"
    bytes = serial_bytes(I)
    field(key) = serial_value(bytes, key)
    dae = C.get_function(I, "dae")
    # The serial form gives values of unknown type; typed fields keep the code that reads them inferable.
    real(key) = Float64(field(key))::Float64
    whole(key) = Int(field(key))::Int
    tolerances = C.class_name(I) in VARIABLE_STEP_CLASSES ?
                 (; abstol = real("SundialsInterface::abstol"), reltol = real("SundialsInterface::reltol")) : (;)
    (; name, dae, t0 = real("Integrator::t0"), times = Vector{Float64}(field("Integrator::tout"))::Vector{Float64},
       nx = whole("Integrator::nx"), nz = whole("Integrator::nz"), nq = whole("Integrator::nq"), np = whole("Integrator::np"), nu = whole("Integrator::nu"),
       events = Int(C.nnz_out(dae, 3))::Int, nfwd = whole("Integrator::nfwd"), nadj = whole("Integrator::nadj"), tolerances)
end

"""
    sensitivity_function(dae) -> CasADiCodegen.Function

Return the forward derivative of the equations `dae` of an integrator in one direction:
`(t, x, z, p, u, dx, dz, dp, du) -> (d ode, d alg, d quad)`.
"""
function sensitivity_function(dae::C.Function)
    symbols(prefix) = MX[MX.sym(prefix * C.name_in(dae, i), C.sparsity_in(dae, i)) for i in 0:4]
    point, direction = symbols(""), symbols("d")
    values = as_list(C.call(dae, point, false, false))
    slope(value) = reduce(+, (C.jtimes(value, point[i], direction[i]) for i in 2:5 if C.nnz(C.sparsity(point[i])) > 0); init = MX(C.DM(zeros(size(value)...))))
    Front.Function("d" * C.name(dae), MX[point; direction[2:5]], [slope(column_of(values[k])) for k in 1:3])
end

"""
    right_side(model) -> CasADiCodegen.Function

Return the function `(u, p, t) -> du` of the equations of an integrator of CasADi (see
`integrator_model`). `u` has the differential states, the algebraic states and the
quadratures, in that order. `p` has the parameters and then the controls. `du` has the
derivatives of the differential states, the residuals of the algebraic equations and the
integrands of the quadratures.

An integrator that CasADi made for forward sensitivities has, in each of these groups, the
values of the model and then one block per direction. Its equations are those of the model
with their forward derivatives.
"""
function right_side(model)
    dae, directions = model.dae, model.nfwd
    u, p, t = MX.sym("u", model.nx + model.nz + model.nq), MX.sym("p", model.np + model.nu), MX.sym("t")
    time = C.nnz_in(dae, 0) == 0 ? MX(C.sparsity_in(dae, 0)) : t
    block(k) = model_block(model, u, p, k)
    parts = Vector{MX}[MX[column_of(value)] for value in as_list(C.call(dae, MX[time; block(0)], false, false))[1:3]]
    if directions > 0
        forward = sensitivity_function(dae)
        for k in 1:directions
            # Loop invariant: `parts` holds the equations of the model and of the directions before `k`.
            foreach(push!, parts, as_list(C.call(forward, MX[time; block(0); block(k)], false, false)))
        end
    end
    du = C.vertcat(reduce(vcat, parts))
    @ensures size(du, 1) == size(u, 1) "the equations of integrator \"$(model.name)\" must give one number per state"
    Front.Function((model.name::String) * "_rhs", [u, p, t], [du])
end

"""
    model_block(model, u, p, k) -> Vector{MX}

Return the differential states, the algebraic states, the parameters and the controls of
an integrator of CasADi in the vectors `u` and `p` of `right_side`: those of the model for
`k = 0`, and those of the forward direction `k` otherwise.
"""
function model_block(model, u, p, k::Int)
    dae = model.dae
    x1, z1, p1, u1 = (C.nnz_in(dae, i) for i in 1:4)
    MX[piece(u, k * x1, C.sparsity_in(dae, 1)), piece(u, model.nx + k * z1, C.sparsity_in(dae, 2)),
       piece(p, k * p1, C.sparsity_in(dae, 3)), piece(p, model.np + k * u1, C.sparsity_in(dae, 4))]
end

"""
    event_functions(model, transition) -> NamedTuple

Return the functions of the events of an integrator of CasADi (see `integrator_model`), on
the state `u` and the parameters `p` of `right_side`:

- `zero`: `(u, p, t) -> e`, the event functions. Event `i` is a change of sign of `e[i]`
  from positive to negative.
- `jump`: `(u, p, t, index, z) -> u`, the state after the event `index`, which counts from
  zero. The new states are those of `transition`, a function with the inputs of
  `event_in()` and the results of `event_out()`. Without it (`nothing`) the states do not
  change. `z` has the size of the algebraic states of `u`, whose first block, that of the
  model, holds the algebraic states that satisfy the algebraic equations after the event.
  The algebraic states of the result are only a guess, which the caller makes consistent.

With forward directions, the event time has the derivative `-de / (de/dt)` along the
trajectory (the implicit function theorem), and `jump` adds its effect to the derivatives
of the states and of the quadratures: the rate before the event times that derivative, the
derivative of `transition`, then minus the rate after the event, which depends on `z`. The
rate of the algebraic states, which `de/dt` needs, comes from the derivative of the
algebraic equations along the trajectory.
"""
function event_functions(model, transition)
    dae, directions, ne = model.dae, model.nfwd, model.events
    point = input_symbols(dae)
    time = MX.sym("t")
    C.nnz(C.sparsity(point[1])) == 0 || (point[1] = time)
    index = MX.sym("index")
    evaluate(at) = map(column_of, as_list(C.call(dae, at, false, false)))
    ode, alg, quad, e = evaluate(point)
    nx1, nz1, nq1 = C.nnz_in(dae, 1), C.nnz_in(dae, 2), C.nnz_out(dae, 2)
    # The states after the event, with the sparsity patterns of the model; an empty result keeps the states.
    after = point[2:3]
    if transition !== nothing
        given = MX[index, time, point[2:5]...]
        arguments = MX[C.nnz_in(transition, i) == 0 ? MX(C.sparsity_in(transition, i)) : piece(column_of(given[i+1]), 0, C.sparsity_in(transition, i)) for i in 0:5]
        made = as_list(C.call(transition, arguments, false, false))
        after = MX[C.nnz_out(transition, o) == 0 ? after[o+1] : piece(column_of(made[o+1]), 0, C.sparsity_in(dae, o + 1)) for o in 0:1]
    end
    consistent = nz1 == 0 ? after[2] : MX.sym("z_after", C.sparsity_in(dae, 2))
    seeds = [MX[MX.sym("d$(k)_" * C.name_in(dae, i)::String, C.sparsity_in(dae, i)) for i in 1:4] for k in 1:directions]
    results = MX[column_of(after[1]), column_of(after[2])]
    shifts = MX[]
    if directions > 0
        rate_after, quad_after = evaluate(MX[point[1]; after[1]; consistent; point[4:5]])[[1, 3]]
        along(value, seed) = reduce(+, (C.jtimes(value, point[i+1], seed[i]) for i in eachindex(seed) if C.nnz(C.sparsity(point[i+1])) > 0); init = MX(C.DM(zeros(size(value)...))))
        chosen = C.vertcat(MX[C.eq(index, MX(Float64(i - 1))) for i in 1:ne])::MX
        # The rate of the algebraic states, from the derivative of the algebraic equations along the trajectory.
        z_rate = nz1 == 0 ? after[2] :
                 piece(-C.solve(C.jacobian(alg, point[3]), C.jacobian(alg, time) + C.jtimes(alg, point[2], ode))::MX, 0, C.sparsity_in(dae, 2))::MX
        # The rate of the event function along the trajectory.
        slope = C.jacobian(e, time) + C.jtimes(e, point[2], ode)
        nz1 == 0 || (slope += C.jtimes(e, point[3], z_rate))
        for seed in seeds
            # Loop invariant: `results` and `shifts` hold the states after the event for the model and the directions before `seed`.
            shift = -C.dot(chosen, along(e, seed)) / C.dot(chosen, slope)
            moved = MX[seed[1] + shift * ode; nz1 == 0 ? seed[2] : seed[2] + shift * z_rate; seed[3:4]]
            change = along(after[1], moved) + C.jtimes(column_of(after[1]), time, shift)
            push!(results, column_of(change) - shift * rate_after)
            push!(shifts, shift * (quad - quad_after))
        end
        # A guess of the derivatives of the algebraic states, which the caller makes consistent.
        foreach(seed -> push!(results, column_of(seed[2])), seeds)
    end
    u, p, t = MX.sym("u", model.nx + model.nz + model.nq), MX.sym("p", model.np + model.nu), MX.sym("t")
    i, z = MX.sym("index"), MX.sym("z", model.nz)
    known = nz1 == 0 ? MX[] : MX[consistent]
    core = Front.Function(model.name * "_jump", MX[time; point[2:5]; index; reduce(vcat, seeds; init = MX[]); known], MX[C.vertcat(results); shifts])
    at = model_block(model, u, p, 0)
    called = as_list(C.call(core, MX[t; at; i; reduce(vcat, (model_block(model, u, p, k) for k in 1:directions); init = MX[]);
                                     nz1 == 0 ? MX[] : MX[piece(z, 0, C.sparsity_in(dae, 2))]], false, false))
    states = called[1]
    # The order of `u`: the states of the model and of the directions, the algebraic states of the model and of the directions, then the quadratures.
    x_part, z_part = part(states, 0, nx1), part(states, nx1, nz1)
    x_dirs, z_dirs = part(states, nx1 + nz1, directions * nx1), part(states, nx1 + nz1 + directions * nx1, directions * nz1)
    quadratures = part(u, model.nx + model.nz, model.nq)
    moved_sums = directions == 0 || nq1 == 0 ? quadratures :
                 C.vertcat(MX[part(quadratures, 0, nq1); [part(quadratures, k * nq1, nq1) + called[1+k] for k in 1:directions]])
    zero_values = evaluate(MX[C.nnz_in(dae, 0) == 0 ? MX(C.sparsity_in(dae, 0)) : t; at])[4]
    (; zero = Front.Function(model.name * "_zero", [u, p, t], [zero_values]),
       jump = Front.Function(model.name * "_after", [u, p, t, i, z], [C.vertcat(MX[x_part, x_dirs, z_part, z_dirs, moved_sums])]))
end

"""
    rootfinder_model(R::CasADiCodegen.Function) -> NamedTuple

Return the problem of the rootfinder `R` of CasADi: the function `oracle`, whose result
`iout` (from zero) is the residual and whose argument `iin` is the unknown of `n` numbers.
"""
function rootfinder_model(R::C.Function)
    name = C.name(R)
    @requires C.class_name(R) in ROOTFINDER_CLASSES "function \"$name\" is not a rootfinder of CasADi"
    bytes = serial_bytes(R)
    (; name, oracle = C.oracle(R), iin = serial_value(bytes, "Rootfinder::iin"), iout = serial_value(bytes, "Rootfinder::iout"),
       n = serial_value(bytes, "Rootfinder::n"))
end

other_sizes(model) = Int[C.nnz_in(model.oracle, i) for i in 0:C.n_in(model.oracle)-1 if i != model.iin]

"""
    residual_function(model) -> CasADiCodegen.Function

Return the function `(u, p) -> residual` of a rootfinder of CasADi (see
`rootfinder_model`). `u` is the unknown, and `p` has the nonzeros of the other arguments,
in their order.
"""
function residual_function(model)
    oracle = model.oracle
    u, p = MX.sym("u", model.n), MX.sym("p", sum(other_sizes(model); init = 0))
    arguments, from = MX[], 0
    for i in 0:C.n_in(oracle)-1
        # Loop invariant: `from` numbers of `p` belong to the arguments before `i`.
        pattern = C.sparsity_in(oracle, i)
        push!(arguments, i == model.iin ? piece(u, 0, pattern) : piece(p, from, pattern))
        i == model.iin || (from += C.nnz(pattern))
    end
    residual = column_of(as_list(C.call(oracle, arguments, false, false))[model.iout+1])
    @ensures size(residual, 1) == model.n "the residual of rootfinder \"$(model.name)\" must have the size of the unknown"
    Front.Function(model.name * "_residual", [u, p], [residual])
end

"""
    optimization_model(S::CasADiCodegen.Function) -> NamedTuple

Return the problem of the solver `S` that `nlpsol` of CasADi made: the `objective(x, p)`,
the `constraints(x, p)`, and the numbers of unknowns `nx`, of parameters `np` and of
constraints `ng`.
"""
function optimization_model(S::C.Function)
    name = C.name(S)
    C.is_a(S, "Conic") && return conic_model(S)
    @requires all(in(C.get_function(S)), ("nlp_f", "nlp_g")) "function \"$name\" is not a solver of CasADi for optimization problems"
    objective, constraints = C.get_function(S, "nlp_f"), C.get_function(S, "nlp_g")
    (; name, objective, constraints, nx = C.nnz_in(objective, 0), np = C.nnz_in(objective, 1), ng = C.nnz_out(constraints, 0))
end

part(v, from::Int, n::Int) = n == 0 ? MX(0, 1) : v[from+1:from+n]::MX

nonzero_column(v) = C.is_dense(v) ? C.vec(v) : C.nnz(C.sparsity(v)) == 0 ? MX(0, 1) : C.vertcat(MX[e for e in C.nonzeros(v)])

input_symbols(f::C.Function) = MX[MX.sym(C.name_in(f, i), C.sparsity_in(f, i)) for i in 0:C.n_in(f)-1]

same_interface(f::C.Function, inputs, results) =
    C.Function(C.name(f), inputs, results, [C.name_in(f, i) for i in 0:C.n_in(f)-1], [C.name_out(f, o) for o in 0:C.n_out(f)-1])

"""
    solver_integrator(callee) -> CasADiCodegen.Function or nothing

Return a function with the inputs and the results of the integrator `callee` of CasADi
that integrates with the solver that the user selected for it (see `Solvers`), or `nothing`
when no solver is selected. An integrator with an event function is one call of a Julia
function (see `solver_events`).

The function is one call per output time of one `CasADiCodegen.integrator` with free times, for
the equations of `right_side`; a model with algebraic states has a mass matrix.

An integrator that CasADi made for adjoint sensitivities gets them as the reverse
derivative of that chain of calls (see `integrator_results`): each call then integrates
the adjoint equations backward (`adjoint = true`).
"""
function solver_integrator(callee::C.Function)
    model = integrator_model(callee)
    kind = C.class_name(callee) in VARIABLE_STEP_CLASSES ? (model.nz > 0 ? :dae : :ode) : :none
    solver = selected_solver(callee, kind)
    solver === nothing && return nothing
    model.events > 0 && return solver_events(callee, model, solver)
    # With adjoint sensitivities the chain is that of the model with no direction, and CasADi differentiates it
    # (`outer` forward directions). Without them the directions are in the equations (see `right_side`).
    dae, nadj = model.dae, model.nadj
    outer = nadj > 0 ? model.nfwd : 0
    nadj > 0 && (model = (; model..., nfwd = 0, nx = C.nnz_in(dae, 1), nz = C.nnz_in(dae, 2), nq = C.nnz_out(dae, 2), np = C.nnz_in(dae, 3), nu = C.nnz_in(dae, 4)))
    rhs = right_side(model)
    n, m = model.nx + model.nz + model.nq, model.np + model.nu
    state, parameters, time = MX.sym("x", n), MX.sym("p", m), MX.sym("t")
    equations = Dict{String,Any}("x" => state, "t" => time, "ode" => rhs(state, parameters, time))
    m > 0 && (equations["p"] = parameters)
    mass = model.nz == 0 ? Float64[] : [ones(model.nx); zeros(model.nz); ones(model.nq)]
    options = (; model.tolerances..., solver.options...)
    arc = Front.integrator(C.name(callee) * "_arc", solver.alg, equations; adjoint = nadj > 0, mass, options...)
    inputs = MX[MX.sym(C.name_in(callee, i), C.size1_in(callee, i), C.size2_in(callee, i) ÷ (1 + outer)) for i in 0:C.n_in(callee)-1]
    outer == 0 && nadj == 0 && (inputs = input_symbols(callee))
    dense(i) = C.vec(C.densify(inputs[i]))
    x, p, controls = C.vertcat(MX[dense(1), dense(2), MX(C.DM(zeros(model.nq)))]), dense(3), dense(4)
    states, algebraic, quadratures, t = MX[], MX[], MX[], model.t0
    for (k, t_next) in enumerate(model.times)
        # Loop invariant: `x` has the states and the quadratures at the time `t`, and the lists hold them at the output times before it.
        q = m > 0 ? MX[C.vertcat(MX[p, part(controls, (k - 1) * model.nu, model.nu)])] : MX[]
        x = arc(MX[x; q]..., MX(t), MX(t_next))
        push!(states, part(x, 0, model.nx))
        push!(algebraic, part(x, model.nx, model.nz))
        push!(quadratures, part(x, model.nx + model.nz, model.nq))
        t = t_next
    end
    integrator_results(callee, inputs, [states, algebraic, quadratures], length(model.times), nadj, outer)
end

"""
    event_integrator(name, alg, rhs, zero, jump, problem, options) -> Julia function

Return the Julia function `(x0, p, u) -> path` that integrates an integrator of CasADi with
an event function with the SciML algorithm `alg` (see `solver_events`). The method is in
the extension for SciMLBase.jl.
"""
function event_integrator end

"""
    emitted_right_side(model) -> RightSide

Return the generated code of `right_side` for a solver, with the mass matrix of a model
with algebraic states and the Jacobian of the algebraic equations by those states.
"""
function emitted_right_side(model)
    emitted(g) = Emitted(build_function(g; expression = Val{false}, nonzeros = true), g, :nonzeros)
    rhs = right_side(model)
    model.nz == 0 && return RightSide(emitted(rhs))
    mass = [ones(model.nx); zeros(model.nz); ones(model.nq)]
    rows = findall(iszero, mass)
    u, p, t = MX.sym("u", length(mass)), MX.sym("p", model.np + model.nu), MX.sym("t")
    by_all = C.jacobian(C.vertcat(MX[rhs(u, p, t)[r] for r in rows]), u)
    slope = Front.Function(model.name * "_slope", [u, p, t], [C.densify(C.horzcat(MX[by_all[:, r] for r in rows]))])
    RightSide(emitted(rhs), mass, emitted(slope))
end

"""
    solver_events(callee, model, solver) -> CasADiCodegen.Function

Return the replacement of `solver_integrator` for an integrator `callee` of CasADi with an
event function: one call of the Julia function of `event_integrator`, which integrates to
all output times with the events (see `event_functions`). As in CasADi, a call throws after
more than `max_events` events, and the derivatives come from forward directions only.
"""
function solver_events(callee::C.Function, model, solver)
    label = "integrator \"$(model.name)\""
    @requires !isempty(methods(event_integrator)) "$label: a solver of the SciML ecosystem needs SciMLBase.jl; load it"
    transition = C.has_function(callee, "transition") ? C.get_function(callee, "transition") : nothing
    events = event_functions(model, transition)
    emitted(g) = Emitted(build_function(g; expression = Val{false}, nonzeros = true), g, :nonzeros)
    n, nt = model.nx + model.nz + model.nq, length(model.times)
    problem = (; model.t0, model.times, n, model.np, model.nu, model.events, max_events = serial_value(serial_bytes(callee), "Integrator::max_events"))
    run = event_integrator(model.name, solver.alg, emitted_right_side(model), emitted(events.zero), emitted(events.jump), problem,
                           (; model.tolerances..., solver.options...))
    found = julia_function(run, model.name * "_events", 3, Tuple{Vararg{Int}}[(n,), (model.np,), (model.nu * nt,)], Tuple{Vararg{Int}}[(n * nt,)], nothing)
    inputs = input_symbols(callee)
    dense(i) = C.vec(C.densify(inputs[i]))
    path = found(C.vertcat(MX[dense(1), dense(2), MX(C.DM(zeros(model.nq)))]), dense(3), dense(4))
    lists = [[part(path, (k - 1) * n + from, size) for k in 1:nt] for (from, size) in ((0, model.nx), (model.nx, model.nz), (model.nx + model.nz, model.nq))]
    integrator_results(callee, inputs, lists, nt, 0, 0)
end

"""
    solver_rootfinder(callee) -> CasADiCodegen.Function or nothing

Return a function with the inputs and the results of the rootfinder `callee` of CasADi
that finds the root with the solver that the user selected for it (see `Solvers`), through
`CasADiCodegen.rootfinder`, or `nothing` when no solver is selected. As in CasADi, the other
results are those of the function of the residual at the root.
"""
function solver_rootfinder(callee::C.Function)
    solver = selected_solver(callee, C.class_name(callee) == "Newton" ? :none : :nonlinear)
    solver === nothing && return nothing
    model = rootfinder_model(callee)
    finder = Front.rootfinder(model.name * "_root", solver.alg, residual_function(model); solver.options...)
    inputs = input_symbols(callee)
    others = C.vertcat(MX[nonzero_column(inputs[i+1]) for i in 0:length(inputs)-1 if i != model.iin])
    root = finder(nonzero_column(inputs[model.iin+1]), others)
    arguments = copy(inputs)
    arguments[model.iin+1] = piece(root, 0, C.sparsity_in(callee, model.iin))
    results = as_list(C.call(model.oracle, arguments, false, false))
    results[model.iout+1] = piece(root, 0, C.sparsity_out(callee, model.iout))
    same_interface(callee, inputs, results)
end

"""
    optimizer(name, alg, functions, sizes, options) -> Julia function

Return the Julia function `(x0, p, lbx, ubx, lbg, ubg) -> (x, lam_x, lam_g)` that solves an
optimization problem with the SciML algorithm `alg` (see `optimization_functions` and
`solver_optimization`). The method is in the extension for SciMLBase.jl.
"""
function optimizer end

"""
    initialization_problem(problem)

The initialization problem of a SciML problem; the method is in the extension for SciMLBase.jl.
"""
function initialization_problem end

"""
    GeneratedObjective

The supertype of the objective of a `SciMLBase.OptimizationFunction` that the extension for
SciMLBase.jl makes. The extension for OptimizationBase.jl declares the derivatives of such a
function, so that Optimization.jl does not ask for automatic differentiation.
"""
abstract type GeneratedObjective end

"""
    GeneratedLagrangeHessian

The supertype of the `lag_h(h, x, σ, μ, p)` of a `SciMLBase.OptimizationFunction` that the
extension for SciMLBase.jl makes. The extension for OptimizationBase.jl passes such a `lag_h`
and its `lag_hess_prototype` on to the solvers.
"""
abstract type GeneratedLagrangeHessian end

"""
    needs_hessians(alg) -> Bool

Return whether the SciML algorithm `alg` calls the Hessians of the objective or of the
constraints, from the traits of SciMLBase.jl. The method is in the extension for SciMLBase.jl.
"""
function needs_hessians end

"""
    optimization_functions(model; hessians = true) -> NamedTuple

Return generated code for the problem of a solver of CasADi (see `optimization_model`), as
functions on nonzeros of the unknowns `x` and the parameters `p`, which are column vectors:

- `cost`, `grad`, `hess`: the objective, its gradient and its Hessian;
- `cons`, `cons_j`: the constraints and their Jacobian;
- `cons_h`: one result per constraint, the Hessian of that constraint.

Each matrix is dense. With `hessians = false`, `hess` and `cons_h` are `nothing`.
"""
function optimization_functions(model; hessians::Bool = true)
    x, p = MX.sym("x", model.nx), MX.sym("p", model.np)
    arguments = MX[piece(x, 0, C.sparsity_in(model.objective, 0)), piece(p, 0, C.sparsity_in(model.objective, 1))]
    value(f) = column_of(as_list(C.call(f, arguments, false, false))[1])
    cost, cons = value(model.objective), value(model.constraints)
    function emitted(name, results)
        g = Front.Function(model.name * "_" * name, [x, p], map(C.densify, results))
        Emitted(build_function(g; nonzeros = true, expression = Val{false}), g, :nonzeros)
    end
    hessian(value) = Front.hessian(value, x)
    (; cost = emitted("cost", [cost]), grad = emitted("grad", [C.gradient(cost, x)]),
       hess = hessians ? emitted("hess", [hessian(cost)]) : nothing,
       cons = emitted("cons", [cons]), cons_j = emitted("cons_j", [C.jacobian(cons, x)]),
       cons_h = hessians ? emitted("cons_h", MX[hessian(cons[j]) for j in 1:model.ng]) : nothing)
end

"""
    conic_model(S::CasADiCodegen.Function) -> NamedTuple

Return the problem of the solver `S` that `conic` of CasADi made, in the form of
`optimization_model`: the objective `x' H x / 2 + g' x` and the constraints `A x` as
functions of the unknowns and of the parameters. The parameters are the nonzeros of `h`,
then `g`, then the nonzeros of `a`. Throws an `ArgumentError` for a problem with cone
constraints (the inputs `q` and `p` of `S`).
"""
function conic_model(S::C.Function)
    name = C.name(S)
    @requires C.nnz_in(S, 10) == 0 && C.nnz_in(S, 11) == 0 "solver \"$name\": cone constraints (the inputs q and p) are not supported"
    hs, as, nx = C.sparsity_in(S, 0), C.sparsity_in(S, 2), C.size1_in(S, 1)
    nh, na = C.nnz(hs), C.nnz(as)
    x, p = MX.sym("x", nx), MX.sym("p", nh + nx + na)
    H, g, A = piece(p, 0, hs), piece(p, nh, C.sparsity_in(S, 1)), piece(p, nh + nx, as)
    objective = Front.Function(name * "_f", [x, p], [C.bilin(H, x, x) / 2 + C.dot(g, x)])
    constraints = Front.Function(name * "_g", [x, p], [C.densify(C.mtimes(A, x))])
    (; name, objective, constraints, nx, np = nh + nx + na, ng = C.size1(as))
end

"""
    solver_conic(callee) -> CasADiCodegen.Function or nothing

Return a function with the inputs and the results of the solver `callee` that `conic` of
CasADi made, which solves the problem with the solver that the user selected for it (see
`Solvers`), or `nothing` when no solver is selected. The first guesses of the multipliers
are not used. CasADi has no derivatives of such a solver, so the replacement needs none.
"""
solver_conic

"""
    solver_optimization(callee) -> CasADiCodegen.Function or nothing

Return a function with the inputs and the results of the solver `callee` that `nlpsol` of
CasADi made, which solves the problem with the solver that the user selected for it (see
`Solvers`), or `nothing` when no solver is selected. The extension for SciMLBase.jl must be
loaded, and the unknowns and the parameters must be dense column vectors.

The Julia function of `optimizer` gives the solution and the multipliers of the bounds.
The objective, the constraints and the multipliers of the parameters are those of the
functions of the problem at the solution, as in CasADi. The first guesses of the
multipliers are not used. CasADi makes the derivatives of such a solver from its results,
so they need no solver of their own.
"""
function solver_conic(callee::C.Function)
    solver = selected_solver(callee, :optimization)
    solver === nothing && return nothing
    model = optimization_model(callee)
    label = "solver \"$(model.name)\""
    @requires !isempty(methods(optimizer)) "$label: a solver of the SciML ecosystem needs SciMLBase.jl; load it"
    nx, np, ng = model.nx, model.np, model.ng
    solve = optimizer(model.name, solver.alg, optimization_functions(model; hessians = needs_hessians(solver.alg)), (nx, np, ng),
                      solver.options)
    shapes_in, shapes_out = Tuple{Vararg{Int}}[(nx,), (np,), (nx,), (nx,), (ng,), (ng,)], Tuple{Vararg{Int}}[(nx,), (nx,), (ng,)]
    found = julia_function(solve, model.name * "_solve", 6, shapes_in, shapes_out, nothing)
    inputs = input_symbols(callee)
    h, g, a, lba, uba, lbx, ubx, x0 = inputs[1:8]
    data = C.vertcat(MX[nonzero_column(h), column_of(g), nonzero_column(a)])
    x, lam_x, lam_a = found(column_of(x0), data, column_of(lbx), column_of(ubx), column_of(lba), column_of(uba))
    cost = column_of(model.objective(x, data))
    results = MX[x, cost, lam_a, lam_x]
    same_interface(callee, inputs, MX[C.project(C.reshape(results[o+1], C.size1_out(callee, o), C.size2_out(callee, o)), C.sparsity_out(callee, o)) for o in 0:3])
end

function solver_optimization(callee::C.Function)
    solver = selected_solver(callee, :optimization)
    solver === nothing && return nothing
    model = optimization_model(callee)
    label = "solver \"$(model.name)\""
    @requires !isempty(methods(optimizer)) "$label: a solver of the SciML ecosystem needs SciMLBase.jl; load it"
    @requires all(C.is_dense(C.sparsity_in(callee, i)) && C.size2_in(callee, i) <= 1 for i in 0:1) "$label: the unknowns and the parameters must be dense column vectors"
    nx, np, ng = model.nx, model.np, model.ng
    solve = optimizer(model.name, solver.alg, optimization_functions(model; hessians = needs_hessians(solver.alg)), (nx, np, ng),
                      solver.options)
    shapes_in, shapes_out = Tuple{Vararg{Int}}[(nx,), (np,), (nx,), (nx,), (ng,), (ng,)], Tuple{Vararg{Int}}[(nx,), (nx,), (ng,)]
    found = julia_function(solve, model.name * "_solve", 6, shapes_in, shapes_out, nothing)
    inputs = input_symbols(callee)
    x, lam_x, lam_g = found(map(column_of, inputs[1:6])...)
    at = MX[x, inputs[2]]
    cost, cons = (as_list(C.call(f, at, false, false))[1] for f in (model.objective, model.constraints))
    # The multipliers of the parameters are the gradient of the Lagrange function by them.
    w, q, mu = MX.sym("x", nx), MX.sym("p", np), MX.sym("mu", ng)
    lagrange = column_of(model.objective(w, q)) + (ng == 0 ? 0 : C.dot(mu, column_of(model.constraints(w, q))))
    by_parameters = Front.Function(model.name * "_lam_p", [w, q, mu], [C.densify(C.gradient(lagrange, q))])
    results = MX[x, cost, cons, lam_x, lam_g, by_parameters(x, inputs[2], lam_g)]
    replaced = same_interface(callee, inputs, MX[C.project(C.reshape(column_of(results[o+1]), C.size1_out(callee, o), C.size2_out(callee, o)), C.sparsity_out(callee, o)) for o in 0:5])
    @ensures all(sparsity_tuple(C.sparsity_out(replaced, o)) == sparsity_tuple(C.sparsity_out(callee, o)) for o in 0:C.n_out(callee)-1) "the replacement of $label must have its results"
    replaced
end
