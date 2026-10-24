# Switches in a right side. CasADi differentiates an operation such as `if_else` or `sign`
# branch by branch, so the derivatives of an integration through a switch that moves with
# the state miss the jump of the STM at the switching time. The scan here finds such
# operations in the graph of a function, and the integrators of this package and of CasADi
# warn about them when they are built.

"""
The operations that switch at a surface of their operand, with the kind of the switch:
`:jump` of the value, which spoils the first derivatives of a flow through it, or `:kink`
of the slope, which spoils the derivatives of order 2 and higher. `OP_COPYSIGN` is apart
(see `record_switch!`). The equality tests, the logical operations and `OP_IF_ELSE_ZERO`
switch on a set of measure zero of a plain value, or combine comparisons that are counted
already, so they are not here.
"""
const NONSMOOTH = Dict{Symbol,Symbol}(
    :OP_LT => :jump, :OP_LE => :jump, :OP_SIGN => :jump,
    :OP_FLOOR => :jump, :OP_CEIL => :jump, :OP_FMOD => :jump, :OP_REMAINDER => :jump,
    :OP_FABS => :kink, :OP_FMIN => :kink, :OP_FMAX => :kink, :OP_LOW => :kink,
    :OP_NORM1 => :kink, :OP_NORMINF => :kink, :OP_MMIN => :kink, :OP_MMAX => :kink)

"One switching surface: the operation, its kind, and the inputs of the scanned function on which its operand depends."
const Switching = NamedTuple{(:op, :kind, :inputs),Tuple{Symbol,Symbol,BitSet}}

const NO_INPUTS = BitSet()

"The union of the sets `sets`, without a copy when they are one set."
function joined(sets)
    isempty(sets) && return NO_INPUTS
    first_set = first(sets)
    all(s -> s === first_set || s == first_set, sets) ? first_set : union(NO_INPUTS, sets...)
end

"Add to `found` the switching surfaces of the operation `op` on operands that depend on the inputs `operands`."
function record_switch!(found::Vector{Switching}, op::Symbol, operands)
    if op === :OP_COPYSIGN
        # copysign(a, b) jumps where b changes its sign, and has a kink where a does.
        isempty(operands[2]) || push!(found, (op = op, kind = :jump, inputs = operands[2]))
        isempty(operands[1]) || push!(found, (op = op, kind = :kink, inputs = operands[1]))
    elseif haskey(NONSMOOTH, op)
        inputs = joined(operands)
        isempty(inputs) || push!(found, (op = op, kind = NONSMOOTH[op], inputs = inputs))
    end
    found
end

"""
    nonsmooth_findings(f::CasADiCodegen.Function; expand = true) -> Vector{Switching}

Return the switching surfaces in the graph of the SX or MX function `f`: for each, the
operation, its kind (see `NONSMOOTH`) and the set of the input places of `f`, from 1, on
which its operand depends. A function that `f` calls is scanned with the sets of its
arguments. An MX function is first expanded to scalar code when `expand` is true and
CasADi can expand it; else the scan reads its matrix nodes, and a set then holds the inputs
of the whole matrix. The body of a Julia callback, an external function or a solver object
is not scanned: each of its results depends on all its arguments.
"""
function nonsmooth_findings(f::C.Function; expand::Bool = true)
    @requires C.is_a(f, "SXFunction") || C.is_a(f, "MXFunction") "function \"$(C.name(f))\" is not an SX or MX function"
    sets = [BitSet(i) for i in 1:C.n_in(f)]
    found = Switching[]
    expanded = expand && C.is_a(f, "MXFunction") ? scalar_function(f) : nothing
    scan_function!(found, expanded === nothing ? f : expanded, sets, Dict{Any,Any}())
    @ensures all(s -> issubset(s.inputs, 1:C.n_in(f)), found) "function \"$(C.name(f))\": a switching surface depends on an input that the function does not have"
    found
end

"""
    scalar_function(f) -> CasADiCodegen.Function or nothing

The expansion of the MX function `f` to an SX function, or `nothing` when CasADi cannot
expand it, or when the expansion still calls a function (an interpolant, for example),
whose body the scalar scan cannot read.
"""
function scalar_function(f::C.Function)
    g = try
        C.expand(f)
    catch err
        err isa InterruptException && rethrow()
        return nothing
    end
    any(k -> C.instruction_id(g, k) == C.OP_CALL, 0:C.n_instructions(g)-1) ? nothing : g
end

"""
The operations whose result is a truth: a comparison or a logical operation. The index of a
`Switch` that is a truth, as that of a short-circuit `if_else`, switches where the
comparison does, which is counted already.
"""
const TRUTHS = (:OP_LT, :OP_LE, :OP_EQ, :OP_NE, :OP_AND, :OP_OR, :OP_NOT)

"""
    scan_function!(found, g, sets, cache; truth = false) -> Vector{BitSet}

Add to `found` the switching surfaces of a call of the function `g` whose arguments
depend on the inputs `sets`, one set per argument, and return the set of each result.
`cache` holds the result of each pair of a function and its sets, so that repeated calls
are read once. `truth` tells that the first argument is a truth (see `TRUTHS`); a `Switch`
then does not count its index again.
"""
function scan_function!(found::Vector{Switching}, g::C.Function, sets::Vector{BitSet}, cache::Dict{Any,Any}; truth::Bool = false)
    key = (C.__hash__(g), sets, truth)
    if haskey(cache, key)
        inner, results = cache[key]
        append!(found, inner)
        return results
    end
    inner = Switching[]
    class = C.class_name(g)
    results = if C.is_a(g, "SXFunction")
        scan_scalar!(inner, g, sets)
    elseif C.is_a(g, "MXFunction")
        scan_matrix!(inner, g, sets, cache)
    elseif class == "Switch"
        # The index selects a branch: a jump, unless the comparison that gives it is counted. Each branch is scanned with the other arguments.
        index, arguments = sets[1], sets[2:end]
        truth || isempty(index) || push!(inner, (op = :Switch, kind = :jump, inputs = index))
        info = C.info(g)
        branches = [b for b in [info["f"]; info["f_def"]] if b isa C.Function && !C.is_null(b)]
        outcomes = [scan_function!(inner, b, arguments, cache) for b in branches]
        [joined([[index]; [o[j] for o in outcomes]]) for j in 1:C.n_out(g)]
    elseif class in ("Map", "ThreadMap", "OmpMap")
        scan_function!(inner, C.info(g)["f"]::C.Function, sets, cache; truth)
    else
        class in ("LinearInterpolant", "LinearInterpolantJac") && !isempty(sets[1]) &&
            push!(inner, (op = :LinearInterpolant, kind = :kink, inputs = sets[1]))
        fill(joined(sets), C.n_out(g))
    end
    cache[key] = (inner, results)
    append!(found, inner)
    results
end

"Scan the instructions of the SX function `g` (see `scan_function!`)."
function scan_scalar!(found::Vector{Switching}, g::C.Function, sets::Vector{BitSet})
    names = op_names()
    work = fill(NO_INPUTS, C.sz_w(g))
    results = fill(NO_INPUTS, C.n_out(g))
    for k in 0:C.n_instructions(g)-1
        # Loop invariant: each slot of `work` written so far holds the inputs on which its value depends.
        op = names[C.instruction_id(g, k)]
        i, o = C.instruction_input(g, k), C.instruction_output(g, k)
        if op === :OP_INPUT
            work[o[1]+1] = sets[i[1]+1]
        elseif op === :OP_CONST
            work[o[1]+1] = NO_INPUTS
        elseif op === :OP_OUTPUT
            results[o[1]+1] = joined((results[o[1]+1], work[i[1]+1]))
        else
            operands = BitSet[work[s+1] for s in i if s >= 0]
            record_switch!(found, op, operands)
            value = joined(operands)
            foreach(s -> s >= 0 && (work[s+1] = value), o)
        end
    end
    results
end

"Scan the instructions of the MX function `g`, one set per matrix value (see `scan_function!`)."
function scan_matrix!(found::Vector{Switching}, g::C.Function, sets::Vector{BitSet}, cache::Dict{Any,Any})
    names = op_names()
    work = Dict{Int,BitSet}()
    truths = Set{Int}()                     # the slots of `work` that hold a truth
    results = fill(NO_INPUTS, C.n_out(g))
    for k in 0:C.n_instructions(g)-1
        # Loop invariant: each slot of `work` written so far holds the inputs on which its matrix depends, and `truths` those that hold a truth.
        op = names[C.instruction_id(g, k)]
        i, o = C.instruction_input(g, k), C.instruction_output(g, k)
        if op === :OP_OUTPUT
            results[o[1]+1] = joined((results[o[1]+1], work[i[1]]))
        elseif op === :OP_INPUT
            work[o[1]] = sets[Int(C.info(C.instruction_MX(g, k))["ind"])+1]
            delete!(truths, o[1])
        else
            operands = BitSet[get(work, s, NO_INPUTS) for s in i]
            if op === :OP_CALL
                callee = C.which_function(C.instruction_MX(g, k))
                truth = !isempty(i) && i[1] in truths
                called = length(operands) == C.n_in(callee) ? scan_function!(found, callee, operands, cache; truth) : fill(joined(operands), length(o))
                foreach(((j, s),) -> s >= 0 && (work[s] = called[j]; delete!(truths, s)), enumerate(o))
            else
                record_switch!(found, op, operands)
                value = joined(operands)
                foreach(s -> s >= 0 && (work[s] = value; op in TRUTHS ? push!(truths, s) : delete!(truths, s)), o)
            end
        end
    end
    results
end

"The places from 1 of the inputs `inputs` of `f`: all with `nothing`, else given by place or by name, one or a collection."
function input_places(f::C.Function, inputs)
    n = C.n_in(f)
    inputs === nothing && return collect(1:n)
    names = [C.name_in(f, i) for i in 0:n-1]
    map(inputs isa Union{AbstractString,Symbol,Integer} ? [inputs] : collect(inputs)) do input
        if input isa Union{AbstractString,Symbol}
            place = findfirst(==(String(input)), names)
            @requires place !== nothing "function \"$(C.name(f))\" has no input \"$input\"; its inputs are $(join(repr.(names), ", "))"
            place
        else
            @requires input isa Integer && 1 <= input <= n "function \"$(C.name(f))\" has $n inputs; the input $(repr(input)) is not one of them"
            Int(input)
        end
    end
end

"""
    nonsmooth_operations(f; inputs = nothing, expand = true) -> Vector{Pair{Symbol,Int}}

Count the switching surfaces in the CasADi function `f` whose operand depends on `inputs`,
by operation. Return the pairs `operation => count`, the most frequent first. An operation
is CasADi's operation code, such as `:OP_SIGN`, or the class of a called function.

A switching surface is an operation whose derivative CasADi takes branch by branch:

- a jump of the value: `<` and `>` (both `:OP_LT`), `<=` and `>=` (`:OP_LE`), `sign`,
  `floor`, `ceil`, `fmod`, `remainder`, the sign operand of `copysign` (`:OP_COPYSIGN`),
  and the index of `conditional` (`:Switch`);
- a kink of the slope: `abs` (`:OP_FABS`), `fmin`, `fmax`, the magnitude operand of
  `copysign`, the MX `norm_1`, `norm_inf`, `mmin` and `mmax`, CasADi's `low`, and a
  `linear` interpolant (`:LinearInterpolant`).

Equality tests and logical operations are not counted. An `if_else` or a `conditional`
whose condition is a comparison counts once, at the comparison. The integrators warn about
a surface whose operand depends on the state, or on the time and a parameter together (see
[Discontinuous models](@ref discontinuities)).

`inputs` is `nothing` for all inputs, an input place from 1 or an input name, or a
collection of them. With `expand = true` an MX function is first expanded to scalar code
when CasADi can expand it; otherwise a matrix counts as depending on every input of any of
its elements. The bodies of Julia callbacks and external functions are not scanned.

Throws `ArgumentError` for an input that `f` does not have.

# Examples
```jldoctest
julia> using CasADiCodegen

julia> x, p = SX.sym("x", 2), SX.sym("p");

julia> f = CasadiFunction("f", [x, p], [sign(x[1]) + min(p, 2) * x[2]], ["x", "p"], ["r"]);

julia> CasADiCodegen.nonsmooth_operations(f; inputs = ["x"])
1-element Vector{Pair{Symbol, Int64}}:
 :OP_SIGN => 1
```
"""
function nonsmooth_operations(f::C.Function; inputs = nothing, expand::Bool = true)
    places = input_places(f, inputs)
    counts = Dict{Symbol,Int}()
    for s in nonsmooth_findings(f; expand)
        any(in(places), s.inputs) && (counts[s.op] = get(counts, s.op, 0) + 1)
    end
    sort!(collect(counts); by = entry -> (-entry.second, entry.first))
end

"""
    with_outputs(f, places) -> CasADiCodegen.Function

An MX function with the inputs of `f` whose results are the results `places` of `f`, from
0, with their names.
"""
function with_outputs(f::C.Function, places)
    inputs = MX[MX.sym(C.name_in(f, i), C.sparsity_in(f, i)) for i in 0:C.n_in(f)-1]
    results = as_list(C.call(f, inputs, false, false))
    Front.Function(C.name(f), inputs, results[collect(places).+1], [C.name_in(f, i) for i in 0:C.n_in(f)-1], [C.name_out(f, o) for o in places])
end

const NONSMOOTH_SETTINGS = (:warn, :error, :ignore)

check_setting(label, nonsmooth) =
    @requires nonsmooth in NONSMOOTH_SETTINGS "$label: nonsmooth must be :warn, :error or :ignore; got $(repr(nonsmooth))"

plural(n, word) = n == 1 ? "1 $word" : "$n $(word)s"

"The text of the switching surfaces `kept` of one result, by operation: \"OP_LT: 1 jump, OP_FABS: 2 kinks\"."
function surface_counts(kept)
    ops = unique(s.op for s in kept)
    join(map(ops) do op
        jumps, kinks = count(s -> s.op === op && s.kind === :jump, kept), count(s -> s.op === op && s.kind === :kink, kept)
        "$op: " * join([plural(n, w) for (n, w) in ((jumps, "jump"), (kinks, "kink")) if n > 0], " and ")
    end, ", ")
end

"The remedies that the warning of an integrator of this package names."
const JULIA_REMEDY = "Give the switch with the keyword `modes`, smooth it with `CasADiCodegen.smoothed`, or pass `nonsmooth = :ignore`."

"""
    check_nonsmooth(label, f, states, times, parameters; nonsmooth, exclude = Int[], remedy = JULIA_REMEDY)

Warn, or throw `ArgumentError`, when a result of `f`, the right side of the integrator
`label`, switches on the state. `states`, `times` and `parameters` are input places of `f`
from 1. A switching surface counts when its operand depends on a state, or on both the
time and a parameter; a surface at a fixed time does not move. The places in `exclude` do
not count. `nonsmooth` is `:warn`, `:error` or `:ignore`, which does not scan.
"""
function check_nonsmooth(label::AbstractString, f::C.Function, states, times, parameters; nonsmooth::Symbol,
                         exclude = Int[], remedy::AbstractString = JULIA_REMEDY)
    check_setting(label, nonsmooth)
    nonsmooth === :ignore && return nothing
    moves(inputs) = (s = setdiff(inputs, exclude); any(in(states), s) || (any(in(times), s) && any(in(parameters), s)))
    parts, total, any_jump, any_kink, on_state = String[], 0, false, false, false
    for o in 0:C.n_out(f)-1
        C.nnz_out(f, o) == 0 && continue
        kept = filter(s -> moves(s.inputs), nonsmooth_findings(C.n_out(f) == 1 ? f : with_outputs(f, [o])))
        isempty(kept) && continue
        total += length(kept)
        any_jump |= any(s -> s.kind === :jump, kept)
        any_kink |= any(s -> s.kind === :kink, kept)
        on_state |= any(s -> any(in(states), setdiff(s.inputs, exclude)), kept)
        push!(parts, "\"$(C.name_out(f, o))\" at $(plural(length(kept), "surface")) ($(surface_counts(kept)))")
    end
    total == 0 && return nothing
    consequence = any_jump ?
        "Values are right, but derivatives by the initial state and the parameters miss the jump of the state transition matrix at the switching time" *
        (any_kink ? " (at a kink, only derivatives of order 2 and higher)." : ".") :
        "Values and first derivatives are right, but derivatives of order 2 and higher by the initial state and the parameters miss the terms of the switching time."
    message = "$label switches on $(on_state ? "the state" : "the time and the parameters"): $(join(parts, "; ")). $consequence $remedy " *
              "A switch whose branches agree at the surface, such as the guard of a removable singularity, is reported too."
    nonsmooth === :error && throw(ArgumentError(message))
    @warn message
    nothing
end

"""
    check_model(name, dae, nonsmooth; modes = nothing)

Run `check_nonsmooth` on the model dictionary `dae` of an integrator of this package: the
right side `"ode"` and the integrands `"quad"` as functions of the state, the parameters
and the time. `modes` is `nothing`, or the column of modes of a switched model (see
`mode_integrator`), on which a switch does not count.
"""
function check_model(name::AbstractString, dae, nonsmooth::Symbol; modes = nothing)
    label = "integrator \"$name\""
    check_setting(label, nonsmooth)
    nonsmooth === :ignore && return nothing
    keys_in = filter(key -> haskey(dae, key), ["x", "p", "t"])
    outputs = filter(key -> haskey(dae, key), ["ode", "quad"])
    symbols = typeof(dae["x"])[dae[key] for key in keys_in]
    modes === nothing || push!(symbols, modes)
    f = Front.Function("rhs", symbols, [Front.densify(dae[key]) for key in outputs], modes === nothing ? keys_in : [keys_in; "modes"], outputs)
    place(key) = findall(==(key), keys_in)
    check_nonsmooth(label, f, place("x"), place("t"), place("p"); nonsmooth, exclude = modes === nothing ? Int[] : [length(symbols)])
end

"The remedies that the warning of an integrator of CasADi names."
const CASADI_REMEDY = "Write the switch as an event function (\"zero\" and \"transition\"), smooth it with `CasADiCodegen.smoothed`, or pass the option \"nonsmooth\" => \"ignore\"."
const REPLACED_REMEDY = "Write the switch as an event function (\"zero\" and \"transition\"), smooth it with `CasADiCodegen.smoothed`, or pass `nonsmooth = :ignore` to `CasADiCodegen.Solvers`."

"""
    check_casadi_model(label, dae::CasADiCodegen.Function; nonsmooth, remedy)

Run `check_nonsmooth` on the DAE function `dae` of an integrator of CasADi, whose inputs
are named `"t"`, `"x"`, `"z"`, `"p"` and `"u"`, and whose results `"ode"`, `"alg"` and
`"quad"` are scanned.
"""
function check_casadi_model(label::AbstractString, dae::C.Function; nonsmooth::Symbol, remedy::AbstractString)
    check_setting(label, nonsmooth)
    nonsmooth === :ignore && return nothing
    names_in = [C.name_in(dae, i) for i in 0:C.n_in(dae)-1]
    place(keys...) = findall(in(keys), names_in)
    outputs = [o for o in 0:C.n_out(dae)-1 if C.name_out(dae, o) in ("ode", "alg", "quad")]
    check_nonsmooth(label, with_outputs(dae, outputs), place("x", "z"), place("t"), place("p", "u"); nonsmooth, remedy)
end

"""
    check_native_integrator(name, dae, options) -> options

Read the option `"nonsmooth"` (`"warn"`, the default, `"error"` or `"ignore"`) of the
integrator `name` of CasADi, check the model `dae` (a dictionary of SX or MX, or a
function) with it, and return the options without that key, which CasADi does not know.
The front end calls it first in each method of `integrator` with a plugin name.
"""
function check_native_integrator(name::AbstractString, dae, options)
    label = "integrator \"$name\""
    given = options isa AbstractDict ? [k for k in keys(options) if string(k) == "nonsmooth"] : []
    setting = isempty(given) ? :warn : Symbol(string(options[only(given)]))
    @requires setting in NONSMOOTH_SETTINGS "$label: the option \"nonsmooth\" must be \"warn\", \"error\" or \"ignore\"; got $(repr(options[only(given)]))"
    rest = isempty(given) ? options : Dict{String,Any}(string(k) => v for (k, v) in options if string(k) != "nonsmooth")
    setting === :ignore && return rest
    model = casadi_model(dae)
    model === nothing || check_casadi_model(label, model; nonsmooth = setting, remedy = CASADI_REMEDY)
    rest
end

"""
    casadi_model(dae) -> CasADiCodegen.Function or nothing

The DAE function of the model `dae` of an integrator of CasADi, with the inputs `"t"`,
`"x"`, `"z"`, `"p"`, `"u"` and the results `"ode"`, `"alg"`, `"quad"` that it has; `nothing`
when `dae` is neither a function nor a dictionary of SX or MX, or when CasADi cannot make
the function, an error that CasADi reports when it builds the integrator.
"""
casadi_model(dae::C.Function) = dae
function casadi_model(dae::AbstractDict)
    entries = Dict(string(k) => v for (k, v) in dae)
    x = get(entries, "x", nothing)
    x isa Symbolic || return nothing
    keys_in = filter(key -> haskey(entries, key) && entries[key] isa typeof(x), ["t", "x", "z", "p", "u"])
    outputs = filter(key -> haskey(entries, key) && entries[key] isa typeof(x), ["ode", "alg", "quad"])
    isempty(outputs) && return nothing
    try
        Front.Function("dae", [entries[key] for key in keys_in], [entries[key] for key in outputs], keys_in, outputs)
    catch err
        err isa InterruptException && rethrow()
        nothing
    end
end
casadi_model(dae) = nothing
