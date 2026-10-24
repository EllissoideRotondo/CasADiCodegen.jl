# Smooth forms of the operations that switch: a step, the sign, the absolute value, the
# maximum and the minimum of a switching value, over a layer of a given width.

const SMOOTH_KINDS = (:tanh, :algebraic)

function check_smooth(width, kind::Symbol)
    @requires kind in SMOOTH_KINDS "the kind of a smooth switch must be :tanh or :algebraic; got $(repr(kind))"
    if width isa Real
        @requires width > 0 "the width of a smooth switch must be positive; got $width"
    else
        @requires width isa Symbolic && C.is_scalar(width) "the width of a smooth switch must be a positive number or a scalar SX or MX; got $(typeof(width))"
    end
end

"The arguments as numbers, or all as the symbol type among them (MX before SX)."
function symbolic_arguments(arguments...)
    has_sx, has_mx = any(a -> a isa SX, arguments), any(a -> a isa MX, arguments)
    @requires !(has_sx && has_mx) "a smooth switch takes SX or MX, not both"
    T = has_mx ? MX : has_sx ? SX : nothing
    @requires all(a -> a isa Real || a isa Symbolic, arguments) "a smooth switch takes numbers, SX or MX; got $(join(map(typeof, arguments), ", "))"
    T === nothing ? arguments : map(a -> a isa T ? a : convert(T, Float64(a)), arguments)
end

product(a::Real, b::Real) = a * b
product(a::Symbolic, b::Symbolic) = C.times(a, b)

sign_value(s::Real, width::Real, kind::Symbol) = kind === :tanh ? tanh(s / width) : s / hypot(s, width)
sign_value(s::Symbolic, width::Symbolic, kind::Symbol) = kind === :tanh ? C.tanh(C.rdivide(s, width)) : C.rdivide(s, C.hypot(s, width))

abs_value(s, width, kind) = product(s, sign_value(s, width, kind))
max_value(a, b, width, kind) = (a + b + abs_value(a - b, width, kind)) / 2
min_value(a, b, width, kind) = (a + b - abs_value(a - b, width, kind)) / 2

"""
    smooth_step(s, width; kind = :tanh)
    smooth_sign(s, width; kind = :tanh)
    smooth_abs(s, width; kind = :tanh)
    smooth_max(a, b, width; kind = :tanh)
    smooth_min(a, b, width; kind = :tanh)

Smooth forms of a step, `sign`, `abs`, `max` and `min` of the switching value `s`, or of
`a - b`, over a layer of width `width` around zero. Outside the layer each form tends to
the operation that it replaces; inside it the form is smooth, so that the derivatives of a
flow through it include the switch.

The arguments are numbers (any `Real`, dual numbers included), or SX or MX expressions,
on which the forms act element by element. `width` is a positive number or a scalar
symbol, so that it can be a parameter of the model and fall in a continuation. A number
among symbolic arguments becomes a constant of their type.

| Function            | `kind = :tanh`                | `kind = :algebraic`  |
|:------------------- |:----------------------------- |:-------------------- |
| `smooth_sign(s, ε)` | `tanh(s/ε)`                   | `s / hypot(s, ε)`    |
| `smooth_step(s, ε)` | `(1 + smooth_sign(s, ε)) / 2` | same                 |
| `smooth_abs(s, ε)`  | `s * smooth_sign(s, ε)`       | same                 |
| `smooth_max(a, b, ε)` | `(a + b + smooth_abs(a - b, ε)) / 2` | same       |
| `smooth_min(a, b, ε)` | `(a + b - smooth_abs(a - b, ε)) / 2` | same       |

So `smooth_step(0, ε) == 1/2`, `smooth_abs(0, ε) == 0` and `smooth_max(a, a, ε) == a`.
The error of `smooth_abs` against `abs` is at most `0.28ε` with `:tanh` and `0.31ε` with
`:algebraic`. The tails of `tanh` decay exponentially and those of the algebraic kind as
`ε²/s²`, so for a flow that crosses the surface the `:tanh` kind converges faster as `ε`
falls.

Throws `ArgumentError` for a `width` that is not positive, a `kind` other than `:tanh` and
`:algebraic`, or SX and MX arguments together.

# Examples
```jldoctest
julia> using CasADiCodegen

julia> CasADiCodegen.smooth_sign(0.5, 0.1) ≈ tanh(5)
true

julia> x, width = SX.sym("x"), SX.sym("width");

julia> CasADiCodegen.smooth_step(x, width)
1×1 SX:
((1+tanh((x/width)))/2)
```
"""
function smooth_step(s, width; kind::Symbol = :tanh)
    check_smooth(width, kind)
    s, width = symbolic_arguments(s, width)
    (1 + sign_value(s, width, kind)) / 2
end

"""
    smooth_sign(s, width; kind = :tanh)

Smooth form of `sign(s)` over a layer of width `width`: `tanh(s/width)`, or
`s / hypot(s, width)` with `kind = :algebraic`. See [`smooth_step`](@ref).
"""
function smooth_sign(s, width; kind::Symbol = :tanh)
    check_smooth(width, kind)
    sign_value(symbolic_arguments(s, width)..., kind)
end

"""
    smooth_abs(s, width; kind = :tanh)

Smooth form of `abs(s)` over a layer of width `width`: `s * smooth_sign(s, width; kind)`.
See [`smooth_step`](@ref).
"""
function smooth_abs(s, width; kind::Symbol = :tanh)
    check_smooth(width, kind)
    abs_value(symbolic_arguments(s, width)..., kind)
end

"""
    smooth_max(a, b, width; kind = :tanh)

Smooth form of `max(a, b)` over a layer of width `width` in `a - b`:
`(a + b + smooth_abs(a - b, width; kind)) / 2`. See [`smooth_step`](@ref).
"""
function smooth_max(a, b, width; kind::Symbol = :tanh)
    check_smooth(width, kind)
    max_value(symbolic_arguments(a, b, width)..., kind)
end

"""
    smooth_min(a, b, width; kind = :tanh)

Smooth form of `min(a, b)` over a layer of width `width` in `a - b`:
`(a + b - smooth_abs(a - b, width; kind)) / 2`. See [`smooth_step`](@ref).
"""
function smooth_min(a, b, width; kind::Symbol = :tanh)
    check_smooth(width, kind)
    min_value(symbolic_arguments(a, b, width)..., kind)
end

# The graph transform `smoothed`: each operation that switches on the chosen inputs becomes
# its smooth form. A value of the graph carries three flags: it depends on the chosen inputs
# (SELECTED), it depends on the result of a rewritten operation (SOFT), or it is a soft
# condition in (0, 1), written by a rewritten comparison or by a logical operation on such
# conditions (TRUTH). A TRUTH value is SOFT, and a SOFT value is SELECTED.

const SELECTED, SOFT, TRUTH = 0x01, 0x02, 0x04

"Operations that copy or rearrange values, so that a matrix of conditions stays one."
const TRUTH_PRESERVING = (:OP_ASSIGN, :OP_GETNONZEROS, :OP_RESHAPE, :OP_TRANSPOSE, :OP_HORZCAT, :OP_VERTCAT, :OP_DIAGCAT,
                          :OP_HORZSPLIT, :OP_VERTSPLIT, :OP_DIAGSPLIT, :OP_SUBREF, :OP_SPARSITY_CAST)

"Operations that `smoothed` keeps, and that would move their switch on a soft operand."
const KEPT_SWITCHES = (:OP_EQ, :OP_NE, :OP_FLOOR, :OP_CEIL, :OP_FMOD, :OP_REMAINDER, :OP_LOW,
                       :OP_IF_ELSE_ZERO, :OP_NOT, :OP_AND, :OP_OR)

"The state of one run of `smoothed`: the kind, the switches kept on the chosen inputs, the transformed callees, and the count of rewrites."
mutable struct Smoothing
    const kind::Symbol
    const kept::Set{Symbol}
    const callees::Dict{Any,Any}
    rewrites::Int
end
Smoothing(kind::Symbol) = Smoothing(kind, Set{Symbol}(), Dict{Any,Any}(), 0)

selected(flag::UInt8) = flag & SELECTED != 0
soft(flag::UInt8) = flag & SOFT != 0
truth(flag::UInt8) = flag & TRUTH != 0

"The elements of the MX `a` as scalars, structural zeros included, in column order."
elements(a::MX) = (v = C.vec(C.densify(a)); MX[v[k] for k in 1:C.numel(v)])

"""
    smooth_operation(op, operands, flags, width, s, T) -> (value, flag) or nothing

The smooth replacement of the operation `op` of the type `T` (SX or MX) on `operands`,
whose flags are `flags`, and the flag of the result; `nothing` when `op` is kept.
"""
function smooth_operation(op::Symbol, operands::Vector, flags::Vector{UInt8}, width, s::Smoothing, ::Type{T}) where {T}
    kind, on = s.kind, any(selected, flags)
    a = operands[1]
    value, flag = if on && op in (:OP_LT, :OP_LE)
        (1 + sign_value(operands[2] - a, width, kind)) / 2, SELECTED | SOFT | TRUTH
    elseif on && op === :OP_SIGN
        sign_value(a, width, kind), SELECTED | SOFT
    elseif on && op === :OP_FABS
        abs_value(a, width, kind), SELECTED | SOFT
    elseif on && op === :OP_FMAX
        max_value(a, operands[2], width, kind), SELECTED | SOFT
    elseif on && op === :OP_FMIN
        min_value(a, operands[2], width, kind), SELECTED | SOFT
    elseif on && op === :OP_COPYSIGN
        product(abs_value(a, width, kind), sign_value(operands[2], width, kind)), SELECTED | SOFT
    elseif on && op === :OP_NORM1
        C.sum1(C.vec(abs_value(a, width, kind))), SELECTED | SOFT
    elseif on && op in (:OP_NORMINF, :OP_MMAX, :OP_MMIN) && C.numel(a) > 0
        terms = elements(op === :OP_NORMINF ? abs_value(a, width, kind) : a)
        foldl((m, e) -> op === :OP_MMIN ? min_value(m, e, width, kind) : max_value(m, e, width, kind), terms), SELECTED | SOFT
    elseif op in (:OP_NOT, :OP_AND, :OP_OR) && any(truth, flags)
        @requires !any(f -> soft(f) && !truth(f), flags) "smoothed: $op of a condition and of a smoothed value that is not a condition; the value is never exactly 0, so the switch would move"
        # A plain operand is 0 or not 0, as a logical operation reads it.
        c = [truth(f) ? v : C.binary(T, C.OP_NE, v, T(0.0)) for (v, f) in zip(operands, flags)]
        (op === :OP_NOT ? 1 - c[1] : op === :OP_AND ? product(c[1], c[2]) : c[1] + c[2] - product(c[1], c[2])), SELECTED | SOFT | TRUTH
    elseif op === :OP_IF_ELSE_ZERO && truth(flags[1])
        # Where the soft condition is exactly 0, the branch stays out, even when it is NaN.
        C.binary(T, C.OP_IF_ELSE_ZERO, a, product(a, operands[2])), (flags[2] & SELECTED) | SELECTED | SOFT
    else
        return nothing
    end
    s.rewrites += 1
    value, flag
end

"""
    kept_operation!(op, flags, s) -> flag

Check the operation `op` that `smoothed` keeps, on operands with the flags `flags`, and
return the flag of its result. A switch on a soft operand throws `ArgumentError`, since a
soft value is never exactly 0 or an integer and the switch would move. A switch on the
chosen inputs is recorded in `s.kept`.
"""
function kept_operation!(op::Symbol, flags::Vector{UInt8}, s::Smoothing)
    isempty(flags) && return 0x00
    if op in KEPT_SWITCHES
        interest = op === :OP_IF_ELSE_ZERO ? flags[1:1] : flags
        @requires !any(soft, interest) "smoothed: $op of a smoothed value that is not a condition; the value is never exactly 0 or an integer, so the switch would move. Write the test on the switching function itself."
        haskey(NONSMOOTH, op) && any(selected, interest) && push!(s.kept, op)
    end
    flag = reduce(|, flags) & (SELECTED | SOFT)
    op in TRUTH_PRESERVING && all(truth, flags) ? flag | TRUTH : flag
end

"""
    scalar_symbols(name, sparsity) -> (symbol, nonzeros)

A symbolic SX input of the pattern `sparsity` whose nonzeros are the scalar symbols
`nonzeros`, so that a replay of an SX function reads its nonzeros by place.
"""
function scalar_symbols(name::AbstractString, sparsity::C.Sparsity)
    n = C.nnz(sparsity)
    nonzeros = SX[SX.sym("$(name)_$j") for j in 0:n-1]
    (n == 0 ? SX.sym(name, sparsity) : SX(sparsity, C.vertcat(nonzeros))), nonzeros
end

"""
    smooth_scalar(g, nonzeros, flags, width, s) -> (results, flags)

Replay the instructions of the SX function `g` on the scalar expressions `nonzeros`, one
vector per input, with each switch replaced by its smooth form of width `width`. Return the
results with the sparsity patterns of `g`, and the flag of each. A value that `g` shares
stays shared, since each work slot becomes one expression.
"""
function smooth_scalar(g::C.Function, nonzeros::Vector{Vector{SX}}, flags_in::Vector{UInt8}, width::SX, s::Smoothing)
    @requires length(nonzeros) == C.n_in(g) == length(flags_in) "smoothed: function \"$(C.name(g))\" has $(C.n_in(g)) inputs"
    names = op_names()
    work, flags = Vector{SX}(undef, C.sz_w(g)), zeros(UInt8, C.sz_w(g))
    out = [Vector{SX}(undef, C.nnz_out(g, o)) for o in 0:C.n_out(g)-1]
    out_flags = [UInt8[] for _ in 0:C.n_out(g)-1]
    for k in 0:C.n_instructions(g)-1
        # Loop invariant: each slot written so far holds the smoothed value of the slot of `g`, and its flag.
        id = C.instruction_id(g, k)
        op = names[id]
        i, o = C.instruction_input(g, k), C.instruction_output(g, k)
        if op === :OP_INPUT
            work[o[1]+1], flags[o[1]+1] = nonzeros[i[1]+1][i[2]+1], flags_in[i[1]+1]
        elseif op === :OP_OUTPUT
            out[o[1]+1][o[2]+1] = work[i[1]+1]
            push!(out_flags[o[1]+1], flags[i[1]+1])
        elseif op === :OP_CONST
            work[o[1]+1], flags[o[1]+1] = SX(C.instruction_constant(g, k)), 0x00
        else
            @requires haskey(OPERATIONS, op) "smoothed: function \"$(C.name(g))\" has the operation $op, which the transform does not read"
            slots = OPERATIONS[op].arity == 2 ? [i[1] + 1, i[2] + 1] : [i[1] + 1]
            operands, operand_flags = work[slots], flags[slots]
            replaced = smooth_operation(op, operands, operand_flags, width, s, SX)
            work[o[1]+1], flags[o[1]+1] = replaced !== nothing ? replaced :
                (length(slots) == 1 ? C.unary(SX, id, operands[1]) : C.binary(SX, id, operands[1], operands[2]), kept_operation!(op, operand_flags, s))
        end
    end
    @ensures all(v -> all(k -> isassigned(v, k), eachindex(v)), out) "smoothed: function \"$(C.name(g))\": a nonzero of a result was not written"
    results = [C.nnz_out(g, o) == 0 ? SX(C.size1_out(g, o), C.size2_out(g, o)) : SX(C.sparsity_out(g, o), C.vertcat(out[o+1])) for o in 0:C.n_out(g)-1]
    results, [isempty(f) ? 0x00 : (reduce(|, f) & (SELECTED | SOFT)) | (all(truth, f) ? TRUTH : 0x00) for f in out_flags]
end

"The results of the MX function `g` as expressions of its inputs `C.mx_in(g)`."
function defining_outputs(g::C.Function)
    outputs = MX[MX(C.size1_out(g, o), C.size2_out(g, o)) for o in 0:C.n_out(g)-1]
    for k in 0:C.n_instructions(g)-1
        C.instruction_id(g, k) == C.OP_OUTPUT && (outputs[C.instruction_output(g, k)[1]+1] = C.dep(C.instruction_MX(g, k), 0))
    end
    outputs
end

"`value` with the sparsity pattern of `node`, which a substitution needs."
fitted_to(value::MX, node::MX) =
    sparsity_tuple(C.sparsity(value)) == sparsity_tuple(C.sparsity(node)) ? value : C.project(value, C.sparsity(node))

"""
    smooth_graph(outputs, symbols, flags, width, s) -> (results, flags)

Rewrite the switches in the MX expressions `outputs` of the symbols `symbols`, whose flags
are `flags`, into smooth forms of width `width`. Each switch node gets its replacement, a
call of an SX or MX function gets a call of its transform, and `C.graph_substitute` puts
the replacements in, so that the rest of the graph stays as it is.
"""
function smooth_graph(outputs::Vector{MX}, symbols::Vector{MX}, flags_in::Vector{UInt8}, width::MX, s::Smoothing)
    g = Front.Function("smoothing", symbols, outputs)
    names = op_names()
    flags, depth = Dict{Int,UInt8}(), Dict{Int,Int}()
    # A replacement is written on new symbols in place of the operands of its node: MX folds
    # a product with an `if_else_zero` operand into a new node, which would hide that operand
    # from a substitution.
    nodes, replacements, placeholders, operand_nodes, levels = MX[], MX[], MX[], MX[], Int[]
    out_flags = zeros(UInt8, length(outputs))
    for k in 0:C.n_instructions(g)-1
        # Loop invariant: each slot written so far holds its flag, and the depth of nesting of the replacements it depends on.
        op = names[C.instruction_id(g, k)]
        i, o = C.instruction_input(g, k), C.instruction_output(g, k)
        node = C.instruction_MX(g, k)
        if op === :OP_OUTPUT
            out_flags[o[1]+1] = get(flags, i[1], 0x00)
            continue
        elseif op === :OP_INPUT
            flags[o[1]], depth[o[1]] = flags_in[Int(C.info(node)["ind"])+1], 0
            continue
        end
        operand_flags = UInt8[get(flags, j, 0x00) for j in i]
        level = maximum((get(depth, j, 0) for j in i); init = 0)
        operands = MX[C.dep(node, j) for j in 0:C.n_dep(node)-1]
        stand_ins = MX[MX.sym("operand", C.sparsity(a)) for a in operands]
        if op === :OP_CALL
            replaced, results = smooth_call(node, operand_flags, width, s; arguments = stand_ins)
            replaced === nothing || (append!(placeholders, stand_ins); append!(operand_nodes, operands))
            for (j, slot) in enumerate(o)
                slot >= 0 || continue
                flags[slot], depth[slot] = results[j], level
                if replaced !== nothing
                    push!(nodes, C.get_output(node, j - 1))
                    push!(replacements, fitted_to(replaced[j], nodes[end]))
                    push!(levels, level + 1)
                    depth[slot] = level + 1
                end
            end
        else
            replaced = !isempty(operands) && length(operands) == length(i) ? smooth_operation(op, stand_ins, operand_flags, width, s, MX) : nothing
            if replaced === nothing
                flag, d = kept_operation!(op, operand_flags, s), level
            else
                push!(nodes, node)
                push!(replacements, fitted_to(replaced[1], node))
                append!(placeholders, stand_ins)
                append!(operand_nodes, operands)
                flag, d = replaced[2], level + 1
                push!(levels, d)
            end
            foreach(slot -> slot >= 0 && (flags[slot] = flag; depth[slot] = d), o)
        end
    end
    isempty(nodes) && return outputs, out_flags
    # Level by level, the operands get the replacements of the levels below, and then go in
    # place of their symbols. The operand graphs are those of `g`, so each node is found.
    done, values = MX[], MX[]
    for level in 1:maximum(levels)
        # Loop invariant: `values` holds the final replacement of each node in `done`, the nodes of the levels below `level`.
        here = findall(==(level), levels)
        isempty(here) && continue
        current = isempty(done) || isempty(operand_nodes) ? operand_nodes : as_list(C.graph_substitute(operand_nodes, done, values))
        written = isempty(placeholders) ? replacements[here] : as_list(C.graph_substitute(replacements[here], placeholders, current))
        append!(done, nodes[here])
        append!(values, written)
    end
    @ensures length(values) == length(nodes) "smoothed: a replacement of the graph was not written"
    as_list(C.graph_substitute(outputs, done, values)), out_flags
end

"""
    smooth_call(node, flags, width, s) -> (results or nothing, flags)

The smooth replacement of the results of the call `node` whose arguments have the flags
`flags`, or `nothing` when the call stays, and the flag of each result. An SX or MX
function, a `map` of one and the branches of a switch are transformed; the index of a
switch and the point of a `linear` interpolant are kept; the body of any other function,
such as a Julia callback, an external function or a solver, is not read.
"""
function smooth_call(node::MX, flags::Vector{UInt8}, width::MX, s::Smoothing; arguments::Vector{MX} = MX[C.dep(node, j) for j in 0:C.n_dep(node)-1])
    callee = C.which_function(node)
    n = C.n_out(callee)
    union_flag = isempty(flags) ? 0x00 : reduce(|, flags) & (SELECTED | SOFT)
    (any(selected, flags) && length(flags) == C.n_in(callee)) || return nothing, fill(union_flag, n)
    class = C.class_name(callee)
    if C.is_a(callee, "SXFunction") || C.is_a(callee, "MXFunction")
        h, changed, results = smoothed_callee(callee, flags, s)
        changed ? (as_list(C.call(h, [arguments; width], false, false)), results) : (nothing, results)
    elseif class in ("Map", "ThreadMap", "OmpMap")
        info = C.info(callee)
        h, changed, results = smoothed_callee(info["f"]::C.Function, flags, s)
        count = Int(info["n"])
        changed ? (as_list(C.call(C.map(h, count), [arguments; C.repmat(width, 1, count)], false, false)), results) : (nothing, results)
    elseif class == "Switch"
        @requires !soft(flags[1]) "smoothed: the index of a switch (\"$(C.name(callee))\") is a smoothed value, so the switch would move"
        selected(flags[1]) && push!(s.kept, :Switch)
        info = C.info(callee)
        transform(b) = b isa C.Function && !C.is_null(b) ? smoothed_callee(b, flags[2:end], s) : nothing
        cases, default = [transform(b) for b in info["f"]], transform(info["f_def"])
        branches = filter(!isnothing, [cases; [default]])
        results = [reduce(|, (b[3][j] for b in branches); init = flags[1] & (SELECTED | SOFT)) & (SELECTED | SOFT) for j in 1:n]
        any(b -> b[2], branches) || return nothing, results
        @requires default !== nothing && all(!isnothing, cases) "smoothed: switch \"$(C.name(callee))\" has a branch that is not a function"
        switch = C.conditional(C.Function, C.name(callee) * "_smoothed", C.Function[b[1] for b in cases], default[1])
        as_list(C.call(switch, [arguments; width], false, false)), results
    else
        if class in ("LinearInterpolant", "LinearInterpolantJac")
            @requires !soft(flags[1]) "smoothed: the point of the interpolant \"$(C.name(callee))\" is a smoothed value"
            selected(flags[1]) && push!(s.kept, :LinearInterpolant)
        end
        nothing, fill(union_flag, n)
    end
end

"""
    smoothed_callee(g, flags, s) -> (h, changed, flags)

The transform `h` of the SX or MX function `g` whose inputs have the flags `flags`: the
inputs of `g` and the width at the end. `changed` tells whether a switch was rewritten.
Each pair of a function and its flags is transformed once.
"""
function smoothed_callee(g::C.Function, flags::Vector{UInt8}, s::Smoothing)
    key = (C.__hash__(g), flags)
    # The table keeps `g`, since CasADi gives the identity of a freed function to a later one.
    haskey(s.callees, key) && return s.callees[key][1]
    entry = transformed_function(g, flags, s, C.name(g) * "_smoothed", "width")
    s.callees[key] = (entry, g)
    entry
end

"""
    transformed_function(g, flags, s, name, input) -> (h, changed, flags)

The function `name` with the inputs of the SX or MX function `g` and the scalar input
`input` at the end, the width, whose results are those of `g` with the switches on the
inputs with the flag SELECTED rewritten. `changed` tells whether a switch was rewritten,
and `flags` holds the flag of each result.
"""
function transformed_function(g::C.Function, flags::Vector{UInt8}, s::Smoothing, name::AbstractString, input::AbstractString)
    names_in = [C.name_in(g, i) for i in 0:C.n_in(g)-1]
    names_out = [C.name_out(g, j) for j in 0:C.n_out(g)-1]
    before = s.rewrites
    if C.is_a(g, "SXFunction")
        width = SX.sym(input)
        symbols = [scalar_symbols(names_in[i+1], C.sparsity_in(g, i)) for i in 0:C.n_in(g)-1]
        results, out_flags = smooth_scalar(g, [last(x) for x in symbols], flags, width, s)
        h = Front.Function(name, [SX[first(x) for x in symbols]; width], results, [names_in; input], names_out)
    else
        @requires C.is_a(g, "MXFunction") "smoothed: function \"$(C.name(g))\" is not an SX or MX function"
        width = MX.sym(input)
        symbols = MX[x for x in C.mx_in(g)]
        results, out_flags = smooth_graph(defining_outputs(g), symbols, flags, width, s)
        h = Front.Function(name, [symbols; width], results, [names_in; input], names_out)
    end
    h, s.rewrites > before, out_flags
end

function check_width(ex, width, kind::Symbol)
    @requires kind in SMOOTH_KINDS "smoothed: the kind must be :tanh or :algebraic; got $(repr(kind))"
    @requires C.is_symbolic(width) && C.is_scalar(width) "smoothed: the width must be a scalar symbol"
    @requires !C.depends_on(ex, width) "smoothed: the expression already depends on the width"
end

"""
    ensure_smoothed(h, places, s)

Check that the switches that `nonsmooth_operations` finds in `h` on the inputs `places` are
switches that the transform kept.
"""
function ensure_smoothed(h::C.Function, places, s::Smoothing)
    left = setdiff(first.(nonsmooth_operations(h; inputs = places)), s.kept)
    @ensures isempty(left) "smoothed: function \"$(C.name(h))\" still switches at $(join(left, ", "))"
    nothing
end

"""
    smoothed(f::CasADiCodegen.Function; kind = :tanh, input = "width", inputs = nothing) -> CasADiCodegen.Function
    smoothed(ex::SX, width::SX; kind = :tanh, inputs = nothing) -> SX
    smoothed(ex::MX, width::MX; kind = :tanh, inputs = nothing) -> MX

Rewrite each switch of an SX or MX function or expression into its smooth form of width
`width`, with the forms of [`smooth_step`](@ref) of the kind `kind`.

The function form returns the function named `C.name(f) * "_smoothed"`, with the inputs and
the outputs of `f` and one more scalar input named `input`, the width, at the end. The
expression form returns an expression of the symbols of `ex` and of `width`, the form for a
model dictionary: `"ode" => smoothed(ode, width; inputs = x)`, with `width` among the
parameters.

`inputs` chooses the switches to rewrite: those whose operand depends on `inputs`. For a
function it is `nothing` (all inputs), or input places from 1 or input names; for an
expression, `nothing` (all symbols) or an expression of symbols, such as the state. A
switch on other inputs is kept.

| Operation                             | Replacement                                       |
|:------------------------------------- |:------------------------------------------------- |
| `a < b`, `a <= b`                     | `smooth_step(b - a, width)`, a soft condition c   |
| `!c`, `c & d`, `c \\| d`              | `1 - c`, `c d`, `c + d - c d`                     |
| `if_else(c, a, b)`                    | `c a + (1 - c) b`                                 |
| `sign`, `abs`, `fmin`, `fmax`, `copysign` | `smooth_sign`, `smooth_abs`, `smooth_min`, `smooth_max`, `smooth_abs(a) smooth_sign(b)` |
| MX `norm_1`, `norm_inf`, `mmin`, `mmax` | the sum or a fold of the smooth forms           |
| `==`, `!=`, `floor`, `ceil`, `fmod`, `remainder`, the index of `conditional`, a `linear` interpolant | kept |

An equality test is kept since it switches only on a set of measure zero. A value that the
graph shares stays shared, and a call of an SX or MX function, also inside `map` or a
switch, becomes a call of its transform. A Julia callback, an external function and a
solver are kept as they are. An MX graph is read one matrix at a time, so a switch on an
element of a matrix that depends on `inputs` is rewritten.

Both branches of `if_else` are weighted by the soft condition within about 19 widths of
the surface with `:tanh`, so both must be finite there. Farther away the `:tanh` condition rounds to exactly 0 or 1, and a branch of weight 0 does not
enter the result, even when it is NaN. The `:algebraic` condition never reaches 0, so
with it both branches must be finite everywhere. One width serves all switches, so the
switching functions need comparable scales.

Throws `ArgumentError` for a `kind` other than `:tanh` and `:algebraic`, a `width` that is
not a scalar symbol or that `ex` depends on, SX with MX, an `input` name that `f` has, an
input that `f` does not have, and a kept operation on a smoothed value, such as
`if_else((x > 0) + (y > 0), a, b)`, whose switch would move.

# Examples
```jldoctest
julia> using CasADiCodegen

julia> x, width = SX.sym("x"), SX.sym("width");

julia> CasADiCodegen.smoothed(sign(x - 1), width)
1×1 SX:
tanh(((x-1)/width))
```
"""
function smoothed(f::C.Function; kind::Symbol = :tanh, input::AbstractString = "width", inputs = nothing)
    @requires kind in SMOOTH_KINDS "smoothed: the kind must be :tanh or :algebraic; got $(repr(kind))"
    @requires C.is_a(f, "SXFunction") || C.is_a(f, "MXFunction") "smoothed: function \"$(C.name(f))\" is not an SX or MX function"
    @requires !(input in [C.name_in(f, i) for i in 0:C.n_in(f)-1]) "smoothed: function \"$(C.name(f))\" already has an input named \"$input\"; choose another with the keyword `input`"
    places = input_places(f, inputs)
    s = Smoothing(kind)
    h, _, _ = transformed_function(f, UInt8[i in places ? SELECTED : 0x00 for i in 1:C.n_in(f)], s, C.name(f) * "_smoothed", input)
    ensure_smoothed(h, places, s)
    h
end

"The places of the symbols `symbols` on which `inputs` depends, from 1: all with `nothing`."
function chosen_symbols(symbols, inputs, ::Type{T}) where {T}
    inputs === nothing && return collect(eachindex(symbols))
    @requires inputs isa T "smoothed: `inputs` must be an expression of the type of the expression, $T"
    findall(v -> C.depends_on(inputs, v), symbols)
end

function smoothed(ex::SX, width::SX; kind::Symbol = :tanh, inputs = nothing)
    check_width(ex, width, kind)
    symbols = SX[v for v in C.symvar(ex)]
    places = chosen_symbols(symbols, inputs, SX)
    s = Smoothing(kind)
    flags = UInt8[k in places ? SELECTED : 0x00 for k in eachindex(symbols)]
    results, _ = smooth_scalar(Front.Function("smoothing", symbols, [ex]), [[v] for v in symbols], flags, width, s)
    ensure_smoothed(Front.Function("smoothed", [symbols; width], results), places, s)
    only(results)
end

function smoothed(ex::MX, width::MX; kind::Symbol = :tanh, inputs = nothing)
    check_width(ex, width, kind)
    symbols = MX[v for v in C.symvar(ex)]
    places = chosen_symbols(symbols, inputs, MX)
    s = Smoothing(kind)
    flags = UInt8[k in places ? SELECTED : 0x00 for k in eachindex(symbols)]
    results, _ = smooth_graph([ex], symbols, flags, width, s)
    ensure_smoothed(Front.Function("smoothed", [symbols; width], results), places, s)
    only(results)
end

smoothed(ex::Symbolic, width; kind::Symbol = :tanh, inputs = nothing) =
    throw(ArgumentError("smoothed: an $(typeof(ex)) expression takes a width of the same type; got $(typeof(width))"))
