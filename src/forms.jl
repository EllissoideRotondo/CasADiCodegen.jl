# Functions on arrays in matrix shape, in place and out of place, next to the
# function on vectors of nonzeros; all of them run the same sub-functions.

"Linear indices, in the matrix shape, of the structural nonzeros of the pattern."
function nonzero_indices(sp::C.Sparsity)
    nrow, ncol, colptr, rowval = sparsity_tuple(sp)
    [(c - 1) * nrow + rowval[k] for c in 1:ncol for k in colptr[c]:colptr[c+1]-1]
end

dimensions(sp::C.Sparsity) = C.size2(sp) == 1 ? (C.size1(sp),) : (C.size1(sp), C.size2(sp))

is_dense(sp::C.Sparsity) = C.nnz(sp) == C.size1(sp) * C.size2(sp)

"""
    outofplace_name(name) -> Symbol

Return the name of the out-of-place function: `name` without its last character, which
must be `!`.
"""
function outofplace_name(name::Symbol)
    text = string(name)
    @requires endswith(text, "!") && Base.isidentifier(chop(text)) "the name of the in-place function must end with \"!\", got \"$name\""
    Symbol(chop(text))
end

"Statements that check the number of elements of each array of `res` (when `results`) and `arg`."
function shape_checks(alg, ref; results::Bool)
    check(array, kind, i, name, sp) = :(length($array[$i]) == $(prod(dimensions(sp))) ||
        $(ref(:Helpers, :shape_error))($("$(alg.name): $kind $i ($name) needs $(prod(dimensions(sp))) elements, got "), length($array[$i])))
    checks = Any[check(:arg, "argument", i, alg.names_in[i], sp) for (i, sp) in enumerate(alg.sparsity_in)]
    results && append!(checks, (check(:res, "result", j, alg.names_out[j], sp) for (j, sp) in enumerate(alg.sparsity_out)))
    checks
end

count_checks(alg, ref) = Any[
    :(length(arg) == $(length(alg.sparsity_in)) || $(ref(:Helpers, :shape_error))($("$(alg.name): expected $(length(alg.sparsity_in)) arguments, got "), length(arg))),
    :(length(res) == $(length(alg.sparsity_out)) || $(ref(:Helpers, :shape_error))($("$(alg.name): expected $(length(alg.sparsity_out)) results, got "), length(res))),
]

zero_fills(alg) = Any[:(fill!(res[$j], 0)) for (j, sp) in enumerate(alg.sparsity_out) if !is_dense(sp)]

"""
    nonzero_call(alg, core, check, ref) -> Expr

Return the call of `core`, the function on vectors of nonzeros, with the structural
nonzeros of each array of `res` and `arg`. A dense array is passed itself. Another array
is passed as a view, or as the stored values of a sparse matrix with the sparsity pattern
of the function. `check(kind, k)` is the expression for the `Helpers.PatternCheck` of
input or output `k`, which holds the indices of its nonzeros.
"""
function nonzero_call(alg, core, check, ref)
    part(array, kind, k, sp) = is_dense(sp) ? :($array[$k]) : C.nnz(sp) == 0 ? :(()) :
                               :($(ref(:Helpers, :stored))($array[$k], $(check(kind, k))))
    arguments = [part(:arg, :in, i, sp) for (i, sp) in enumerate(alg.sparsity_in)]
    results = [part(:res, :out, j, sp) for (j, sp) in enumerate(alg.sparsity_out)]
    :($core($(Expr(:tuple, results...)), $(Expr(:tuple, arguments...))))
end

"Expressions that read and write a nonzero of scalar code at its place in the arrays in matrix shape."
function matrix_access(alg)
    index_in, index_out = nonzero_indices.(alg.sparsity_in), nonzero_indices.(alg.sparsity_out)
    (load = (i, k) -> :(arg[$i][$(index_in[i][k])]), store = (j, k, value) -> :(res[$j][$(index_out[j][k])] = $value))
end

"The keywords `place_in` and `place_out` of `framed_body` for arrays in matrix shape; `index` is that of `nonzero_call`."
matrix_places(alg, index) =
    (place_in = i -> is_dense(alg.sparsity_in[i]) ? :k : :($(index(:in, i))[k]),
     place_out = j -> is_dense(alg.sparsity_out[j]) ? :k : :($(index(:out, j))[k]))

"""
    inplace_function(alg, signature, frame, core, index, check, ref, key) -> Expr

Return the definition of the function `signature(res, arg)` on arrays in matrix shape,
which reads and writes the arrays directly. `frame` is that of `entry_function`. A sparse
matrix among the arrays, which is possible when a sparsity pattern has structural zeros,
sends the call to `core`, the function on vectors of nonzeros; `index` and `check` are
those of `matrix_places` and `nonzero_call`. A sparse matrix of `res` is not filled with
zeros first, since `core` writes each of its stored values.

The generated function leaves one result of `alg` in each array of `res`, in column-major
order, with zero at each structural zero.
"""
function inplace_function(alg, signature, frame, core, index, check, ref, key)
    tables = index_tables(alg)
    # `ref` records each helper that the code uses, so it names `is_stored` only for code that calls it.
    is_stored = isempty(tables) ? nothing : ref(:Helpers, :is_stored)
    sparse_fills = Any[:($is_stored(res[$j]) || fill!(res[$j], 0)) for (j, sp) in enumerate(alg.sparsity_out) if !is_dense(sp)]
    sparse = isempty(tables) ? Any[] : Any[Expr(:if,
        foldl((a, b) -> :($a || $b), (:($is_stored($(kind === :in ? :arg : :res)[$k])) for ((kind, k), _) in tables)),
        Expr(:block, sparse_fills..., nonzero_call(alg, core, check, ref), :(return res)))]
    body = frame === nothing ? Any[inbounds_block(whole_statements(alg, ref; matrix_access(alg)...))] :
           framed_body(alg, frame, ref, key; matrix_places(alg, index)...)
    Expr(:function, signature, Expr(:block, count_checks(alg, ref)..., shape_checks(alg, ref; results = true)...,
        sparse..., zero_fills(alg)..., body..., :(return res)))
end

argument_names(alg) = [Symbol(:arg, i) for i in eachindex(alg.sparsity_in::Vector{C.Sparsity})]

returned(results) = length(results) == 1 ? only(results) : Expr(:tuple, results...)

static_array(ref) = ref(:StaticArraysCore, :SArray)

static_type(type, dims) = :($type{Tuple{$(dims...)},T,$(length(dims)),$(prod(dims))})

const ELEMENT_TYPE = :(T = promote_type(Float64, map(eltype, arg)...))

"""
    outofplace_function(alg, signature, inplace, ref) -> Expr

Return the definition of the function `signature(arg1, arg2, ...)` that makes a `Vector` or
a `Matrix` per result, fills them with `inplace`, the function on arrays in matrix shape,
and returns them. The statements of `warning`, from `static_warning`, come first.
"""
outofplace_function(alg, signature, inplace, ref; warning = Any[]) =
    Expr(:function, signature, Expr(:block, warning..., :(arg = $(Expr(:tuple, argument_names(alg)...))), ELEMENT_TYPE,
        :(res = $(Expr(:tuple, (:(Array{T}(undef, $(dimensions(sp)...))) for sp in alg.sparsity_out)...))),
        :($inplace(res, arg)),
        :(return $(returned([:(res[$j]) for j in eachindex(alg.sparsity_out)])))))

"""
    static_function(alg, signature, frame, index, ref, key) -> Expr

Return the definition of the out-of-place function with static arrays. It builds the static
arrays from local variables, or from the work vector of the `frame`, so it allocates no
memory.
"""
function static_function(alg, signature, frame, index, ref, key)
    index_out = nonzero_indices.(alg.sparsity_out)
    values = [Any[0.0 for _ in 1:prod(dimensions(sp))] for sp in alg.sparsity_out]
    body = if frame === nothing
        keep(j, k, value) = (values[j][index_out[j][k]] = value; nothing)
        Any[inbounds_block(whole_statements(alg, ref; load = matrix_access(alg).load, store = keep))]
    else
        offset_out = frame[1][2]
        for (j, indices) in enumerate(index_out), (k, place) in enumerate(indices)
            values[j][place] = :(w[$(offset_out[j] + k)])
        end
        framed_body(alg, frame, ref, key; place_in = matrix_places(alg, index).place_in, results = false)
    end
    SArray = static_array(ref)
    results = [:($(static_type(SArray, dimensions(sp)))($(Expr(:tuple, values[j]...))))
               for (j, sp) in enumerate(alg.sparsity_out)]
    Expr(:function, signature, Expr(:block, :(arg = $(Expr(:tuple, argument_names(alg)...))),
        shape_checks(alg, ref; results = false)..., ELEMENT_TYPE, body...,
        :(return $(Expr(:macrocall, Symbol("@inbounds"), nothing, returned(results))))))
end

"Inputs and outputs that need a table of indices: `(kind, k) => indices`."
index_tables(alg) = [(kind, k) => nonzero_indices(sp)
                     for (kind, patterns) in ((:in, alg.sparsity_in), (:out, alg.sparsity_out))
                     for (k, sp) in enumerate(patterns) if !is_dense(sp) && C.nnz(sp) > 0]

core_name(name::Symbol) = Symbol(outofplace_name(name), :_nonzeros!)

"""
Largest number of elements, summed over the outputs, of the out-of-place method with
static arrays. The method writes each element of each output into its code. Measured on
dense and diagonal Jacobians, above about 1000 elements it compiles more slowly than the
method with ordinary arrays, and for a dense Jacobian its call is also slower; far above,
Julia cannot parse it.
"""
const STATIC_LIMIT = 1024

output_elements(alg) = sum(sp -> prod(dimensions(sp)), alg.sparsity_out; init = 0)

"""
    methods_wanted(alg, force_SA) -> (ordinary, static)

Whether the out-of-place function has a method for ordinary arrays, and one for static
arrays. A function without inputs, or with more than `STATIC_LIMIT` output elements, has
no method for static arrays unless `force_SA` asks for it, which the second case refuses.
"""
function methods_wanted(alg, force_SA::Bool)
    elements = output_elements(alg)
    @requires !force_SA || elements <= STATIC_LIMIT "$(alg.name): the outputs have $elements elements, and static arrays (`force_SA = true`) take at most $STATIC_LIMIT"
    (!force_SA, force_SA || (!isempty(alg.sparsity_in) && elements <= STATIC_LIMIT))
end

"""
    static_warning(alg, static, ref, id) -> Vector{Any}

Return the statement, for the method with ordinary arrays, that warns once when its first
argument is a static array, which happens when the outputs exceed `STATIC_LIMIT`; an empty
vector when `static` (the method with static arrays exists) or `alg` has no input. `id` is the
`_id` of the log record: a symbol unique in the session, or `nothing` in a source file,
where each expansion of `@warn` gets its own.
"""
function static_warning(alg, static::Bool, ref, id)
    (static || isempty(alg.sparsity_in)) && return Any[]
    message = "$(alg.name): the outputs have $(output_elements(alg)) elements, and static arrays hold at most " *
              "$STATIC_LIMIT, so a static array as argument gives `Vector` and `Matrix` results; pass ordinary arrays " *
              "to avoid this warning"
    options = Any[Expr(:(=), :maxlog, 1)]
    id === nothing || push!(options, Expr(:(=), :_id, QuoteNode(id)))
    # `@warn` needs a line number to expand; a source file gets its own when Julia parses it.
    line = id === nothing ? nothing : LineNumberNode(0, :generated)
    warn = Expr(:macrocall, GlobalRef(Base, Symbol("@warn")), line, message, options...)
    Any[:($(first(argument_names(alg))) isa $(ref(:StaticArraysCore, :StaticArray)) && $warn)]
end

"""
    form_definitions(alg, name, chunk_size, ref, force_SA) -> Vector{Expr}

Return the named definitions that follow those of the function on nonzeros
`core_name(name)`, whose sub-functions they share: the tables of indices, the in-place
function `name(res, arg)`, and the out-of-place function, whose name is `name` without `!`.

The out-of-place function has a method for a static array as its first argument, which
returns static arrays. With `force_SA` that is its only method, for any argument. Above
`STATIC_LIMIT` the method with ordinary arrays warns once for a static argument, with the
log id `log_id` (see `static_warning`). `julia` is that of `mx_definitions`.
"""
function form_definitions(alg, name::Symbol, chunk_size::ChunkSize, ref, force_SA::Bool; key = identity, julia = identity,
                          log_id::Union{Nothing,Symbol} = nothing)
    base, arguments = outofplace_name(name), argument_names(alg)
    table(kind, k) = Symbol(base, :_index_, kind, k)
    check(kind, k) = Symbol(base, :_check_, kind, k)
    definitions = Expr[Expr(:const, :($(table(kind, k)) = $(Expr(:vect, indices...))))
                       for ((kind, k), indices) in index_tables(alg)]
    append!(definitions, (Expr(:const, :($(check(kind, k)) = $(ref(:Helpers, :PatternCheck))($(table(kind, k)))))
                          for ((kind, k), _) in index_tables(alg)))
    frame = named_frame(alg, core_name(name), chunk_size, ref; julia)
    ordinary, static = methods_wanted(alg, force_SA)
    typed = force_SA || !static ? arguments : Any[:($(arguments[1])::$(ref(:StaticArraysCore, :StaticArray))), arguments[2:end]...]
    push!(definitions, inplace_function(alg, :($name(res, arg)), frame, core_name(name), table, check, ref, key(name)))
    warning = static_warning(alg, static, ref, log_id)
    ordinary && push!(definitions, outofplace_function(alg, :($base($(arguments...))), name, ref; warning))
    static && push!(definitions, static_function(alg, :($base($(typed...))), frame, table, ref, key(name)))
    named_calls(alg, definitions, julia)
end

"""
    runtime_inplace(alg, chunk_size, ref) -> NamedTuple

Return the in-place function on arrays in matrix shape as a runtime generated function
`inplace`, which holds the function on nonzeros that it calls, together with the `frame`,
the function `index` and the `key` that the out-of-place forms share with it.
"""
function runtime_inplace(alg, chunk_size::ChunkSize, ref)
    generate(ex) = RuntimeGeneratedFunction(@__MODULE__, CONTEXT[], ex)
    tables = Dict(index_tables(alg))
    checks = Dict(place => Helpers.PatternCheck(indices) for (place, indices) in tables)
    index(kind, k) = tables[(kind, k)]
    check(kind, k) = checks[(kind, k)]
    key = unique_key(alg.name)
    frame = runtime_frame(alg, chunk_size, ref)
    # Only a sparse matrix, which needs a pattern with structural zeros, sends a call to the function on nonzeros.
    core = isempty(tables) ? nothing : RuntimeGeneratedFunctions.drop_expr(generate(entry_function(alg, :((res, arg)), frame, ref, key)))
    inplace = generate(inplace_function(alg, :((res, arg)), frame, core, index, check, ref, key))
    (; inplace, frame, index, key)
end

"""
    runtime_forms(alg, chunk_size, ref, force_SA) -> (out_of_place, in_place)

Return the two functions as runtime generated functions. Each holds the functions that it
calls and the tables of indices in its body. The out-of-place function returns static
arrays when its first argument is a static array, or always with `force_SA`.
"""
function runtime_forms(alg, chunk_size::ChunkSize, ref, force_SA::Bool)
    generate(ex) = RuntimeGeneratedFunction(@__MODULE__, CONTEXT[], ex)
    hold(g) = RuntimeGeneratedFunctions.drop_expr(g)
    (; inplace, frame, index, key) = runtime_inplace(alg, chunk_size, ref)
    arguments = argument_names(alg)
    signature = Expr(:tuple, arguments...)
    ordinary, static = methods_wanted(alg, force_SA)
    warning = static_warning(alg, static, ref, Symbol(key, :_static))
    with_arrays = ordinary ? generate(outofplace_function(alg, signature, hold(inplace), ref; warning)) : nothing
    with_static = static ? generate(static_function(alg, signature, frame, index, ref, key)) : nothing
    outofplace = !ordinary ? with_static : !static ? with_arrays :
        generate(Expr(:function, signature, Expr(:block,
            :(return $(arguments[1]) isa $(ref(:StaticArraysCore, :StaticArray)) ?
                     $(hold(with_static))($(arguments...)) : $(hold(with_arrays))($(arguments...))))))
    # Without its expression a function is a value without fields, which a call passes without allocation.
    hold(outofplace), hold(inplace)
end
