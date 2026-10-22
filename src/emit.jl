const DEFAULT_CHUNK_SIZE = 1000

"A scalar function of at most this number of instructions is one function by default."
const UNSPLIT_LIMIT = 50_000

unique_key(name) = gensym(string(name))

"A scalar function of at most this number of instructions, in one piece, has its calls by `build_inplace` compiled inline."
const INLINE_LIMIT = 256

"""
    Split(chunk_size, split_above)

The keywords `chunk_size` and `split_above` of `build_function`. Each is a number of
instructions, or `nothing` for its default.
"""
struct Split
    chunk_size::Union{Nothing,Int}
    split_above::Union{Nothing,Int}
    function Split(chunk_size, split_above)
        @requires chunk_size === nothing || chunk_size >= 1 "chunk_size must be at least 1"
        @requires split_above === nothing || split_above >= 0 "split_above must not be negative"
        new(chunk_size, split_above)
    end
end

"How to split a function: a `Split`, or the keyword `chunk_size` alone."
const ChunkSize = Union{Nothing,Integer,Split}

"""
    chunk_size_for(n, chunk_size; limit = UNSPLIT_LIMIT) -> Integer

Return the chunk size for a function of `n` instructions: one chunk when `n` is at most
`split_above`, and `chunk_size` otherwise.

`chunk_size` defaults to `DEFAULT_CHUNK_SIZE`. `split_above` defaults to `chunk_size` when
that is given, and to `limit` otherwise.
"""
function chunk_size_for(n::Integer, split::Split; limit::Integer = UNSPLIT_LIMIT)
    above = something(split.split_above, split.chunk_size, limit)
    n <= above ? max(n, 1) : something(split.chunk_size, DEFAULT_CHUNK_SIZE)
end
chunk_size_for(n::Integer, chunk_size::Union{Nothing,Integer}; kwargs...) =
    chunk_size_for(n, Split(chunk_size, nothing); kwargs...)

"The number of sub-functions of scalar code: 0 when the algorithm is one function."
function chunk_count(alg::Algorithm, chunk_size::ChunkSize)
    n = length(alg.instructions)
    count = length(chunk_ranges(n, chunk_size_for(n, chunk_size)))
    count <= 1 ? 0 : count
end

function chunk_ranges(n::Integer, chunk_size::Integer)
    @requires chunk_size >= 1 "chunk_size must be at least 1"
    [lo:min(lo + chunk_size - 1, n) for lo in 1:chunk_size:n]
end

"Work slots that an instruction reads, and the slot that it writes (0 for none)."
function slots(ins::Instruction)
    if ins.op === :OP_INPUT || ins.op === :OP_CONST
        ((), ins.out)
    elseif ins.op === :OP_OUTPUT
        ((ins.a,), 0)
    elseif OPERATIONS[ins.op].arity == 1
        ((ins.a,), ins.out)
    else
        ((ins.a, ins.b), ins.out)
    end
end

"""
    liveness(alg, ranges) -> (live_in, live_out)

Return, for each chunk, the work slots that it loads from the shared work vector and those
that it stores to it. `live_in[c]` lists the slots that chunk `c` reads before it writes
them. `live_out[c]` lists the slots that chunk `c` writes and that a later chunk has in its
`live_in`. The list is conservative: it can hold a slot that another chunk writes again first.
"""
function liveness(alg::Algorithm, ranges)
    live_in = Vector{Vector{Int}}(undef, length(ranges))
    written = Vector{Vector{Int}}(undef, length(ranges))
    for (c, range) in enumerate(ranges)
        reads, writes = Int[], BitSet()
        for k in range
            ins_reads, ins_write = slots(alg.instructions[k])
            for s in ins_reads
                s in writes || s in reads || push!(reads, s)
            end
            ins_write == 0 || push!(writes, ins_write)
        end
        live_in[c] = sort!(reads)
        written[c] = collect(writes)
    end
    live_out = Vector{Vector{Int}}(undef, length(ranges))
    needed_later = BitSet()
    for c in length(ranges):-1:1
        # Loop invariant: `needed_later` holds every slot in `live_in` of chunks c+1:end.
        live_out[c] = [s for s in written[c] if s in needed_later]
        union!(needed_later, live_in[c])
    end
    live_in, live_out
end

"""
    chunk_statements(alg, range, live_in, live_out, ref; load, store, slot, prefix) -> Vector{Any}

Return the Julia statements for the instructions in `range`. They use the variables `res`,
`arg` and, when `live_in` or `live_out` is not empty, `w`. Each value gets a fresh variable,
and a constant is a literal where it is used.

- `ref(owner, name)`: the expression that names a library function.
- `load(i, k)`: the expression that reads nonzero `k` of input `i`.
- `store(j, k, value)`: the statement that writes nonzero `k` of output `j`, or `nothing`
  when the caller keeps the value itself.
- `slot(s)`: the place of work slot `s` in `w`.
- `prefix`: the start of each variable name.
"""
function chunk_statements(alg::Algorithm, range, live_in, live_out, ref;
                          load = (i, k) -> :(arg[$i][$k]),
                          store = (j, k, value) -> :(res[$j][$k] = $value),
                          slot = identity, prefix = Symbol(""))
    current = Vector{Any}(nothing, alg.nwork)   # slot => variable name or literal
    statements = Any[]
    for s in live_in
        current[s] = Symbol(prefix, :l, s)
        push!(statements, :($(current[s]) = w[$(slot(s))]))
    end
    for k in range
        ins = alg.instructions[k]
        op = ins.op
        if op === :OP_CONST
            current[ins.out] = ins.constant
        elseif op === :OP_OUTPUT
            statement = store(ins.out, ins.b, current[ins.a])
            statement === nothing || push!(statements, statement)
        else
            value = if op === :OP_INPUT
                load(ins.a, ins.b)
            elseif OPERATIONS[op].arity == 1
                OPERATIONS[op].build(ref, current[ins.a])
            else
                OPERATIONS[op].build(ref, current[ins.a], current[ins.b])
            end
            current[ins.out] = Symbol(prefix, :v, k)
            push!(statements, :($(current[ins.out]) = $value))
        end
    end
    for s in live_out
        push!(statements, :(w[$(slot(s))] = $(current[s])))
    end
    statements
end

inbounds_block(statements) =
    Expr(:macrocall, Symbol("@inbounds"), nothing, Expr(:block, statements...))

"Statements that check the shape of `res` and `arg` in generated code."
function io_checks(alg, ref)
    label = alg.name
    fail(parts...) = :($(ref(:Helpers, :shape_error))($(parts...)))
    checks = Any[
        :(length(arg) == $(length(alg.nnz_in)) ||
          $(fail("$label: expected $(length(alg.nnz_in)) arguments, got ", :(length(arg))))),
        :(length(res) == $(length(alg.nnz_out)) ||
          $(fail("$label: expected $(length(alg.nnz_out)) results, got ", :(length(res))))),
    ]
    for (i, n) in enumerate(alg.nnz_in)
        push!(checks, :(length(arg[$i]) >= $n ||
            $(fail("$label: argument $i ($(alg.names_in[i])) needs $n nonzeros, got ", :(length(arg[$i]))))))
    end
    for (i, n) in enumerate(alg.nnz_out)
        push!(checks, :(length(res[$i]) >= $n ||
            $(fail("$label: result $i ($(alg.names_out[i])) needs $n nonzeros, got ", :(length(res[$i]))))))
    end
    checks
end

"""
    work_vector(ref, key, n) -> Expr

Return the statement that binds `w` to the work vector of `n` elements. The vector belongs
to the current task and to `key`, a symbol for this function only, so a call does not
allocate it. A function that calls itself through a callback would use one vector twice,
which is not supported.
"""
work_vector(ref, key::Symbol, n::Integer) =
    :(w = $(ref(:Helpers, :workspace))(Val{$(QuoteNode(key))}, promote_type(Float64, map(eltype, arg)...), $n))

"""
    work_layout(alg) -> (offset_in, offset_out, total)

Return the layout of the work vector of split code. The nonzeros of each input and of each
output follow the `nwork` work slots, so a sub-function needs the work vector only. Nonzero
`k` of input `i` is element `offset_in[i] + k`. `total` is the length of the vector.
"""
function work_layout(alg::Algorithm)
    sizes = [alg.nnz_in; alg.nnz_out]
    offsets = alg.nwork .+ cumsum(sizes) .- sizes
    offsets[1:length(alg.nnz_in)], offsets[length(alg.nnz_in)+1:end], alg.nwork + sum(sizes)
end

"""
    split_statements(alg, chunk_size, ref; base = 0) -> Vector{Vector{Any}}

One statement list per sub-function of split code, over the work vector `w`
only, in the layout of `work_layout` after `w[base]`; no list when the
algorithm fits in one chunk.
"""
function split_statements(alg::Algorithm, chunk_size::ChunkSize, ref; base::Integer = 0)
    n = length(alg.instructions)
    ranges = chunk_ranges(n, chunk_size_for(n, chunk_size))
    length(ranges) <= 1 && return Vector{Any}[]
    offset_in, offset_out, _ = work_layout(alg)
    live_in, live_out = liveness(alg, ranges)
    Vector{Any}[chunk_statements(alg, ranges[c], live_in[c], live_out[c], ref;
                      load = (i, k) -> :(w[$(base + offset_in[i] + k)]),
                      store = (j, k, value) -> :(w[$(base + offset_out[j] + k)] = $value),
                      slot = s -> base + s)
     for c in eachindex(ranges)]
end

whole_statements(alg::Algorithm, ref; kwargs...) =
    chunk_statements(alg, eachindex(alg.instructions), Int[], Int[], ref; kwargs...)

"""
    framed_body(alg, frame, ref, key; place_in, place_out, results = true) -> Vector{Any}

Return the statements that run code which works in the work vector only. `frame` is
`((offset_in, offset_out, total), statements)`: the layout of `work_layout` and the
statements over `w`. The result binds `w`, copies the nonzeros of each array of `arg` into
it, runs the statements and, with `results`, copies the nonzeros of each result to `res`.

`place_in(i)` is the expression for the position in `arg[i]` of nonzero `k`, and
`place_out(j)` is that in `res[j]`. Both are `k` by default.
"""
function framed_body(alg, frame, ref, key; place_in = i -> :k, place_out = j -> :k, results::Bool = true)
    (offset_in, offset_out, total), statements = frame
    loop(n, statement) = :(for k in 1:$n; $statement; end)
    loads = Any[loop(n, :(w[$(offset_in[i]) + k] = arg[$i][$(place_in(i))])) for (i, n) in enumerate(alg.nnz_in) if n > 0]
    stores = !results ? Any[] :
        Any[loop(n, :(res[$j][$(place_out(j))] = w[$(offset_out[j]) + k])) for (j, n) in enumerate(alg.nnz_out) if n > 0]
    Any[work_vector(ref, key, total), inbounds_block(Any[loads..., statements..., stores...])]
end

"""
    scalar_frame(alg, calls) -> frame or nothing

Return the frame of scalar code for `framed_body`: `nothing` when `calls` is empty, because
one function needs no work vector, and otherwise the calls of the sub-functions that `calls`
names. A sub-function takes the work vector only, because a call between runtime generated
functions can allocate memory for a tuple of arrays.
"""
scalar_frame(alg::Algorithm, calls) = isempty(calls) ? nothing : (work_layout(alg), Any[:($c(w)) for c in calls])

named_frame(alg::Algorithm, name::Symbol, chunk_size::ChunkSize, ref; julia = identity) =
    scalar_frame(alg, [chunk_name(name, c) for c in 1:chunk_count(alg, chunk_size)])

"The sub-functions of split code as runtime generated functions `(w) -> nothing`; none when the algorithm is one chunk."
function runtime_frame(alg::Algorithm, chunk_size::ChunkSize, ref)
    generate(ex) = RuntimeGeneratedFunctions.drop_expr(RuntimeGeneratedFunction(@__MODULE__, CONTEXT[], ex))
    scalar_frame(alg, Any[generate(chunk_function(:((w,)), statements)) for statements in split_statements(alg, chunk_size, ref)])
end

"""
    entry_function(alg, signature, frame, ref, key) -> Expr

The function `signature(res, arg)` on vectors of nonzeros: shape checks, then
all statements of scalar code without a frame, or the statements of
`framed_body`; `key` is that of `work_vector`.
"""
function entry_function(alg, signature, frame, ref, key)
    body = frame === nothing ? Any[inbounds_block(whole_statements(alg, ref))] : framed_body(alg, frame, ref, key)
    Expr(:function, signature, Expr(:block, io_checks(alg, ref)..., body..., :(return res)))
end

chunk_function(signature, statements) =
    Expr(:function, signature, Expr(:block, inbounds_block(statements), :(return nothing)))

chunk_name(name::Symbol, c::Int) = Symbol(name, :_chunk, c)

"""
    function_definitions(alg, name, chunk_size, ref) -> Vector{Expr}

Named function definitions that implement the algorithm: the entry point
`name(res, arg)` last, preceded by the functions `(w)` that it calls: one per
chunk of scalar code that needs more than one chunk, and those of matrix code.
`key(name)` gives the key of `work_vector`.
"""
function function_definitions(alg::Algorithm, name::Symbol, chunk_size::ChunkSize, ref; key = identity, julia = identity)
    @requires Base.isidentifier(name) "\"$name\" is not a valid function name"
    definitions = Expr[chunk_function(:($(chunk_name(name, c))(w)), statements)
                       for (c, statements) in enumerate(split_statements(alg, chunk_size, ref))]
    push!(definitions, entry_function(alg, :($name(res, arg)), named_frame(alg, name, chunk_size, ref), ref, key(name)))
end

"""
    runtime_function(alg, chunk_size, ref) -> RuntimeGeneratedFunction

One runtime generated function `(res, arg) -> res`, which holds the functions
that it calls in its body.
"""
runtime_function(alg, chunk_size::ChunkSize, ref) =
    RuntimeGeneratedFunctions.drop_expr(RuntimeGeneratedFunction(@__MODULE__, CONTEXT[],
        entry_function(alg, :((res, arg)), runtime_frame(alg, chunk_size, ref), ref, unique_key(alg.name))))

function_name(alg) = alg.name
function_name(f::C.Function) = C.name(f)
default_name(f) = Symbol(function_name(f), "!")

"""
    algorithm(f; expand = true, solvers = nothing) -> Algorithm, StagedAlgorithm or MXAlgorithm

Return the copy of `f` that the emitter reads:

- an `Algorithm` for a scalar (SX) function and for a matrix (MX) function that is expanded;
- a `StagedAlgorithm` for a matrix function that is expanded around its calls of Julia
  functions;
- an `MXAlgorithm` otherwise.

A function of another class, such as an interpolant, is read as a matrix function that
calls it.
"""
algorithm(alg; expand::Bool = true, solvers::Union{Nothing,Solvers} = nothing) = alg
function algorithm(f::C.Function; expand::Bool = true, solvers::Union{Nothing,Solvers} = nothing)
    C.is_a(f, "SXFunction") && return Algorithm(f)
    # The barrier keeps the readers of matrix functions out of the inferred code of this method,
    # so that another package that invalidates them does not invalidate the path of scalar functions.
    Base.inferencebarrier(matrix_algorithm)(f, expand, solvers)
end

function matrix_algorithm(f::C.Function, expand::Bool, solvers::Union{Nothing,Solvers})
    f = matrix_function(f)
    # The readers of the called functions find the solvers of the user in `SOLVERS`.
    Base.ScopedValues.with(SOLVERS => solvers) do
        expand || return MXAlgorithm(f; expand)
        expanded = expanded_algorithm(f)
        expanded === nothing || return expanded
        staged = staged_algorithm(f)
        staged === nothing ? MXAlgorithm(f; expand) : staged
    end
end

"The function `f` when it is a matrix (MX) function, and else a matrix function with its inputs and results that calls it."
function matrix_function(f::C.Function)
    C.is_a(f, "MXFunction") && return f
    inputs = MX[MX.sym(C.name_in(f, i), C.sparsity_in(f, i)) for i in 0:C.n_in(f)-1]
    C.Function(C.name(f), inputs, as_list(C.call(f, inputs, false, true)),
               [C.name_in(f, i) for i in 0:C.n_in(f)-1], [C.name_out(f, j) for j in 0:C.n_out(f)-1])
end

"The scalar algorithm of the expanded matrix function `f`, or `nothing` when there is none."
function expanded_algorithm(f::C.Function)
    try
        Algorithm(C.expand(f))
    catch err
        err isa Union{ArgumentError,C.SwigError} || rethrow()
        nothing
    end
end

check_optlevel(optlevel) =
    @requires optlevel === nothing || 0 <= optlevel <= 3 "optlevel must be nothing or an integer from 0 to 3, got $optlevel"

expression_flag(::Type{Val{B}}) where {B} = B::Bool
expression_flag(::Val{B}) where {B} = B::Bool
expression_flag(other) =
    throw(ArgumentError("expression must be Val{true} or Val{false}, got $(repr(other))"))

"""
    build_function(f; expression = Val{true}, force_SA = false, nonzeros = false,
                   name = Symbol(name_of_f, "!"), chunk_size = nothing, split_above = nothing,
                   nanmath = true, expand = true, cache = false, solvers = nothing, optlevel = nothing)

Generate Julia code for the CasADi function `f`, in the manner of
`Symbolics.build_function`. Return an out-of-place function `g(arg1, arg2, ...)` and an
in-place function `g!(res, arg)`: as one `Expr` with the two definitions by default, or as
the pair `(g, g!)` of compiled functions with `expression = Val{false}`.

Both functions take arrays in matrix shape; a column is a vector.

- `g` takes one array per input. It returns one array for one output, and a tuple of
  arrays otherwise. The arrays are `SVector` and `SMatrix` when the first argument is a
  static array, and `Vector` and `Matrix` otherwise. Outputs with more than
  4096 elements in all give `Vector` and `Matrix` for any argument.
- `g!` fills the arrays of the tuple `res` from the tuple `arg` and returns `res`. An array
  of `res` can be a mutable static array.
- A structural zero of an output is set to zero. A structural zero of an input is not read.
- With SparseArrays.jl loaded, an array of `res` or `arg` can be a `SparseMatrixCSC` with
  the sparsity pattern of that output or input, which `SparseMatrixCSC(sparsity)` makes.
  `g!` then reads and writes the stored values only. It compares the whole pattern of a
  matrix whose index vectors (`colptr` and `rowval`) are not those of the last matrix that
  passed this comparison. Otherwise it compares the number of rows and of stored values
  only, so a call costs as much as the function on nonzeros. Do not change the index
  vectors of a matrix in place.

`f` must have no free symbols and only supported operations; see
[Scope and differences from CasADi](@ref).

# Keywords
- `expression = Val{true}`: `Val{true}` returns an `Expr`, and `Val{false}` returns
  compiled functions.
- `force_SA = false`: with `true`, `g` returns static arrays for any argument. It throws
  an `ArgumentError` for outputs with more than 4096 elements in all.
- `nonzeros = false`: with `true`, return the single function `(res, arg) -> res` on which
  the others are built. `arg[i]` holds the structural nonzeros of input `i`, and `res[j]`
  receives those of output `j`, as CasADi evaluates a function.
- `name`: the name of `g!` in the returned `Expr`, a valid identifier. It must end with
  `!` unless `nonzeros` is set. The name of `g` is `name` without `!`.
- `chunk_size = nothing`, `split_above = nothing`: a function with more than `split_above`
  instructions is split into sub-functions of at most `chunk_size` instructions. This
  shortens the compilation of a large function and costs some speed. `chunk_size` is
  $DEFAULT_CHUNK_SIZE by default. `split_above` is `chunk_size` when that is given;
  otherwise it is $UNSPLIT_LIMIT for scalar code and $DEFAULT_CHUNK_SIZE operations for
  matrix code. See [Large functions and the cache on disk](@ref).
- `nanmath = true`: an operation outside its domain returns NaN through NaNMath.jl, as in
  CasADi. With `false` it throws a `DomainError`, as in Base, and the code accepts any
  real number type.
- `expand = true`: a matrix (MX) function is expanded to scalar operations where CasADi
  can do that. With `false` the code keeps the matrix operations. See
  [Matrix code and functions without instructions](@ref).
- `cache = false`: with `true` or a directory, the compiled functions come from a package
  on disk that Julia compiles once, so a later session compiles nothing. It needs
  `expression = Val{false}`. Code that runs already must call the functions with
  `Base.invokelatest`. A Julia function in the graph must belong to a package. See
  [Large functions and the cache on disk](@ref).
- `optlevel = nothing`: with an integer from 0 to 3, Julia compiles the functions at this
  optimization level, as `Base.Experimental.@optlevel` sets it, in place of the level of
  the session (`-O`, 2 by default). A lower level shortens the compilation at the first
  call of a large function and can slow the later calls. It needs `expression = Val{false}`,
  and with `cache` it sets the level of the cached package.
- `solvers = nothing`: a [`CasADiCodegen.Solvers`](@ref Solvers) with the SciML solvers that the
  code calls in place of the solver objects of CasADi in the graph. Generated code never
  calls CasADi, so such an object with no solver throws an `ArgumentError`.

# Examples
```jldoctest
julia> using CasADiCodegen

julia> x = SX.sym("x", 2);

julia> f = CasadiFunction("f", [x], [x[1] * x[2], jacobian(x[1] * x[2], x)]);

julia> g, g! = build_function(f; expression = Val{false});

julia> g([2.0, 3.0])
([6.0], [3.0 2.0])

julia> out = (zeros(1), zeros(1, 2));

julia> g!(out, ([2.0, 3.0],));

julia> out[2]
1×2 Matrix{Float64}:
 3.0  2.0
```
"""
function build_function(f; expression = Val{true},
                        force_SA::Bool = false, nonzeros::Bool = false, name::Symbol = default_name(f),
                        chunk_size::Union{Nothing,Integer} = nothing,
                        split_above::Union{Nothing,Integer} = nothing, nanmath::Bool = true,
                        expand::Bool = true, cache::Union{Bool,AbstractString} = false,
                        solvers::Union{Nothing,Solvers} = nothing, optlevel::Union{Nothing,Integer} = nothing)
    check_optlevel(optlevel)
    @requires optlevel === nothing || !expression_flag(expression) "optlevel sets the compilation of the returned functions, so it needs expression = Val{false}"
    if cache !== false
        @requires !expression_flag(expression) "the cache holds compiled functions, so it needs expression = Val{false}"
        return cached_functions(f, cache_directory(cache); nonzeros, force_SA, name, expand, solvers, chunk_size, split_above, nanmath, optlevel)
    end
    optlevel === nothing ||
        return Base.ScopedValues.with(CONTEXT => OPTLEVEL_CONTEXTS[optlevel+1]) do
            build_function(f; expression, force_SA, nonzeros, name, chunk_size, split_above, nanmath, expand, solvers)
        end
    chunk_size = Split(chunk_size, split_above)
    alg = one_function(algorithm(f; expand, solvers), chunk_size)
    ref = nanmath ? global_ref : throwing(global_ref)
    if nonzeros
        return expression_flag(expression) ? Expr(:block, function_definitions(alg, name, chunk_size, ref; key = unique_key)...) :
               runtime_function(alg, chunk_size, ref)
    end
    if expression_flag(expression)
        Expr(:block, function_definitions(alg, core_name(name), chunk_size, ref; key = unique_key)...,
             form_definitions(alg, name, chunk_size, ref, force_SA; key = unique_key)...)
    else
        runtime_forms(alg, chunk_size, ref, force_SA)
    end
end

"""
    Inlined(g!)

The function `g!(res, arg)` with each call compiled inline. A call of a small generated
function otherwise costs more than its arithmetic, as in the right side of an ODE.
"""
struct Inlined{G<:RuntimeGeneratedFunction}
    g!::G
end
# The call of a runtime generated function goes through a method that the compiler does not
# inline when the body is large enough to matter; its inner function takes the inline request.
(i::Inlined)(res, arg) = @inline RuntimeGeneratedFunctions.generated_callfunc(i.g!, res, arg)

"""
    build_inplace(f; chunk_size = nothing, split_above = nothing, nanmath = true, expand = true, optlevel = nothing) -> g!

Return the compiled in-place function `g!(res, arg)` of `build_function(f; expression =
Val{false})` alone. The out-of-place forms are not generated: their static method has one
element per place of each output matrix, which for a large sparse Jacobian costs more
time and memory than the rest of the code. A scalar function of at most `INLINE_LIMIT`
instructions in one piece comes as an `Inlined` function.
"""
function build_inplace(f; chunk_size::Union{Nothing,Integer} = nothing, split_above::Union{Nothing,Integer} = nothing,
                       nanmath::Bool = true, expand::Bool = true, optlevel::Union{Nothing,Integer} = nothing)
    check_optlevel(optlevel)
    optlevel === nothing ||
        return Base.ScopedValues.with(CONTEXT => OPTLEVEL_CONTEXTS[optlevel+1]) do
            build_inplace(f; chunk_size, split_above, nanmath, expand)
        end
    split = Split(chunk_size, split_above)
    alg = one_function(algorithm(f; expand), split)
    (; inplace, frame) = runtime_inplace(alg, split, nanmath ? global_ref : throwing(global_ref))
    g! = RuntimeGeneratedFunctions.drop_expr(inplace)
    alg isa Algorithm && frame === nothing && length(alg.instructions) <= INLINE_LIMIT ? Inlined(g!) : g!
end
