# Scalar code around Julia functions. CasADi cannot expand a function that
# calls a Julia function, but it can expand what lies between the calls: the
# arguments of each call are a scalar function of the inputs and of the
# results of the earlier calls, and so are the outputs. The emitter writes
# these scalar functions and the calls into one Julia function with local
# variables, which runs much faster than matrix code in a work vector.

"""
    Stage

One call of a Julia function in a `StagedAlgorithm`.

- `pre`: the scalar algorithm `(inputs..., earlier results...) -> arguments`.
- `arguments[d]`: the result of `pre` that is argument `d` of the Julia
  function.
- `results`: the results of the call that have a place among the inputs of the
  later scalar algorithms.
- `live`: false when nothing reads the results. The value of a function at the
  point where CasADi takes its derivative is such a case, because the
  derivative functions of this package do not read it.
"""
struct Stage
    pre::Algorithm
    call::JuliaCall
    arguments::Vector{Int}
    results::Vector{Int}
    live::Bool
end

inputs_read(alg::Algorithm) = Set(ins.a for ins in alg.instructions if ins.op === :OP_INPUT)

"""
    mark_live(stages, post, n_in) -> Vector{Stage}

Return the stages with `live` set. A stage is live when `post` or the `pre` of
a later live stage reads one of its results. Result `p` of all calls, in the
order of the stages, is input `n_in + p` of the scalar algorithms.
"""
function mark_live(stages, post::Algorithm, n_in::Integer)
    needed = inputs_read(post)
    last = n_in + sum(stage -> length(stage.results), stages; init = 0)
    marked = Vector{Stage}(undef, length(stages))
    for s in length(stages):-1:1
        # Loop invariant: `needed` holds each input that `post` or a live stage after `s` reads.
        stage = stages[s]
        own = last-length(stage.results)+1:last
        last -= length(stage.results)
        live = any(in(needed), own)
        live && union!(needed, inputs_read(stage.pre))
        marked[s] = Stage(stage.pre, stage.call, stage.arguments, stage.results, live)
    end
    marked
end

"""
    StagedAlgorithm

A matrix (MX) function whose only operations that CasADi cannot expand are
calls of Julia functions at its top level: the `stages` in the order of the
calls, and the scalar algorithm `post`, which gives the outputs from the
inputs and the results of all calls. `matrix` is the same function as matrix
code, for the case that the code must be split into sub-functions.
"""
struct StagedAlgorithm
    name::String
    stages::Vector{Stage}
    post::Algorithm
    matrix::MXAlgorithm
    nnz_in::Vector{Int}
    nnz_out::Vector{Int}
    sparsity_in::Vector{C.Sparsity}
    sparsity_out::Vector{C.Sparsity}
    names_in::Vector{String}
    names_out::Vector{String}
end

instruction_count(alg::StagedAlgorithm) =
    sum(stage -> stage.live ? length(stage.pre.instructions) : 0, alg.stages; init = 0) + length(alg.post.instructions)

as_list(r) = (r isa MX ? MX[r] : collect(MX, r))::Vector{MX}

"""
    staged_algorithm(f) -> StagedAlgorithm or nothing

Return the staged form of the MX function `f`. Return `nothing` when `f` has no
call of a Julia function at its top level, or when CasADi cannot expand what
lies between the calls.
"""
function staged_algorithm(f::C.Function)
    (C.is_a(f, "MXFunction") && !C.has_free(f)) || return nothing
    names = op_names()
    n_in, n_out = C.n_in(f), C.n_out(f)
    inputs = MX[MX.sym(C.name_in(f, i), C.sparsity_in(f, i)) for i in 0:n_in-1]
    outputs = as_list(C.call(f, inputs, true, false))        # the graph of `f` on new symbols
    g = C.Function(C.name(f), inputs, outputs)
    # The nodes that carry the results of the calls: (call node, result) => node.
    carriers = Dict{Tuple{Int,Int},MX}()
    note(x) = C.is_output(x) && (carriers[(C.__hash__(C.dep(x, 0)), C.which_output(x))] = x)
    foreach(note, outputs)
    calls = Int[]
    for k in 0:C.n_instructions(g)-1
        op = names[C.instruction_id(g, k)]
        op in (:OP_INPUT, :OP_OUTPUT) && continue
        node = C.instruction_MX(g, k)
        foreach(d -> note(C.dep(node, d)), 0:C.n_dep(node)-1)
        op === :OP_CALL && C.class_name(C.which_function(node)) == "CallbackInternal" && push!(calls, k)
    end
    isempty(calls) && return nothing
    from, placeholders, actual = MX[], MX[], MX[]
    pure(name, expressions) = C.expand(C.Function(name, MX[inputs; placeholders],
        isempty(from) ? expressions : as_list(C.graph_substitute(expressions, from, placeholders))))
    apply(p) = as_list(p(MX[inputs; actual]...))
    stages = Stage[]
    try
        for k in calls
            node = C.instruction_MX(g, k)
            callee = C.which_function(node)
            haskey(JULIA_CALLS, C.__hash__(callee)) || return nothing
            call = JULIA_CALLS[C.__hash__(callee)][1]
            all(d -> C.nnz_in(callee, d) > 0, 0:call.nargs-1) || return nothing
            pre = pure("pre$k", MX[C.dep(node, d) for d in 0:call.nargs-1])
            computed = apply(pre)
            # The Julia function reads its arguments only, so the other inputs get zeros.
            arguments = MX[d < call.nargs ? computed[d+1] : MX.zeros(C.sparsity_in(callee, d)) for d in 0:C.n_in(callee)-1]
            results = as_list(callee(arguments...))
            read = Int[]
            for o in 0:C.n_out(callee)-1
                carrier = get(carriers, (C.__hash__(node), o), nothing)
                (carrier === nothing || C.nnz_out(callee, o) == 0) && continue    # nothing reads it, or it has no value
                push!(from, carrier)
                push!(placeholders, MX.sym("c$(k)_$o", C.sparsity_out(callee, o)))
                push!(actual, results[o+1])
                push!(read, o + 1)
            end
            push!(stages, Stage(Algorithm(pre), call, collect(1:call.nargs), read, true))
        end
        post = pure("post", outputs)
        names_in, names_out = [C.name_in(f, i) for i in 0:n_in-1], [C.name_out(f, j) for j in 0:n_out-1]
        staged = C.Function(C.name(f), inputs, apply(post), names_in, names_out)
        post = Algorithm(post)
        StagedAlgorithm(C.name(f), mark_live(stages, post, n_in), post, MXAlgorithm(staged; expand = true),
                        Int[C.nnz_in(f, i) for i in 0:n_in-1], Int[C.nnz_out(f, j) for j in 0:n_out-1],
                        [C.sparsity_in(f, i) for i in 0:n_in-1], [C.sparsity_out(f, j) for j in 0:n_out-1],
                        names_in, names_out)
    catch err
        err isa Union{ArgumentError,C.SwigError} || rethrow()
        nothing
    end
end

"""
    whole_statements(alg::StagedAlgorithm, ref; load, store) -> Vector{Any}

Return the statements of the staged algorithm in one piece, with local
variables only: the scalar code of each stage, the call of its Julia function
with static arrays as arguments, and the scalar code of the outputs. `load`
and `store` are those of `chunk_statements`.
"""
function whole_statements(alg::StagedAlgorithm, ref; load = (i, k) -> :(arg[$i][$k]),
                          store = (j, k, value) -> :(res[$j][$k] = $value))
    n_in = length(alg.nnz_in)
    sizes = [length(stage.results) for stage in alg.stages]
    first_result = cumsum(sizes) .- sizes
    carried = Vector{Vector{Any}}(undef, sum(sizes))     # the values of each result of a call that later code reads
    read(i, k) = i <= n_in ? load(i, k) : carried[i-n_in][k]
    sites = call_sites([stage.call.f for stage in alg.stages if stage.live])
    statements = first_statements(sites)
    for (s, stage) in enumerate(alg.stages)
        stage.live || continue
        call, pre = stage.call, stage.pre
        places = [nonzero_indices(sp) for sp in pre.sparsity_out]
        values = [Any[0.0 for _ in 1:C.size1(sp)*C.size2(sp)] for sp in pre.sparsity_out]
        append!(statements, chunk_statements(pre, eachindex(pre.instructions), Int[], Int[], ref; load = read,
            store = (j, k, value) -> (values[j][places[j][k]] = value; nothing), prefix = Symbol(:s, s)))
        arguments = [:($(ref(:Helpers, :static))(Tuple{$(call.shapes_in[d]...)}, $(Expr(:tuple, values[stage.arguments[d]]...))))
                     for d in 1:call.nargs]
        y = Symbol(:y, s)
        # The call is not a part of the code that runs without bounds checks.
        push!(statements, :($y = $(ref(:Helpers, :checked))($(sites[call.f]), $(arguments...))))
        returned = cumsum(call.nnz_out .> 0)
        for (position, o) in enumerate(stage.results)
            n, value = call.nnz_out[o], Symbol(y, :_, o)
            push!(statements, :($value = $(returned[end] == 1 ? :($y isa Tuple ? $y[1] : $y) : :($y[$(returned[o])]))))
            push!(statements, :(length($value) == $n || throw(DimensionMismatch(
                string($("Julia function \"$(call.name)\": result $o needs $n elements, got "), length($value))))))
            locals = Any[Symbol(:c, s, :_, o, :_, k) for k in 1:n]
            append!(statements, (:($(locals[k]) = $value[$k]) for k in 1:n))
            carried[first_result[s]+position] = locals
        end
    end
    append!(statements, chunk_statements(alg.post, eachindex(alg.post.instructions), Int[], Int[], ref;
                                         load = read, store, prefix = :p))
    statements
end

"Replace each Julia function of the stages by `julia(f)`, which names it in a source file."
function named_calls(alg::StagedAlgorithm, definitions, julia)
    sites = call_sites([stage.call.f for stage in alg.stages if stage.live])
    # In the order of the stages, so that a source file is the same text in each session.
    names = IdDict{Any,Any}()
    for stage in alg.stages
        stage.live && get!(() -> julia(sites[stage.call.f]), names, sites[stage.call.f])
    end
    isempty(first_statements(sites)) || (names[new_call!] = julia(new_call!))
    rename(ex::Expr) = Expr(ex.head, map(rename, ex.args)...)
    rename(other) = get(names, other, other)
    Expr[rename(definition) for definition in definitions]
end
named_calls(alg, definitions, julia) = definitions

named_frame(alg::StagedAlgorithm, name::Symbol, chunk_size::ChunkSize, ref; julia = identity) = nothing
runtime_frame(alg::StagedAlgorithm, chunk_size::ChunkSize, ref) = nothing

function_definitions(alg::StagedAlgorithm, name::Symbol, chunk_size::ChunkSize, ref; key = identity, julia = identity) =
    named_calls(alg, Expr[entry_function(alg, :($name(res, arg)), nothing, ref, key(name))], julia)

"""
    one_function(alg, split) -> algorithm

Return the algorithm that the emitter reads for the keywords `split`. A staged
algorithm is one function, so when it must be split it gives its matrix code.
"""
one_function(alg, split::Split) = alg
one_function(alg::StagedAlgorithm, split::Split) =
    (n = instruction_count(alg); chunk_size_for(n, split) >= n ? alg : alg.matrix)
