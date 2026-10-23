# Julia code for a matrix algorithm. As in the CasADi virtual machine, every
# value is written to one work vector `w`, and each instruction becomes a few
# loops over its nonzeros. A called function is emitted once, with the signature
# `(w)`: it has a part of the work vector of its own, its frame, which starts
# with the nonzeros of its inputs and of its outputs. A function takes the
# work vector only, because a call between runtime generated functions can
# allocate memory for a tuple of arrays. An input and a constant are not copied
# to a work slot: the instructions read them where they are (see `Place`), so
# that a call does not cost time in proportion to a table that it reads.

"""
    Emission

State that one call of the emitter collects.

- `definitions`: `name => statements` of each function `name(w)`, each after
  the functions that it calls.
- `constants`, `indices`: the values of the constant table and of the index
  table. The statements name them `constants_name` and `indices_name`.
- `callees`, `julia`, `free`: see the comments on the fields.
"""
struct Emission
    prefix::String
    chunk_size::Split
    ref::Any
    definitions::Vector{Pair{Symbol,Vector{Any}}}
    constants::Vector{Float64}
    constants_name::Symbol
    indices::Vector{Int}
    indices_name::Symbol
    callees::IdDict{Any,Tuple{Symbol,Vector{Int},Vector{Int}}}   # algorithm => (function name, offsets of inputs, of outputs)
    julia::Vector{Pair{Symbol,Any}}              # name in the statements => Julia function
    free::Base.RefValue{Int}                     # length of the work vector that the frames use so far
end

function Emission(name::Symbol, chunk_size::ChunkSize, ref)
    @requires Base.isidentifier(name) "\"$name\" is not a valid function name"
    Emission(String(chopsuffix(String(name), "!")), chunk_size isa Split ? chunk_size : Split(chunk_size, nothing), ref, Pair{Symbol,Vector{Any}}[],
             Float64[], Symbol(name, :_constants), Int[], Symbol(name, :_indices),
             IdDict{Any,Tuple{Symbol,Vector{Int},Vector{Int}}}(), Pair{Symbol,Any}[], Ref(0))
end

element(offset::Int, i::Int) = :(w[$(offset + i)])
element(offset::Int, i) = offset == 0 ? :(w[$i]) : :(w[$offset + $i])

"""
    Place(array, offset, convert)
    Place(offset)
    Place(value)

Where a value of a matrix algorithm is read: nonzero `i` is `array[offset + i]`. The
array is the work vector `w` (the second form), the constant table, or the argument
`input_name(k)`, which the entry point reads in place. With `convert`, a read gives
the element type of `w`, as a copy to the work vector would. In the third form, every
nonzero is the number `value`.
"""
struct Place
    array::Union{Symbol,Float64}
    offset::Int
    convert::Bool
end
Place(offset::Int) = Place(:w, offset, false)
Place(value::Float64) = Place(value, 0, false)

shifted(p::Place, by::Int) = Place(p.array, p.offset + by, p.convert)
in_work(p::Place, offset::Int) = p.array === :w && p.offset == offset

"The name under which the entry point reads its argument `k` in place."
input_name(k::Int) = Symbol(:input_, k)

"The element at the index expression `index` of the array of `p`."
function element_at(p::Place, index)
    p.array isa Float64 && return p.array
    read = :($(p.array)[$index])
    p.convert ? :(convert(eltype(w), $read)) : read
end
element(p::Place, i) = p.array === :w ? element(p.offset, i) :
    element_at(p, i isa Int ? p.offset + i : p.offset == 0 ? i : :($(p.offset) + $i))

loop(n::Int, statement) = n == 0 ? Any[] : n == 1 ? Any[statement(1)] :
    Any[Expr(:for, :(i = 1:$n), Expr(:block, statement(:i)))]

stride(first::Int, step::Int, i::Int) = first + step * (i - 1)
stride(first::Int, step::Int, i) =
    step == 1 ? (first == 1 ? i : :($(first - 1) + $i)) : :($(first - step) + $step * $i)

"""
    runs(source, keep) -> Vector{NTuple{4,Int}}

Split the positions `k` with `keep[k]` into runs `(k, first, step, length)`:
positions `k:k+length-1` over which `source` advances from `first` by the fixed
`step`. The runs cover each kept position once, in rising order.
"""
function runs(source::AbstractVector{<:Integer}, keep)
    found = NTuple{4,Int}[]
    k, n = 1, length(source)
    while k <= n
        # Loop invariant: `found` covers exactly the kept positions below k.
        if !keep[k]
            k += 1
            continue
        end
        len, step = 1, 1
        if k < n && keep[k+1]
            step = source[k+1] - source[k]
            len = 2
            while k + len <= n && keep[k+len] && source[k+len] - source[k+len-1] == step
                len += 1
            end
        end
        push!(found, (k, source[k], step, len))
        k += len
    end
    @ensures sum(r -> r[4], found; init = 0) == count(keep) "runs must cover every kept position once"
    found
end

"""
    blocks(found) -> Vector{NTuple{7,Int}}

Group consecutive runs of `found` (see `runs`) that have one step and one length, and
whose positions and first values advance by fixed strides, into blocks
`(k, first, step, length, count, k_stride, first_stride)`: run `j` of a block, from 0 to
`count - 1`, is `(k + k_stride * j, first + first_stride * j, step, length)`. A transpose
gives one block where it gives one run per column.
"""
function blocks(found::Vector{NTuple{4,Int}})
    grouped = NTuple{7,Int}[]
    r = 1
    while r <= length(found)
        # Loop invariant: `grouped` holds the runs before r, in order.
        k, first, step, len = found[r]
        count, dk, dfirst = 1, 0, 0
        if r < length(found) && found[r+1][3] == step && found[r+1][4] == len
            dk, dfirst = found[r+1][1] - k, found[r+1][2] - first
            count = 2
            while r + count <= length(found) &&
                  found[r+count] == (k + dk * count, first + dfirst * count, step, len)
                count += 1
            end
        end
        push!(grouped, (k, first, step, len, count, dk, dfirst))
        r += count
    end
    @ensures sum(b -> b[5], grouped; init = 0) == length(found) "blocks must hold every run once"
    grouped
end

"""
    copy_loop(statement, b) -> Vector{Any}

Return the statements that run `statement(i, j)` for each position `i` in `1:length` of
each run `j` in `0:count-1` of the block `b` of `blocks`. An index is an integer when it
takes one value.
"""
function copy_loop(statement, b::NTuple{7,Int})
    _, _, _, len, count, _, _ = b
    count == 1 && return loop(len, i -> statement(i, 0))
    len == 1 && return Any[Expr(:for, :(j = 0:$(count - 1)), Expr(:block, statement(1, :j)))]
    Any[Expr(:for, Expr(:block, :(j = 0:$(count - 1)), :(i = 1:$len)), Expr(:block, statement(:i, :j)))]
end

"""
    indexed(em, tuples, statement) -> Vector{Any}

Return statements that run `statement(indices...)` for each tuple of indices.
Up to 8 tuples are written out one by one. More become one loop over the
variable `t` that reads the index table of `em`.
"""
function indexed(em::Emission, tuples::Vector{NTuple{N,Int}}, statement) where {N}
    length(tuples) <= 8 && return Any[statement(t...) for t in tuples]
    base = length(em.indices)
    foreach(t -> append!(em.indices, t), tuples)
    entry(j) = :($(em.indices_name)[$(base - N + j) + $N * t])
    Any[Expr(:for, :(t = 1:$(length(tuples))), Expr(:block, statement(ntuple(entry, N)...)))]
end

"""
    copy_statements(maps, operands, result_offset) -> Vector{Any}

Return the statements of a copy operation for one result, which starts after
`w[result_offset]`. `maps[d]` is the map of operand `d`, which is read at the
`Place` `operands[d]`, or after `w[operands[d]]` for an integer. An operand at the
offset of the result stays in place.
"""
copy_statements(maps::Vector{Vector{Int}}, operand_offsets::Vector{Int}, result_offset::Int) =
    copy_statements(maps, Place.(operand_offsets), result_offset)

function copy_statements(maps::Vector{Vector{Int}}, operands::Vector{Place}, result_offset::Int)
    n = length(first(maps))
    statements = Any[]
    fed = [any(m -> m[k] != 0, maps) for k in 1:n]
    for (k, _, _, len) in runs(1:n, .!fed)
        append!(statements, loop(len, i -> :($(element(result_offset + k - 1, i)) = zero(eltype(w)))))
    end
    seen = falses(n)
    order = sortperm([!in_work(p, result_offset) for p in operands])    # in place first
    for d in order
        from = maps[d]
        if !in_work(operands[d], result_offset)
            for first_use in (true, false)
                keep = [from[k] != 0 && seen[k] != first_use for k in 1:n]
                for b in blocks(runs(from, keep))
                    k, source, step, _, _, dk, dsource = b
                    shift(index, by, j) = j == 0 ? index : index isa Int && by == 0 ? index : :($index + $by * $j)
                    target(i, j) = :(w[$(shift(i isa Int ? result_offset + k - 1 + i : :($(result_offset + k - 1) + $i), dk, j))])
                    value(i, j) = element_at(operands[d], shift(stride(operands[d].offset + source, step, i), dsource, j))
                    append!(statements, copy_loop(b) do i, j
                        first_use ? :($(target(i, j)) = $(value(i, j))) : :($(target(i, j)) += $(value(i, j)))
                    end)
                end
            end
        end
        seen .|= from .!= 0
    end
    statements
end

"""
    reads_in_place(ins) -> Bool

Whether the statements of `ins` read each operand at its `Place`, so that an operand
in the constant table or in an argument needs no copy to its work slot first. The
grid of `:OP_LOW` must be an array (see `needs_copy`).
"""
function reads_in_place(ins::MXInstruction)
    op = ins.op
    (op === :OP_OUTPUT || op in COPY_OPERATIONS || haskey(INDEX_OPERATIONS, op)) && return true
    op === :OP_CALL && return !(ins.data isa JuliaCall)
    !(op in (:OP_INPUT, :OP_CONST) || haskey(LINEAR_OPERATIONS, op) || haskey(PASS_OPERATIONS, op) || haskey(TABLE_OPERATIONS, op))
end

"Whether `ins` reads its operand slot `s`, now at `place`, in the work vector only."
needs_copy(ins::MXInstruction, s::Int, place::Place) = place.array !== :w &&
    (!reads_in_place(ins) || ins.op === :OP_LOW && s == ins.in[1] && place.array isa Float64)

"""
    instruction_statements(em, alg, ins, reads, writes, scratch, offset_out) -> Vector{Any}

Return the statements of one instruction other than `:OP_INPUT` and `:OP_CONST`, which
give an operand a `Place` and no statements (see `body!`). Operand slot `s` is
read at `reads[s]`, which is in the work vector unless `reads_in_place(ins)`; result slot
`s` is written after `w[writes[s]]`. Nonzero `k` of result `j` of the function is
`w[offset_out[j] + k]`, and `scratch` is the offset of the work space of `scratch_size`.
"""
function instruction_statements(em::Emission, alg::MXAlgorithm, ins::MXInstruction, reads::Vector{Place}, writes::Vector{Int}, scratch, offset_out)
    op = ins.op
    nnz(s) = alg.slot_nnz[s]
    @requires !(op in (:OP_INPUT, :OP_CONST)) "$op has no statements"
    in_place = reads_in_place(ins)
    @requires !any(s -> needs_copy(ins, s, reads[s]), ins.in) "$op reads its operands in the work vector only"
    at = in_place ? Int[] : Int[reads[s].offset for s in ins.in]
    if op === :OP_OUTPUT
        slot, (result, offset) = ins.in[1], ins.data
        loop(nnz(slot), i -> :($(element(offset_out[result] + offset, i)) = $(element(reads[slot], i))))
    elseif op === :OP_CALL && ins.data isa JuliaCall
        julia_call_statements(em, ins, at, writes)
    elseif op === :OP_CALL
        # Copy the arguments to the frame of the function, call it, and copy the results that are read.
        name, callee_in, callee_out = callee!(em, ins.data)
        statements = Any[]
        for (d, slot) in enumerate(ins.in)
            append!(statements, loop(ins.data.nnz_in[d], i -> :($(element(callee_in[d], i)) = $(element(reads[slot], i)))))
        end
        push!(statements, :($name(w)))
        for (o, slot) in enumerate(ins.out)
            slot == 0 || append!(statements, loop(ins.data.nnz_out[o], i -> :($(element(writes[slot], i)) = $(element(callee_out[o], i)))))
        end
        statements
    elseif op in COPY_OPERATIONS
        statements = Any[]
        for (o, slot) in enumerate(ins.out)
            slot == 0 || append!(statements, copy_statements((ins.data::Vector{Vector{Vector{Int}}})[o], reads[ins.in], writes[slot]))
        end
        statements
    elseif haskey(LINEAR_OPERATIONS, op)
        linear_statements(em, ins, writes[ins.out[1]], at, alg.slot_nnz[ins.in], scratch)
    elseif haskey(INDEX_OPERATIONS, op)
        index_statements(em, ins, writes[ins.out[1]], reads[ins.in], alg.slot_nnz[ins.in])
    elseif haskey(PASS_OPERATIONS, op)
        pass_statements(em, ins, writes[ins.out[1]], at, nnz(ins.out[1]))
    elseif haskey(TABLE_OPERATIONS, op)
        table_statements(em, ins, writes[ins.out[1]], at, alg.slot_nnz[ins.in], nnz(ins.out[1]), scratch)
    else
        slot = ins.out[1]
        n = nnz(slot)
        operand(s, i) = element(reads[s], nnz(s) == n ? i : 1)
        loop(n, i -> :($(element(writes[slot], i)) = $(OPERATIONS[op].build(em.ref, (operand(s, i) for s in ins.in)...))))
    end
end

"""
    julia_call_statements(em, ins, at, writes) -> Vector{Any}

Return statements that call a Julia function directly and copy each result
that is read to its work slot. Operand `d` is read after `w[at[d]]`, and result
slot `s` is written after `w[writes[s]]`. The function gets views of the work
vector, as vectors or as matrices. It returns only the results that have
nonzeros. A result of the wrong length throws `DimensionMismatch`.
"""
function julia_call_statements(em::Emission, ins::MXInstruction, at::Vector{Int}, writes::Vector{Int})
    call = ins.data
    name = Symbol(em.prefix, :_julia_, length(em.julia) + 1)
    known = findfirst(entry -> entry[2] === call.f, em.julia)
    known === nothing ? push!(em.julia, name => call.f) : (name = em.julia[known][1])
    part(start, n) = :(view(w, $(start + 1):$(start + n)))
    arguments = map(1:call.nargs) do d
        dims = call.shapes_in[d]
        values = part(at[d], call.nnz_in[d])
        length(dims) == 1 ? values : :(reshape($values, $(dims[1]), $(dims[2])))
    end
    copies = Any[]
    returned = cumsum(call.nnz_out .> 0)     # the function does not return a structural zero
    for (o, slot) in enumerate(ins.out)
        n = call.nnz_out[o]
        (slot == 0 || n == 0) && continue
        value = returned[end] == 1 ? :(y isa Tuple ? y[1] : y) : :(y[$(returned[o])])
        push!(copies, :(length($value) == $n || throw(DimensionMismatch(
            string($("Julia function \"$(call.name)\": result $o needs $n elements, got "), length($value))))))
        push!(copies, :(copyto!($(part(writes[slot], n)), $value)))
    end
    # The call is not a part of the code that runs without bounds checks.
    Any[Expr(:let, :(y = $(em.ref(:Helpers, :checked))($name, $(arguments...))), Expr(:block, copies...))]
end

"""
    linear_statements(em, ins, out, at, sizes, scratch) -> Vector{Any}

Return the statements of a linear algebra operation. The result starts after
`w[out]`, and operand `d` has `sizes[d]` nonzeros after `w[at[d]]`. Operations
that factorize a matrix use the work space after `w[scratch]`.

The sums run in the order of the CasADi runtime, so they are rounded in the
same way. Determinant, inverse and solve use the LU factorization of
LinearAlgebra, where CasADi uses a QR factorization. A solve with a large
sparse matrix uses the sparse QR factorization of CasADi (see `qr_tables`).
"""
function linear_statements(em::Emission, ins::MXInstruction, out, at, sizes, scratch)
    op, data = ins.op, ins.data
    build(name, arguments...) = OPERATIONS[name].build(em.ref, arguments...)
    result = element(out, 1)
    # An accumulator that is not the result slot is copied to it first.
    keep() = at[1] == out ? Any[] : loop(sizes[1], i -> :($(element(out, i)) = $(element(at[1], i))))
    fold(start, update) = Any[:($result = $start), loop(sizes[1], i -> :($result = $(update(i))))...]
    nested(ranges, statement) = Any[Expr(:for, Expr(:block, ranges...), Expr(:block, statement))]
    library(name) = em.ref(:LinearAlgebra, name)
    if op === :OP_MTIMES
        product(kz, kx, ky) = :($(element(out, kz)) += $(element(at[2], kx)) * $(element(at[3], ky)))
        data.dims === nothing && return Any[keep()..., indexed(em, data.terms, product)...]
        nr, nk, nc = data.dims
        Any[keep()..., nested((:(j = 1:$nc), :(k = 1:$nk), :(i = 1:$nr)),
                              product(:(i + $nr * (j - 1)), :(i + $nr * (k - 1)), :(k + $nk * (j - 1))))...]
    elseif op === :OP_DOT
        fold(:(zero(eltype(w))), i -> :($result + $(element(at[1], i)) * $(element(at[2], i))))
    elseif op === :OP_NORMF
        Any[fold(:(zero(eltype(w))), i -> :($result + $(element(at[1], i)) * $(element(at[1], i))))...,
            :($result = $(build(:OP_SQRT, result)))]
    elseif op === :OP_NORM1
        fold(:(zero(eltype(w))), i -> :($result + $(build(:OP_FABS, element(at[1], i)))))
    elseif op === :OP_NORMINF
        fold(:(zero(eltype(w))), i -> build(:OP_FMAX, result, build(:OP_FABS, element(at[1], i))))
    elseif op === :OP_MMIN
        fold(data.dense ? Inf : 0.0, i -> build(:OP_FMIN, result, element(at[1], i)))
    elseif op === :OP_MMAX
        fold(data.dense ? -Inf : 0.0, i -> build(:OP_FMAX, result, element(at[1], i)))
    elseif op === :OP_KRON
        indexed(em, data.terms, (kz, kx, ky) -> :($(element(out, kz)) = $(element(at[1], kx)) * $(element(at[2], ky))))
    elseif op === :OP_LOGSUMEXP
        # As the CasADi routine `casadi_logsumexp`: the largest element is taken out of the sum.
        n = sizes[1]
        largest = Expr(:for, :(i = 2:$n), :(if $(element(at[1], :i)) > m
                                               m, first = $(element(at[1], :i)), i
                                           end))
        others = Expr(:for, :(i = 1:$n), :(i == first || (r += $(build(:OP_EXP, :($(element(at[1], :i)) - m))))))
        Any[Expr(:let, Expr(:block, :(m = $(element(at[1], 1))), :(first = 1), :(r = zero(eltype(w)))),
                 Expr(:block, largest, others, :($result = $(build(:OP_LOG1P, :r)) + m)))]
    elseif op in (:OP_BILIN, :OP_RANK1)
        term(el, r, c) = op === :OP_BILIN ?
            :($result += $(element(at[2], r)) * $(element(at[1], el)) * $(element(at[3], c))) :
            :($(element(out, el)) += $(element(at[2], 1)) * $(element(at[3], r)) * $(element(at[4], c)))
        start = op === :OP_BILIN ? Any[:($result = zero(eltype(w)))] : keep()
        data.dims === nothing &&
            return Any[start..., indexed(em, NTuple{3,Int}[(k, data.rows[k], data.columns[k]) for k in eachindex(data.rows)], term)...]
        nr, nc = data.dims
        Any[start..., nested((:(c = 1:$nc), :(r = 1:$nr)), term(:(r + $nr * (c - 1)), :r, :c))...]
    elseif op === :OP_SOLVE && data.qr !== nothing
        base = length(em.indices)
        append!(em.indices, data.qr)
        helper(name) = em.ref(:Helpers, name)
        Any[:($(helper(Symbol("qr_factor!")))(w, $(em.indices_name), $base, $(at[2]), $scratch)), keep()...,
            :($(helper(Symbol("qr_solve!")))(w, $(em.indices_name), $base, $scratch, $out, $(data.nrhs), $(data.tr)))]
    else
        n = data.n
        matrix(first, columns) = :(reshape(view(w, $(first + 1):$(first + n * columns)), $n, $columns))
        dense = copy_statements(Vector{Int}[data.place], Int[at[op === :OP_SOLVE ? 2 : 1]], Int(scratch))
        factors = :($(library(:lu!))($(matrix(scratch, n)); check = false))
        if op === :OP_DETERMINANT
            Any[dense..., :($result = $(library(:det))($factors))]
        elseif op === :OP_INVERSE
            Any[dense..., loop(n * n, i -> :($(element(out, i)) = zero(eltype(w))))...,
                loop(n, i -> :(w[$(stride(out + 1, n + 1, i))] = one(eltype(w))))...,
                :($(library(:ldiv!))($factors, $(matrix(out, n))))]
        else
            factors = data.tr ? :($(library(:transpose))($factors)) : factors
            Any[dense..., keep()..., :($(library(:ldiv!))($factors, $(matrix(out, data.nrhs))))]
        end
    end
end

"""
    index_statements(em, ins, out, at, sizes) -> Vector{Any}

Return the statements of an operation of `INDEX_OPERATIONS`, with the arguments
of `linear_statements`, except that operand `d` is read at the `Place` `at[d]`.
They follow `index_model`: an index out of range reads NaN and writes nothing.
"""
function index_statements(em::Emission, ins::MXInstruction, out, at::Vector{Place}, sizes)
    op, data = ins.op, ins.data
    helper(name) = em.ref(:Helpers, name)
    op === :OP_LOW &&
        return loop(sizes[2], i -> :($(element(out, i)) = $(helper(:low))($(at[1].array), $(at[1].offset), $(sizes[1]), $(element(at[2], i)), $(data.mode))))
    form = data::IndexForm
    reads = op === :OP_GETNONZEROS_PARAM
    keys = at[(reads ? 2 : 3):end]
    two = !isempty(form.second)
    function index(offset, a, b)
        sum = :($(helper(:whole))($(element(keys[1], a))))
        two && (sum = :($sum + $(helper(:whole))($(element(keys[2], b)))))
        offset isa Integer && offset == 0 ? sum : :($offset + $sum)
    end
    tuples = [(e, form.offsets[e], form.first[e], two ? form.second[e] : 0) for e in eachindex(form.offsets)]
    uniform = at[1].array isa Float64
    reads && return indexed(em, tuples, (e, offset, a, b) -> :($(element(out, e)) = $(uniform ?
        :(ifelse(0 <= $(index(offset, a, b)) < $(sizes[1]), $(at[1].array), NaN)) :
        :($(helper(:gather))($(at[1].array), $(at[1].offset), $(sizes[1]), $(index(offset, a, b)))))))
    copied = in_work(at[1], out) ? Any[] : loop(sizes[1], i -> :($(element(out, i)) = $(element(at[1], i))))
    update = op === :OP_SETNONZEROS_PARAM ? :(=) : :(+=)
    Any[copied..., indexed(em, tuples, (e, offset, a, b) ->
        Expr(:let, :(index = $(index(offset, a, b))),
             :(if 0 <= index < $(sizes[1])
                   $(Expr(update, :(w[$(out + 1) + index]), element(at[2], e)))
               end)))...]
end

"""
    pass_statements(em, ins, out, at, n) -> Vector{Any}

Return the statements of an operation of `PASS_OPERATIONS`: the action on the
`n` nonzeros after `w[at[1]]`, then their copy to the result after `w[out]`.
"""
function pass_statements(em::Emission, ins::MXInstruction, out, at, n)
    op, data = ins.op, ins.data
    helper(name) = em.ref(:Helpers, name)
    values = :(view(w, $(at[1] + 1):$(at[1] + n)))
    action = op === :OP_MONITOR ? Any[:($(helper(:monitor))($data, $values))] :
             op === :OP_ASSERTION ? Any[:($(helper(:assertion))($(element(at[2], 1)), $data))] :
             op === :OP_DUMP ? Any[:($(helper(:write_matrix))($(data.base), $(data.dims[1]), $(data.dims[2]), $(Tuple(data.rows)), $(Tuple(data.columns)), $values))] : Any[]
    copied = at[1] == out ? Any[] : loop(n, i -> :($(element(out, i)) = $(element(at[1], i))))
    Any[action..., copied...]
end

"""
    table_statements(em, ins, out, at, sizes, n, scratch) -> Vector{Any}

Return the statements of an operation of `TABLE_OPERATIONS`, with the arguments
of `linear_statements`; the result has `n` nonzeros. The knots and the fixed
coefficients of a B-spline go to the constant table. The integers of each
operation go to the index table.
"""
function table_statements(em::Emission, ins::MXInstruction, out, at, sizes, n, scratch)
    op, data = ins.op, ins.data
    helper(name) = em.ref(:Helpers, name)
    if op === :OP_BSPLINE
        knots, table = length(em.constants), length(em.indices)
        append!(em.constants, data.knots)
        append!(em.indices, bspline_table(data))
        fixed = data.coefficients !== nothing
        coefficients = fixed ? length(em.constants) : at[2]
        fixed && append!(em.constants, data.coefficients)
        Any[:($(helper(Symbol("bspline!")))(w, $(em.constants_name), $knots, $(em.indices_name), $table,
                                           $(fixed ? em.constants_name : :w), $coefficients, $(at[1]), $out, $scratch))]
    elseif op === :OP_EINSTEIN
        copied = at[1] == out ? Any[] : loop(sizes[1], i -> :($(element(out, i)) = $(element(at[1], i))))
        Any[copied..., indexed(em, data.terms, (kz, kx, ky) -> :($(element(out, kz)) += $(element(at[2], kx)) * $(element(at[3], ky))))...]
    elseif op === :OP_KRON_CONTRACT
        ma, na, mb, nb = data.dims
        data.inner ? Any[:($(helper(Symbol("kron_inner!")))(w, $(at[1]), $ma, $na, $(at[2]), $mb, $nb, $out))] :
                     Any[:($(helper(Symbol("kron_outer!")))(w, $(at[1]), $mb, $nb, $(at[2]), $ma, $na, $out))]
    elseif op === :OP_CONVEXIFY
        base = length(em.indices)
        append!(em.indices, data.table)
        Any[:($(helper(Symbol("convexify!")))(w, $(em.indices_name), $base, $(data.margin), $(at[1]), $out, $scratch))]
    elseif op === :OP_HORZREPSUM
        # As CasADi: the result is cleared, and the blocks are added one after the other.
        Any[loop(n, i -> :($(element(out, i)) = zero(eltype(w))))...,
            Expr(:for, Expr(:block, :(r = 0:$(data.n - 1)), :(i = 1:$n)), :(w[$out + i] += w[$(at[1]) + i + $n * r]))]
    else
        base = length(em.indices)
        append!(em.indices, data.rows)
        Any[:($(element(out, 1)) = $(helper(:find))(w, $(at[1]), $(sizes[1]), $(em.indices_name), $base, $(data.last)))]
    end
end

scratch_size(ins::MXInstruction) = ins.op === :OP_SOLVE && ins.data.qr !== nothing ? ins.data.qr[4] + ins.data.qr[5] + 2 * ins.data.n :
    ins.op in (:OP_DETERMINANT, :OP_INVERSE, :OP_SOLVE) ? ins.data.n^2 : ins.op === :OP_BSPLINE ? bspline_scratch(ins.data) : ins.op === :OP_CONVEXIFY ? ins.data.scratch : 0

"""
    frame!(em, alg, own) -> (offset_in, offset_out, start)

Reserve the frame of one function in the work vector: the nonzeros of each
input, those of each output, and `own` elements after `w[start]`. Nonzero `k`
of input `i` is `w[offset_in[i] + k]`.
"""
function frame!(em::Emission, alg, own::Int)
    sizes = [alg.nnz_in; alg.nnz_out]
    offsets = em.free[] .+ cumsum(sizes) .- sizes
    start = em.free[] + sum(sizes)
    em.free[] = start + own
    offsets[1:length(alg.nnz_in)], offsets[length(alg.nnz_in)+1:end], start
end

"""
    callee!(em, alg) -> (name, offset_in, offset_out)

Return the name of the function `name(w)` that runs the called algorithm
`alg`, and the work vector offsets of its inputs and outputs. The first call
adds the definition to `em`; later calls with the same algorithm reuse it.
"""
function callee!(em::Emission, alg)
    haskey(em.callees, alg) && return em.callees[alg]
    taken = Set(first(entry) for entry in values(em.callees))
    name = Symbol(em.prefix, :_, alg.name, "!")
    count = 1
    while name in taken
        count += 1
        name = Symbol(em.prefix, :_, alg.name, :_, count, "!")
    end
    em.callees[alg] = (name, Int[], Int[])        # reserve the name
    statements, offset_in, offset_out = body!(em, alg, name)
    push!(em.definitions, name => statements::Vector{Any})
    em.callees[alg] = (name, offset_in, offset_out)
end

"The ranges of the instructions of each sub-function of an algorithm of `n` instructions; one range for none."
chunks(em::Emission, n::Int) = chunk_ranges(n, chunk_size_for(n, em.chunk_size; limit = DEFAULT_CHUNK_SIZE))

"""
    chunked!(em, name, parts) -> Vector{Any}

Return the statements of `parts`, which has one statement list per instruction.
When they exceed the chunk size, add sub-functions to `em` and return the calls
to them.
"""
function chunked!(em::Emission, name::Symbol, parts)
    ranges = chunks(em, length(parts))
    length(ranges) <= 1 && return reduce(vcat, parts; init = Any[])
    calls = Any[]
    for (c, range) in enumerate(ranges)
        push!(em.definitions, chunk_name(name, c) => reduce(vcat, parts[range]; init = Any[]))
        push!(calls, :($(chunk_name(name, c))(w)))
    end
    calls
end

"""
    body!(em, alg, name) -> (statements, offset_in, offset_out)

Return statements over `w` that run `alg` in a new frame, and the offsets of
its inputs and outputs (see `frame!`). Sub-functions are added to `em` under
names derived from `name`.
"""
function body!(em::Emission, alg::Algorithm, name::Symbol)
    base = em.free[]
    offset_in, offset_out, total = work_layout(alg)
    em.free[] += total
    chunks = split_statements(alg, em.chunk_size, em.ref; base)
    if isempty(chunks)
        statements = whole_statements(alg, em.ref; load = (i, k) -> :(w[$(base + offset_in[i] + k)]),
                                      store = (j, k, value) -> :(w[$(base + offset_out[j] + k)] = $value))
        return (statements, base .+ offset_in, base .+ offset_out)
    end
    for (c, statements) in enumerate(chunks)
        push!(em.definitions, chunk_name(name, c) => statements)
    end
    (Any[:($(chunk_name(name, c))(w)) for c in eachindex(chunks)], base .+ offset_in, base .+ offset_out)
end

# A value is written to its work slot, but an input and a constant are read where they
# are: input `i` in the frame, or as `input_name(i)` with `direct`, a constant of several
# values in the constant table, and one of equal values as that number. `reads[s]` is the
# place of the value of slot `s`, until an instruction writes the slot. An instruction
# that reads in the work vector only (see `needs_copy`) first copies such an operand to
# its slot.
function body!(em::Emission, alg::MXAlgorithm, name::Symbol; direct::Bool = false)
    own = sum(alg.slot_nnz)
    # Instructions run one after the other, so they share one work space after the slots.
    unused = maximum(scratch_size, alg.instructions; init = 0)
    offset_in, offset_out, start = frame!(em, alg, own + unused)
    home = start .+ cumsum(alg.slot_nnz) .- alg.slot_nnz
    inputs = direct ? [Place(input_name(i), 0, true) for i in eachindex(offset_in)] : Place.(offset_in)
    reads = Place.(home)
    parts = map(alg.instructions) do ins
        if ins.op === :OP_INPUT
            (argument, offset), slot = ins.data, ins.out[1]
            reads[slot] = shifted(inputs[argument], offset)
            return Any[]
        elseif ins.op === :OP_CONST
            values = ins.data::Vector{Float64}
            uniform = isempty(values) || all(isequal(first(values)), values)
            reads[ins.out[1]] = uniform ? Place(isempty(values) ? 0.0 : first(values)) : Place(em.constants_name, length(em.constants), false)
            uniform || append!(em.constants, values)
            return Any[]
        end
        statements = Any[]
        for s in unique(ins.in)
            needs_copy(ins, s, reads[s]) || continue
            append!(statements, loop(alg.slot_nnz[s], i -> :($(element(home[s], i)) = $(element(reads[s], i)))))
            reads[s] = Place(home[s])
        end
        append!(statements, instruction_statements(em, alg, ins, reads, home, start + own, offset_out))
        for s in ins.out
            s == 0 || (reads[s] = Place(home[s]))
        end
        # Loop invariant: `reads[s]` holds the value that slot `s` has after this instruction.
        statements
    end
    (chunked!(em, name, parts), offset_in, offset_out)
end

# A switch copies its arguments to the frame of the function that it selects,
# runs that function only, and copies the results back.
function body!(em::Emission, switch::SwitchCall, name::Symbol)
    offset_in, offset_out, _ = frame!(em, switch, 0)
    function branch(alg)
        callee, callee_in, callee_out = callee!(em, alg)
        statements = Any[]
        for d in eachindex(callee_in)
            append!(statements, loop(switch.nnz_in[d+1], i -> :($(element(callee_in[d], i)) = $(element(offset_in[d+1], i)))))
        end
        push!(statements, :($callee(w)))
        for o in eachindex(callee_out)
            append!(statements, loop(switch.nnz_out[o], i -> :($(element(offset_out[o], i)) = $(element(callee_out[o], i)))))
        end
        Expr(:block, statements...)
    end
    chain = branch(switch.default)
    for k in length(switch.cases):-1:1
        chain = Expr(k == 1 ? :if : :elseif, :(trunc(selector) == $(k - 1)), branch(switch.cases[k]), chain)
    end
    (Any[:(selector = $(element(offset_in[1], 1))), chain], offset_in, offset_out)
end

# A map runs its body in one loop over the blocks: each turn copies one block of
# each input to the frame of the body, calls it, and copies one block of each output back.
function body!(em::Emission, map::MapCall, name::Symbol)
    offset_in, offset_out, _ = frame!(em, map, 0)
    callee, callee_in, callee_out = callee!(em, map.body)
    statements = Any[]
    for (d, n) in enumerate(map.body.nnz_in)
        append!(statements, loop(n, i -> :($(element(callee_in[d], i)) = w[$(offset_in[d]) + $n * k + $i])))
    end
    push!(statements, :($callee(w)))
    for (o, n) in enumerate(map.body.nnz_out)
        append!(statements, loop(n, i -> :(w[$(offset_out[o]) + $n * k + $i] = $(element(callee_out[o], i)))))
    end
    (Any[Expr(:for, :(k = 0:$(map.n - 1)), Expr(:block, statements...))], offset_in, offset_out)
end

"""
    emission(alg::MXAlgorithm, name, chunk_size, ref) -> (em, frame)

Emit the matrix algorithm `alg` with the entry point `name`. `em` holds the
sub-functions and the tables. `frame` is
`((offset_in, offset_out, total, direct), statements)`: the layout of the work vector
and the statements that run the algorithm in it (see `framed_body`). With `direct`,
the statements read argument `i` as `input_name(i)`, because they are not split into
sub-functions, which get the work vector only.
"""
function emission(alg::MXAlgorithm, name::Symbol, chunk_size::ChunkSize, ref)
    em = Emission(name, chunk_size, ref)
    direct = length(chunks(em, length(alg.instructions))) <= 1
    statements, offset_in, offset_out = body!(em, alg, name; direct)
    em, ((offset_in, offset_out, em.free[], direct), statements)
end

"""
    mx_definitions(alg, name, chunk_size, ref; julia = identity) -> (definitions, frame)

Return the named definitions that the matrix algorithm needs, and the frame of
the entry point (see `emission`). The definitions are the constant table and
the index table, when not empty, then each called function and sub-function.

`julia(f)` gives what the definitions hold in place of the Julia function `f`
of a callback: the function itself by default, or an expression that names it.
"""
function mx_definitions(alg::MXAlgorithm, name::Symbol, chunk_size::ChunkSize, ref; julia = identity)
    em, (layout, statements) = emission(alg, name, chunk_size, ref)
    sites = call_sites(last.(em.julia))
    functions = Dict{Symbol,Any}()
    for (placeholder, f) in em.julia
        functions[placeholder] = julia(sites[f])
    end
    first = Any[:($(julia(new_call!))()) for _ in first_statements(sites)]
    definitions = Expr[]
    for (table, values) in (em.constants_name => em.constants, em.indices_name => em.indices)
        isempty(values) || push!(definitions, Expr(:const, :($table = $(Expr(:vect, values...)))))
    end
    for (callee, body) in em.definitions
        push!(definitions, chunk_function(:($callee(w)), Any[substitute(s, functions) for s in body]))
    end
    definitions, (layout, Any[first..., (substitute(s, functions) for s in statements)...])
end

named_frame(alg::MXAlgorithm, name::Symbol, chunk_size::ChunkSize, ref; julia = identity) =
    mx_definitions(alg, name, chunk_size, ref; julia)[2]

function function_definitions(alg::MXAlgorithm, name::Symbol, chunk_size::ChunkSize, ref; key = identity, julia = identity)
    definitions, frame = mx_definitions(alg, name, chunk_size, ref; julia)
    push!(definitions, entry_function(alg, :($name(res, arg)), frame, ref, key(name)))
end

substitute(ex::Expr, values) = Expr(ex.head, (substitute(a, values) for a in ex.args)...)
substitute(s::Symbol, values) = get(values, s, s)
substitute(other, values) = other

"""
    runtime_frame(alg::MXAlgorithm, chunk_size, ref) -> frame

Return the frame of the entry point for runtime-generated functions. Each
called function is a runtime-generated function of its own, placed in the
statements of its callers, as are the constant table and the index table.
"""
function runtime_frame(alg::MXAlgorithm, chunk_size::ChunkSize, ref)
    em, (layout, statements) = emission(alg, :entry, chunk_size, ref)
    generate(ex) = RuntimeGeneratedFunction(@__MODULE__, CONTEXT[], ex)
    values = Dict{Symbol,Any}()
    values[em.constants_name], values[em.indices_name] = em.constants, em.indices
    sites = call_sites(last.(em.julia))
    for (placeholder, f) in em.julia
        values[placeholder] = sites[f]
    end
    for (callee, body) in em.definitions
        ex = chunk_function(:((w,)), Any[substitute(s, values) for s in body])
        values[callee] = RuntimeGeneratedFunctions.drop_expr(generate(ex))
    end
    (layout, Any[first_statements(sites)..., (substitute(s, values) for s in statements)...])
end
