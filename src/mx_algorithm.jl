"""
    MXInstruction(op, out, in, data)

One step of a matrix algorithm. A work slot holds the nonzeros of one matrix
value; slots are 1-based, and 0 in `out` marks a result that nothing reads.

| `op`           | `out`       | `in`          | `data`                                   |
|----------------|-------------|---------------|------------------------------------------|
| `:OP_INPUT`    | one slot    | none          | `(argument index, nonzero offset)`       |
| `:OP_OUTPUT`   | none        | one slot      | `(result index, nonzero offset)`         |
| `:OP_CONST`    | one slot    | none          | `Vector{Float64}` of nonzeros            |
| elementwise    | one slot    | one or two    | `nothing`                                |
| copy operation | one or more | one or more   | `maps`, see below                        |
| linear algebra | one slot    | one to four   | patterns and sizes, see `linear_data`    |
| indexed        | one slot    | two to four   | see `index_data`                         |
| with tables    | one slot    | one to three  | see `table_data`                         |
| passing        | one slot    | one or two    | see `pass_data`                          |
| `:OP_CALL`     | one or more | one or more   | the `Algorithm`, `MXAlgorithm`, `SwitchCall`, `MapCall` or `JuliaCall` called |

An elementwise operation is an entry of the scalar operation table, applied to
each nonzero; an operand with one nonzero is used for every element.

A copy operation (`COPY_OPERATIONS`: slices, joins, splits, transpose, reshape,
projection, assignment to nonzeros) moves nonzeros and computes nothing.
`maps[o][d][k]` is the nonzero of operand `d` that goes to nonzero `k` of
result `o`, or 0 for none. A nonzero that no operand feeds is zero, and a
nonzero that two operands feed is their sum.
"""
struct MXInstruction
    op::Symbol
    out::Vector{Int}
    in::Vector{Int}
    data::Any
end

"Operations of CasADi that only move nonzeros."
const COPY_OPERATIONS = Set([:OP_GETNONZEROS, :OP_SETNONZEROS, :OP_ADDNONZEROS, :OP_PROJECT,
    :OP_TRANSPOSE, :OP_RESHAPE, :OP_SPARSITY_CAST, :OP_VERTCAT, :OP_HORZCAT, :OP_DIAGCAT,
    :OP_VERTSPLIT, :OP_HORZSPLIT, :OP_DIAGSPLIT, :OP_HORZREPMAT, :OP_SUBREF, :OP_SUBASSIGN])

"Copy operations in which one result nonzero can be the sum of two operands."
const ADDING_OPERATIONS = Set([:OP_ADDNONZEROS])

"""
Operations of linear algebra, with their number of operands. `:OP_INVERSE` is
the matrix inverse; `:OP_INV` is the elementwise reciprocal.
"""
const LINEAR_OPERATIONS = Dict(:OP_MTIMES => 3, :OP_DOT => 2, :OP_BILIN => 3, :OP_RANK1 => 4,
    :OP_NORMF => 1, :OP_NORM1 => 1, :OP_NORMINF => 1, :OP_MMIN => 1, :OP_MMAX => 1,
    :OP_DETERMINANT => 1, :OP_INVERSE => 1, :OP_SOLVE => 2, :OP_KRON => 2, :OP_LOGSUMEXP => 1)

"""
Operations whose indices are values of the graph, with the operand counts that
each can have. `:OP_LOW` gives the place of each number in a rising grid. The
others read, write or add nonzeros at computed indices (see `index_data`).
"""
const INDEX_OPERATIONS = Dict(:OP_LOW => (2,), :OP_GETNONZEROS_PARAM => (2, 3),
    :OP_SETNONZEROS_PARAM => (3, 4), :OP_ADDNONZEROS_PARAM => (3, 4))

"""
Operations that need tables that the CasADi interface does not expose, with the
operand counts that each can have. `table_data` builds the tables, most of them
from the C code that CasADi generates for the node.
"""
const TABLE_OPERATIONS = Dict(:OP_BSPLINE => (1, 2), :OP_EINSTEIN => (3,), :OP_KRON_CONTRACT => (2,),
    :OP_HORZREPSUM => (1,), :OP_FIND => (1,), :OP_CONVEXIFY => (1,))

"""
Operations whose result is their first operand, with their operand count (see
`pass_data`).

| `op`            | effect                                                    |
|-----------------|-----------------------------------------------------------|
| `:OP_MONITOR`   | prints the operand                                        |
| `:OP_ASSERTION` | checks its second operand                                 |
| `:OP_DUMP`      | writes the operand to a file                              |
| `:OP_LIFT`      | none; the second operand is a guess that is not evaluated |
"""
const PASS_OPERATIONS = Dict(:OP_MONITOR => 1, :OP_ASSERTION => 2, :OP_DUMP => 1, :OP_LIFT => 2)

"""
    MXAlgorithm

The instruction list of a matrix (MX) CasADi function, copied into Julia
values. Work slot `s` holds `slot_nnz[s]` nonzeros.

Invariants (checked by the constructor):
- every work slot lies in `1:length(slot_nnz)` and is written before it is read;
- every `:OP_INPUT` reads a range of nonzeros that its argument has;
- the `:OP_OUTPUT` instructions of one result write its nonzeros once, in order;
- the size of each operand and result fits the operation;
- every operation is elementwise, a copy operation, linear algebra, an operation
  with indices or with tables, a passing operation or a call.
"""
struct MXAlgorithm
    name::String
    instructions::Vector{MXInstruction}
    slot_nnz::Vector{Int}
    nnz_in::Vector{Int}
    nnz_out::Vector{Int}
    sparsity_in::Vector{C.Sparsity}
    sparsity_out::Vector{C.Sparsity}
    names_in::Vector{String}
    names_out::Vector{String}
    function MXAlgorithm(name, instructions, slot_nnz, sparsity_in, sparsity_out, names_in, names_out)
        @requires length(names_in) == length(sparsity_in) "one name per input is required"
        @requires length(names_out) == length(sparsity_out) "one name per output is required"
        alg = new(String(name), collect(MXInstruction, instructions), collect(Int, slot_nnz),
                  Int[C.nnz(sp) for sp in sparsity_in], Int[C.nnz(sp) for sp in sparsity_out],
                  collect(C.Sparsity, sparsity_in), collect(C.Sparsity, sparsity_out),
                  collect(String, names_in), collect(String, names_out))
        check_algorithm(alg)
        alg
    end
end

"""
    check_algorithm(alg::MXAlgorithm) -> nothing

Verify the `MXAlgorithm` invariants; throw `ArgumentError` on the first violation.
"""
function check_algorithm(alg::MXAlgorithm)
    nslot = length(alg.slot_nnz)
    defined = falses(nslot)
    written = zeros(Int, length(alg.nnz_out))     # nonzeros of each result written so far
    readable(s) = 1 <= s <= nslot && defined[s]
    for (k, ins) in enumerate(alg.instructions)
        op = ins.op
        at = "instruction $k ($op)"
        @requires all(s -> 0 <= s <= nslot, ins.out) "$at: result slot out of range"
        @requires all(readable, ins.in) "$at: reads an undefined work slot"
        size_in = alg.slot_nnz[ins.in]
        size_out = [s == 0 ? -1 : alg.slot_nnz[s] for s in ins.out]
        if op === :OP_INPUT
            i, offset = ins.data
            @requires 1 <= i <= length(alg.nnz_in) "$at: no argument $i"
            @requires 0 <= offset && offset + size_out[1] <= alg.nnz_in[i] "$at: argument $i is too short"
        elseif op === :OP_OUTPUT
            j, offset = ins.data
            @requires 1 <= j <= length(alg.nnz_out) "$at: no result $j"
            @requires offset == written[j] "$at: result $j is not written in order"
            written[j] += size_in[1]
        elseif op === :OP_CONST
            @requires length(ins.data) == size_out[1] "$at: wrong number of values"
        elseif op === :OP_CALL
            @requires size_in == ins.data.nnz_in "$at: arguments do not fit \"$(ins.data.name)\""
            @requires all(size_out[o] in (-1, n) for (o, n) in enumerate(ins.data.nnz_out)) "$at: results do not fit \"$(ins.data.name)\""
        elseif op in COPY_OPERATIONS
            @requires length(ins.data) == length(ins.out) "$at: one group of maps per result is required"
            for (o, maps) in enumerate(ins.data)
                ins.out[o] == 0 && continue
                @requires length(maps) == length(ins.in) "$at: one map per operand is required"
                @requires all(length(m) == size_out[o] for m in maps) "$at: wrong map length"
                @requires all(all(i -> 0 <= i <= size_in[d], m) for (d, m) in enumerate(maps)) "$at: map out of range"
                # A result that shares its slot with an operand keeps those nonzeros in place.
                @requires all(ins.in[d] != ins.out[o] || all(i -> m[i] in (0, i), eachindex(m))
                              for (d, m) in enumerate(maps)) "$at: a result that overwrites its operand must keep the order"
            end
        elseif haskey(LINEAR_OPERATIONS, op)
            @requires length(ins.in) == LINEAR_OPERATIONS[op] "$at: wrong number of operands"
            @requires op in (:OP_MTIMES, :OP_RANK1, :OP_INVERSE, :OP_SOLVE, :OP_KRON) || size_out[1] == 1 "$at: the result must be a scalar"
            @requires op !== :OP_KRON || size_out[1] == length(ins.data.terms) == prod(size_in) "$at: one term per nonzero of the result is required"
            @requires !(op in (:OP_INVERSE, :OP_SOLVE)) || size_out[1] == ins.data.n * ins.data.nrhs "$at: the result must be dense"
        elseif haskey(INDEX_OPERATIONS, op)
            @requires length(ins.in) in INDEX_OPERATIONS[op] "$at: wrong number of operands"
            if op === :OP_LOW
                @requires size_in[1] >= 1 && size_out[1] == size_in[2] "$at: one place per number is required, in a grid that is not empty"
                @requires ins.data.mode in 0:2 "$at: the lookup mode must be 0, 1 or 2"
            else
                reads = op === :OP_GETNONZEROS_PARAM
                keys, data = size_in[(reads ? 2 : 3):end], ins.data
                @requires reads || size_out[1] == size_in[1] "$at: the result must have the size of the first operand"
                @requires length(data.offsets) == length(data.first) == (reads ? size_out[1] : size_in[2]) "$at: one index per nonzero is required"
                @requires all(i -> 1 <= i <= keys[1], data.first) "$at: index operand out of range"
                @requires length(keys) == 1 ? isempty(data.second) :
                          length(data.second) == length(data.first) && all(i -> 1 <= i <= keys[2], data.second) "$at: second index operand out of range"
            end
        elseif haskey(PASS_OPERATIONS, op)
            @requires length(ins.in) == PASS_OPERATIONS[op] && size_out[1] == size_in[1] "$at: the result must have the size of the first operand"
            @requires !(op in (:OP_MONITOR, :OP_ASSERTION)) || ins.data isa String "$at: a text is required"
            @requires op !== :OP_ASSERTION || size_in[2] == 1 "$at: the condition must be a scalar"
            @requires op !== :OP_DUMP || length(ins.data.rows) == length(ins.data.columns) == size_in[1] "$at: one place per nonzero is required"
        elseif haskey(TABLE_OPERATIONS, op)
            @requires length(ins.in) in TABLE_OPERATIONS[op] "$at: wrong number of operands"
            data = ins.data
            if op === :OP_BSPLINE
                @requires size_in[1] == data.n && size_out[1] == data.m "$at: one number per dimension and one result per value are required"
                @requires length(data.offsets) == data.n + 1 && length(data.degrees) == length(data.strides) == length(data.lookup) == data.n "$at: one entry per dimension is required"
                @requires data.strides[1] == data.m && all(d -> data.offsets[d+1] - data.offsets[d] >= 2 * data.degrees[d] + 2, 1:data.n) "$at: the knots do not fit the degrees"
                count = data.strides[end] * (data.offsets[end] - data.offsets[end-1] - data.degrees[end] - 1)
                @requires (data.coefficients === nothing ? size_in[2] : length(data.coefficients)) == count "$at: wrong number of coefficients"
            elseif op === :OP_EINSTEIN
                @requires size_out[1] == size_in[1] && all(t -> 1 <= t[1] <= size_out[1] && 1 <= t[2] <= size_in[2] && 1 <= t[3] <= size_in[3], data.terms) "$at: term out of range"
            elseif op === :OP_KRON_CONTRACT
                ma, na, mb, nb = data.dims
                @requires size_in[1] == ma * na * mb * nb && size_in[2] == (data.inner ? mb * nb : ma * na) &&
                          size_out[1] == (data.inner ? ma * na : mb * nb) "$at: the sizes do not fit the blocks"
            elseif op === :OP_CONVEXIFY
                @requires size_out[1] == data.table[3] && size_in[1] == (data.table[8] == 0 ? size_out[1] : data.table[8]) "$at: the nonzeros do not fit the pattern"
            elseif op === :OP_HORZREPSUM
                @requires size_in[1] == data.n * size_out[1] "$at: the operand must have the nonzeros of the result, repeated"
            else
                @requires size_out[1] == 1 && length(data.rows) == size_in[1] "$at: one row per nonzero and a scalar result are required"
            end
        else
            @requires haskey(OPERATIONS, op) "$at: unsupported operation"
            @requires length(ins.in) == OPERATIONS[op].arity "$at: wrong number of operands"
            @requires all(n -> n == 1 || n == size_out[1], size_in) "$at: operand sizes do not fit"
        end
        for s in ins.out
            s == 0 || (defined[s] = true)
        end
        # Loop invariant: `defined` marks exactly the slots written by instructions 1:k,
        # and `written[j]` counts the nonzeros of result j written by instructions 1:k.
    end
    @requires written == alg.nnz_out "function \"$(alg.name)\": not every result nonzero is written"
    nothing
end

isfull(sp::C.Sparsity) = C.nnz(sp) == C.size1(sp) * C.size2(sp)

"""
    positions(sp) -> (rows, columns)

Return the 1-based row and column of each nonzero of the sparsity pattern `sp`,
in nonzero order.
"""
function positions(sp::C.Sparsity)
    colind, rows = C.colind(sp), C.row(sp) .+ 1
    columns = [c for c in 1:C.size2(sp) for _ in colind[c]+1:colind[c+1]]
    rows, columns
end

"""
    kronecker_terms(z, x, y) -> Vector{NTuple{3,Int}}

Return the terms `(kz, kx, ky)` of the Kronecker product `z = kron(x, y)`:
nonzero `kz` of `z` is `x[kx] * y[ky]`. The arguments are sparsity patterns.

The terms are in the nonzero order of `z`: by column of `x`, then by column of
`y`, then by row of `x`, then by row of `y`.
"""
function kronecker_terms(z::C.Sparsity, x::C.Sparsity, y::C.Sparsity)
    (rows_x, columns_x), (rows_y, columns_y) = positions(x), positions(y)
    terms = NTuple{3,Int}[]
    for cx in 1:C.size2(x), cy in 1:C.size2(y), kx in findall(==(cx), columns_x), ky in findall(==(cy), columns_y)
        # Loop invariant: `terms` holds the nonzeros of the columns of `z` before this one, and of this one above this row.
        push!(terms, (length(terms) + 1, kx, ky))
    end
    rows, columns = positions(z)
    nr, nc = C.size1(y), C.size2(y)
    @ensures length(terms) == length(rows) && all(rows[kz] == (rows_x[kx] - 1) * nr + rows_y[ky] &&
                                                 columns[kz] == (columns_x[kx] - 1) * nc + columns_y[ky]
                                                 for (kz, kx, ky) in terms) "the terms must follow the pattern of the Kronecker product"
    terms
end

"""
    product_terms(z, x, y) -> Vector{NTuple{3,Int}}

Return the terms `(kz, kx, ky)` of the sparse product `z += x * y`: nonzero
`kz` of `z` receives `x[kx] * y[ky]`. The arguments are sparsity patterns.

The order is that of the CasADi routine `casadi_mtimes`: by column of `y`, then
by nonzero of `y`, then by nonzero of `x`. Each sum is then rounded as in CasADi.
"""
function product_terms(z::C.Sparsity, x::C.Sparsity, y::C.Sparsity)
    colind_z, row_z = C.colind(z), C.row(z)
    colind_x, row_x = C.colind(x), C.row(x)
    colind_y, row_y = C.colind(y), C.row(y)
    terms = NTuple{3,Int}[]
    for c in 1:C.size2(y)
        place = Dict(row_z[k] => k for k in colind_z[c]+1:colind_z[c+1])     # row => nonzero of z
        for ky in colind_y[c]+1:colind_y[c+1], kx in colind_x[row_y[ky]+1]+1:colind_x[row_y[ky]+2]
            kz = get(place, row_x[kx], 0)
            kz == 0 || push!(terms, (kz, kx, ky))
        end
    end
    terms
end

"""
    linear_data(op, node) -> NamedTuple

Return the data that the emitter needs for a linear algebra operation, read
from the sparsity patterns of the operands of `node`.

| `op`                          | fields                                                        |
|-------------------------------|---------------------------------------------------------------|
| `:OP_MTIMES`                  | `dims = (rows of x, columns of x, columns of y)` when all three operands are dense, else `terms` (see `product_terms`) |
| `:OP_BILIN`, `:OP_RANK1`      | `rows`, `columns` of each nonzero of the matrix, and `dims` when it is dense |
| `:OP_MMIN`, `:OP_MMAX`        | `dense`: a sparse operand has zeros that take part            |
| `:OP_KRON`                    | `terms` (see `kronecker_terms`)                               |
| `:OP_DETERMINANT`, `:OP_INVERSE`, `:OP_SOLVE` | `n`, and `place[k]`: the nonzero of the matrix at dense position `k`, or 0; `:OP_SOLVE` also has `nrhs`, `tr` (solve with the transpose) and `qr` (see `qr_tables`) |

CasADi itself ensures that the matrix of a determinant, inverse or solve is
square, and that the result of an inverse and the right side of a solve
are dense.
"""
function linear_data(op::Symbol, node)
    pattern(d) = C.sparsity(C.dep(node, d))
    if op === :OP_MTIMES
        z, x, y = pattern(0), pattern(1), pattern(2)
        all(isfull, (z, x, y)) ? (; dims = (C.size1(x), C.size2(x), C.size2(y)), terms = NTuple{3,Int}[]) :
                                 (; dims = nothing, terms = product_terms(z, x, y))
    elseif op in (:OP_BILIN, :OP_RANK1)
        A = pattern(0)
        rows, columns = positions(A)
        (; rows, columns, dims = isfull(A) ? (C.size1(A), C.size2(A)) : nothing)
    elseif op in (:OP_MMIN, :OP_MMAX)
        (; dense = isfull(pattern(0)))
    elseif op === :OP_KRON
        (; terms = kronecker_terms(C.sparsity(node), pattern(0), pattern(1)))
    elseif op in (:OP_DETERMINANT, :OP_INVERSE, :OP_SOLVE)
        A = pattern(op === :OP_SOLVE ? 1 : 0)
        n = C.size1(A)
        rows, columns = positions(A)
        place = zeros(Int, n * n)
        place[rows .+ n .* (columns .- 1)] = 1:length(rows)
        op === :OP_DETERMINANT && return (; n, place)
        nrhs = op === :OP_SOLVE ? C.size2(pattern(0)) : n
        (; n, place, nrhs, tr = op === :OP_SOLVE && C.info(node)["tr"], qr = op === :OP_SOLVE ? qr_tables(A) : nothing)
    else
        (;)
    end
end

"""
    whole(x) -> Int

Return the integer part of `x` as an index, as CasADi takes it. Return -1, which
is no index, when `x` is NaN or too large.
"""
whole(x) = abs(x) < 9.0e15 ? trunc(Int, x) : -1

"""
    index_model(op, data, values) -> Vector{Float64}

Return the result nonzeros of an indexed operation (see `index_data`). `values`
holds one vector of nonzeros per operand.

An index out of range reads NaN and writes nothing. This function is the
specification of the emitted code; `index_data` checks it against CasADi.
"""
function index_model(op::Symbol, data, values)
    reads = op === :OP_GETNONZEROS_PARAM
    keys = values[(reads ? 2 : 3):end]
    index(e) = data.offsets[e] + whole(keys[1][data.first[e]]) + (isempty(data.second) ? 0 : whole(keys[2][data.second[e]]))
    if reads
        source = values[1]
        return [0 <= index(e) < length(source) ? source[index(e)+1] : NaN for e in eachindex(data.offsets)]
    end
    result = copy(values[1])
    for e in eachindex(data.offsets)
        # Loop invariant: `result` holds the first operand with elements 1:e-1 of the second one written.
        i = index(e)
        0 <= i < length(result) || continue
        op === :OP_SETNONZEROS_PARAM ? (result[i+1] = values[2][e]) : (result[i+1] += values[2][e])
    end
    result
end

"The data of an indexed operation other than `:OP_LOW` (see `index_data`)."
const IndexForm = @NamedTuple{offsets::Vector{Int}, first::Vector{Int}, second::Vector{Int}}

"""
    index_data(op, node, slots) -> NamedTuple

Return the data that the emitter needs for an operation of `INDEX_OPERATIONS`.
`slots` gives the work slot of each operand.

For `:OP_LOW` the data is the lookup `mode` only (see the helper `low`). For
the others, element `e` of the result that is read, or of the operand that is
written, has the zero-based index `offsets[e] + k1[first[e]] + k2[second[e]]`.
`k1` and `k2` are the index operands; `second` is empty when there is only one.

CasADi gives an index as the sum of two parts, each an operand or a fixed
slice, and its interface does not say which. The function tries each form on
evaluations of the node, rebuilt on fresh symbols. It keeps the form that
reproduces them with `index_model`, and throws `ArgumentError` when none does.

CasADi 3.8.1 evaluates a write at a fixed slice plus indices wrongly, so that
node is read from its adjoint (see `slice_write_data`).
"""
function index_data(op::Symbol, node, slots)
    if op === :OP_LOW
        # CasADi does not keep the lookup mode when it rebuilds the node, so the C code is that of the node with what it depends on.
        source = c_source(C.Function("node", C.symvar(node), [node]))
        return (; mode = parse(Int, c_call(source, "low")[4]))
    end
    occursin("SetNonzerosSliceParam", C.class_name(node)) && return slice_write_data(op, node, slots)
    reads = op === :OP_GETNONZEROS_PARAM
    distinct = [d for d in eachindex(slots) if findfirst(==(slots[d]), slots) == d]
    same = [findfirst(==(s), slots) for s in slots]              # operand => the first operand with its slot
    sizes = [C.nnz(C.sparsity(C.dep(node, d - 1))) for d in eachindex(slots)]
    evaluate = rebuild(node, [C.dep(node, d - 1) for d in distinct])
    casadi(values) = only(evaluate(values[distinct]))
    range = sizes[1]                                             # the indices are those of the first operand
    keys = (reads ? 2 : 3):length(slots)
    count = reads ? C.nnz(C.sparsity(node)) : sizes[2]
    # Operands for an evaluation: distinct values that are no whole numbers, and indices `shift + 3j` in and near the range.
    function operands(shift)
        values = [[0.37 * k + 0.11 * d + 1.3 for k in 1:sizes[d]] for d in eachindex(slots)]
        for d in keys
            values[d] = shift === nothing ? zeros(sizes[d]) : [Float64(mod(3 * j + shift + d, range + 2) - 1) for j in 1:sizes[d]]
        end
        values[same]
    end
    forms = IndexForm[]
    if length(keys) == 2
        ni = sizes[keys[1]]
        push!(forms, (; offsets = zeros(Int, count), first = [mod1(e, ni) for e in 1:count], second = [(e - 1) ÷ ni + 1 for e in 1:count]))
    elseif sizes[keys[1]] > 0 && count % sizes[keys[1]] == 0
        nk = sizes[keys[1]]
        ns = count ÷ nk                                          # the length of the fixed slice
        inner, outer = [mod1(e, nk) for e in 1:count], [(e - 1) ÷ ns + 1 for e in 1:count]
        if reads
            # With all indices zero and the tags 1, 2, ... as values, each result is its offset plus one.
            tags = operands(nothing)
            tags[1] = collect(1.0:sizes[1])
            found = casadi(tags)
            if all(isfinite, found)
                offsets = round.(Int, found) .- 1
                push!(forms, (; offsets, first = inner, second = Int[]), (; offsets, first = outer, second = Int[]))
            end
        else
            # With all indices zero, the places that change are those of the slice.
            ones_in = operands(nothing)
            ones_in[1], ones_in[2] = zeros(sizes[1]), ones(sizes[2])
            touched = findall(!iszero, casadi(ones_in)) .- 1
            if length(touched) == ns
                for slice in (touched, reverse(touched))
                    push!(forms, (; offsets = [slice[(e - 1) ÷ nk + 1] for e in 1:count], first = inner, second = Int[]),
                                 (; offsets = [slice[mod1(e, ns)] for e in 1:count], first = outer, second = Int[]))
                end
            end
        end
    end
    probes = [operands(shift) for shift in (nothing, 0, 1, 2)]
    expected = map(casadi, probes)
    k = findfirst(form -> all(isequal(index_model(op, form, x), y) for (x, y) in zip(probes, expected)), forms)
    @requires k !== nothing "operation $op has indices of a form that this package cannot read"
    forms[k]
end

"""
    slice_write_data(op, node, slots) -> NamedTuple

Return the data of `index_data` for a write at a fixed slice plus index values.
Element `e` of the written operand goes to the index `slice[s] + k[j]`, with
`e = (j - 1) * length(slice) + s`.

CasADi 3.8.1 evaluates this node wrongly, with values from other memory, so the
function does not evaluate it. The reverse derivative of the node by the
written operand is a read at the same slice plus the same indices. CasADi
evaluates that read correctly, and the slice is read from it.
"""
function slice_write_data(op::Symbol, node, slots)
    refusal = "operation $op has indices of a form that this package cannot read"
    @requires length(slots) == 3 && allunique(slots) refusal
    f, place = rebuilt_function(node, [C.dep(node, d - 1) for d in 1:3])
    x = [MX.sym("x$i", C.sparsity_in(f, i - 1)) for i in 1:C.n_in(f)]
    seed = MX.sym("seed", C.sparsity_out(f, 0))
    read = C.Function("read", MX[seed; x], [Front.jtimes(f(x...), x[place[2]], seed, true) / 2])
    range, count, nk = C.nnz(C.sparsity_out(f, 0)), C.nnz(C.sparsity_in(f, place[2] - 1)), C.nnz(C.sparsity_in(f, place[3] - 1))
    @requires nk > 0 && count % nk == 0 refusal
    function casadi(tags, keys)
        values = [zeros(C.nnz(C.sparsity_in(read, i))) for i in 0:C.n_in(read)-1]
        values[1], values[1+place[3]] = tags, keys
        C.nonzeros(read((sparsity_values(C.sparsity_in(read, i - 1), v) for (i, v) in enumerate(values))...))
    end
    # With all indices zero and the tags 1, 2, ... as the seed, each element reads its offset plus one.
    found = casadi(collect(1.0:range), zeros(nk))
    @requires all(isfinite, found) refusal
    ns = count ÷ nk
    data = IndexForm((round.(Int, found) .- 1, [(e - 1) ÷ ns + 1 for e in 1:count], Int[]))
    tags = [0.37 * k + 1.3 for k in 1:range]
    @requires all(0:2) do shift
        keys = [Float64(mod(3 * j + shift, range + 2) - 1) for j in 1:nk]
        isequal(index_model(:OP_GETNONZEROS_PARAM, data, [tags, keys]), casadi(tags, keys))
    end refusal
    data
end

distinct_operands(node, slots) = [C.dep(node, d - 1) for d in eachindex(slots) if findfirst(==(slots[d]), slots) == d]

node_source(node, slots) = c_source(rebuilt_function(node, distinct_operands(node, slots))[1])

"""
    pass_data(op, node, slots) -> data

Return the data that the emitter needs for an operation of `PASS_OPERATIONS`.

| `op`            | data                                                                          |
|-----------------|-------------------------------------------------------------------------------|
| `:OP_MONITOR`   | the text that the monitor prints                                              |
| `:OP_ASSERTION` | the text of the failed assertion                                              |
| `:OP_DUMP`      | `base` (file name), `rows`, `columns`, `dims` (see the helper `write_matrix`) |
| `:OP_LIFT`      | `nothing`                                                                     |
"""
function pass_data(op::Symbol, node, slots)
    op === :OP_MONITOR && return monitor_text(node)
    op === :OP_ASSERTION && return assertion_text(node, slots)
    op === :OP_DUMP || return nothing
    rows, columns = positions(C.sparsity(node))
    base = match(r"snprintf\(dump_fname, \d+, \"\./(.*)\.%06d\.mtx\"", node_source(node, slots).body)[1]
    (; base = String(base), rows, columns, dims = (C.size1(node), C.size2(node)))
end

function monitor_text(node)
    operand = C.dep(node, 0)
    named = C.graph_substitute([node], [operand], [MX.sym("monitored", C.sparsity(operand))])[1]::MX
    text, head = C.str(named), "monitor(monitored, "
    startswith(text, head) ? String(chop(text; head = length(head), tail = 1)) : "monitor"
end

"""
    assertion_text(node, slots) -> String

Return the message of an assertion node. CasADi exposes it only in the error
of an evaluation with a false condition, so the node is evaluated at zeros.
"""
function assertion_text(node, slots)
    operands = distinct_operands(node, slots)
    evaluate = rebuild(node, operands)
    text = try evaluate([zeros(C.nnz(C.sparsity(x))) for x in operands]); "" catch err; sprint(showerror, err) end
    String(strip(last(split(text, "Assertion error:"; limit = 2))))
end

"""
    table_data(op, node, slots) -> NamedTuple

Return the data that the emitter needs for an operation of `TABLE_OPERATIONS`.
`slots` gives the work slot of each operand.

| `op`                | fields                                                                  |
|---------------------|-------------------------------------------------------------------------|
| `:OP_BSPLINE`       | see `bspline_data`                                                      |
| `:OP_EINSTEIN`      | `terms`: nonzero `kz` of the result receives `x[kx] * y[ky]`, in the order of CasADi |
| `:OP_KRON_CONTRACT` | `inner`, and `dims = (ma, na, mb, nb)`: `na` by `nb` blocks of `ma` by `mb` elements (see the helper `kron_inner!`) |
| `:OP_HORZREPSUM`    | `n`: the number of blocks that are added                                |
| `:OP_FIND`          | `rows`: the row of each nonzero, from zero, and `last`: the number of rows |
| `:OP_CONVEXIFY`     | see `convexify_data`                                                    |

The data of a B-spline and of a tensor product are checked against evaluations
of the node, rebuilt on fresh symbols.
"""
function table_data(op::Symbol, node, slots)
    if op === :OP_HORZREPSUM
        return (; n = C.nnz(C.sparsity(C.dep(node, 0))) ÷ max(C.nnz(C.sparsity(node)), 1))
    elseif op === :OP_FIND
        operand = C.sparsity(C.dep(node, 0))
        return (; rows = Vector{Int}(C.row(operand)), last = C.size1(operand))
    end
    source = node_source(node, slots)
    op === :OP_CONVEXIFY && return convexify_data(source)
    operands = distinct_operands(node, slots)
    sizes = [C.nnz(C.sparsity(x)) for x in operands]
    evaluate = rebuild(node, operands)
    if op === :OP_BSPLINE
        data = bspline_data(source)
        check_bspline(data, (x, c) -> only(evaluate(c === nothing ? Vector{Float64}[x] : Vector{Float64}[x, c])))
        data
    elseif op === :OP_EINSTEIN
        @requires allunique(slots) "operation $op: an operand that is used twice is not supported"
        terms = einstein_terms(source)
        values = [[0.37 * k + 0.11 * d + 1.3 for k in 1:n] for (d, n) in enumerate(sizes)]
        model = copy(values[1])
        foreach(((kz, kx, ky),) -> model[kz] += values[2][kx] * values[3][ky], terms)
        @ensures isapprox(model, only(evaluate(values)); rtol = 1e-12) "the terms of the tensor product must reproduce CasADi"
        (; terms)
    else
        inner = c_call(source, "kron_contract_inner_dense")
        a = inner === nothing ? c_call(source, "kron_contract_outer_dense") : inner
        # The routines take the sizes of the result first: those of a block of blocks (inner), or of a block (outer).
        first_dims, second_dims = (parse(Int, a[2]), parse(Int, a[3])), (parse(Int, a[5]), parse(Int, a[6]))
        (; inner = inner !== nothing, dims = inner !== nothing ? (first_dims..., second_dims...) : (second_dims..., first_dims...))
    end
end

"""
Strategies of the CasADi convexify node, which makes a matrix positive
definite, and the matrix parts that it accepts (`CONVEXIFY_PARTS`). Each tuple
is in the order of the CasADi numeric codes.
"""
const CONVEXIFY_STRATEGIES = ("CVX_REGULARIZE", "CVX_EIGEN_CLIP", "CVX_EIGEN_REFLECT")
const CONVEXIFY_PARTS = ("CVX_SYMM", "CVX_TRIL", "CVX_TRIU")

"""
    convexify_data(source) -> NamedTuple

Return the data of the one convexify node in the C code `source`, from the
configuration that CasADi writes before its call of `casadi_convexify_eval`.

- `margin`: the eigenvalue margin.
- `table`: the integers that the helper `convexify!` reads. The comment above
  its source in `HELPER_SOURCE` gives the layout.
- `scratch`: the work space that the helper needs.
"""
function convexify_data(source::CSource)
    config = Dict{String,String}(String(m[1]) => String(strip(m[2])) for m in eachmatch(r"cvx_config\.(\w+) = ([^;]+);", source.body))
    whole(name) = parse(Int, config[name])
    table(name) = get(source.ints, config[name], Int[])
    pattern, operand = table("Hsp"), table("Hrsp")
    ncol = pattern[2]
    colind, row = pattern[3:ncol+3], pattern[ncol+4:end]
    count = length(row)
    nblocks = max(whole("scc_offset_size") - 1, 0)
    offsets = nblocks == 0 ? [0] : table("scc_offset")
    mapping = whole("scc_transform") == 0 ? Int[] : table("scc_mapping")
    projection, nin = Int[], 0
    if whole("Hsp_project") != 0
        # The nonzero of the operand at the place of each nonzero of the result.
        columns, rows = operand[3:operand[2]+3], operand[operand[2]+4:end]
        nin = length(rows)
        place = Dict{Tuple{Int,Int},Int}((rows[k], c) => k for c in 0:operand[2]-1 for k in columns[c+1]+1:columns[c+2])
        projection = Int[get(place, (row[k], c), 0) for c in 0:ncol-1 for k in colind[c+1]+1:colind[c+2]]
    end
    strategy = findfirst(==(config["strategy"]), CONVEXIFY_STRATEGIES) - 1
    nmax = maximum(diff(offsets); init = 0)
    header = [strategy, findfirst(==(config["type_in"]), CONVEXIFY_PARTS) - 1, count, ncol, whole("scc_transform"), nblocks,
              whole("max_iter_eig"), nin, length(mapping), nmax]
    data = (; margin = c_real(config["margin"]), table = [header; colind; row; offsets; mapping; projection],
            scratch = nin + (strategy == 0 ? 0 : nmax^2 + 2 * max(nmax - 1, 0) * header[7] + 2 * nmax))
    @ensures length(data.table) == 10 + (ncol + 1) + count + (nblocks + 1) + length(mapping) + (nin == 0 ? 0 : count) "the tables of the node must be complete"
    data
end

"""
    bspline_data(source) -> NamedTuple

Return the tables of the one B-spline in the C code `source`, from the
arguments of its call of `casadi_nd_boor_eval`.

- `n`, `m`: the numbers of dimensions and of results.
- `knots`: the knots of all dimensions. Those of dimension `d` start after
  `knots[offsets[d]]`.
- `degrees`, `strides`, `lookup`: the degree, the step in the coefficients and
  the lookup mode of each dimension.
- `coefficients`: `nothing` when the coefficients are the second operand.
"""
function bspline_data(source::CSource)
    a = c_call(source, "nd_boor_eval")
    @requires a !== nothing "the C code of CasADi has no B-spline"
    n, m = parse(Int, a[2]), parse(Int, a[8])
    (; n, m, knots = source.reals[a[3]], offsets = source.ints[a[4]][1:n+1], degrees = source.ints[a[5]][1:n],
       strides = source.ints[a[6]][1:n], lookup = source.ints[a[10]][1:n], coefficients = get(source.reals, a[7], nothing))
end

bspline_table(data) = Int[data.n; data.m; data.offsets; data.degrees; data.strides; data.lookup]

bspline_scratch(data) = sum(data.degrees .+ 1) + maximum(data.degrees)

"""
    bspline_value(data, x, c) -> Vector{Float64}

Evaluate the B-spline with the tables `data` at the point `x`, with the helper
that the generated code calls. `c` holds the coefficients when they are an
operand.
"""
function bspline_value(data, x, c)
    coefficients = data.coefficients === nothing ? c : data.coefficients
    w = [Vector{Float64}(x); zeros((data.m + bspline_scratch(data))::Int)]
    Helpers.var"bspline!"(w, data.knots, 0, bspline_table(data), 0, coefficients, 0, 0, data.n, data.n + data.m)
    w[data.n+1:data.n+data.m]
end

"""
    check_bspline(data, casadi) -> nothing

Check that the tables of a B-spline reproduce CasADi; throw `AssertionError`
otherwise. `casadi(x, c)` returns the CasADi value at the point `x` for the
coefficients `c`, which are `nothing` when the tables hold them.

The points lie in a knot span, at the first and the last knot, and out of
range. A spline of degree zero is checked in a span only: at a knot CasADi
writes out of its work space, and the result is not defined.
"""
function check_bspline(data, casadi)
    span(d) = data.knots[data.offsets[d]+1:data.offsets[d+1]]
    count = data.strides[end] * (length(span(data.n)) - data.degrees[end] - 1)
    c = data.coefficients === nothing ? [sin(0.7 * k) + 0.1 * k for k in 1:count] : nothing
    inside(d, fraction) = (knots = span(d); knots[1] + fraction * (knots[end] - knots[1]))
    points = [[inside(d, fraction) for d in 1:data.n] for fraction in (0.37, 0.81, 0.0, 1.0, -0.2, 1.3)]
    for x in (all(>(0), data.degrees) ? points : points[1:2])
        @ensures isapprox(bspline_value(data, x, c)::Vector{Float64}, casadi(x, c); rtol = 1e-9, atol = 1e-12, nans = true) "the tables of the B-spline must reproduce CasADi at the point $x"
    end
    nothing
end

"""
    einstein_terms(source) -> Vector{NTuple{3,Int}}

Return the terms `(kz, kx, ky)` of the tensor product in the C code `source`:
nonzero `kz` of the result receives `x[kx] * y[ky]`.

CasADi writes one loop over all index values. Each index is a digit `j` of the
loop counter, and the place in each operand moves by `j` times a step. The
terms are in the order of that loop, so each sum is rounded as in CasADi.
"""
function einstein_terms(source::CSource)
    loop = match(r"for \(i=0; i<(\d+); \+\+i\) \{(.*?)\*rr \+= \*cr\*\*cs;"s, source.body)
    @requires loop !== nothing "the C code of CasADi has no tensor product"
    count, lines = parse(Int, loop[1]), strip.(split(loop[2], '\n'))
    # Each digit of the counter: its base, and the step of the result, of the first and of the second factor.
    digits = Vector{Int}[]
    for line in lines
        base = match(r"^j = k % (\d+);$", line)
        base === nothing || push!(digits, [parse(Int, base[1]), 0, 0, 0])
        step = match(r"^(rr|cr|cs) \+= j\*(\d+);$", line)
        step === nothing || (digits[end][findfirst(==(step[1]), ("rr", "cr", "cs"))+1] = parse(Int, step[2]))
    end
    map(0:count-1) do i
        place, k = [1, 1, 1], i
        for (base, steps...) in digits
            # Loop invariant: `place` holds the steps of the digits before this one, and `k` the digits that remain.
            place .+= (k % base) .* steps
            k ÷= base
        end
        (place[1], place[2], place[3])
    end
end

"""
    qr_tables(A) -> Vector{Int} or nothing

Return the index table of the CasADi sparse QR factorization for a square
matrix with the sparsity pattern `A`, in the layout that the helper
`qr_factor!` reads.

Return `nothing` when the dense factorization is the better one: the matrix is
dense or has fewer than 8 rows, has a structural rank defect, or has factors
with more than half the nonzeros of a dense matrix.
"""
function qr_tables(A::C.Sparsity)
    n = C.size1(A)
    (n >= 8 && !isfull(A)) || return nothing
    V, R, prinv, pc = C.qr_sparse(A)::Tuple{C.Sparsity,C.Sparsity,Vector,Vector}
    (C.size1(V) == n && 2 * (C.nnz(V) + C.nnz(R)) <= n^2) || return nothing
    vcat(Int[n, n, C.nnz(A), C.nnz(V), C.nnz(R)], C.colind(A), C.row(A), C.colind(V), C.row(V), C.colind(R), C.row(R),
         Vector{Int}(prinv), Vector{Int}(pc))
end

sparsity_values(sp::C.Sparsity, values) = C.nnz(sp) == 0 ? C.DM(sp) : C.DM(sp, C.DM(values))

"""
    rebuild(node, operands) -> evaluate

Rebuild `node` as a CasADi function of fresh symbols, one per operand.
`evaluate(values)` takes one vector of nonzeros per operand and returns one
vector of nonzeros per result of the node.

CasADi replaces a node only when the node is a step of the expression, so two
cases need care:
- An operand that is one result of a multi-output node (a split or a call) is
  not a step. That node is replaced as a whole, by a call to a function that
  returns its arguments.
- CasADi takes a result that is a join or a reshape apart before it looks at
  it. Each result is doubled, which keeps the node whole, and `evaluate`
  halves the numbers.
"""
function rebuild(node, operands)
    f, place = rebuilt_function(node, operands)
    patterns = [C.sparsity_in(f, i - 1) for i in 1:C.n_in(f)]
    function evaluate(values::Vector{Vector{Float64}})
        all = [zeros(C.nnz(sp)) for sp in patterns]
        all[place] = values
        r = f(map(sparsity_values, patterns, all)...)
        results = (r isa C.DM ? C.DM[r] : collect(C.DM, r))::Vector{C.DM}
        [C.nonzeros(x) ./ 2 for x in results]
    end
end

"""
    rebuilt_function(node, operands) -> (f, place)

Return the CasADi function behind `rebuild`. Its results are twice those of
`node`, and operand `d` is its input `place[d]`. When an operand is one result
of a multi-output node, the other results of that node are inputs too.
"""
function rebuilt_function(node, operands)
    inputs, from, to = MX[], MX[], MX[]
    place = zeros(Int, length(operands))          # operand => position in `inputs`
    parents = Dict{Int,Int}()                     # node with many results => position of its first result
    for (d, x) in enumerate(operands)
        if C.is_output(x)
            parent = C.dep(x, 0)
            first_result = get!(parents, C.__hash__(parent)) do
                patterns = [C.sparsity(C.get_output(parent, o)) for o in 0:C.n_out(parent)-1]
                arguments = [MX.sym("a$o", sp) for (o, sp) in enumerate(patterns)]
                identity = C.Function("identity", arguments, arguments, Dict("never_inline" => true))
                fresh = [MX.sym("r$o", sp) for (o, sp) in enumerate(patterns)]
                returned = identity(fresh...)
                push!(from, parent)
                # A result with no nonzeros is a constant and not a result of the call.
                listed = as_list(returned)
                push!(to, C.dep(listed[findfirst(C.is_output, listed)::Int], 0))
                append!(inputs, fresh)
                length(inputs) - length(fresh) + 1
            end
            place[d] = first_result + C.which_output(x)
        else
            push!(inputs, MX.sym("d$d", C.sparsity(x)))
            push!(from, x)
            push!(to, inputs[end])
            place[d] = length(inputs)
        end
    end
    results = C.n_out(node) == 1 ? [node] : [C.get_output(node, o) for o in 0:C.n_out(node)-1]
    (C.Function("node", inputs, C.graph_substitute([2 * r for r in results], from, to)), place)
end

"""
    copy_maps(node, slots, adding) -> maps

Return the maps of a copy operation (see `MXInstruction`), found by evaluation.
The node is rebuilt on fresh symbols, and each operand nonzero carries its own
number as its value, so every result nonzero tells where it came from.

`slots` gives the work slot of each operand. When one value is an operand more
than once, the maps name its first place only. With `adding`, a result nonzero
can be the sum of two operands, so each operand has an evaluation of its own.
The maps are checked on a second, random evaluation.
"""
function copy_maps(node, slots, adding::Bool)
    distinct = [d for d in eachindex(slots) if findfirst(==(slots[d]), slots) == d]
    operands = [C.dep(node, d - 1) for d in distinct]
    sizes = [C.nnz(C.sparsity(x)) for x in operands]
    evaluate = rebuild(node, operands)
    base = cumsum(sizes) .- sizes            # operand d carries the tags base[d]+1 : base[d]+sizes[d]
    maps = if adding
        # One evaluation per operand, because a result nonzero can hold the sum of two tags.
        columns = [evaluate([e == d ? collect(1.0:sizes[e]) : zeros(sizes[e]) for e in eachindex(operands)])
                   for d in eachindex(operands)]
        [[round.(Int, columns[d][o]) for d in eachindex(operands)] for o in eachindex(first(columns))]
    else
        tagged = evaluate([collect(base[d] .+ (1.0:sizes[d])) for d in eachindex(operands)])
        map(tagged) do tags
            found = [zeros(Int, length(tags)) for _ in operands]
            for (k, tag) in enumerate(tags)
                tag == 0 && continue
                d = searchsortedlast(base, tag - 1)
                found[d][k] = round(Int, tag) - base[d]
            end
            found
        end
    end
    values = [rand(n) .+ 1 for n in sizes]
    expected = evaluate(values)
    for o in eachindex(expected), k in eachindex(expected[o])
        actual = sum((m[k] == 0 ? 0.0 : values[d][m[k]] for (d, m) in enumerate(maps[o])); init = 0.0)
        @ensures actual == expected[o][k] "operation $(op_names()[C.op(node)]) is not a copy of nonzeros"
    end
    map(maps) do found
        every = [zeros(Int, length(first(found))) for _ in slots]
        every[distinct] = found
        every
    end
end

"""
    MXAlgorithm(f::CasADiCodegen.Function; expand = false)

Copy the instruction list of the matrix (MX) function `f`. A function that `f`
calls is copied once. With `expand = true`, a called MX function becomes a
scalar algorithm when CasADi can expand it.

`f` must be an MX function with no free symbols. Each node must be elementwise,
a call, or an operation of `COPY_OPERATIONS`, `LINEAR_OPERATIONS`,
`INDEX_OPERATIONS`, `TABLE_OPERATIONS` or `PASS_OPERATIONS`. A called function
that is neither SX nor MX must be one that `read_callee` replaces; `refuse`
throws for any other. A call of a matrix function that has more argument nonzeros
than instructions, such as an interpolant, is replaced by those instructions (see
`inline_calls`), because a call copies its arguments.
"""
MXAlgorithm(f::C.Function; expand::Bool = false) = read_mx(f, expand, Dict{Int,Any}())

# `callees` maps the identity of a CasADi function to its copy (see `callee_copy`).
function read_mx(f::C.Function, expand::Bool, callees::Dict{Int,Any})
    label = "function \"$(C.name(f))\""
    @requires C.is_a(f, "MXFunction") "$label is not a matrix (MX) function"
    @requires !C.has_free(f) "$label has free symbols: $(C.get_free(f))"
    names = op_names()
    n = C.n_instructions(f)
    instructions = Vector{MXInstruction}(undef, n)
    slot_nnz = Int[]
    inlined = Set{Int}()                         # the calls that are replaced by the instructions of the callee
    function write_slots(slots, sizes)
        for (s, size) in zip(slots, sizes)
            s == 0 && continue
            s > length(slot_nnz) && append!(slot_nnz, fill(-1, s - length(slot_nnz)))
            @ensures slot_nnz[s] in (-1, size) "$label: work slot $s changes its size"
            slot_nnz[s] = size
        end
    end
    for k in 1:n
        op = names[C.instruction_id(f, k - 1)]
        operands = C.instruction_input(f, k - 1) .+ 1
        out = C.instruction_output(f, k - 1) .+ 1
        if op === :OP_OUTPUT
            # CasADi writes the segments of one result in order, so the offset is a running sum.
            offset = sum((slot_nnz[ins.in[1]] for ins in @view(instructions[1:k-1])
                          if ins.op === :OP_OUTPUT && ins.data[1] == out[1]); init = 0)
            instructions[k] = MXInstruction(op, Int[], operands, (out[1], offset))
            continue
        end
        node = C.instruction_MX(f, k - 1)
        callee = op === :OP_CALL ? callee_copy(C.which_function(node), expand, callees) : nothing
        sizes = op === :OP_CALL ? callee.nnz_out :
                C.n_out(node) == 1 ? [C.nnz(C.sparsity(node))] :
                [C.nnz(C.sparsity(C.get_output(node, o))) for o in 0:C.n_out(node)-1]
        write_slots(out, sizes)
        instructions[k] = if op === :OP_INPUT
            info = C.info(node)
            MXInstruction(op, out, Int[], (info["ind"] + 1, info["offset"]))
        elseif op === :OP_CONST
            MXInstruction(op, out, Int[], Vector{Float64}(C.nonzeros(C.evalf(node))))
        elseif op === :OP_CALL
            callee isa MXAlgorithm && sum(callee.nnz_in; init = 0) > length(callee.instructions) && push!(inlined, k)
            MXInstruction(op, out, operands, callee)
        elseif op in COPY_OPERATIONS
            MXInstruction(op, out, operands, copy_maps(node, operands, op in ADDING_OPERATIONS))
        elseif haskey(LINEAR_OPERATIONS, op)
            MXInstruction(op, out, operands, linear_data(op, node))
        elseif haskey(INDEX_OPERATIONS, op)
            MXInstruction(op, out, operands, index_data(op, node, operands))
        elseif haskey(PASS_OPERATIONS, op)
            MXInstruction(op, out, operands, pass_data(op, node, operands))
        elseif haskey(TABLE_OPERATIONS, op)
            MXInstruction(op, out, operands, table_data(op, node, operands))
        else
            @requires haskey(OPERATIONS, op) && (C.is_unary(node) || C.is_binary(node)) "$label: unsupported operation $op at instruction $k"
            MXInstruction(op, out, operands, nothing)
        end
    end
    replace!(slot_nnz, -1 => 0)                  # a slot that no instruction writes
    alg = MXAlgorithm(C.name(f), instructions, slot_nnz,
                      [C.sparsity_in(f, j) for j in 0:C.n_in(f)-1],
                      [C.sparsity_out(f, j) for j in 0:C.n_out(f)-1],
                      [C.name_in(f, j) for j in 0:C.n_in(f)-1],
                      [C.name_out(f, j) for j in 0:C.n_out(f)-1])
    inline_calls(alg, inlined)
end

"""
    inline_calls(alg, calls) -> MXAlgorithm

Return `alg` with each call at a position in `calls`, which must call an
`MXAlgorithm`, replaced by the instructions of the called algorithm (see
`spliced`). The result computes the same values without a frame for the called
function, so that its arguments are not copied.
"""
function inline_calls(alg::MXAlgorithm, calls::Set{Int})
    isempty(calls) && return alg
    @requires all(k -> alg.instructions[k].op === :OP_CALL && alg.instructions[k].data isa MXAlgorithm, calls) "only a call of a matrix algorithm can be inlined"
    slot_nnz = copy(alg.slot_nnz)
    instructions = MXInstruction[]
    for (k, ins) in enumerate(alg.instructions)
        k in calls ? append!(instructions, spliced(ins, slot_nnz)) : push!(instructions, ins)
    end
    MXAlgorithm(alg.name, instructions, slot_nnz, alg.sparsity_in, alg.sparsity_out, alg.names_in, alg.names_out)
end

"""
    spliced(call, slot_nnz) -> Vector{MXInstruction}

Return the instructions of the algorithm that the instruction `call` calls, in
the work slots of the caller. The slots of the callee follow those in
`slot_nnz`, which gains them (mutated). An input of the callee is read in the
slot of the argument, without a copy. The results are copied to the result
slots of `call` after the last instruction, because a result slot can be that
of an argument which a later instruction of the callee reads.
"""
function spliced(call::MXInstruction, slot_nnz::Vector{Int})
    callee = call.data::MXAlgorithm
    base = length(slot_nnz)
    append!(slot_nnz, callee.slot_nnz)
    current = Dict{Int,Int}()                    # callee slot => caller slot that holds its value
    at(s) = get(current, s, base + s)
    last_write = zeros(Int, length(callee.slot_nnz))
    for (k, ins) in enumerate(callee.instructions), s in ins.out
        s == 0 || (last_write[s] = k)
    end
    identity_copy(to, from, n) = MXInstruction(:OP_GETNONZEROS, [to], [from], [[collect(1:n)]])
    segments = [Tuple{Int,Int}[] for _ in call.out]      # (offset, slot) of each part of each result
    result = MXInstruction[]
    for (k, ins) in enumerate(callee.instructions)
        if ins.op === :OP_INPUT
            (d, offset), s = ins.data, ins.out[1]
            if offset == 0 && callee.slot_nnz[s] == callee.nnz_in[d]
                current[s] = call.in[d]
            else
                delete!(current, s)
                push!(result, MXInstruction(:OP_GETNONZEROS, [base + s], [call.in[d]], [[collect(offset+1:offset+callee.slot_nnz[s])]]))
            end
        elseif ins.op === :OP_OUTPUT
            (j, offset), s = ins.data, ins.in[1]
            call.out[j] == 0 && continue
            source = at(s)
            if haskey(current, s) || last_write[s] > k
                # A slot of the caller, or one that the callee writes again: the result is kept in a slot of its own.
                push!(slot_nnz, callee.slot_nnz[s])
                push!(result, identity_copy(length(slot_nnz), source, callee.slot_nnz[s]))
                source = length(slot_nnz)
            end
            push!(segments[j], (offset, source))
        else
            push!(result, MXInstruction(ins.op, [s == 0 ? 0 : base + s for s in ins.out], [at(s) for s in ins.in], ins.data))
            foreach(s -> delete!(current, s), ins.out)
        end
        # Loop invariant: the value of callee slot s after instruction k is in caller slot `at(s)`.
    end
    for (j, slot) in enumerate(call.out)
        slot == 0 && continue
        n = slot_nnz[slot]
        maps = [[offset < i <= offset + slot_nnz[source] ? i - offset : 0 for i in 1:n] for (offset, source) in segments[j]]
        push!(result, MXInstruction(:OP_VERTCAT, [slot], [source for (_, source) in segments[j]], [maps]))
    end
    result
end

"""
Class names of the CasADi interpolants, and of the derivative of the linear one.
"""
const INTERPOLANT_CLASSES = ("BSplineInterpolant", "LinearInterpolant", "LinearInterpolantJac")

"""
Names of the CasADi lookup modes, in the order of their numeric codes (see the
helper `low`).
"""
const LOOKUP_MODES = ("linear", "exact", "binary")

"""
    interpolant_function(f) -> CasADiCodegen.Function

Return an MX function with the inputs and the results of the interpolant `f`,
made of nodes that this package has code for. A CasADi interpolant has no
instruction list, so its grid, values and lookup modes are read from the C
code that CasADi generates for it (see `c_source`). A grid or values that are
arguments of `f` stay arguments of the function that replaces it; CasADi has
no B-spline on a grid that is an argument.

| class of `f`           | the function that replaces it                                        |
|------------------------|----------------------------------------------------------------------|
| `BSplineInterpolant`   | a B-spline node; values that are an argument are first fitted, with the matrix of CasADi; with the option `inline`, MX code (see `inline_bspline`) |
| `LinearInterpolant`    | the linear interpolant of CasADi with the option `inline`           |
| `LinearInterpolantJac` | the Jacobian of that interpolant by the point                        |

The function is checked against `f` at points in and out of the grid. The
linear forms agree to rounding only, because CasADi computes the inline form
with other operations.
"""
function interpolant_function(f::C.Function)
    source, class, name = c_source(f), C.class_name(f), C.name(f)
    inputs = MX[MX.sym(C.name_in(f, i), C.sparsity_in(f, i)) for i in 0:C.n_in(f)-1]
    modes(lookup) = Dict{String,Any}("lookup_mode" => [LOOKUP_MODES[mode+1] for mode in lookup])
    split(values, offsets) = [values[offsets[d]+1:offsets[d+1]] for d in 1:length(offsets)-1]
    if class == "BSplineInterpolant" && c_call(source, "nd_boor_eval") === nothing
        grid, value = inline_bspline(f, inputs)
        results = MX[value]
    elseif class == "BSplineInterpolant"
        data = bspline_data(source)
        coefficients = data.coefficients === nothing ? fitted_coefficients(source, inputs[2], data.m) : MX(C.DM(data.coefficients))
        grid = split(data.knots, data.offsets)
        results = MX[C.bspline(inputs[1], coefficients, grid, data.degrees, data.m, modes(data.lookup))]
    else
        gradient = class == "LinearInterpolantJac"
        a = c_call(source, gradient ? "interpn_grad" : "interpn")
        @requires a !== nothing "the C code of CasADi has no linear interpolant for function \"$name\""
        n, m = parse(Int, a[2]), parse(Int, a[8])
        offsets = source.ints[a[4]][1:n+1]
        # A grid or values that are not a table of the C code are arguments of `f`, after the point.
        fixed_grid, fixed = get(source.reals, a[3], nothing), get(source.reals, a[5], nothing)
        grid = fixed_grid === nothing ? diff(offsets) : split(fixed_grid, offsets)
        table = C.interpolant(name, "linear", grid, fixed === nothing ? m : fixed, merge(modes(source.ints[a[7]][1:n]), Dict("inline" => true)))
        value = table(inputs[1:1+(fixed_grid === nothing)+(fixed === nothing)]...)
        # CasADi gives no derivative by the values, so each other result of the derivative is a structural zero.
        results = gradient ? MX[C.project(C.jacobian(value, inputs[1]), C.sparsity_out(f, 0)); [MX(C.sparsity_out(f, j)) for j in 1:C.n_out(f)-1]] : MX[value]
    end
    g = C.Function(name, inputs, results, [C.name_in(f, i) for i in 0:C.n_in(f)-1], [C.name_out(f, j) for j in 0:C.n_out(f)-1])
    # The check takes a grid argument of increasing points, 0, 1, 2, ... in each dimension.
    parametric_grid = grid isa Vector{Int}
    knots = parametric_grid ? [collect(0.0:length-1) for length in grid] : grid
    for fraction in (0.37, 0.81, 0.0, 1.0, -0.2, 1.3)
        values = [C.DM(C.sparsity_in(f, i), C.DM([sin(0.7 * k) + 0.1 * k for k in 1:C.nnz_in(f, i)])) for i in 0:C.n_in(f)-1]
        values[1] = C.DM([first(points) + fraction * (last(points) - first(points)) for points in knots])
        parametric_grid && (values[2] = C.DM(reduce(vcat, knots)))
        expected, actual = f(values...), g(values...)
        nonzeros(r) = reduce(vcat, (Vector{Float64}(C.nonzeros(C.densify(x))) for x in (r isa C.DM ? (r,) : r)))
        @ensures isapprox(nonzeros(actual), nonzeros(expected); rtol = 1e-7, atol = 1e-10, nans = true) "the function that replaces the interpolant \"$name\" must agree with it at the point $(Vector{Float64}(C.nonzeros(values[1])))"
    end
    g
end

"""
    inline_bspline(f, inputs) -> (grid, value)

Return the grid of the B-spline interpolant `f` made with the option `inline`, and its
value at the symbols `inputs`, as MX code without a B-spline node. CasADi writes such an
interpolant as inline code, which extends the polynomials of the first and the last
interval beyond the grid, whereas the B-spline node is zero there. The value of a
B-spline is linear in the values at the grid points, so that it equals the derivative
in the direction of those values, `J c`, which CasADi gives as MX code. The grid, the
values and the lookup modes come from the serial form of `f`, and the interpolant is
made again with the values as an argument and the default degree and algorithm;
`interpolant_function` checks the result against `f`, so that other options of `f`
give an error instead of another function.
"""
function inline_bspline(f::C.Function, inputs::Vector{MX})
    bytes = serial_bytes(f)
    points, offsets = Float64.(serial_value(bytes, "Interpolant::grid")), Int.(serial_value(bytes, "Interpolant::offset"))
    values, modes = serial_value(bytes, "Interpolant::values"), serial_value(bytes, "Interpolant::lookup_modes")
    m = Int(serial_value(bytes, "Interpolant::m"))
    grid = [points[offsets[d]+1:offsets[d+1]] for d in 1:length(offsets)-1]
    options = Dict{String,Any}("inline" => true)
    isempty(modes) || (options["lookup_mode"] = String.(modes))
    table = C.interpolant(C.name(f), "bspline", grid, m, options)
    c = MX.sym("c", prod(length, grid) * m)
    linear = C.Function("linear_" * C.name(f), [inputs[1], c], [C.jtimes(table(inputs[1], c), c, c)])
    # The graph of `linear` itself, not a call, so that the values are not copied to a frame.
    grid, only(as_list(C.call(linear, MX[inputs[1], isempty(values) ? inputs[2] : MX(C.DM(Float64.(values)))], true, false)))
end

"""
    fitted_coefficients(source, values, m) -> MX

Return the coefficients of the B-spline through `values`, which has `m`
numbers per grid point, as CasADi computes them. CasADi solves a linear system
with a fixed sparse matrix: one row per grid point and one column per basis
function. The matrix is read from the C code `source`, where CasADi solves the
system with `casadi_lsqr_solve`.
"""
function fitted_coefficients(source::CSource, values::MX, m::Integer)
    a = c_call(source, "lsqr_solve")
    @requires a !== nothing "the C code of CasADi has no fit of the B-spline coefficients"
    table = source.ints[a[5]]
    nrow, ncol = table[1], table[2]
    pattern = C.Sparsity(nrow, ncol, table[3:3+ncol], table[4+ncol:end])
    # The matrix is the table of real numbers that the statements give to the first argument of the routine.
    holder = match(Regex("\\b$(a[1]) = (\\w+);"), source.body)[1]
    matrix = source.reals[match(Regex("\\b$holder = (\\w+);"), source.body)[1]]
    fitted = C.solve(MX(C.DM(pattern, C.DM(matrix))), C.transpose(C.reshape(values, m, nrow)))
    C.vec(C.transpose(fitted))
end

"""
    forward_columns(f, inputs, results, directions) -> (inputs, results)

Return the inputs and the results of the CasADi function `f`, which is the
forward derivative in `directions` directions of the function with the symbols
`inputs` and the expressions `results`.

Each input and result of `f` holds, for each column of the original, that
column followed by its derivative in each direction. CasADi differentiates the
results.
"""
function forward_columns(f::C.Function, inputs::Vector{MX}, results::Vector{MX}, directions::Int)
    seeds = MX[MX.sym("d" * C.name_in(f, i - 1), C.sparsity(inputs[i])) for i in eachindex(inputs)]
    slope(value) = reduce(+, (C.jtimes(value, inputs[i], seeds[i]) for i in eachindex(inputs) if C.nnz(C.sparsity(inputs[i])) > 0);
                          init = MX(C.DM(zeros(size(value)...))))
    name = C.name(f)
    value_function = C.Function(name * "_value", inputs, results)
    slope_function = C.Function(name * "_slope", MX[inputs; seeds], MX[slope(r) for r in results])
    actual = MX[MX.sym(C.name_in(f, i), C.sparsity_in(f, i)) for i in 0:C.n_in(f)-1]
    width = 1 + directions
    # Direction `d` of an input, with `d = 0` for the values: one column of each group of `width` columns.
    chosen(i, d) = (n = size(inputs[i], 2); n == 0 || size(inputs[i], 1) == 0 ? inputs[i] * 0 : C.horzcat(MX[C.densify(actual[i])[:, (k-1)*width+d+1] for k in 1:n]))
    empty_safe(i, d) = C.nnz(C.sparsity(inputs[i])) == 0 ? MX(C.sparsity(inputs[i])) : chosen(i, d)
    point = MX[empty_safe(i, 0) for i in eachindex(inputs)]
    groups = [as_list(C.call(value_function, point, false, true))]
    for d in 1:directions
        # Loop invariant: `groups` holds the values and the derivatives in the directions before `d`.
        push!(groups, as_list(C.call(slope_function, MX[point; [empty_safe(i, d) for i in eachindex(inputs)]], false, true)))
    end
    joined(o) = C.nnz(C.sparsity_out(f, o - 1)) == 0 ? MX(C.sparsity_out(f, o - 1)) :
                C.project(C.horzcat(MX[C.densify(group[o])[:, k] for k in 1:size(results[o], 2) for group in groups]), C.sparsity_out(f, o - 1))
    actual, MX[joined(o) for o in eachindex(results)]
end

"""
    integrator_results(f, inputs, lists, count, nadj, outer) -> CasADiCodegen.Function

Return a function with the inputs and the results of the CasADi integrator
`f`, built from a written integration. `lists` holds the states, the algebraic
states and the quadratures at each of the `count` output times, as expressions
of the symbols `inputs`.

With `nadj > 0` the function also has the adjoint results: the reverse
derivative of the results by the initial state, the parameters and the
controls, once per direction. With `outer > 0` forward directions, which a
second-order derivative needs, the function is the forward derivative of that
one (see `forward_columns`); `inputs` then have the columns of one direction.
"""
function integrator_results(f::C.Function, inputs::Vector{MX}, lists, count::Int, nadj::Int, outer::Int)
    shape_out(o) = (C.size1_out(f, o), C.size2_out(f, o) ÷ (1 + outer))
    fitted(value, o) = outer == 0 ? C.project(C.reshape(value, shape_out(o)...), C.sparsity_out(f, o)) : C.reshape(value, shape_out(o)...)
    results = MX[fitted(C.vertcat(lists[o+1]), o) for o in 0:2]
    if nadj > 0
        # A seed has one column per output time and direction, and so have the results for the controls; the
        # others have one column per direction.
        value = C.vertcat(MX[C.vec(C.densify(r)) for r in results])
        empty(i) = C.nnz(C.sparsity(inputs[i])) == 0
        seed(a) = C.vertcat(MX[C.vec(C.horzcat(MX[C.densify(inputs[i])[:, (k-1)*nadj+a] for k in 1:count])) for i in 5:7 if !empty(i)])
        for (i, o) in ((1, 3), (2, 4), (3, 5), (4, 6))
            if empty(i)
                push!(results, outer == 0 ? MX(C.sparsity_out(f, o)) : MX(shape_out(o)...))
                continue
            end
            slopes = MX[C.densify(C.jtimes(value, inputs[i], seed(a), true)) for a in 1:nadj]
            columns = i == 4 ? MX[slopes[a][:, k] for k in 1:count for a in 1:nadj] : slopes
            push!(results, fitted(C.vec(C.horzcat(columns)), o))
        end
    else
        append!(results, MX[MX(C.sparsity_out(f, o)) for o in 3:6])
    end
    outer > 0 && ((inputs, results) = forward_columns(f, inputs, results, outer))
    g = C.Function(C.name(f), inputs, results, [C.name_in(f, i) for i in 0:C.n_in(f)-1], [C.name_out(f, o) for o in 0:C.n_out(f)-1])
    @ensures all(sparsity_tuple(C.sparsity_out(g, o)) == sparsity_tuple(C.sparsity_out(f, o)) for o in 0:C.n_out(f)-1) "the written integration must have the results of the integrator \"$(C.name(f))\""
    g
end

"""
Class names of the fixed-step CasADi integrators.
"""
const FIXED_STEP_CLASSES = ("RungeKutta", "Collocation")

"""
    fixed_step_function(f) -> CasADiCodegen.Function or nothing

Return an MX function with the inputs and the results of the fixed-step CasADi
integrator `f` (`rk` or `collocation`), or `nothing` for an integrator with
events. It ports the loop of CasADi 3.8.1 (`Integrator::eval` and
`FixedStepIntegrator::advance_noevent`): a `fold` over the CasADi step function
between two output times.

- For collocation the step function is a Newton rootfinder (see `NewtonCall`),
  which starts from the state at each collocation point.
- The start time, the output times and the number of steps between them are
  read from the serial form of `f` (see `serial_value`).
- With forward sensitivities the loop also calls the forward derivative of the
  step function, as CasADi does.
- With adjoint sensitivities the function has the reverse derivative of the
  written loop. With forward directions too, CasADi differentiates that
  function (see `forward_columns`).

The function is checked against `f` on one evaluation.
"""
function fixed_step_function(f::C.Function)
    bytes = serial_bytes(f)
    field(name) = serial_value(bytes, name)
    integer(name) = Int(field(name))::Int
    directions, nadj = integer("Integrator::nfwd"), integer("Integrator::nadj")
    field("Integrator::ne") == 0 || return nothing
    # An integrator with adjoint sensitivities and forward directions is the forward derivative of the one with no
    # direction: the loop is written for that one, and CasADi differentiates it (`outer` directions). Without
    # adjoint sensitivities the directions are in the loop, with the forward derivative of the step.
    outer, nfwd = nadj > 0 ? (directions, 0) : (0, directions)
    shape_in(i) = (C.size1_in(f, i), C.size2_in(f, i) ÷ (1 + outer))
    nx1, nz1, np1, nu1, nv1 = integer("Integrator::nx1"), integer("Integrator::nz1"), integer("Integrator::np1"), integer("Integrator::nu1"),
                              integer("FixedStepIntegrator::nv1")
    t0 = Float64(field("Integrator::t0"))::Float64
    times, disc = Vector{Float64}(field("Integrator::tout"))::Vector{Float64}, Vector{Int}(field("FixedStepIntegrator::disc"))::Vector{Int}
    step = C.get_function(f, "step")
    forward = nfwd > 0 ? C.get_function(f, "fwd$(nfwd)_step") : nothing
    inputs = MX[outer == 0 ? MX.sym(C.name_in(f, i), C.sparsity_in(f, i)) : MX.sym(C.name_in(f, i), shape_in(i)...) for i in 0:C.n_in(f)-1]
    dense(i) = C.vec(C.densify(inputs[i]))
    first_part(v, n) = v[1:n]::MX
    seeds(v, n) = C.reshape(v[n+1:n*(1+nfwd)]::MX, n, nfwd)
    x, p, controls = dense(1), dense(3), dense(4)
    q = MX(C.DM(zeros(C.nnz_out(step, 2) * (1 + nfwd))))
    # The collocation method starts its unknowns from the state at each point; the sensitivities of a first guess are not used.
    v = nv1 == 0 ? MX(0, 1) : C.vertcat(MX[C.repmat(C.vertcat(MX[first_part(x, nx1), first_part(dense(2), nz1)]), nv1 ÷ (nx1 + nz1), 1); MX(C.DM(fill(NaN, nv1 * nfwd)))])
    states, algebraic, quadratures = MX[], MX[], MX[]
    t, nu = t0, nu1 * (1 + nfwd)
    # One step with its sensitivities as a function of its own, so that the steps between two output times are a loop.
    # The state of the loop is one vector, because `fold` passes on its first argument only.
    nx, nv, nq = C.size1(x), C.size1(v), C.size1(q)
    one = let state = MX.sym("state", nx + nv + nq), t = MX.sym("t"), h = MX.sym("h"), p = MX.sym("p", C.size1(p)), u = MX.sym("u", nu)
        xs, vs, qs = state[1:nx]::MX, state[nx+1:nx+nv]::MX, state[nx+nv+1:nx+nv+nq]::MX
        point = MX[t, h, first_part(xs, nx1), first_part(vs, nv1), first_part(p, np1), first_part(u, nu1)]
        next = as_list(C.call(step, point, false, true))
        if forward !== nothing
            slopes = as_list(C.call(forward, MX[point; next; MX(1, 0); MX(1, 0); seeds(xs, nx1); seeds(vs, nv1); seeds(p, np1); seeds(u, nu1)], false, true))
            next = MX[C.vertcat(MX[C.vec(next[o]), C.vec(slopes[o])]) for o in 1:3]
        end
        C.Function("$(C.name(f))_step", MX[state, t, h, p, u], MX[C.vertcat(MX[C.vec(next[1]), C.vec(next[2]), qs + C.vec(next[3])])])
    end
    loops = Dict{Int,C.Function}()
    for (k, t_next) in enumerate(times)
        # Loop invariant: `x`, `v` and `q` are the values at the time `t`, and the lists hold the results at the output times before it.
        nj = disc[k+1] - disc[k]
        h, start = (t_next - t) / nj, t      # the comprehension below reads `start`: capturing `t` would box it
        u = controls[(k-1)*nu+1:k*nu]
        steps = get!(() -> nj == 1 ? one : C.fold(one, nj), loops, nj)
        each(value) = C.repmat(value, 1, nj)
        state = C.vec(only(as_list(C.call(steps, MX[C.vertcat(MX[x, v, q]), MX(C.DM(reshape([start + j * h for j in 0:nj-1], 1, nj))), each(MX(h)), each(p), each(u)], false, true))))
        x, v, q = state[1:nx], state[nx+1:nx+nv], state[nx+nv+1:nx+nv+nq]
        push!(states, x)
        push!(algebraic, C.vertcat(MX[v[(d+1)*nv1-nz1+1:(d+1)*nv1] for d in 0:nfwd]))
        push!(quadratures, q)
        t = t_next
    end
    g = integrator_results(f, inputs, [states, algebraic, quadratures], length(times), nadj, outer)
    values = C.DM[C.DM(C.sparsity_in(f, i), C.DM([0.37 + 0.11 * k / (1 + C.nnz_in(f, i)) for k in 1:C.nnz_in(f, i)])) for i in 0:C.n_in(f)-1]
    nonzeros(r) = reduce(vcat, (Vector{Float64}(C.nonzeros(C.densify(x))) for x in r))
    @ensures isapprox(nonzeros(g(values...)), nonzeros(f(values...)); rtol = 1e-9, atol = 1e-12, nans = true) "the written loop must agree with the integrator \"$(C.name(f))\" of CasADi"
    g
end

"""
    SwitchCall

A function that runs one of several functions, as the CasADi switch that
`if_else` and `conditional` make with `short_circuit`. The first input selects
`cases[k+1]` when its integer part is `k`, and `default` for any other value.
The other inputs and the outputs are those of the selected function.
"""
struct SwitchCall
    name::String
    cases::Vector{Any}
    default::Any
    nnz_in::Vector{Int}
    nnz_out::Vector{Int}
end

"""
    callee_copy(g, expand, callees) -> copy

Return the copy of the CasADi function `g` from the table `callees`; the first
request reads it (see `read_callee`).

The table keeps `g` with its copy, because the key is the identity of `g`.
CasADi gives the identity of a freed function to a later function, and some
functions in the table are made during the read and have no other owner.
"""
callee_copy(g::C.Function, expand::Bool, callees) = first(get!(() -> (read_callee(g, expand, callees), g), callees, C.__hash__(g)))

"""
    fitted(g, switch) -> CasADiCodegen.Function

Return the branch `g` of a switch with the sparsity patterns of the switch:
`g` itself when they agree, else a function that calls `g` and projects.
"""
function fitted(g::C.Function, switch::C.Function)
    agree = all(sparsity_tuple(C.sparsity_in(g, i)) == sparsity_tuple(C.sparsity_in(switch, i + 1)) for i in 0:C.n_in(g)-1) &&
            all(sparsity_tuple(C.sparsity_out(g, j)) == sparsity_tuple(C.sparsity_out(switch, j)) for j in 0:C.n_out(g)-1)
    agree && return g
    arguments = [MX.sym("i$i", C.sparsity_in(switch, i + 1)) for i in 0:C.n_in(g)-1]
    results = C.call(g, arguments, false, false)
    Front.Function(C.name(g), arguments, [C.project(results[j+1], C.sparsity_out(switch, j)) for j in 0:C.n_out(g)-1])
end

function read_switch(switch::C.Function, expand::Bool, callees)
    info, name = C.info(switch), C.name(switch)
    branch(g) = callee_copy(fitted(g, switch), expand, callees)
    cases, default = Any[branch(g) for g in info["f"]], branch(info["f_def"])
    nnz_in = [C.nnz_in(switch, i) for i in 0:C.n_in(switch)-1]
    nnz_out = [C.nnz_out(switch, j) for j in 0:C.n_out(switch)-1]
    for alg in [cases; default]
        @requires !(alg isa JuliaCall) "function \"$name\": a branch that is a Julia function itself is not supported; call it from a CasADi function"
    end
    SwitchCall(name, cases, default, nnz_in, nnz_out)
end

"""
    MapCall

A function that calls `body` on each of `n` blocks of columns of its inputs, as
the CasADi `map` (serial, on threads or with OpenMP). Block `k` of input `i`
is the nonzeros `(k - 1) * body.nnz_in[i] + 1` to `k * body.nnz_in[i]`, and
the same holds for the outputs. The generated code runs the blocks in one loop
in one thread.
"""
struct MapCall
    name::String
    body::Any
    n::Int
    nnz_in::Vector{Int}
    nnz_out::Vector{Int}
end

"""
    read_map(map, expand, callees) -> MapCall, MXAlgorithm or Algorithm

Return the copy of a CasADi map. A map of a Julia function, which has no frame
in the work vector, is unrolled instead: CasADi writes it again as a function
that calls the body once per block.
"""
function read_map(map::C.Function, expand::Bool, callees)
    info, name = C.info(map), C.name(map)
    f, n = info["f"]::C.Function, Int(info["n"])
    body = callee_copy(f, expand, callees)
    body isa JuliaCall && return read_callee(C.map(f, n, "unroll"), expand, callees)
    nnz_in = [C.nnz_in(map, i) for i in 0:C.n_in(map)-1]
    nnz_out = [C.nnz_out(map, j) for j in 0:C.n_out(map)-1]
    @ensures nnz_in == n .* body.nnz_in && nnz_out == n .* body.nnz_out "map \"$name\": each input and output must hold $n blocks of the body"
    MapCall(name, body, n, nnz_in, nnz_out)
end

function read_callee(callee::C.Function, expand::Bool, callees)
    C.is_a(callee, "SXFunction") && return Algorithm(callee)
    if C.class_name(callee) == "CallbackInternal"
        # A callback without a record was made with the type of the front end.
        haskey(JULIA_CALLS, C.__hash__(callee)) || refuse(callee)
        return JULIA_CALLS[C.__hash__(callee)][1]
    end
    C.class_name(callee) in ("Map", "ThreadMap", "OmpMap") && return read_map(callee, expand, callees)
    C.class_name(callee) == "Switch" && return read_switch(callee, expand, callees)
    C.class_name(callee) in INTERPOLANT_CLASSES && return read_callee(interpolant_function(callee), expand, callees)
    C.class_name(callee) == "External" && return external_call(callee)
    # A solver that the user selected has priority over the method of CasADi. Its reader calls the SciML
    # extension, so it runs behind a barrier: loading the extension does not invalidate this method.
    if C.class_name(callee) in ROOTFINDER_CLASSES || C.class_name(callee) in INTEGRATOR_CLASSES
        replaced = Base.inferencebarrier(C.class_name(callee) in ROOTFINDER_CLASSES ? solver_rootfinder : solver_integrator)(callee)
        replaced === nothing || return read_callee(replaced::C.Function, expand, callees)
    end
    if C.is_a(callee, "Nlpsol") || C.is_a(callee, "Conic")
        replaced = Base.inferencebarrier(C.is_a(callee, "Nlpsol") ? solver_optimization : solver_conic)(callee)
        replaced === nothing || return read_callee(replaced::C.Function, expand, callees)
    end
    C.class_name(callee) == "Newton" && return newton_call(callee)
    if C.class_name(callee) in FIXED_STEP_CLASSES
        written = fixed_step_function(callee)
        written === nothing && refuse(callee)
        return read_callee(written, expand, callees)
    end
    C.is_a(callee, "MXFunction") || refuse(callee)
    expanded = expand ? expanded_algorithm(callee) : nothing
    expanded === nothing ? read_mx(callee, expand, callees) : expanded
end
