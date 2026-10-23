# What CasADi does not tell through its interface, it writes in the C code that
# it generates: the knots of a B-spline, the grid of an interpolant, the index
# lists of a tensor product. This file reads those tables and the arguments of
# the calls to the CasADi runtime from that code, so that this package can
# emit Julia code for a node or a function that has no instruction list.

"""
    CSource

The C code that CasADi generates for one function: the statements of the
function (`body`), and its tables of integers (`ints`) and of real
numbers (`reals`), by name.
"""
struct CSource
    body::String
    ints::Dict{String,Vector{Int}}
    reals::Dict{String,Vector{Float64}}
end

"""
Parse a number of a real table. CasADi writes a C literal, or a `casadi_` name
for an infinite value or NaN.
"""
c_real(token) = parse(Float64, replace(lowercase(token), "casadi_" => ""))

"""
    c_source(f::CasADiCodegen.Function) -> CSource

Generate the C code of `f` and read its tables and its statements. CasADi must
be able to generate C code for `f`. CasADi leaves out the zeros at the end of a
table, so each table is padded to the length of its declaration.
"""
function c_source(f::C.Function)
    generator = C.CodeGenerator("code", Dict("with_header" => false))
    C.add(generator, f)
    text = C.dump(generator)
    ints, reals = Dict{String,Vector{Int}}(), Dict{String,Vector{Float64}}()
    for found in eachmatch(r"static const casadi_(int|real) (\w+)\[(\d+)\] =\s*\{([^}]*)\};", text)
        kind, name, count, list = found.captures
        tokens = filter(!isempty, strip.(split(list, ',')))
        values = kind == "int" ? parse.(Int, tokens) : c_real.(tokens)
        append!(values, zeros(eltype(values), parse(Int, count) - length(values)))
        kind == "int" ? (ints[name] = values) : (reals[name] = values)
    end
    # The entry point is the function `casadi_f0`; the functions that it calls follow it.
    start = findfirst("static int casadi_f0(", text)
    @requires start !== nothing "CasADi generated no C code for function \"$(C.name(f))\""
    stop = findnext("\n}\n", text, last(start))
    CSource(text[last(start):first(stop)], ints, reals)
end

"""
    c_call(source, routine) -> Vector{String} or nothing

Return the arguments of the last call of the CasADi runtime routine `routine`
(without the prefix `casadi_`) in the statements of `source`, or `nothing`
when the statements do not call it.
"""
function c_call(source::CSource, routine::AbstractString)::Union{Nothing,Vector{String}}
    found = collect(eachmatch(Regex("(?:casadi_|CASADI_PREFIX\\()$routine\\)?\\(([^;]*)\\);"), source.body))
    isempty(found) ? nothing : String.(strip.(split(last(found)[1], ',')))
end

# CasADi also writes each object in a serial form. With the option `debug` each
# field has its name before its value, so the options of a solver, which CasADi
# does not tell through its interface, are read from that form by name.

"""
    serial_bytes(f::CasADiCodegen.Function) -> Vector{UInt8}

Return the serial form of `f` with the names of its fields. CasADi gives it as
text, in which each byte is two letters, `a` to `p`, for its two halves.
"""
function serial_bytes(f::C.Function)
    code = codeunits(C.serialize(f, Dict("debug" => true)))
    UInt8[(code[k+1] - 0x61) << 4 | (code[k] - 0x61) & 0x0f for k in 1:2:length(code)-1]
end

"""
    serial_read(bytes, i) -> (value, next)

Return the value that starts at `bytes[i]` in a serial form, and the index
after it. A value starts with a letter for its type.

| letter | type                                   |
|--------|----------------------------------------|
| `d`    | `Float64`                              |
| `J`    | 8-byte integer                         |
| `i`    | 4-byte integer                         |
| `b`    | `Bool`                                 |
| `s`    | `String`: its length, then its bytes   |
| `V`    | vector: its length, then its elements  |
"""
function serial_read(bytes::Vector{UInt8}, i::Integer)
    tag = Char(bytes[i])
    tag == 'd' && return (reinterpret(Float64, bytes[i+1:i+8])[1], i + 9)
    tag == 'J' && return (Int(reinterpret(Int64, bytes[i+1:i+8])[1]), i + 9)
    tag == 'i' && return (Int(reinterpret(Int32, bytes[i+1:i+4])[1]), i + 5)
    tag == 'b' && return (bytes[i+1] != 0, i + 2)
    if tag == 's'
        n, start = serial_read(bytes, i + 1)
        return (String(bytes[start:start+n-1]), start + n)
    end
    @requires tag == 'V' "the serial form of CasADi has a value of the unknown type '$tag'"
    count, i = serial_read(bytes, i + 1)
    values = Any[]
    for _ in 1:count
        # Loop invariant: `values` holds the elements before `bytes[i]`.
        value, i = serial_read(bytes, i)
        push!(values, value)
    end
    (values, i)
end

"""
    serial_value(bytes, name; first = false) -> value

Return the value of the field `name` in the serial form `bytes` (see
`serial_bytes`). Throws `ArgumentError` when the form has no such field.

A function holds the functions that it uses, which can have the same field,
and writes the fields of its own class after them. The last field of that name
therefore belongs to the function itself. `first = true` selects the first
one, for a field of the base class, which comes before the other functions.
"""
function serial_value(bytes::Vector{UInt8}, name::AbstractString; first::Bool = false)
    key = [UInt8[0x73, 0x69]; Vector{UInt8}(reinterpret(UInt8, [Int32(ncodeunits(name))])); Vector{UInt8}(codeunits(name))]
    at = first ? findfirst(key, bytes) : findlast(key, bytes)
    @requires at !== nothing "CasADi does not write the field \"$name\""
    serial_read(bytes, last(at) + 1)[1]
end
