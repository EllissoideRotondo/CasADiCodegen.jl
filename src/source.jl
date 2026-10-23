sparsity_tuple(sp::C.Sparsity) = (C.size1(sp), C.size2(sp), C.colind(sp) .+ 1, C.row(sp) .+ 1)

# Print one function definition with a stable, readable layout.
function print_function(io::IO, definition::Expr)
    signature, body = definition.args
    println(io, "function ", signature)
    for statement in body.args
        if statement isa Expr && statement.head === :macrocall && statement.args[1] === Symbol("@inbounds")
            println(io, "    @inbounds begin")
            for inner in statement.args[end].args
                println(io, "        ", replace(string(inner), "\n" => "\n        "))
            end
            println(io, "    end")
        else
            println(io, "    ", replace(string(statement), "\n" => "\n    "))
        end
    end
    println(io, "end")
end

"""
    full_name(f, roots) -> Expr

Return the expression `Module.Submodule.name` that names the Julia function `f` in a source
file. The first module of the path is added to `roots`.
"""
function full_name(f, roots)
    @requires f isa Base.Function "a source file can call a named function of a module only, not $(repr(f))"
    owner, name = parentmodule(f), nameof(f)
    @requires isdefined(owner, name) && getfield(owner, name) === f "a source file can call a named function of a module only, not $(repr(f))"
    push!(roots, first(fullname(owner)))
    foldl((path, part) -> Expr(:., path, QuoteNode(part)), (fullname(owner)..., name))
end

"""
    Emitted

The generated function `f` of the CasADi function `model`, which a solver calls. `form` is
`:nonzeros` for the function on nonzeros, or `:outofplace` for the out-of-place function.
A source file gets the code of `model` again, under a name of its own.
"""
struct Emitted{F}
    f::F
    model::C.Function
    form::Symbol
end
(e::Emitted)(arguments...) = e.f(arguments...)

"""
    SourceWriter

What a source file collects for the Julia objects that its code calls.

- `roots`: the modules that the file imports by name.
- `required`: the modules that the file loads by their identity, because a package of the
  user need not depend on them.
- `definitions`: the definitions that come before the code of the function.
- `names`: the name of each object that has a definition.
"""
struct SourceWriter
    roots::Set{Symbol}
    required::Dict{Symbol,Base.PkgId}
    definitions::Vector{Any}
    names::IdDict{Any,Any}
    split::Split
    ref::Any
end

"Can a source file make this object again? See `object_source`."
writable(object) = false

"Compile what the object of a source file needs at its first call; nothing for most objects."
warm(object) = nothing

"""
    source(writer, f) -> expression

Return what a source file holds in place of the Julia object `f` that generated code calls:
the path of a named function, or the name of a definition that makes the object again (see
`object_source`). The writer keeps that definition. `f` must be a named function of a
module, or `writable(f)` must hold.
"""
function source(writer::SourceWriter, f)
    f isa Base.Function && return full_name(f, writer.roots)
    @requires writable(f) "a source file can call a named function of a module only, not $(repr(f))"
    get!(writer.names, f) do
        made = object_source(writer, f)
        name = Symbol(:object_, length(writer.names) + 1)
        push!(writer.definitions, Expr(:const, :($name = $made)))
        name
    end
end

function source(writer::SourceWriter, e::Emitted)
    get!(writer.names, e) do
        name = Symbol(:emitted_, length(writer.names) + 1, :!)
        alg = one_function(algorithm(e.model; expand = true), writer.split)
        julia(f) = source(writer, f)
        definitions = function_definitions(alg, core_name(name), writer.split, writer.ref; julia)
        e.form === :outofplace && append!(definitions, form_definitions(alg, name, writer.split, writer.ref, false; julia))
        append!(writer.definitions, definitions)
        e.form === :outofplace ? outofplace_name(name) : core_name(name)
    end
end

"The path of the module `m` in a source file, which then loads the package of `m`."
function module_path(writer::SourceWriter, m::Module)
    root = Base.moduleroot(m)
    id = Base.PkgId(root)
    if id.uuid === nothing
        root in (Base, Core) || push!(writer.roots, nameof(root))
    else
        writer.required[nameof(root)] = id
    end
    foldl((path, part) -> Expr(:., path, QuoteNode(part)), fullname(m))
end

function function_path(writer::SourceWriter, f)
    owner, name = parentmodule(f), nameof(f)
    @requires isdefined(owner, name) && getfield(owner, name) === f "a source file can call a named function of a module only, not $(repr(f))"
    Expr(:., module_path(writer, owner), QuoteNode(name))
end

"""
    type_source(writer, T) -> expression

Return the expression that names the concrete type `T` in a source file. The parameters of
`T` can be types, the types of named functions, symbols, numbers and tuples.
"""
function type_source(writer::SourceWriter, T)
    @requires T isa DataType "a source file cannot name the type $T"
    T <: Base.Function && isdefined(T, :instance) && return :(typeof($(function_path(writer, T.instance))))
    parameter(p) = p isa Type ? type_source(writer, p) : p isa Symbol ? QuoteNode(p) :
                   p isa Tuple ? Expr(:tuple, map(parameter, p)...) : p
    T <: Tuple && return Expr(:curly, :Tuple, map(parameter, T.parameters)...)
    owner, name = parentmodule(T), nameof(T)
    @requires isdefined(owner, name) && !startswith(String(name), "#") && all(p -> p isa Type || isbits(p) || p isa Symbol || p isa Tuple, T.parameters) "a source file cannot name the type $T"
    path = Expr(:., module_path(writer, owner), QuoteNode(name))
    isempty(T.parameters) ? path : Expr(:curly, path, map(parameter, T.parameters)...)
end

"""
    value_source(writer, value) -> expression

Return the expression that gives `value` again in a source file. A value can be:

- a number, a text, a symbol, a type, a module or a named function;
- a tuple, a named tuple or a vector of such values, or a `Dict` of such keys and values,
  as the options of a solver;
- an object with a definition (see `source`);
- an object of an immutable type whose fields are such values, as the algorithm of a solver.
"""
function value_source(writer::SourceWriter, value)
    value isa Union{Bool,Int,Float64,Float32,String,Nothing} && return value
    value isa Symbol && return QuoteNode(value)
    value isa Module && return module_path(writer, value)
    value isa Type && return type_source(writer, value)
    value isa Base.Function && return function_path(writer, value)
    value isa Tuple && return Expr(:tuple, (value_source(writer, x) for x in value)...)
    value isa NamedTuple &&
        return Expr(:tuple, Expr(:parameters, (Expr(:kw, key, value_source(writer, x)) for (key, x) in pairs(value))...))
    value isa Vector && return Expr(:ref, type_source(writer, eltype(value)), (value_source(writer, x) for x in value)...)
    # The entries in the order of their keys, so that the same Dict gives the same source.
    value isa Dict && return Expr(:call, type_source(writer, typeof(value)),
                                  (Expr(:call, :(=>), value_source(writer, k), value_source(writer, x)) for (k, x) in sort!(collect(value); by = repr ∘ first))...)
    (value isa Emitted || writable(value)) && return source(writer, value)
    T = typeof(value)
    @requires isstructtype(T) && !ismutabletype(T) "a source file cannot hold the value $(repr(value)) of the type $T"
    fields = (:(QuoteNode($(value_source(writer, getfield(value, k))))) for k in 1:fieldcount(T))
    :(eval(Expr(:new, $(type_source(writer, T)), $(fields...))))
end

"""
    generate_julia(f, path; module_name, force_SA = false, name = Symbol(name_of_f, "!"),
                   chunk_size = nothing, split_above = nothing, nanmath = true,
                   expand = true, solvers = nothing, optlevel = nothing) -> path
    generate_julia(f, io::IO; kwargs...) -> nothing

Write Julia source code for the CasADi function `f` to the file `path` or to the stream
`io`, as CasADi `generate` writes C code. The output is one module with the functions of
[`build_function`](@ref), which describes them.

The module holds:

- the in-place function `name`;
- the out-of-place function, whose name is `name` without `!`;
- the function on nonzeros, whose name is that name followed by `_nonzeros!`;
- the constants `NAME`, `NAMES_IN`, `NAMES_OUT`, `NNZ_IN`, `NNZ_OUT`, `NWORK`, `SPARSITY_IN`,
  `SPARSITY_OUT` and `DEPENDENCIES`. Each sparsity pattern is `(nrow, ncol, colptr, rowval)`,
  1-based. `NWORK` is the length of the work vector. `DEPENDENCIES` is the sorted tuple of
  the names of the packages that the module imports.

The module does not need CasADiCodegen.jl. A package that includes the file must have each
package of `DEPENDENCIES` as a dependency. The module imports:

- StaticArraysCore.jl, for the method with static arrays, unless `f` has no input;
- NaNMath.jl, when the code has an operation with a limited domain, such as `sqrt` or
  `log`, unless `nanmath = false`;
- SparseArrays.jl, when an input or an output has structural zeros, as most Jacobians have,
  for the methods with sparse matrices;
- SpecialFunctions.jl for `erfinv`, LinearAlgebra for the factorizations of matrix code,
  StaticArrays.jl and the package of each Julia function of a `callback`.

A Julia function of a `callback` must be a named function of a module. The module calls it
by its full name and imports its package.

A function with an integrator, a rootfinder or an optimization solver of the SciML
extension gives a module that makes the solvers again. That module loads CasADiCodegen.jl,
SciMLBase.jl and the packages of the algorithm, which must be installed.

# Keywords
- `module_name`: the name of the module, a valid identifier. The default is the name of
  `f` with its first letter in upper case.
- `optlevel = nothing`: with an integer from 0 to 3, the module sets its optimization
  level with `Base.Experimental.@optlevel`.
- The other keywords of the signature have the meaning that [`build_function`](@ref)
  gives them.

# Examples
```julia
generate_julia(f, "f.jl"; module_name = :F)
include("f.jl")
F.f([2.0, 3.0])         # the out-of-place function
```
"""
function generate_julia(f, io::IO;
                        module_name::Symbol = Symbol(uppercasefirst(function_name(f))),
                        force_SA::Bool = false, name::Symbol = default_name(f),
                        chunk_size::Union{Nothing,Integer} = nothing,
                        split_above::Union{Nothing,Integer} = nothing, nanmath::Bool = true,
                        expand::Bool = true, seen::Union{Nothing,Set{Symbol}} = nothing,
                        solvers::Union{Nothing,Solvers} = nothing, optlevel::Union{Nothing,Integer} = nothing)
    @requires Base.isidentifier(module_name) "\"$module_name\" is not a valid module name"
    check_optlevel(optlevel)
    chunk_size = Split(chunk_size, split_above)
    alg = one_function(algorithm(f; expand, solvers), chunk_size)
    used = Set{Tuple{Symbol,Symbol}}()
    source_ref(owner, fname) = (push!(used, (owner, fname));
                                owner === :Helpers ? fname : Expr(:., owner, QuoteNode(fname)))
    roots = Set{Symbol}()
    ref = nanmath ? source_ref : throwing(source_ref)
    writer = SourceWriter(roots, Dict{Symbol,Base.PkgId}(), Any[], IdDict{Any,Any}(), chunk_size, ref)
    julia(f) = source(writer, f)
    definitions = function_definitions(alg, core_name(name), chunk_size, ref; julia)
    append!(definitions, form_definitions(alg, name, chunk_size, ref, force_SA; julia))
    frame = named_frame(alg, core_name(name), chunk_size, ref; julia)
    nwork = frame === nothing ? 0 : frame[1][3]
    # `is_stored` and `PatternCheck` are defined with `stored`.
    helpers = sort!(unique!([fname in (:is_stored, :PatternCheck) ? :stored : fname for (owner, fname) in used if owner === :Helpers]))
    helpers = sort!(unique!([helpers; (needed for helper in helpers for needed in get(HELPER_NEEDS, helper, Symbol[]))...]))
    imports = Set(owner for (owner, _) in used if owner !== :Helpers)
    :erfinv in helpers && push!(imports, :SpecialFunctions)
    :stored in helpers && push!(imports, :SparseArrays)
    :static in helpers && push!(imports, :StaticArrays)   # a Julia function gets static arrays
    union!(imports, setdiff(roots, (:Main, :Base, :Core)))    # modules of the Julia functions of callbacks
    required = sort!([name => id for (name, id) in writer.required if !(name in imports)]; by = first)
    seen === nothing || union!(seen, imports, roots, keys(writer.required))    # for the cache, which must find the packages

    println(io, "# Generated by CasADiCodegen.jl from the CasADi function \"", alg.name, "\". Do not edit.")
    println(io, "module ", module_name, "\n")
    optlevel === nothing || println(io, "Base.Experimental.@optlevel ", optlevel, "\n")
    for owner in sort!(collect(imports))
        println(io, "import ", owner)
    end
    # A package that only the objects of a solver need is loaded by its identity: the user need not depend on it.
    for (name, id) in required
        println(io, "const ", name, " = Base.require(Base.PkgId(Base.UUID(\"", id.uuid, "\"), \"", id.name, "\"))")
    end
    isempty(imports) && isempty(required) || println(io)
    for helper in helpers
        println(io, HELPER_SOURCE[helper])
        helper === :stored && println(io, SPARSE_HELPER_SOURCE)
    end
    isempty(helpers) || println(io)
    println(io, "const NAME = ", repr(alg.name))
    println(io, "const NAMES_IN = ", repr(Tuple(alg.names_in)))
    println(io, "const NAMES_OUT = ", repr(Tuple(alg.names_out)))
    println(io, "const NNZ_IN = ", repr(Tuple(alg.nnz_in)))
    println(io, "const NNZ_OUT = ", repr(Tuple(alg.nnz_out)))
    println(io, "const NWORK = ", nwork)
    println(io, "const SPARSITY_IN = ", repr(Tuple(sparsity_tuple.(alg.sparsity_in))))
    println(io, "const SPARSITY_OUT = ", repr(Tuple(sparsity_tuple.(alg.sparsity_out))))
    println(io, "const DEPENDENCIES = ", repr(Tuple(sort!(collect(imports)))))
    # The objects of the solvers, which the cache on disk compiles (see `warm`).
    objects = sort!([name for name in values(writer.names) if startswith(String(name)::String, "object_")])
    isempty(objects) || push!(writer.definitions, Expr(:const, :(OBJECTS = $(Expr(:tuple, objects...)))))
    for definition in [writer.definitions; definitions]
        println(io)
        ex = definition::Expr
        ex.head === :function ? print_function(io, ex) : println(io, ex)
    end
    println(io, "\nend # module ", module_name)
    nothing
end

function generate_julia(f, path::AbstractString;
                        expand::Bool = true, solvers::Union{Nothing,Solvers} = nothing, kwargs...)
    alg = algorithm(f; expand, solvers)     # fail before the file is created
    open(io -> generate_julia(alg, io; kwargs...), path, "w")
    path
end
