# Julia functions as CasADi functions. The front end offers the type
# `CasadiCallback`; this file adds a method of `callback` that takes a plain
# Julia function, and a record of each such function, so that the emitter can
# replace the call through CasADi by a direct Julia call.

"""
    JuliaCall

The record of a Julia function that `callback` made a CasADi function. The emitter calls
`f` with the first `nargs` arguments of the CasADi function. An argument is a vector when
its shape is `(n,)` and a matrix when it is `(r, c)`. `f` returns one array, or a tuple
with one array per result.

A result with `nnz_out` 0 is a structural zero, which `f` does not return: the derivative
of a function has one for each input that is not an argument.
"""
struct JuliaCall
    name::String
    f::Any
    nargs::Int
    shapes_in::Vector{Tuple{Vararg{Int}}}
    shapes_out::Vector{Tuple{Vararg{Int}}}
    nnz_in::Vector{Int}
    nnz_out::Vector{Int}
end

"A callback that has a record: `call` is the record and `evaluate` takes every argument."
abstract type RecordedCallback <: Front.CasadiCallback end

# The CasADi side of a Julia function. `evaluate` takes every argument;
# `derivative` is `nothing`, or the Julia function that gives the Jacobian
# blocks with what gives the derivative of that function in turn: the rest of
# a chain of functions, or an ADTypes backend.
struct JuliaCallback <: RecordedCallback
    call::JuliaCall
    evaluate::Any
    derivative::Union{Nothing,Tuple{Any,Any}}
end

# Identity of a CasADi function => its record. The entries also keep the
# callback objects alive, which CasADi itself does not do.
const JULIA_CALLS = Dict{Int,Tuple{JuliaCall,Any}}()

shape(n::Integer) = (Int(n),)
shape(dims::Tuple{Integer}) = (Int(dims[1]),)
shape(dims::Tuple{Integer,Integer}) = (Int(dims[1]), Int(dims[2]))
shape(other) = throw(ArgumentError("a size must be an integer or a tuple (rows, columns), got $(repr(other))"))

dense(dims) = C.Sparsity.dense(dims[1], length(dims) == 2 ? dims[2] : 1)
pattern(dims, n) = n == 0 ? C.Sparsity(dims[1], length(dims) == 2 ? dims[2] : 1) : dense(dims)
shaped(values, dims) = length(dims) == 1 ? values : reshape(values, dims)
results(y) = y isa Tuple ? y : (y,)

Front.cb_n_in(cb::RecordedCallback) = length(cb.call.shapes_in)
Front.cb_n_out(cb::RecordedCallback) = length(cb.call.shapes_out)
Front.cb_sparsity_in(cb::RecordedCallback, i) = pattern(cb.call.shapes_in[i], cb.call.nnz_in[i])
Front.cb_sparsity_out(cb::RecordedCallback, i) = pattern(cb.call.shapes_out[i], cb.call.nnz_out[i])

function Front.cb_eval(cb::RecordedCallback, arg)
    call = cb.call
    # An input that is a structural zero has no values to shape; no function reads it.
    y = results(cb.evaluate((n == 0 ? Float64[] : shaped(C.nonzeros(a), dims)
                             for (a, dims, n) in zip(arg, call.shapes_in, call.nnz_in))...))
    returned = 0        # the function returns the results that are not structural zeros
    map(call.shapes_out, call.nnz_out) do dims, n
        n == 0 ? C.DM(pattern(dims, 0)) : C.DM(dense(dims), C.DM(Vector{Float64}(vec(y[returned += 1]))))
    end
end

C.Callback_has_jacobian(cb::JuliaCallback, args...) = cb.derivative !== nothing

# CasADi asks for the function (inputs..., results...) -> one Jacobian block
# per pair of result and input, ordered by result first. A block is a
# structural zero when the result is one or when the input is not an argument.
function C.Callback_get_jacobian(cb::JuliaCallback, name, names_in, names_out, options)
    call = cb.call
    pairs = [(o, i) for o in eachindex(call.shapes_out) for i in eachindex(call.shapes_in)]
    shapes = Tuple{Vararg{Int}}[(prod(call.shapes_out[o]), prod(call.shapes_in[i])) for (o, i) in pairs]
    nnz_out = [call.nnz_out[o] > 0 && i <= call.nargs ? prod(dims) : 0 for ((o, i), dims) in zip(pairs, shapes)]
    f, next = cb.derivative
    julia_function(f, String(name), call.nargs, vcat(call.shapes_in, call.shapes_out), shapes, next;
                   nnz_in = vcat(call.nnz_in, call.nnz_out), nnz_out)
end

"""
    derivative_function(jacobian, call) -> (f, next) or nothing

Return the Julia function `f` that gives the Jacobian blocks of `call` that are not
structural zeros, and `next`, which gives the derivative of `f` in the same way: the rest
of a chain of Julia functions, or the ADTypes backend itself.
"""
derivative_function(::Nothing, call::JuliaCall) = nothing
derivative_function(jacobian::Base.Callable, call::JuliaCall) = (jacobian, nothing)
function derivative_function(chain::Tuple, call::JuliaCall)
    @requires !isempty(chain) && all(f -> f isa Base.Callable, chain) "jacobian must be a Julia function, a tuple of Julia functions or an ADTypes.jl backend"
    (first(chain), length(chain) == 1 ? nothing : Base.tail(chain))
end
derivative_function(backend, call::JuliaCall) =
    (backend_jacobian(call.f, backend, call.shapes_in[1:call.nargs], filter(>(0), call.nnz_out)), backend)

"""
    backend_jacobian(f, backend, shapes_in, sizes_out) -> Julia function

Return the function `(arguments...) -> tuple of Jacobian blocks` of `f`, computed with the
automatic differentiation `backend` (an ADTypes.jl object). There is one block per pair of
result and argument, ordered by result first. `f` takes one array per shape of `shapes_in`
and returns one array per size of `sizes_out`.

The method is in the package extension that loads with DifferentiationInterface.jl.
"""
backend_jacobian(f, backend, shapes_in, sizes_out) =
    throw(ArgumentError("jacobian = $(repr(backend)) needs DifferentiationInterface.jl and an ADTypes.jl " *
                        "backend; load DifferentiationInterface, or give a Julia function"))

# Make the CasADi function and record it. `f` takes the first `nargs`
# arguments and returns the results whose `nnz_out` is not 0.
function julia_function(f, name::String, nargs::Int, shapes_in, shapes_out, jacobian;
                        nnz_in = map(prod, shapes_in), nnz_out = map(prod, shapes_out), options...)
    @requires Base.isidentifier(name) "\"$name\" is not a valid function name"
    call = JuliaCall(name, f, nargs, shapes_in, shapes_out, nnz_in, nnz_out)
    evaluate = (arguments...) -> f(arguments[1:nargs]...)
    proxy = Front.callback(JuliaCallback(call, evaluate, derivative_function(jacobian, call)); name, options...)
    proxy_function = Front.Function(proxy)
    JULIA_CALLS[C.__hash__(proxy_function)] = (call, proxy)
    proxy_function
end

"""
    callback(f, name, sizes_in, sizes_out; jacobian = nothing, options...) -> CasADiCodegen.Function

Make the Julia function `f` a CasADi function that a matrix (MX) graph can call. Code from
[`build_function`](@ref) and [`generate_julia`](@ref) calls `f` and its Jacobian directly,
with no CasADi in between.

# Arguments
- `f`: a function that takes one array per size of `sizes_in`. It returns one array, or a
  tuple with one array per size of `sizes_out`. A result can be any array with the right
  number of elements, for example a static array.
- `name`: the name of the CasADi function, a valid identifier.
- `sizes_in`, `sizes_out`: tuples with one size per argument and per result. A size is an
  integer `n` or a tuple `(n,)` for a vector, which `f` receives as a `Vector`, or a
  tuple `(rows, columns)` for a matrix.

# Keywords
- `jacobian = nothing`: how CasADi differentiates `f`.
    - `nothing`: no derivative.
    - A Julia function with the arguments of `f`. It returns the Jacobian matrix, or a
      tuple with one matrix per pair of result and argument, ordered by result first.
    - A tuple of such functions, for derivatives of higher order. Each function returns
      the Jacobian blocks of the function before it: one matrix per pair of a block and
      an argument, ordered by block first. A block counts as the vector of its elements
      in column-major order.
    - An ADTypes.jl backend, such as `AutoForwardDiff()`, for any order. Load
      DifferentiationInterface.jl and the package of the backend.
- `options...`: options of CasADi for the function, for example `enable_fd = true`.

`generate_julia` needs `f` and the Jacobian functions to be named functions of a module.
See [Call a Julia function from the model](@ref).

# Examples
```jldoctest
julia> using CasADiCodegen

julia> square(v) = v .^ 2;

julia> c = callback(square, "square", (2,), (2,); jacobian = v -> [2v[1] 0; 0 2v[2]]);

julia> v = MX.sym("v", 2);

julia> f = CasadiFunction("f", [v], [c(v), jacobian(c(v), v)]);

julia> g, g! = build_function(f; expression = Val{false});

julia> g([3.0, 4.0])
([9.0, 16.0], [6.0 0.0; 0.0 8.0])
```
"""
function Front.callback(f, name::AbstractString, sizes_in::Tuple, sizes_out::Tuple; jacobian = nothing, options...)
    shapes_in = Tuple{Vararg{Int}}[shape(s) for s in sizes_in]
    shapes_out = Tuple{Vararg{Int}}[shape(s) for s in sizes_out]
    julia_function(f, String(name), length(shapes_in), shapes_in, shapes_out, jacobian; options...)
end
