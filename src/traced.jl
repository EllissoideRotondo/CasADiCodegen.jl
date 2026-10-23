"""
    Traced{S}(value::S) <: Real

A real number that records the operations on it as a CasADi expression of type `S`, which
is `SX` or `MX`, so that generic Julia code traces into a CasADi graph.

Arithmetic, the elementary functions, `min`, `max`, `clamp`, `ifelse`, `floor`, `ceil`,
`trunc`, `rem`, `mod`, `copysign`, `cbrt` and `norm` give `Traced` values; a plain number
promotes to `Traced`. `det`, `inv` and `A \\ b` of a square matrix of `Traced` values are
formed by CasADi, since the factorizations of LinearAlgebra choose pivots by comparing
values. A comparison gives a traced condition, not a `Bool`, so an `if`, `&&` or `? :` on
traced values throws a `TypeError` instead of following one branch. Use `ifelse`, `min`, `max` or `clamp` for a choice that
depends on the input, or [`CasADiCodegen.callback`](@ref) for a Julia function with branches.

See [`CasADiCodegen.traced`](@ref traced) and [`CasADiCodegen.untraced`](@ref untraced).
"""
struct Traced{S<:Symbolic} <: Real
    value::S
    Traced{S}(value::S) where {S<:Symbolic} = new{S}(value)
end
Traced(value::S) where {S<:Symbolic} = Traced{S}(value)
Traced{S}(x::Traced{S}) where {S<:Symbolic} = x
Traced{S}(x::Real) where {S<:Symbolic} = Traced{S}(convert(S, Float64(x)))

Base.convert(::Type{Traced{S}}, x::Real) where {S<:Symbolic} = Traced{S}(x)
Base.convert(::Type{Traced{S}}, x::Traced{S}) where {S<:Symbolic} = x
Base.promote_rule(::Type{Traced{S}}, ::Type{<:Real}) where {S<:Symbolic} = Traced{S}

Base.zero(::Type{Traced{S}}) where {S<:Symbolic} = Traced{S}(0.0)
Base.one(::Type{Traced{S}}) where {S<:Symbolic} = Traced{S}(1.0)
Base.zero(x::Traced) = zero(typeof(x))
Base.one(x::Traced) = one(typeof(x))
Base.float(x::Traced) = x
Base.show(io::IO, x::Traced) = print(io, "Traced(", x.value, ")")

for f in (:+, :-, :*, :/, :^, :atan, :hypot, :min, :max, :&, :|)
    @eval Base.$f(a::Traced{S}, b::Traced{S}) where {S<:Symbolic} = Traced{S}($f(a.value, b.value))
end
for (f, c) in ((:<, :lt), (:<=, :le), (:(==), :eq))
    @eval Base.$f(a::Traced{S}, b::Traced{S}) where {S<:Symbolic} = Traced{S}(C.$c(a.value, b.value))
end
Base.:^(a::Traced{S}, n::Integer) where {S<:Symbolic} = Traced{S}(a.value^n)

for f in (:-, :sqrt, :exp, :log, :log2, :log10, :log1p, :expm1, :abs, :abs2, :inv, :!,
          :sin, :cos, :tan, :asin, :acos, :atan, :sinh, :cosh, :tanh, :asinh, :acosh, :atanh)
    @eval Base.$f(a::Traced{S}) where {S<:Symbolic} = Traced{S}($f(a.value))
end
Base.sincos(a::Traced) = (sin(a), cos(a))
Base.clamp(x::Traced, lo::Real, hi::Real) = min(max(x, lo), hi)
# A condition joins a plain `Bool`, as `(x > 0) & true` in generic code.
for f in (:&, :|)
    @eval Base.$f(a::Traced{S}, b::Real) where {S<:Symbolic} = $f(a, convert(Traced{S}, b))
    @eval Base.$f(a::Real, b::Traced{S}) where {S<:Symbolic} = $f(convert(Traced{S}, a), b)
end

# `floor`, `ceil` and `trunc` of a real number call `round` with a rounding mode.
Base.round(x::Traced{S}, ::RoundingMode{:Down}) where {S<:Symbolic} = Traced{S}(C.floor(x.value))
Base.round(x::Traced{S}, ::RoundingMode{:Up}) where {S<:Symbolic} = Traced{S}(C.ceil(x.value))
Base.round(x::Traced{S}, ::RoundingMode{:ToZero}) where {S<:Symbolic} =
    Traced{S}(C.if_else(C.lt(x.value, S(0.0)), C.ceil(x.value), C.floor(x.value)))
Base.rem(a::Traced{S}, b::Traced{S}) where {S<:Symbolic} = Traced{S}(C.fmod(a.value, b.value))
# As in Julia, the result of `mod` has the sign of `b`.
Base.mod(a::Traced{S}, b::Traced{S}) where {S<:Symbolic} = a - b * floor(a / b)
Base.copysign(a::Traced{S}, b::Traced{S}) where {S<:Symbolic} = Traced{S}(C.copysign(a.value, b.value))
Base.cbrt(a::Traced{S}) where {S<:Symbolic} = Traced{S}(C.sign(a.value) * C.power(C.abs(a.value), S(1 / 3)))

# The factorizations of LinearAlgebra choose pivots by comparing values, which a trace
# cannot do; CasADi forms the determinant, the inverse and the solution from the graph.
function casadi_matrix(A::AbstractMatrix{Traced{S}}) where {S<:Symbolic}
    C.reshape(C.vertcat(S[x.value for x in vec(collect(A))]), size(A, 1), size(A, 2))
end
casadi_column(::Type{S}, b::AbstractVector) where {S<:Symbolic} = C.vertcat(S[convert(Traced{S}, x).value for x in collect(b)])
square_traced(A) = (@requires size(A, 1) == size(A, 2) "the matrix must be square; got the size $(size(A))"; A)

LinearAlgebra.det(A::AbstractMatrix{Traced{S}}) where {S<:Symbolic} = Traced{S}(C.det(casadi_matrix(square_traced(A))))
function traced_inverse(A::AbstractMatrix{Traced{S}}) where {S<:Symbolic}
    n = size(square_traced(A), 1)
    reshape(traced(C.reshape(C.inv(casadi_matrix(A)), n * n, 1)), n, n)
end
# LinearAlgebra has a method for strided matrices of any element type.
LinearAlgebra.inv(A::AbstractMatrix{Traced{S}}) where {S<:Symbolic} = traced_inverse(A)
LinearAlgebra.inv(A::StridedMatrix{Traced{S}}) where {S<:Symbolic} = traced_inverse(A)
function Base.:\(A::AbstractMatrix{Traced{S}}, b::AbstractVector{<:Real}) where {S<:Symbolic}
    @requires size(square_traced(A), 1) == length(b) "the matrix has $(size(A, 1)) rows and the vector $(length(b)) elements"
    traced(C.solve(casadi_matrix(A), casadi_column(S, b)))
end

"""
    ifelse(condition::Traced, a, b)

Return the traced selection: `a` where `condition` is not zero, else `b`. Both values stay
in the graph, as with CasADi `if_else`, and the derivative is that of the selected value.
"""
Base.ifelse(condition::Traced{S}, a::Real, b::Real) where {S<:Symbolic} =
    Traced{S}(C.if_else(condition.value, convert(Traced{S}, a).value, convert(Traced{S}, b).value))

# The norms of LinearAlgebra and StaticArrays compare values to avoid overflow, which a trace cannot do.
LinearAlgebra.norm(v::AbstractArray{<:Traced}) = sqrt(sum(abs2, v))
LinearAlgebra.norm(v::StaticArrays.StaticArray{<:Tuple,<:Traced}) = sqrt(sum(abs2, v))

"""
    traced(x::Union{SX,MX}) -> Vector{Traced}

Return the elements of the symbolic column vector `x` as a vector of [`CasADiCodegen.Traced`](@ref Traced),
the argument of a generic Julia function to trace.

# Examples
```julia
using CasADiCodegen, LinearAlgebra

acceleration(r, μ) = -μ * r / norm(r)^3          # generic Julia code
r = SX.sym("r", 3)
a = acceleration(CasADiCodegen.traced(r), 398600.4418)
f = CasadiFunction("a", [r], [CasADiCodegen.untraced(a)])
g, g! = build_function(f; expression = Val{false})
```
"""
function traced(x::S) where {S<:Symbolic}
    @requires size(x, 2) == 1 "traced takes a column vector; got a matrix of size $(size(x))"
    Traced{S}[Traced{S}(x[i]) for i in 1:size(x, 1)]
end

"""
    untraced(v) -> SX or MX

Return the CasADi expression of the traced value `v`, or the column vector of the
expressions of the elements of the vector `v`. A plain number in `v` becomes a constant.
"""
untraced(x::Traced) = x.value
function untraced(v::AbstractVector)
    S = mapreduce(x -> x isa Traced ? typeof(x.value) : Union{}, typejoin, v; init = Union{})
    @requires S === SX || S === MX "untraced needs a vector with traced values of one type, SX or MX; got the element types $(unique(typeof.(v)))"
    column = C.vertcat(S[convert(Traced{S}, x).value for x in collect(v)])
    @ensures size(column) == (length(v), 1) "untraced must give one element per element of the vector"
    column
end

"""
    traced_function(name, f, u0, p = nothing; inplace = true, time = true, symbolic = SX,
                    states = nothing, parameters = nothing) -> CasADiCodegen.Function

Trace the Julia function `f` of a SciML problem into the CasADi function `name(u, p, t)`,
or `name(u, p)` with `time = false`, whose result is the derivative or the residual.

`f` is `f(du, u, p, t)` in place, or `f(u, p, t)` out of place with `inplace = false`, and
without `t` when `time = false`. It must be generic Julia code, as for
[`traced`](@ref traced). It gets the state as an array of [`Traced`](@ref Traced) values
with the shape of `u0`, and the parameters in the container of `p` with traced elements:
`p` may be a number, an array of numbers, a tuple or a named tuple of numbers, or
`nothing` for a problem without parameters. In place, `du`
starts at zero.

The arguments of the result are column vectors: `u` and `du` hold the elements in the
order of `vec(u0)`, and `p` those of `vec(p)`, or of the tuple, empty for `nothing`.
`states` and `parameters` name the scalar symbols of these elements, as vectors of
`Symbol` or `String`; the names become those of the variables of a SciML function made
from the result (see [`ODEFunction`](@ref SciMLBase.ODEFunction(::CasADiCodegen.Function))).
By default the parameters of a named tuple have its keys as names, and the other
elements are `u_0`, `u_1`, ... and `p_0`, `p_1`, ...
`symbolic` is `SX` or `MX`. A method for a SciML problem comes with SciMLBase.jl.

# Examples
```julia
using CasADiCodegen
import SciMLBase

decay!(du, u, p, t) = (du .= -p.k .* u)
f = CasADiCodegen.traced_function("decay", decay!, [1.0, 2.0], (k = 0.5,); states = [:x, :y])
problem = SciMLBase.ODEProblem(SciMLBase.ODEFunction(f), [1.0, 2.0], (0.0, 1.0), [0.5])
```
"""
function traced_function(name::AbstractString, f, u0::AbstractArray{<:Real}, p = nothing;
                         inplace::Bool = true, time::Bool = true, symbolic::Type{<:Symbolic} = SX,
                         states::Union{Nothing,AbstractVector} = nothing, parameters::Union{Nothing,AbstractVector} = nothing)
    @requires p === nothing || p isa Real || p isa AbstractArray{<:Real} || p isa Tuple{Vararg{Real}} || p isa NamedTuple{<:Any,<:Tuple{Vararg{Real}}} "traced_function: the parameters must be a number, an array, a tuple or a named tuple of numbers, or nothing; got $(typeof(p))"
    S = symbolic
    T = Traced{S}
    parameters = parameters === nothing && p isa NamedTuple ? collect(keys(p)) : parameters
    u = named_symbol(S, "u", length(u0), states, "states")
    q = named_symbol(S, "p", p === nothing ? 0 : length(p), parameters, "parameters")
    t = S.sym("t")
    given = String[string(x) for x in [something(states, String[]); something(parameters, String[])]]
    @requires allunique(given) && "t" ∉ given "traced_function: the names of the states and the parameters must be distinct and not \"t\"; got $given"
    state = reshape(traced(u), size(u0))
    elements = traced(q)
    parameter_values = p === nothing ? nothing : p isa Real ? only(elements) : p isa Tuple ? Tuple(elements) :
                       p isa NamedTuple ? NamedTuple{keys(p)}(Tuple(elements)) : reshape(elements, size(p))
    rest = time ? (parameter_values, only(traced(t))) : (parameter_values,)
    value = if inplace
        du = fill(zero(T), size(u0))
        f(du, state, rest...)
        du
    else
        f(state, rest...)
    end
    @requires length(value) == length(u0) "traced_function: \"$name\" must give one element per element of the state, $(length(u0)); got $(length(value))"
    result = untraced(T[convert(T, v) for v in vec(collect(value))])
    Front.Function(String(name), time ? [u, q, t] : [u, q], [result], time ? ["u", "p", "t"] : ["u", "p"], ["f"])
end

"The column vector of `n` symbols: `S.sym(default, n)`, or one scalar symbol per name of `names`."
function named_symbol(::Type{S}, default::String, n::Int, names, kind::String) where {S<:Symbolic}
    names === nothing && return S.sym(default, n)
    @requires length(names) == n "traced_function: $kind needs $n names, one per element; got $(length(names))"
    n == 0 ? S.sym(default, 0) : C.vertcat(S[S.sym(string(x)) for x in names])
end

"""
    traced_problem(name, problem; symbolic = SX, states = nothing, parameters = nothing, options...)

Return a SciML problem of generated code for the function of `problem`, an `ODEProblem` or
a `NonlinearProblem`, traced into CasADi. The methods are in the extension for SciMLBase.jl.
"""
function traced_problem end
